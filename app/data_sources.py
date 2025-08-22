# app/data_sources.py
import os
import time
import datetime as dt
from typing import Dict, List, Tuple
import requests
import pandas as pd

API_BASE = "https://v3.football.api-sports.io"

def _now_utc():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def _season_for_date(d: dt.datetime) -> int:
    # Good enough for major European leagues
    return d.year

def _league_map_from_env() -> Dict[str, str]:
    """
    APIFOOTBALL_LEAGUES like: "EPL:39,LaLiga:140,SerieA:135,Bundesliga:78,Ligue1:61"
    """
    raw = os.getenv("APIFOOTBALL_LEAGUES", "")
    mp: Dict[str, str] = {}
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        if ":" in part:
            name, lid = part.split(":", 1)
            mp[name.strip()] = lid.strip()
    return mp

def _req_headers() -> Dict[str, str]:
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    return {"x-apisports-key": key} if key else {}

def _http_get_json(path: str, params: Dict) -> Tuple[int, dict, dict]:
    url = f"{API_BASE}{path}"
    r = requests.get(url, headers=_req_headers(), params=params, timeout=25)
    try:
        j = r.json()
    except Exception:
        j = {}
    return r.status_code, j, r.headers

def probe_apifootball(leagues: List[str], hours_ahead: int):
    """
    Ping fixtures for each league and return status/headers so we can see rate-limits.
    """
    out = {"ok": True, "headers": {}, "leagues_tried": [], "report": {}, "window": {}}
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key:
        out["ok"] = False
        out["error"] = "API_FOOTBALL_KEY missing"
        return out

    now = _now_utc()
    to_dt = now + dt.timedelta(hours=hours_ahead)
    from_s = now.strftime("%Y-%m-%d")
    to_s = to_dt.strftime("%Y-%m-%d")
    out["window"] = {"from": from_s, "to": to_s}

    league_map = _league_map_from_env()
    season = _season_for_date(now)

    for lname in leagues:
        lid = league_map.get(lname)
        if not lid:
            continue
        params = {
            "league": lid,
            "season": season,
            "from": from_s,
            "to": to_s,
            "timezone": "UTC",
        }
        status, j, hdrs = _http_get_json("/fixtures", params)
        out["leagues_tried"].append({lname: {"league_id": lid, "params": params}})
        out["headers"][lname] = {
            "status": status,
            "x-ratelimit-requests-limit": hdrs.get("x-ratelimit-requests-limit"),
            "x-ratelimit-requests-remaining": hdrs.get("x-ratelimit-requests-remaining"),
        }
        sample = None
        errors = j.get("errors") if isinstance(j, dict) else None
        resp = j.get("response") if isinstance(j, dict) else None
        if isinstance(resp, list) and resp:
            sample = resp[0]
        out["report"][lname] = {
            "status": status,
            "errors": errors or [],
            "first_item_sample": sample,
            "results_count": (j.get("results") if isinstance(j, dict) else None),
            "paging": j.get("paging") if isinstance(j, dict) else None,
        }
        # be nice to their API
        time.sleep(0.2)

    return out

def fetch_fixtures(leagues: List[str], hours_ahead: int) -> pd.DataFrame:
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key:
        # Return empty DF; let caller handle no-key gracefully.
        return pd.DataFrame(columns=["match_id","league","utc_kickoff","home","away"])

    now = _now_utc()
    to_dt = now + dt.timedelta(hours=hours_ahead)
    from_s = now.strftime("%Y-%m-%d")
    to_s = to_dt.strftime("%Y-%m-%d")
    season = _season_for_date(now)
    league_map = _league_map_from_env()

    rows = []
    for lname in leagues:
        lid = league_map.get(lname)
        if not lid:
            continue
        params = {
            "league": lid,
            "season": season,
            "from": from_s,
            "to": to_s,
            "timezone": "UTC",
        }
        status, j, hdrs = _http_get_json("/fixtures", params)
        if status != 200 or not isinstance(j, dict):
            continue
        for it in j.get("response", []) or []:
            fx = it.get("fixture", {})
            league = it.get("league", {})
            teams = it.get("teams", {})
            mid = fx.get("id")
            date = fx.get("date")  # e.g. "2025-08-22T19:00:00+00:00"
            tm_home = (teams.get("home") or {}).get("name")
            tm_away = (teams.get("away") or {}).get("name")
            if not (mid and date and tm_home and tm_away):
                continue
            # normalize to Z
            utc_kick = date.replace("+00:00", "Z")
            rows.append([
                str(mid),
                lname,
                utc_kick,
                tm_home,
                tm_away,
            ])
        time.sleep(0.2)

    df = pd.DataFrame(rows, columns=["match_id","league","utc_kickoff","home","away"])
    return df

def _parse_1x2(bet_obj) -> List[Tuple[str, float]]:
    """
    bet_obj e.g. {"name": "Match Winner", "values": [{"value":"Home","odd":"1.83"}, ...]}
    Returns [("Home",1.83), ("Draw",3.5), ("Away",4.6)]
    """
    out = []
    vals = bet_obj.get("values") or []
    for v in vals:
        sel = (v.get("value") or "").strip()
        odd = v.get("odd")
        try:
            price = float(str(odd))
        except Exception:
            continue
        # Normalize selection labels
        if sel.lower() in ("home", "1"):
            sel = "Home"
        elif sel.lower() in ("away", "2"):
            sel = "Away"
        elif sel.lower() in ("draw", "x"):
            sel = "Draw"
        out.append((sel, price))
    return out

def _parse_over_under(bet_obj) -> List[Tuple[str, str, float]]:
    """
    bet_obj name might contain 'Over/Under' or 'Goals Over/Under'.
    values maybe: [{"value":"Over 2.5","odd":"1.95"}, {"value":"Under 2.5","odd":"1.90"}]
    Returns list of (market_key, selection, price), e.g. ("OU2.5","Over",1.95)
    """
    out = []
    vals = bet_obj.get("values") or []
    for v in vals:
        label = (v.get("value") or "").strip()
        odd = v.get("odd")
        try:
            price = float(str(odd))
        except Exception:
            continue
        sel, line = None, None
        low = label.lower()
        if low.startswith("over "):
            sel = "Over"; line = label[5:].strip()
        elif low.startswith("under "):
            sel = "Under"; line = label[6:].strip()
        if sel and line:
            # keep plain "2.5", "3.25" -> market key OU2.5
            try:
                L = float(line)
                mkt = f"OU{L}".replace(".0", "")
                out.append((mkt, sel, price))
            except Exception:
                continue
    return out

def fetch_odds(fixtures_df: pd.DataFrame,
               leagues: List[str],
               hours_ahead: int,
               book_filter: str = "") -> pd.DataFrame:
    """
    Use API-Football odds endpoint for each fixture id.
    Return columns: match_id, league, utc_kickoff, market, selection, price, book, home, away
    """
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key or fixtures_df.empty:
        return pd.DataFrame(columns=["match_id","league","utc_kickoff","market","selection","price","book","home","away"])

    rows = []
    # Deduplicate fixtures to be safe
    fixtures = fixtures_df[["match_id","league","utc_kickoff","home","away"]].drop_duplicates()

    for _, fx in fixtures.iterrows():
        mid = fx["match_id"]
        params = {"fixture": mid}
        status, j, hdrs = _http_get_json("/odds", params)
        if status != 200 or not isinstance(j, dict):
            continue
        resp = j.get("response") or []
        for item in resp:
            # API-Football nests bookmakers inside item["bookmakers"]
            bms = item.get("bookmakers") or []
            for bm in bms:
                book_name = (bm.get("name") or "").strip()
                if book_filter and book_filter.lower() not in book_name.lower():
                    # if filtering by a book keyword
                    continue
                bets = bm.get("bets") or []
                for bet in bets:
                    bname = (bet.get("name") or "").lower()
                    if "match winner" in bname or bname == "1x2":
                        for sel, price in _parse_1x2(bet):
                            rows.append([
                                mid, fx["league"], fx["utc_kickoff"], "1X2", sel, price, book_name, fx["home"], fx["away"]
                            ])
                    elif "over/under" in bname:
                        for mkt, sel, price in _parse_over_under(bet):
                            rows.append([
                                mid, fx["league"], fx["utc_kickoff"], mkt, sel, price, book_name, fx["home"], fx["away"]
                            ])
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame(columns=["match_id","league","utc_kickoff","market","selection","price","book","home","away"])

    odds = pd.DataFrame(rows, columns=["match_id","league","utc_kickoff","market","selection","price","book","home","away"])

    # Keep best price per outcome per match/market
    odds = odds.sort_values("price", ascending=False).drop_duplicates(
        subset=["match_id","market","selection"], keep="first"
    )
    return odds
