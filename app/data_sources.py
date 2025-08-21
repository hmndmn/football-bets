# app/data_sources.py
import os
import time
import math
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

"""
We fetch BOTH fixtures and odds from The Odds API so we don't depend on API-Football.
This file guarantees the following schemas:

fixtures DF columns:
  ["match_id", "league", "utc_kickoff", "home", "away"]

odds DF columns:
  ["match_id","league","utc_kickoff","market","selection","price","book","home","away"]

Notes:
- markets handled: 1X2 ("h2h"), Totals -> "OU{line}", Spreads -> "AH{handicap}"
- We keep the BEST price per (match_id, market, selection) later in markets.py, but here we emit all rows.
"""

# Map our human league names to The Odds API sport_keys
SPORT_KEY_BY_LEAGUE = {
    "EPL": "soccer_epl",
    "LaLiga": "soccer_spain_la_liga",
    "SerieA": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue1": "soccer_france_ligue_one",
    # add more here if needed
}

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

REGIONS = os.getenv("ODDS_REGIONS", "uk,eu,us").strip() or "uk,eu,us"
ODDS_MARKETS = "h2h,totals,spreads"  # 1X2, totals, (asian-like) spreads


def _now_utc():
    return datetime.now(timezone.utc)


def _within_hours(start_iso: str, hours_ahead: int) -> bool:
    try:
        dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return _now_utc() <= dt <= (_now_utc() + timedelta(hours=hours_ahead))
    except Exception:
        return True  # if parsing fails, don't filter out


def _safe_float(x):
    try:
        f = float(x)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return None


def _league_list(leagues):
    if not leagues:
        return []
    return [s.strip() for s in leagues if s and s.strip()]


def _fetch_odds_api_events(sport_key: str):
    """Call The Odds API once to get upcoming events with bookmakers/markets."""
    if not ODDS_API_KEY:
        return None, {"status": 401, "sample": {"message": "Missing ODDS_API_KEY"}, "headers": {}}

    url = (
        f"{ODDS_API_BASE}/sports/{sport_key}/odds"
        f"?apiKey={ODDS_API_KEY}"
        f"&regions={REGIONS}"
        f"&markets={ODDS_MARKETS}"
        f"&oddsFormat=decimal"
        f"&dateFormat=iso"
    )
    try:
        r = requests.get(url, timeout=25)
        hdr = {
            "x-requests-remaining": r.headers.get("x-requests-remaining", ""),
            "x-requests-used": r.headers.get("x-requests-used", ""),
        }
        if r.status_code != 200:
            return None, {"status": r.status_code, "sample": _try_json(r), "headers": hdr}
        return r.json(), {"status": r.status_code, "headers": hdr}
    except Exception as e:
        return None, {"status": 599, "sample": {"error": str(e)}, "headers": {}}


def _try_json(resp):
    try:
        return resp.json()
    except Exception:
        return {"text": resp.text[:500] if hasattr(resp, "text") else ""}


def fetch_fixtures(leagues, hours_ahead=168) -> pd.DataFrame:
    """
    Build fixtures from the Odds API event list.
    Returns DF with columns: match_id, league, utc_kickoff, home, away
    """
    leagues = _league_list(leagues)
    rows = []

    for lg in leagues:
        sport_key = SPORT_KEY_BY_LEAGUE.get(lg)
        if not sport_key:
            continue
        events, meta = _fetch_odds_api_events(sport_key)
        if not events or not isinstance(events, list):
            continue

        for ev in events:
            cid = ev.get("id", "")
            home = ev.get("home_team", "")
            away = ev.get("away_team", "")
            ko = ev.get("commence_time", "")  # ISO
            if not _within_hours(ko, hours_ahead):
                continue
            rows.append({
                "match_id": cid,
                "league": lg,
                "utc_kickoff": ko,
                "home": home,
                "away": away,
            })

    if not rows:
        return pd.DataFrame(columns=["match_id", "league", "utc_kickoff", "home", "away"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["match_id"]).reset_index(drop=True)
    return df


def fetch_odds(leagues) -> pd.DataFrame:
    """
    Flatten odds rows from the Odds API for the requested leagues.
    Output columns:
      match_id, league, utc_kickoff, market, selection, price, book, home, away
    """
    leagues = _league_list(leagues)
    out_rows = []

    for lg in leagues:
        sport_key = SPORT_KEY_BY_LEAGUE.get(lg)
        if not sport_key:
            continue
        events, meta = _fetch_odds_api_events(sport_key)
        if not events or not isinstance(events, list):
            continue

        for ev in events:
            cid = ev.get("id", "")
            home = ev.get("home_team", "")
            away = ev.get("away_team", "")
            ko = ev.get("commence_time", "")

            bms = ev.get("bookmakers", []) or []
            for bm in bms:
                book = bm.get("title") or bm.get("key") or ""
                mkts = bm.get("markets", []) or []
                for m in mkts:
                    mkey = (m.get("key") or "").lower()
                    outcomes = m.get("outcomes", []) or []
                    # 1) 1X2 ("h2h")
                    if mkey == "h2h":
                        # outcomes typically have names: home team, away team, Draw
                        # Map to selections "Home", "Away", "Draw" by comparing names.
                        for o in outcomes:
                            oname = str(o.get("name", ""))
                            price = _safe_float(o.get("price"))
                            sel = None
                            if oname == home:
                                sel = "Home"
                            elif oname == away:
                                sel = "Away"
                            elif oname.lower() == "draw":
                                sel = "Draw"
                            if sel and price:
                                out_rows.append({
                                    "match_id": cid,
                                    "league": lg,
                                    "utc_kickoff": ko,
                                    "market": "1X2",
                                    "selection": sel,
                                    "price": price,
                                    "book": book,
                                    "home": home,
                                    "away": away,
                                })
                    # 2) Totals -> "OU{line}" selection Over/Under
                    elif mkey == "totals":
                        for o in outcomes:
                            side = str(o.get("name","")).capitalize()  # "Over"/"Under"
                            point = _safe_float(o.get("point"))
                            price = _safe_float(o.get("price"))
                            if side in ("Over","Under") and point and price:
                                market = f"OU{point:g}"
                                out_rows.append({
                                    "match_id": cid,
                                    "league": lg,
                                    "utc_kickoff": ko,
                                    "market": market,
                                    "selection": side,
                                    "price": price,
                                    "book": book,
                                    "home": home,
                                    "away": away,
                                })
                    # 3) Spreads -> treat as Asian Handicap (from home perspective)
                    elif mkey == "spreads":
                        for o in outcomes:
                            oname = str(o.get("name",""))
                            point = _safe_float(o.get("point"))
                            price = _safe_float(o.get("price"))
                            if point is None or price is None:
                                continue
                            # Normalize to market "AH{point_from_home_perspective}"
                            # If outcome is home team, selection=Home, market AH{point}
                            # If outcome is away team, selection=Away, market AH{-point}
                            if oname == home:
                                market = f"AH{point:g}"
                                sel = "Home"
                            elif oname == away:
                                market = f"AH{(-point):g}"
                                sel = "Away"
                            else:
                                continue
                            out_rows.append({
                                "match_id": cid,
                                "league": lg,
                                "utc_kickoff": ko,
                                "market": market,
                                "selection": sel,
                                "price": price,
                                "book": book,
                                "home": home,
                                "away": away,
                            })
                    else:
                        # ignore other market types
                        pass

    if not out_rows:
        return pd.DataFrame(columns=[
            "match_id","league","utc_kickoff","market","selection","price","book","home","away"
        ])
    return pd.DataFrame(out_rows).reset_index(drop=True)
