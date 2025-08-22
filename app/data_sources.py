# app/data_sources.py
import os
import time
import datetime as dt
from typing import Dict, List, Tuple
import requests
import pandas as pd

UTC = dt.timezone.utc

# -----------------------
# Helpers
# -----------------------
def _now_utc() -> dt.datetime:
    return dt.datetime.now(tz=UTC)

def _iso(ts: dt.datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%d")

def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "y")

# -----------------------
# Public API (used by main.py)
# -----------------------
def fetch_fixtures(leagues: List[str], hours_ahead: int = 168) -> pd.DataFrame:
    """
    Returns DataFrame: [match_id, league, utc_kickoff, home, away]
    Chooses provider by ODDS_PROVIDER env.
    """
    provider = os.getenv("ODDS_PROVIDER", "oddsapi").lower()
    if provider == "apifootball":
        return _af_fetch_fixtures(leagues, hours_ahead)
    else:
        # We keep the old “events list” from The Odds API as a rough fixture source.
        return _toa_fetch_fixtures(leagues, hours_ahead)

def fetch_odds(
    leagues: List[str],
    fixtures_df: pd.DataFrame,
    hours_ahead: int = 168,
) -> pd.DataFrame:
    """
    Returns DataFrame: [match_id, league, utc_kickoff, market, selection, price, book, home, away]
    """
    provider = os.getenv("ODDS_PROVIDER", "oddsapi").lower()
    if provider == "apifootball":
        return _af_fetch_odds(leagues, fixtures_df)
    else:
        return _toa_fetch_odds(leagues, fixtures_df, hours_ahead)

# ==========================================================
# Provider: API-FOOTBALL
# Docs portal: https://www.api-football.com/documentation-v3
# We use /fixtures (by date range + league) and /odds (by fixture)
# ==========================================================

def _af_config() -> Tuple[str, Dict[str, int]]:
    api_key = os.getenv("APIFOOTBALL_KEY", "").strip()
    if not api_key:
        raise RuntimeError("APIFOOTBALL_KEY missing")
    # Map short names to league IDs
    raw = os.getenv(
        "APIFOOTBALL_LEAGUES",
        "EPL:39,LaLiga:140,SerieA:135,Bundesliga:78,Ligue1:61",
    )
    id_map: Dict[str, int] = {}
    for part in raw.split(","):
        k, v = [s.strip() for s in part.split(":")]
        id_map[k] = int(v)
    return api_key, id_map

def _af_fetch_fixtures(leagues: List[str], hours_ahead: int) -> pd.DataFrame:
    api_key, id_map = _af_config()
    base = "https://v3.football.api-sports.io"
    # Time window
    start = _now_utc()
    end = start + dt.timedelta(hours=hours_ahead)
    from_d = _iso(start)
    to_d = _iso(end)

    rows = []
    headers = {"x-apisports-key": api_key}

    for lg in leagues:
        if lg not in id_map:
            continue
        league_id = id_map[lg]
        # We call fixtures by league + date range (pre-match)
        # doc hub: fixtures endpoint supports date or between dates (from/to).
        params = {
            "league": league_id,
            "from": from_d,
            "to": to_d,
            "timezone": "UTC",
        }
        r = requests.get(f"{base}/fixtures", headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            continue
        js = r.json()
        for item in js.get("response", []):
            fid = str(item.get("fixture", {}).get("id"))
            date_iso = item.get("fixture", {}).get("date")  # ISO timestamp
            home = item.get("teams", {}).get("home", {}).get("name")
            away = item.get("teams", {}).get("away", {}).get("name")
            # Only upcoming/NS fixtures
            status_short = item.get("fixture", {}).get("status", {}).get("short", "")
            if status_short not in ("NS", "TBD", "PST"):  # not started
                continue
            rows.append(
                {
                    "match_id": fid,
                    "league": lg,
                    "utc_kickoff": date_iso,
                    "home": home,
                    "away": away,
                }
            )
        # be polite to rate limits
        time.sleep(0.15)

    df = pd.DataFrame(rows, columns=["match_id", "league", "utc_kickoff", "home", "away"])
    return df.dropna(subset=["match_id", "home", "away"]).drop_duplicates("match_id")

def _af_fetch_odds(leagues: List[str], fixtures_df: pd.DataFrame) -> pd.DataFrame:
    api_key, id_map = _af_config()
    base = "https://v3.football.api-sports.io"
    headers = {"x-apisports-key": api_key}

    # Markets we care about and their API-Football bet keys (common names).
    # API-Football standardizes bet names like:
    # - "Match Winner" (1X2)
    # - "Goals Over/Under" (totals)
    # - "Both Teams To Score"
    # - "Asian Handicap"
    wanted_bets = {
        "Match Winner": "1X2",
        "Goals Over/Under": "OU",
        "Both Teams To Score": "BTTS",
        "Asian Handicap": "AH",
    }

    out_rows = []

    # Build quick lookup for league by match_id
    league_by_id = {str(r.match_id): r.league for r in fixtures_df.itertuples()}

    for fid in fixtures_df["match_id"].tolist():
        params = {"fixture": fid, "bookmaker": "", "timezone": "UTC"}
        r = requests.get(f"{base}/odds", headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            time.sleep(0.2)
            continue
        js = r.json()
        for resp in js.get("response", []):
            # Structure: response[].bookmakers[].name, bets[].name, bets[].values[]
            for bm in resp.get("bookmakers", []):
                book = bm.get("name")
                for bet in bm.get("bets", []):
                    bet_name = bet.get("name", "")
                    if bet_name not in wanted_bets:
                        continue
                    market_code = wanted_bets[bet_name]

                    # Values structure differs by market; we normalize to (selection, price)
                    for v in bet.get("values", []):
                        label = (v.get("value") or v.get("odd") or "").strip()
                        # API returns price as string under "odd"
                        price_str = v.get("odd") or v.get("price") or ""
                        try:
                            price = float(price_str)
                        except Exception:
                            continue

                        selection = None
                        if bet_name == "Match Winner":
                            # labels typically "Home", "Draw", "Away" or "1", "X", "2"
                            lab = label.lower()
                            if lab in ("home", "1", "1 (home)"):
                                selection = "Home"
                            elif lab in ("draw", "x"):
                                selection = "Draw"
                            elif lab in ("away", "2", "2 (away)"):
                                selection = "Away"
                        elif bet_name == "Goals Over/Under":
                            # label like "Over 2.5" / "Under 2.5"
                            selection = label.title()  # keep "Over 2.5" etc
                            market_code = f"OU{_extract_total_line(label)}"
                        elif bet_name == "Both Teams To Score":
                            selection = "Yes" if label.lower().startswith("yes") else "No"
                        elif bet_name == "Asian Handicap":
                            # label example: "Home -0.25" / "Away +0.25"
                            selection = label  # keep original; we display it as-is

                        if not selection:
                            continue

                        out_rows.append(
                            {
                                "match_id": str(fid),
                                "league": league_by_id.get(str(fid), ""),
                                "utc_kickoff": resp.get("fixture", {}).get("date"),
                                "market": market_code,
                                "selection": selection,
                                "price": price,
                                "book": book,
                            }
                        )

        time.sleep(0.2)

    if not out_rows:
        return pd.DataFrame(
            columns=[
                "match_id",
                "league",
                "utc_kickoff",
                "market",
                "selection",
                "price",
                "book",
                "home",
                "away",
            ]
        )

    odds = pd.DataFrame(out_rows)
    # Attach home/away team names
    odds = odds.merge(
        fixtures_df[["match_id", "home", "away"]],
        on="match_id",
        how="left",
    )
    # Keep best (highest) price per (match, market, selection)
    odds.sort_values(["match_id", "market", "selection", "price"], ascending=[True, True, True, False], inplace=True)
    odds = odds.drop_duplicates(subset=["match_id", "market", "selection"], keep="first")
    return odds.reset_index(drop=True)

def _extract_total_line(label: str) -> str:
    # Turn "Over 2.5" / "Under 3" into "2.5" / "3"
    parts = label.strip().split()
    for p in parts:
        try:
            float(p.replace(",", "."))
            return p.replace(",", ".")
        except Exception:
            continue
    return "?"

# ==========================================================
# Provider: The Odds API (existing fallback)
# ==========================================================
def _toa_key() -> str:
    return os.getenv("ODDS_API_KEY", "").strip()

_TOA_LEAGUE_KEYS = {
    "EPL": "soccer_epl",
    "LaLiga": "soccer_spain_la_liga",
    "SerieA": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue1": "soccer_france_ligue_one",
}

def _toa_fetch_fixtures(leagues: List[str], hours_ahead: int) -> pd.DataFrame:
    api_key = _toa_key()
    if not api_key:
        return pd.DataFrame(columns=["match_id", "league", "utc_kickoff", "home", "away"])
    base = "https://api.the-odds-api.com/v4/sports"
    regions = os.getenv("ODDS_REGIONS", "uk,eu,us")
    end = _now_utc() + dt.timedelta(hours=hours_ahead)

    rows = []
    for lg in leagues:
        skey = _TOA_LEAGUE_KEYS.get(lg)
        if not skey:
            continue
        url = f"{base}/{skey}/odds"
        params = {
            "regions": regions,
            "markets": "h2h,totals,spreads",
            "oddsFormat": "decimal",
            "apiKey": api_key,
        }
        r = requests.get(url, params=params, timeout=25)
        if r.status_code != 200:
            continue
        for ev in r.json():
            utc = ev.get("commence_time")
            try:
                when = dt.datetime.fromisoformat(utc.replace("Z", "+00:00"))
            except Exception:
                continue
            if when > end:
                continue
            rows.append(
                {
                    "match_id": ev.get("id"),
                    "league": lg,
                    "utc_kickoff": utc,
                    "home": ev.get("home_team"),
                    "away": ev.get("away_team"),
                }
            )
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["match_id", "league", "utc_kickoff", "home", "away"])
    return df.drop_duplicates("match_id")

def _toa_fetch_odds(leagues: List[str], fixtures_df: pd.DataFrame, hours_ahead: int) -> pd.DataFrame:
    api_key = _toa_key()
    if not api_key or fixtures_df.empty:
        return pd.DataFrame(columns=["match_id","league","utc_kickoff","market","selection","price","book","home","away"])

    base = "https://api.the-odds-api.com/v4/sports"
    regions = os.getenv("ODDS_REGIONS", "uk,eu,us")

    out_rows = []
    for lg in leagues:
        skey = _TOA_LEAGUE_KEYS.get(lg)
        if not skey:
            continue
        url = f"{base}/{skey}/odds"
        params = {
            "regions": regions,
            "markets": "h2h,totals,spreads",
            "oddsFormat": "decimal",
            "apiKey": api_key,
        }
        r = requests.get(url, params=params, timeout=25)
        if r.status_code != 200:
            continue
        events = r.json()
        # Build quick lookup of fixture rows for this league
        ids_for_league = set(fixtures_df.loc[fixtures_df["league"] == lg, "match_id"].astype(str).tolist())
        for ev in events:
            mid = ev.get("id")
            if str(mid) not in ids_for_league:
                continue
            utc = ev.get("commence_time")
            home = ev.get("home_team")
            away = ev.get("away_team")
            for bm in ev.get("bookmakers", []):
                book = bm.get("title")
                for mk in bm.get("markets", []):
                    key = mk.get("key")
                    if key == "h2h":
                        mkt = "1X2"
                        for oc in mk.get("outcomes", []):
                            sel = oc.get("name")
                            price = oc.get("price")
                            if sel in (home,):
                                out_rows.append( _mk(mid, lg, utc, mkt, "Home", price, book, home, away) )
                            elif sel in (away,):
                                out_rows.append( _mk(mid, lg, utc, mkt, "Away", price, book, home, away) )
                            elif sel == "Draw":
                                out_rows.append( _mk(mid, lg, utc, mkt, "Draw", price, book, home, away) )
                    elif key == "totals":
                        # take a couple of common lines if present (2.5, 3.5)
                        for oc in mk.get("outcomes", []):
                            name = oc.get("name", "")
                            point = oc.get("point", None)
                            price = oc.get("price", None)
                            if point is None or price is None:
                                continue
                            try:
                                pstr = str(float(point))
                            except Exception:
                                pstr = str(point)
                            if name.lower().startswith("over"):
                                mkt = f"OU{pstr}"
                                out_rows.append( _mk(mid, lg, utc, mkt, f"Over {pstr}", price, book, home, away) )
                            elif name.lower().startswith("under"):
                                mkt = f"OU{pstr}"
                                out_rows.append( _mk(mid, lg, utc, mkt, f"Under {pstr}", price, book, home, away) )
                    elif key == "spreads":
                        for oc in mk.get("outcomes", []):
                            name = oc.get("name", "")
                            point = oc.get("point", None)
                            price = oc.get("price", None)
                            if point is None or price is None:
                                continue
                            # Normalize to "Home -0.25" / "Away +0.25"
                            side = "Home" if name == home else ("Away" if name == away else name)
                            sign = "+" if float(point) > 0 else ""
                            sel = f"{side} {sign}{point}"
                            out_rows.append( _mk(mid, lg, utc, "AH", sel, price, book, home, away) )
        time.sleep(0.25)

    if not out_rows:
        return pd.DataFrame(columns=["match_id","league","utc_kickoff","market","selection","price","book","home","away"])
    odds = pd.DataFrame(out_rows)
    # keep best price per (match, market, selection)
    odds.sort_values(["match_id","market","selection","price"], ascending=[True,True,True,False], inplace=True)
    odds = odds.drop_duplicates(subset=["match_id","market","selection"], keep="first")
    return odds.reset_index(drop=True)

def _mk(mid, lg, utc, mkt, sel, price, book, home, away):
    return {
        "match_id": str(mid),
        "league": lg,
        "utc_kickoff": utc,
        "market": mkt,
        "selection": sel,
        "price": float(price),
        "book": book,
        "home": home,
        "away": away,
    }
