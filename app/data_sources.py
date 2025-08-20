import os
import time
import json
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any

import requests
import pandas as pd

# ----------------------------
# Config / constants
# ----------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BASE = "https://api.the-odds-api.com/v4"
ODDS_REGIONS = os.getenv("ODDS_REGIONS", "uk,eu,us")
ODDS_MARKETS = os.getenv("ODDS_MARKETS", "h2h,totals,spreads")  # 1X2, OU, AH
HOURS_AHEAD_DEFAULT = int(os.getenv("HOURS_AHEAD", "168"))

# Map our short league names to The Odds API sport_keys
LEAGUE_MAP: Dict[str, str] = {
    "EPL": "soccer_epl",
    "LaLiga": "soccer_spain_la_liga",
    "SerieA": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue1": "soccer_france_ligue_one",
    # add more when needed
}

# ----------------------------
# Helpers
# ----------------------------
def _hash_match(home: str, away: str, ts: str, league: str) -> str:
    raw = f"{league}|{ts}|{home}|{away}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def _to_iso(ts: float) -> str:
    # The Odds API returns ISO strings already; this is for safety if we ever parse epoch
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return str(ts)

def _ensure_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _book_title(bm: Dict[str, Any]) -> str:
    # Normalize the bookmaker display title a bit (leave as-is if provided)
    return bm.get("title") or bm.get("key") or ""

def _sel_name_to_side(sel_name: str, home: str, away: str) -> Tuple[str, str]:
    """Return (selection_label, normalized_team_name) for H2H market."""
    n = (sel_name or "").strip()
    if n.lower() in ("draw", "x"):
        return ("Draw", "Draw")
    # map to exact team strings if possible
    if n.lower() in (home or "").lower():
        return ("Home", home)
    if n.lower() in (away or "").lower():
        return ("Away", away)
    # default: compare fuzzy
    if n == home:
        return ("Home", home)
    if n == away:
        return ("Away", away)
    # last resort: keep as-is
    return (n, n)

# ----------------------------
# Public API
# ----------------------------
def fetch_fixtures(
    league: Optional[str] = None,
    leagues: Optional[List[str]] = None,
    hours_ahead: int = HOURS_AHEAD_DEFAULT,
) -> pd.DataFrame:
    """
    Build a fixtures table from odds endpoints (only upcoming matches within hours_ahead).
    Columns: [match_id, league, utc_kickoff, home, away]
    """
    if not ODDS_API_KEY:
        return pd.DataFrame(columns=["match_id", "league", "utc_kickoff", "home", "away"])

    ask_leagues = leagues if leagues else _ensure_list(league)
    ask_leagues = [l for l in ask_leagues if l]

    rows = []
    cutoff = datetime.now(timezone.utc) + timedelta(hours=hours_ahead)

    for lg in ask_leagues:
        sport_key = LEAGUE_MAP.get(lg)
        if not sport_key:
            continue

        url = f"{ODDS_BASE}/sports/{sport_key}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": ODDS_REGIONS,
            "markets": "h2h",  # minimal call for fixture list
            "dateFormat": "iso",
            "oddsFormat": "decimal",
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                continue
            events = r.json()
        except Exception:
            continue

        for ev in events or []:
            home = ev.get("home_team")
            away = ev.get("away_team")
            ts = ev.get("commence_time")  # already ISO
            if not (home and away and ts):
                continue
            # Only keep matches within the horizon
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt > cutoff:
                    continue
            except Exception:
                pass

            mid = _hash_match(home, away, ts, lg)
            rows.append({
                "match_id": mid,
                "league": lg,
                "utc_kickoff": ts,
                "home": home,
                "away": away,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["match_id"])
    return df.sort_values(["league", "utc_kickoff"]).reset_index(drop=True)


def fetch_odds(
    league: Optional[str] = None,
    leagues: Optional[List[str]] = None,
    markets: Optional[str] = None,
    limit_only: bool = False,
):
    """
    When limit_only=True:
        Returns (status:int, sample:any, headers:dict) for quota/status probing.
        If 'leagues' is provided, returns a dict keyed by league.

    When limit_only=False:
        Returns a pandas DataFrame with columns:
        [match_id, league, utc_kickoff, market, selection, price, book, home, away]
    """
    if not ODDS_API_KEY:
        if limit_only:
            return 200, {"ok": True, "note": "no ODDS_API_KEY set"}, {}
        return pd.DataFrame(columns=["match_id", "league", "utc_kickoff", "market", "selection", "price", "book", "home", "away"])

    ask_leagues = leagues if leagues else _ensure_list(league)
    ask_leagues = [l for l in ask_leagues if l]
    mkts = markets if markets else ODDS_MARKETS

    if limit_only:
        # Probe each league individually so we can surface per-league quota/errors
        out = {"ok": True, "regions": ODDS_REGIONS, "leagues": {}}
        last_status = 200
        last_headers = {}
        for lg in ask_leagues:
            sport_key = LEAGUE_MAP.get(lg)
            if not sport_key:
                out["leagues"][lg] = {"status": 400, "sample": {"message": "unknown league"}, "events": None}
                continue
            url = f"{ODDS_BASE}/sports/{sport_key}/odds"
            params = {
                "apiKey": ODDS_API_KEY,
                "regions": ODDS_REGIONS,
                "markets": mkts,
                "dateFormat": "iso",
                "oddsFormat": "decimal",
            }
            try:
                r = requests.get(url, params=params, timeout=15)
                last_status = r.status_code
                last_headers = {k.lower(): v for k, v in r.headers.items()}
                try:
                    sample = r.json()
                except Exception:
                    sample = r.text
                # normalize summary
                events_len = None
                if isinstance(sample, list):
                    events_len = len(sample)
                out["leagues"][lg] = {
                    "status": last_status,
                    "sample": sample if not isinstance(sample, list) else (sample[0] if sample else {}),
                    "events": events_len,
                    "x-requests-remaining": last_headers.get("x-requests-remaining"),
                    "x-requests-used": last_headers.get("x-requests-used"),
                }
            except Exception as e:
                out["leagues"][lg] = {"status": 599, "sample": {"error": str(e)}, "events": None}
            # small delay to be polite
            time.sleep(0.25)
        return last_status, out, last_headers

    # Build odds rows across all requested leagues
    rows = []
    for lg in ask_leagues:
        sport_key = LEAGUE_MAP.get(lg)
        if not sport_key:
            continue

        url = f"{ODDS_BASE}/sports/{sport_key}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": ODDS_REGIONS,
            "markets": mkts,
            "dateFormat": "iso",
            "oddsFormat": "decimal",
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200:
                # skip league on error
                continue
            events = r.json()
        except Exception:
            continue

        for ev in events or []:
            home = ev.get("home_team")
            away = ev.get("away_team")
            ts = ev.get("commence_time")
            mid = _hash_match(home or "", away or "", ts or "", lg)
            bms = ev.get("bookmakers") or []

            # iterate all requested markets
            for bm in bms:
                book = _book_title(bm)

                for mkt in (bm.get("markets") or []):
                    key = (mkt.get("key") or "").lower()

                    # --- 1X2 (h2h) ---
                    if key == "h2h":
                        for oc in (mkt.get("outcomes") or []):
                            name = oc.get("name")
                            price = oc.get("price")
                            sel_label, team_label = _sel_name_to_side(name, home or "", away or "")
                            # Keep "selection" as our readable label (Home/Away/Draw)
                            rows.append({
                                "match_id": mid,
                                "league": lg,
                                "utc_kickoff": ts,
                                "market": "1X2",
                                "selection": sel_label,
                                "price": float(price) if price is not None else None,
                                "book": book,
                                "home": home,
                                "away": away,
                            })

                    # --- Totals (OU) ---
                    elif key == "totals":
                        for oc in (mkt.get("outcomes") or []):
                            # outcome has fields: "name" (Over/Under), "price", "point"
                            side = oc.get("name")  # "Over" / "Under"
                            point = oc.get("point")
                            price = oc.get("price")
                            if point is None or side is None:
                                continue
                            market_name = f"OU{point}"
                            rows.append({
                                "match_id": mid,
                                "league": lg,
                                "utc_kickoff": ts,
                                "market": market_name,
                                "selection": side,  # "Over" or "Under"
                                "price": float(price) if price is not None else None,
                                "book": book,
                                "home": home,
                                "away": away,
                            })

                    # --- Spreads (Asian Handicap proxy) ---
                    elif key == "spreads":
                        for oc in (mkt.get("outcomes") or []):
                            # outcome fields: "name" (team), "point" (handicap), "price"
                            who = oc.get("name") or ""
                            point = oc.get("point")
                            price = oc.get("price")
                            if point is None:
                                continue
                            # Normalize selection to "Home" or "Away" based on team
                            sel_label, _ = _sel_name_to_side(who, home or "", away or "")
                            market_name = f"AH{point:+g}"  # e.g. AH-0.5, AH+0.25
                            rows.append({
                                "match_id": mid,
                                "league": lg,
                                "utc_kickoff": ts,
                                "market": market_name,
                                "selection": sel_label,  # Home / Away
                                "price": float(price) if price is not None else None,
                                "book": book,
                                "home": home,
                                "away": away,
                            })

        # be polite to the API
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["match_id", "league", "utc_kickoff", "market", "selection", "price", "book", "home", "away"])

    # Keep best price per (match_id, market, selection, book) — or per book we might just keep one row already.
    # Also drop obvious nulls.
    df = df.dropna(subset=["match_id", "market", "selection", "price"])
    df["price"] = df["price"].astype(float)

    # Sometimes multiple entries per book/market are returned; keep the best (max price)
    df = (
        df.sort_values("price", ascending=False)
          .drop_duplicates(subset=["match_id", "market", "selection", "book"], keep="first")
          .reset_index(drop=True)
    )
    return df
