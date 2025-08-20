import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()

# Map friendly names -> The Odds API sport keys
SPORT_KEYS = {
    "EPL":    "soccer_epl",
    "LaLiga": "soccer_spain_la_liga",
    "SerieA": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue1": "soccer_france_ligue_one",
    "UCL":    "soccer_uefa_champs_league",
}

REGIONS = os.getenv("ODDS_REGIONS", "uk,eu,us")  # broaden coverage
ODDS_FORMAT = "decimal"  # we want decimal odds

def _hours_from_now_iso(hours: int) -> str:
    dt = datetime.now(timezone.utc) + timedelta(hours=hours)
    return dt.isoformat().replace("+00:00", "Z")

def _fetch_odds_for_sport(sport_key: str, hours_ahead: int) -> list:
    """
    Pull odds for a single sport_key. Returns the raw JSON list of events.
    We use odds endpoint (markets=h2h,totals,spreads) so we get everything we need.
    """
    if not ODDS_API_KEY:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "oddsFormat": ODDS_FORMAT,
        "markets": "h2h,totals,spreads",
        "dateFormat": "iso",
    }
    # The Odds API returns only upcoming by default; hoursAhead filter is implicit via the feed.
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return []
    try:
        return r.json()
    except Exception:
        return []

def fetch_fixtures(leagues: list[str], hours_ahead: int = 240) -> pd.DataFrame:
    """
    Build a fixtures table for all requested leagues from The Odds API odds feed.
    Columns: match_id, league, utc_kickoff, home, away
    """
    rows = []
    for lg in leagues:
        sport_key = SPORT_KEYS.get(lg, None)
        if not sport_key:
            continue
        events = _fetch_odds_for_sport(sport_key, hours_ahead)
        for ev in events:
            rows.append({
                "match_id": ev.get("id"),
                "league": lg,
                "utc_kickoff": ev.get("commence_time"),
                "home": ev.get("home_team"),
                "away": ev.get("away_team"),
            })
    if not rows:
        return pd.DataFrame(columns=["match_id","league","utc_kickoff","home","away"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["match_id"]).reset_index(drop=True)
    return df

def fetch_odds(fixtures: pd.DataFrame, book_preference: str = "bet365") -> pd.DataFrame:
    """
    Build normalized odds rows for H2H, Totals, and Asian Handicap (Spreads).
    Columns (common): match_id, market, selection, price, book
    For AH we also include: side ('home'/'away') and line (float)
    - H2H -> market='1X2', selection in {'Home','Draw','Away'}
    - Totals -> market='OU{line}', selection in {'Over','Under'}
    - Spreads -> market='AH' (we also set columns 'side' and 'line')
    """
    if fixtures.empty:
        return pd.DataFrame(columns=["match_id","market","selection","price","book","side","line"])

    # Group fixtures by league so we don’t over-fetch
    needed_keys = {row["league"] for _, row in fixtures.iterrows()}
    rows = []
    for lg in needed_keys:
        sport_key = SPORT_KEYS.get(lg)
        if not sport_key:
            continue
        events = _fetch_odds_for_sport(sport_key, hours_ahead=240)
        for ev in events:
            match_id = ev.get("id")
            if match_id not in set(fixtures["match_id"]):
                continue

            # Choose bookmaker: prefer the 'book_preference' if available, else take best price across books
            bookmakers = ev.get("bookmakers", []) or []
            # If a preferred book exists, keep only that one first; else all
            preferred = [b for b in bookmakers if b.get("title","").lower() == book_preference.lower()]
            bks = preferred if preferred else bookmakers

            for bk in bks:
                book = bk.get("title") or ""
                for mkt in bk.get("markets", []):
                    key = (mkt.get("key") or "").lower()

                    # H2H -> 1X2
                    if key == "h2h":
                        # outcomes: name (team or 'Draw'), price
                        for oc in mkt.get("outcomes", []):
                            nm = oc.get("name")
                            price = oc.get("price")
                            if nm is None or price is None:
                                continue
                            sel = None
                            if nm == ev.get("home_team"):
                                sel = "Home"
                            elif nm == ev.get("away_team"):
                                sel = "Away"
                            elif nm.lower() == "draw":
                                sel = "Draw"
                            else:
                                # fallback: treat as exact name
                                sel = nm
                            rows.append({
                                "match_id": match_id,
                                "market": "1X2",
                                "selection": sel,
                                "price": float(price),
                                "book": book,
                                "side": "",
                                "line": None,
                            })

                    # Totals -> OU{line}
                    elif key == "totals":
                        total = mkt.get("outcomes", [])
                        # Expect two outcomes: Over/Under with 'point'
                        for oc in total:
                            line = oc.get("point")
                            price = oc.get("price")
                            name = (oc.get("name") or "").capitalize()
                            if line is None or price is None or name not in ("Over","Under"):
                                continue
                            rows.append({
                                "match_id": match_id,
                                "market": f"OU{float(line):g}",
                                "selection": name,  # Over or Under
                                "price": float(price),
                                "book": book,
                                "side": "",
                                "line": float(line),
                            })

                    # Spreads -> Asian Handicap (AH)
                    elif key == "spreads":
                        # outcomes: name (team), point (handicap relative to that team), price
                        for oc in mkt.get("outcomes", []):
                            team = oc.get("name")
                            line = oc.get("point")
                            price = oc.get("price")
                            if team is None or line is None or price is None:
                                continue
                            # Determine side (home/away) for the selected team
                            if team == ev.get("home_team"):
                                side = "home"
                            elif team == ev.get("away_team"):
                                side = "away"
                            else:
                                continue
                            rows.append({
                                "match_id": match_id,
                                "market": "AH",          # we’ll keep AH generic; side+line carry specifics
                                "selection": team,       # actual team name
                                "price": float(price),
                                "book": book,
                                "side": side,            # 'home' or 'away'
                                "line": float(line),     # e.g., -1.0 for fav, +1.0 for dog (relative to that team)
                            })

    if not rows:
        return pd.DataFrame(columns=["match_id","market","selection","price","book","side","line"])
    df = pd.DataFrame(rows)

    # If multiple prices per match/market/selection/book, keep the best (highest) price
    df = (df.sort_values("price", ascending=False)
            .drop_duplicates(subset=["match_id","market","selection","book","side","line"])
            .reset_index(drop=True))
    return df
