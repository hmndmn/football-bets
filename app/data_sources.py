import os
import requests
import pandas as pd

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()

# Map friendly names -> The Odds API sport keys (corrected)
SPORT_KEYS = {
    "EPL":        "soccer_epl",
    "LaLiga":     "soccer_spain_la_liga",
    "SerieA":     "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue1":     "soccer_france_ligue_1",            # <- fixed (was ligue_one)
    "UCL":        "soccer_uefa_champions_league",     # <- fixed (was champs_league)
}

REGIONS = os.getenv("ODDS_REGIONS", "uk,eu,us")  # broaden coverage
ODDS_FORMAT = "decimal"  # decimal odds

def _fetch_odds_for_sport(sport_key: str) -> list:
    """
    Pull upcoming odds for a single sport_key.
    We request h2h, totals, spreads in one call.
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
    try:
        r = requests.get(url, params=params, timeout=25)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    try:
        return r.json()
    except Exception:
        return []

def fetch_fixtures(leagues: list[str], hours_ahead: int = 240) -> pd.DataFrame:
    """
    Build fixtures from the odds feed for the requested leagues.
    Columns: match_id, league, utc_kickoff, home, away
    """
    rows = []
    for lg in leagues:
        sport_key = SPORT_KEYS.get(lg)
        if not sport_key:
            continue
        events = _fetch_odds_for_sport(sport_key)
        for ev in events or []:
            rows.append({
                "match_id": ev.get("id"),
                "league": lg,
                "utc_kickoff": ev.get("commence_time"),
                "home": ev.get("home_team"),
                "away": ev.get("away_team"),
            })
    if not rows:
        return pd.DataFrame(columns=["match_id","league","utc_kickoff","home","away"])
    df = pd.DataFrame(rows).dropna(subset=["match_id"]).drop_duplicates(subset=["match_id"]).reset_index(drop=True)
    return df

def fetch_odds(fixtures: pd.DataFrame, book_preference: str = "bet365") -> pd.DataFrame:
    """
    Normalize odds rows for H2H, Totals, and Asian Handicap (Spreads).
    Output columns: match_id, market, selection, price, book, side, line
      - H2H -> market='1X2', selection in {'Home','Draw','Away'}
      - Totals -> market='OU{line}', selection in {'Over','Under'}
      - Spreads -> market='AH', selection = team name, plus side ('home'/'away') and line (float)
    We keep the SINGLE BEST price across ALL books per (match, market, selection[, side, line]).
    """
    if fixtures.empty:
        return pd.DataFrame(columns=["match_id","market","selection","price","book","side","line"])

    wanted_ids = set(fixtures["match_id"])
    wanted_leagues = set(fixtures["league"])

    rows = []
    # Fetch each league once; filter by match ids we actually have
    for lg in wanted_leagues:
        sport_key = SPORT_KEYS.get(lg)
        if not sport_key:
            continue
        events = _fetch_odds_for_sport(sport_key)
        for ev in events or []:
            match_id = ev.get("id")
            if match_id not in wanted_ids:
                continue

            bookmakers = ev.get("bookmakers", []) or []

            for bk in bookmakers:
                book = bk.get("title") or ""
                for mkt in bk.get("markets", []) or []:
                    key = (mkt.get("key") or "").lower()

                    # H2H -> 1X2
                    if key == "h2h":
                        for oc in mkt.get("outcomes", []) or []:
                            nm = oc.get("name")
                            price = oc.get("price")
                            if nm is None or price is None:
                                continue
                            if nm == ev.get("home_team"):
                                sel = "Home"
                            elif nm == ev.get("away_team"):
                                sel = "Away"
                            elif str(nm).lower() == "draw":
                                sel = "Draw"
                            else:
                                sel = str(nm)
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
                        for oc in mkt.get("outcomes", []) or []:
                            name = (oc.get("name") or "").capitalize()  # Over / Under
                            line = oc.get("point")
                            price = oc.get("price")
                            if name not in ("Over","Under") or line is None or price is None:
                                continue
                            try:
                                L = float(line)
                            except Exception:
                                continue
                            rows.append({
                                "match_id": match_id,
                                "market": f"OU{L:g}",
                                "selection": name,
                                "price": float(price),
                                "book": book,
                                "side": "",
                                "line": L,
                            })

                    # Spreads -> Asian Handicap
                    elif key == "spreads":
                        for oc in mkt.get("outcomes", []) or []:
                            team = oc.get("name")
                            line = oc.get("point")
                            price = oc.get("price")
                            if team is None or line is None or price is None:
                                continue
                            if team == ev.get("home_team"):
                                side = "home"
                            elif team == ev.get("away_team"):
                                side = "away"
                            else:
                                continue
                            try:
                                L = float(line)
                            except Exception:
                                continue
                            rows.append({
                                "match_id": match_id,
                                "market": "AH",
                                "selection": str(team),
                                "price": float(price),
                                "book": book,
                                "side": side,
                                "line": L,
                            })

    if not rows:
        return pd.DataFrame(columns=["match_id","market","selection","price","book","side","line"])
    df = pd.DataFrame(rows)

    # Keep the single BEST price across ALL books per unique outcome signature
    # (match_id, market, selection[, side, line])
    group_cols = ["match_id","market","selection"]
    if "side" in df.columns:
        group_cols.append("side")
    if "line" in df.columns:
        group_cols.append("line")

    # Sort so the first in each group is the best price; keep its book too
    df = df.sort_values("price", ascending=False)
    df = df.drop_duplicates(subset=group_cols, keep="first").reset_index(drop=True)

    return df
