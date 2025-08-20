import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()

# Friendly -> The Odds API sport keys (correct)
SPORT_KEYS = {
    "EPL":        "soccer_epl",
    "LaLiga":     "soccer_spain_la_liga",
    "SerieA":     "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue1":     "soccer_france_ligue_1",
    "UCL":        "soccer_uefa_champions_league",
}

REGIONS = os.getenv("ODDS_REGIONS", "uk,eu,us")      # Broaden coverage
ODDS_FORMAT = "decimal"                               # Decimal odds


def _fetch_odds_for_sport(sport_key: str) -> list:
    """
    Pull upcoming odds for a single sport_key.
    We request h2h, totals, spreads in one call.
    Returns a list of event dicts or [].
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


def _within_hours(commence_iso: str | None, hours_ahead: int) -> bool:
    if not commence_iso:
        return False
    try:
        dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        now = datetime.now(timezone.utc)
        return now <= dt <= now + timedelta(hours=hours_ahead)
    except Exception:
        return False


def fetch_fixtures(leagues: list[str], hours_ahead: int = 240) -> pd.DataFrame:
    """
    Build fixtures from the odds feed for the requested leagues.
    Columns: match_id, league, utc_kickoff, home, away
    """
    rows: list[dict] = []
    for lg in leagues or []:
        sport_key = SPORT_KEYS.get(lg)
        if not sport_key:
            continue
        events = _fetch_odds_for_sport(sport_key)
        for ev in events or []:
            if not _within_hours(ev.get("commence_time"), hours_ahead):
                continue
            rows.append({
                "match_id": ev.get("id"),
                "league": lg,
                "utc_kickoff": ev.get("commence_time"),
                "home": ev.get("home_team"),
                "away": ev.get("away_team"),
            })

    if not rows:
        return pd.DataFrame(columns=["match_id", "league", "utc_kickoff", "home", "away"])

    df = (
        pd.DataFrame(rows)
        .dropna(subset=["match_id"])
        .drop_duplicates(subset=["match_id"])
        .reset_index(drop=True)
    )
    return df


def fetch_odds(fixtures: pd.DataFrame, book_preference: str = "bet365", hours_ahead: int = 240) -> pd.DataFrame:
    """
    Normalize odds rows for H2H, Totals, and Asian Handicap (Spreads).
    Output columns: match_id, market, selection, price, book, side, line
      - H2H -> market='1X2', selection in {'Home','Draw','Away'}
      - Totals -> market='OU{line}', selection in {'Over','Under'}
      - Spreads -> market='AH', selection = team name, plus side ('home'/'away') and line (float)

    IMPORTANT: We DO NOT re-filter by time here. We trust the fixtures list already
    limited the window, so we only include events whose match_id appears in fixtures.
    """
    if fixtures.empty:
        return pd.DataFrame(columns=["match_id", "market", "selection", "price", "book", "side", "line"])

    wanted_ids = set(fixtures["match_id"])
    leagues = list(fixtures["league"].dropna().unique())

    odds_rows: list[dict] = []

    # Fetch each league once; only keep events in fixtures
    for lg in leagues:
        sport_key = SPORT_KEYS.get(lg)
        if not sport_key:
            continue

        events = _fetch_odds_for_sport(sport_key)
        for ev in events or []:
            match_id = ev.get("id")
            if match_id not in wanted_ids:
                continue

            home = ev.get("home_team")
            away = ev.get("away_team")

            for bk in ev.get("bookmakers", []) or []:
                book = bk.get("title") or ""
                for mkt in bk.get("markets", []) or []:
                    key = (mkt.get("key") or "").lower()

                    # --- 1X2 (h2h) ---
                    if key == "h2h":
                        for oc in mkt.get("outcomes", []) or []:
                            name, price = oc.get("name"), oc.get("price")
                            if name is None or price is None:
                                continue
                            if name == home:
                                sel = "Home"
                            elif name == away:
                                sel = "Away"
                            elif str(name).lower() == "draw":
                                sel = "Draw"
                            else:
                                continue
                            odds_rows.append({
                                "match_id": match_id,
                                "market": "1X2",
                                "selection": sel,
                                "price": float(price),
                                "book": book,
                                "side": "",
                                "line": None,
                            })

                    # --- Totals (Over/Under) ---
                    elif key == "totals":
                        for oc in mkt.get("outcomes", []) or []:
                            name, line, price = oc.get("name"), oc.get("point"), oc.get("price")
                            if not name or line is None or price is None:
                                continue
                            try:
                                L = float(line)
                            except Exception:
                                continue
                            nm = str(name).capitalize()  # Over / Under
                            if nm not in ("Over", "Under"):
                                continue
                            odds_rows.append({
                                "match_id": match_id,
                                "market": f"OU{L:g}",
                                "selection": nm,
                                "price": float(price),
                                "book": book,
                                "side": "",
                                "line": L,
                            })

                    # --- Asian Handicap (spreads) ---
                    elif key == "spreads":
                        for oc in mkt.get("outcomes", []) or []:
                            team, line, price = oc.get("name"), oc.get("point"), oc.get("price")
                            if not team or line is None or price is None:
                                continue
                            try:
                                L = float(line)
                            except Exception:
                                continue
                            if team == home:
                                side = "home"
                            elif team == away:
                                side = "away"
                            else:
                                continue
                            odds_rows.append({
                                "match_id": match_id,
                                "market": "AH",
                                "selection": str(team),
                                "price": float(price),
                                "book": book,
                                "side": side,
                                "line": L,
                            })

    if not odds_rows:
        return pd.DataFrame(columns=["match_id", "market", "selection", "price", "book", "side", "line"])

    df = pd.DataFrame(odds_rows)

    # Keep ONLY the best price per unique outcome signature (across ALL books)
    # Signature: (match_id, market, selection[, side, line])
    group_cols = ["match_id", "market", "selection"]
    if "side" in df.columns:
        group_cols.append("side")
    if "line" in df.columns:
        group_cols.append("line")

    df = df.sort_values("price", ascending=False)
    df = df.drop_duplicates(subset=group_cols, keep="first").reset_index(drop=True)

    return df
