import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# Map our league names to API-Football league IDs and The Odds API sport keys
LEAGUE_MAP = {
    "EPL":    {"api_football_id": 39,  "odds_key": "soccer_epl"},
    "LaLiga": {"api_football_id": 140, "odds_key": "soccer_spain_la_liga"},
}

AF_BASE = "https://v3.football.api-sports.io"
ODDS_BASE = "https://api.the-odds-api.com/v4"

def _af_headers():
    key = os.environ.get("APIFOOTBALL_KEY")
    if not key:
        return {}
    return {"x-apisports-key": key}

def _norm(s):
    return s.strip().lower() if isinstance(s, str) else s

def fetch_fixtures(leagues, hours_ahead=48):
    """
    Return DataFrame: match_id, league, utc_kickoff, home, away
    Uses API-Football fixtures with 'next' to ensure we get upcoming matches,
    then filters by the [now, now+hours_ahead] window.
    """
    rows = []
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=hours_ahead)

    for lg in leagues:
        info = LEAGUE_MAP.get(lg)
        if not info:
            continue
        league_id = info["api_football_id"]

        # Try without season first (API often handles season with 'next')
        params_list = [
            {"league": league_id, "next": 50},
            {"league": league_id, "season": datetime.now().year, "next": 50},
        ]

        data_accum = []
        for params in params_list:
            try:
                resp = requests.get(
                    f"{AF_BASE}/fixtures",
                    headers=_af_headers(),
                    params=params,
                    timeout=20
                )
                resp.raise_for_status()
                data = resp.json().get("response", [])
                if data:
                    data_accum = data
                    break
            except Exception:
                continue

        for m in data_accum:
            fx = m.get("fixture", {}) or {}
            tm = m.get("teams", {}) or {}
            fid = fx.get("id")
            ts = fx.get("date")  # ISO string
            home = (tm.get("home") or {}).get("name")
            away = (tm.get("away") or {}).get("name")
            if not (fid and ts and home and away):
                continue
            try:
                kick = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except Exception:
                continue
            if now <= kick <= until:
                rows.append({
                    "match_id": str(fid),
                    "league": lg,
                    "utc_kickoff": kick.isoformat(),
                    "home": home,
                    "away": away,
                })

    df = pd.DataFrame(rows).drop_duplicates(subset=["match_id"]).reset_index(drop=True)
    return df

def fetch_odds(fixtures_df, book_preference="bet365"):
    """
    Return DataFrame: match_id, market, selection, price, book
    Uses The Odds API for H2H (1X2) and Totals markets.
    """
    key = os.environ.get("ODDS_API_KEY")
    if fixtures_df.empty or not key:
        return pd.DataFrame(columns=["match_id","market","selection","price","book"])

    rows = []

    # group fixtures by league to query the right sport key
    for lg, group in fixtures_df.groupby("league"):
        odds_key = LEAGUE_MAP.get(lg, {}).get("odds_key")
        if not odds_key:
            continue

        try:
            resp = requests.get(
                f"{ODDS_BASE}/sports/{odds_key}/odds",
                params={
                    "apiKey": key,
                    "regions": "eu",           # 'eu' usually includes bet365
                    "oddsFormat": "decimal",
                    "markets": "h2h,totals"    # we’ll add more later
                },
                timeout=25
            )
            resp.raise_for_status()
            events = resp.json()
        except Exception:
            continue

        # make a lookup from our (home, away) to match_id
        index = {
            (_norm(r["home"]), _norm(r["away"])): r["match_id"]
            for _, r in group.iterrows()
        }

        for ev in events:
            home = _norm(ev.get("home_team"))
            away = _norm(ev.get("away_team"))
            mid = index.get((home, away)) or index.get((away, home))
            if not mid:
                continue

            for bk in ev.get("bookmakers", []) or []:
                markets = bk.get("markets", []) or []
                for mk in markets:
                    keym = mk.get("key")
                    outcomes = mk.get("outcomes", []) or []
                    if keym == "h2h":
                        # 1X2
                        for o in outcomes:
                            name = o.get("name")  # Home / Away / Draw
                            price = o.get("price")
                            if name and price:
                                rows.append({
                                    "match_id": str(mid),
                                    "market": "1X2",
                                    "selection": name,
                                    "price": float(price),
                                    "book": bk.get("title","")
                                })
                    elif keym == "totals":
                        # Over/Under (use provided line 'point')
                        line = None
                        if outcomes and isinstance(outcomes, list):
                            # some books include line per outcome; keep first point
                            line = outcomes[0].get("point", None)
                        for o in outcomes:
                            name = o.get("name")      # Over / Under
                            price = o.get("price")
                            point = o.get("point", line)
                            if name and price and point is not None:
                                rows.append({
                                    "match_id": str(mid),
                                    "market": f"OU{point}",
                                    "selection": name,
                                    "price": float(price),
                                    "book": bk.get("title","")
                                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Prefer a specific book if requested (e.g., bet365)
    if book_preference:
        df["_pref"] = (df["book"].str.lower() == str(book_preference).lower()).astype(int)
        df = (
            df.sort_values(
                ["match_id", "market", "selection", "_pref"],
                ascending=[True, True, True, False]
            )
            .drop_duplicates(subset=["match_id","market","selection"], keep="first")
            .drop(columns=["_pref"])
        )

    return df.reset_index(drop=True)
