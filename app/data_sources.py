import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# Map our league names to API-Football league IDs and The Odds API sport keys
LEAGUE_MAP = {
    "EPL":  {"api_football_id": 39,  "odds_key": "soccer_epl"},
    "LaLiga": {"api_football_id": 140, "odds_key": "soccer_spain_la_liga"},
}

AF_BASE = "https://v3.football.api-sports.io"
ODDS_BASE = "https://api.the-odds-api.com/v4"

def _af_headers():
    key = os.environ.get("APIFOOTBALL_KEY")
    return {"x-apisports-key": key} if key else {}

def fetch_fixtures(leagues, hours_ahead=48):
    """
    Return DataFrame: match_id, league, utc_kickoff, home, away
    Uses API-Football 'fixtures' for next N hours per league.
    """
    rows = []
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=hours_ahead)

    for lg in leagues:
        info = LEAGUE_MAP.get(lg)
        if not info:
            continue
        league_id = info["api_football_id"]

        # pull upcoming fixtures (next 50 by default) and filter by time window
        # You can also use params like 'next=20'; here we pull by date range.
        # API-Football expects dates as YYYY-MM-DD, so we fetch today and tomorrow.
        date_list = sorted({now.date().isoformat(), until.date().isoformat()})
        for d in date_list:
            resp = requests.get(
                f"{AF_BASE}/fixtures",
                headers=_af_headers(),
                params={"league": league_id, "season": datetime.now().year, "date": d}
            )
            resp.raise_for_status()
            data = resp.json().get("response", [])
            for m in data:
                fid = m.get("fixture", {}).get("id")
                ts = m.get("fixture", {}).get("date")  # ISO string
                home = m.get("teams", {}).get("home", {}).get("name")
                away = m.get("teams", {}).get("away", {}).get("name")
                if not (fid and ts and home and away):
                    continue
                # keep only within [now, until]
                try:
                    kick = datetime.fromisoformat(ts.replace("Z","+00:00"))
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
    Uses The Odds API for H2H and Totals markets.
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

        # Request odds for this sport
        # regions: 'eu' often includes bet365; markets: h2h (1X2) and totals
        resp = requests.get(
            f"{ODDS_BASE}/sports/{odds_key}/odds",
            params={
                "apiKey": key,
                "regions": "eu",
                "oddsFormat": "decimal",
                "markets": "h2h,totals"
            },
            timeout=20
        )
        resp.raise_for_status()
        events = resp.json()

        # Make a quick lookup from our (home, away) to match_id
        match_index = {
            (r["home"].strip().lower(), r["away"].strip().lower()): r["match_id"]
            for _, r in group.iterrows()
        }

        def norm(s): return s.strip().lower() if isinstance(s, str) else s

        for ev in events:
            # The Odds API teams:
            home = norm(ev.get("home_team"))
            away = norm(ev.get("away_team"))
            mid = match_index.get((home, away))
            if not mid:
                # sometimes names differ slightly; try swapped just in case
                mid = match_index.get((away, home))
            if not mid:
                continue  # skip if we can't map

            for bk in ev.get("bookmakers", []):
                title = norm(bk.get("title",""))
                markets = bk.get("markets", [])
                # prioritize preferred book; we still record others if preferred not present
                for mk in markets:
                    keym = mk.get("key")
                    outcomes = mk.get("outcomes", [])
                    if keym == "h2h":
                        # 1X2 — outcomes usually "Home", "Away", and maybe "Draw"
                        for o in outcomes:
                            name = o.get("name")  # e.g., "Home", "Away", "Draw"
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
                        # Over/Under — outcomes have "name": "Over" or "Under", with "point" (line)
                        line = mk.get("outcomes", [{}])[0].get("point")
                        for o in outcomes:
                            name = o.get("name")  # "Over" or "Under"
                            price = o.get("price")
                            if name and price and line is not None:
                                rows.append({
                                    "match_id": str(mid),
                                    "market": f"OU{line}",
                                    "selection": name,
                                    "price": float(price),
                                    "book": bk.get("title","")
                                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # If multiple books exist, keep preferred book rows when available
    if book_preference:
        # Tag rows from preferred book and sort so preferred come first
        df["_pref"] = (df["book"].str.lower() == book_preference.lower()).astype(int)
        df = (df.sort_values(["match_id","market","selection","_pref"], ascending=[True, True, True, False])
                .drop_duplicates(subset=["match_id","market","selection"], keep="first")
                .drop(columns=["_pref"]))
    return df.reset_index(drop=True)