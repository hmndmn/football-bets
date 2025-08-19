import os
import requests
import pandas as pd

# League map (we only need The Odds API sport keys now)
LEAGUE_MAP = {
    "EPL":    {"odds_key": "soccer_epl"},
    "LaLiga": {"odds_key": "soccer_spain_la_liga"},
}

ODDS_BASE = "https://api.the-odds-api.com/v4"

def _odds_key():
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ODDS_API_KEY is missing")
    return key

def _norm(s):
    return s.strip().lower() if isinstance(s, str) else s

def fetch_fixtures(leagues, hours_ahead=240):
    """
    Use The Odds API to get upcoming events for each league.
    Returns: match_id (synthetic), league, utc_kickoff, home, away
    Note: The Odds API does not give a unique fixture id that matches other APIs,
    so we synthesize one from the event id if present, else teams+commence_time.
    """
    rows = []
    key = _odds_key()

    for lg in leagues:
        odds_key = LEAGUE_MAP.get(lg, {}).get("odds_key")
        if not odds_key:
            continue

        # Get upcoming events with at least H2H market (that returns the schedule)
        r = requests.get(
            f"{ODDS_BASE}/sports/{odds_key}/odds",
            params={
                "apiKey": key,
                "regions": "eu",          # includes bet365 region
                "oddsFormat": "decimal",
                "markets": "h2h,totals"   # request both so we can reuse later
            },
            timeout=25
        )
        if r.status_code != 200:
            # If quota empty or something else, just skip league
            continue

        events = r.json() or []
        for ev in events:
            event_id = ev.get("id") or ""
            commence = ev.get("commence_time")  # ISO8601
            home = ev.get("home_team")
            away = ev.get("away_team")
            if not (home and away and commence):
                continue

            # Use Odds API event id when available; else synthesize
            match_id = str(event_id) if event_id else f"{_norm(home)}_{_norm(away)}_{commence}"

            rows.append({
                "match_id": match_id,
                "league": lg,
                "utc_kickoff": commence,
                "home": home,
                "away": away,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["match_id"]).reset_index(drop=True)
    return df

def fetch_odds(fixtures_df, book_preference="bet365"):
    """
    Reuse The Odds API results to collect odds for our fixtures.
    Returns: match_id, market, selection, price, book
    """
    key = _odds_key()
    if fixtures_df.empty:
        return pd.DataFrame(columns=["match_id","market","selection","price","book"])

    rows = []

    for lg, group in fixtures_df.groupby("league"):
        odds_key = LEAGUE_MAP.get(lg, {}).get("odds_key")
        if not odds_key:
            continue

        rr = requests.get(
            f"{ODDS_BASE}/sports/{odds_key}/odds",
            params={
                "apiKey": key,
                "regions": "eu",
                "oddsFormat": "decimal",
                "markets": "h2h,totals"
            },
            timeout=25
        )
        if rr.status_code != 200:
            continue
        events = rr.json() or []

        # We’ll match by Odds API event id when present, else name+time key
        def make_key(ev):
            eid = ev.get("id") or ""
            if eid:
                return str(eid)
            return f"{_norm(ev.get('home_team'))}_{_norm(ev.get('away_team'))}_{ev.get('commence_time')}"

        target_ids = set(group["match_id"].tolist())
        for ev in events:
            mid = make_key(ev)
            if mid not in target_ids:
                continue

            for bk in (ev.get("bookmakers") or []):
                for mk in (bk.get("markets") or []):
                    keym = mk.get("key")
                    outs = mk.get("outcomes") or []
                    if keym == "h2h":
                        for o in outs:
                            name, price = o.get("name"), o.get("price")
                            if name and price:
                                rows.append({
                                    "match_id": mid,
                                    "market": "1X2",
                                    "selection": name,   # Home / Away / Draw
                                    "price": float(price),
                                    "book": bk.get("title","")
                                })
                    elif keym == "totals":
                        line = outs[0].get("point") if outs else None
                        for o in outs:
                            name, price = o.get("name"), o.get("price")
                            point = o.get("point", line)
                            if name and price and point is not None:
                                rows.append({
                                    "match_id": mid,
                                    "market": f"OU{point}",
                                    "selection": name,   # Over / Under
                                    "price": float(price),
                                    "book": bk.get("title","")
                                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Prefer bet365 when available
    if book_preference:
        df["_pref"] = (df["book"].str.lower() == str(book_preference).lower()).astype(int)
        df = (df.sort_values(["match_id","market","selection","_pref"], ascending=[True,True,True,False])
                .drop_duplicates(subset=["match_id","market","selection"], keep="first")
                .drop(columns=["_pref"]))
    return df.reset_index(drop=True)
