import os
import requests
import pandas as pd

# Using The Odds API for both fixtures and odds (free-plan friendly)
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

def _fmt_line(x):
    """Normalize lines so '2.0' -> '2', keep '2.5' as '2.5'."""
    try:
        v = float(x)
        if abs(v - int(v)) < 1e-9:
            return str(int(v))
        s = str(v)
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    except Exception:
        return str(x)

def fetch_fixtures(leagues, hours_ahead=240):
    """
    Use The Odds API to get upcoming events for each league.
    Returns: match_id (from Odds API event id when present), league, utc_kickoff, home, away
    """
    rows = []
    key = _odds_key()

    for lg in leagues:
        odds_key = LEAGUE_MAP.get(lg, {}).get("odds_key")
        if not odds_key:
            continue

        r = requests.get(
            f"{ODDS_BASE}/sports/{odds_key}/odds",
            params={
                "apiKey": key,
                "regions": "eu",
                "oddsFormat": "decimal",
                "markets": "h2h,totals,btts",   # BTTS included
            },
            timeout=25
        )
        if r.status_code != 200:
            continue

        events = r.json() or []
        for ev in events:
            event_id = ev.get("id") or ""
            commence = ev.get("commence_time")  # ISO8601
            home = ev.get("home_team")
            away = ev.get("away_team")
            if not (home and away and commence):
                continue

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
    Collect odds for our fixtures (1X2, Totals, BTTS).
    - Normalizes h2h selections to 'Home'/'Away'/'Draw' using event team names.
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
                "markets": "h2h,totals,btts",   # BTTS included
            },
            timeout=25
        )
        if rr.status_code != 200:
            continue
        events = rr.json() or []

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

            home_name = ev.get("home_team")
            away_name = ev.get("away_team")
            home_norm = _norm(home_name)
            away_norm = _norm(away_name)

            for bk in (ev.get("bookmakers") or []):
                for mk in (bk.get("markets") or []):
                    keym = mk.get("key")
                    outs = mk.get("outcomes") or []

                    if keym == "h2h":
                        for o in outs:
                            name, price = o.get("name"), o.get("price")
                            if not (name and price):
                                continue
                            sel_norm = _norm(name)
                            # Normalize to Home/Away/Draw
                            if sel_norm == _norm("draw"):
                                selection = "Draw"
                            elif sel_norm == home_norm:
                                selection = "Home"
                            elif sel_norm == away_norm:
                                selection = "Away"
                            else:
                                selection = name
                            rows.append({
                                "match_id": mid,
                                "market": "1X2",
                                "selection": selection,
                                "price": float(price),
                                "book": bk.get("title","")
                            })

                    elif keym == "totals":
                        base_line = outs[0].get("point") if outs else None
                        for o in outs:
                            name = o.get("name")    # "Over" / "Under"
                            price = o.get("price")
                            point = o.get("point", base_line)
                            if name and price and point is not None:
                                line_str = _fmt_line(point)
                                rows.append({
                                    "match_id": mid,
                                    "market": f"OU{line_str}",
                                    "selection": name,
                                    "price": float(price),
                                    "book": bk.get("title","")
                                })

                    elif keym == "btts":
                        # Selections usually "Yes" / "No"
                        for o in outs:
                            name = o.get("name")    # "Yes" / "No"
                            price = o.get("price")
                            if name and price:
                                rows.append({
                                    "match_id": mid,
                                    "market": "BTTS",
                                    "selection": name,
                                    "price": float(price),
                                    "book": bk.get("title","")
                                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Prefer a specific book (e.g., Bet365) if available per (match, market, selection)
    if book_preference:
        df["_pref"] = (df["book"].str.lower() == str(book_preference).lower()).astype(int)
        df = (
            df.sort_values(["match_id","market","selection","_pref"], ascending=[True,True,True,False])
              .drop_duplicates(subset=["match_id","market","selection"], keep="first")
              .drop(columns=["_pref"])
        )

    return df.reset_index(drop=True)
