import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

LEAGUE_MAP = {
    "EPL":    {"api_football_id": 39,  "odds_key": "soccer_epl"},
    "LaLiga": {"api_football_id": 140, "odds_key": "soccer_spain_la_liga"},
}

AF_BASE = "https://v3.football.api-sports.io"
ODDS_BASE = "https://api.the-odds-api.com/v4"

def _af_headers():
    key = os.environ.get("APIFOOTBALL_KEY")
    return {"x-apisports-key": key} if key else {}

def check_af_status():
    try:
        r = requests.get(f"{AF_BASE}/status", headers=_af_headers(), timeout=20)
        return (r.status_code == 200, r.text)
    except Exception as e:
        return (False, f"EXC: {e}")

def af_probe_dates(league_id=39, days=7):
    """Probe fixtures via from/to date range (free-plan friendly)."""
    now = datetime.now(timezone.utc)
    start = now.date().isoformat()
    end = (now + timedelta(days=days)).date().isoformat()
    # Try likely seasons (current year, then previous year in case API hasn’t rolled)
    for season in [now.year, now.year - 1]:
        try:
            r = requests.get(
                f"{AF_BASE}/fixtures",
                headers=_af_headers(),
                params={"league": league_id, "season": season, "from": start, "to": end},
                timeout=25
            )
            if r.status_code == 200:
                return r.status_code, f"season={season} {r.text[:1000]}"
        except Exception as e:
            return 0, f"EXC: {e}"
    return 200, "no data in both seasons"

def fetch_fixtures(leagues, hours_ahead=48):
    """
    Return DataFrame: match_id, league, utc_kickoff, home, away
    Uses API-Football fixtures with 'from'/'to' date range (free plan OK),
    tries current season, then previous season if needed.
    """
    rows = []
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=hours_ahead)
    start_d = now.date().isoformat()
    end_d = until.date().isoformat()

    for lg in leagues:
        info = LEAGUE_MAP.get(lg)
        if not info:
            continue
        league_id = info["api_football_id"]

        data_accum = []
        for season in [now.year, now.year - 1]:
            try:
                resp = requests.get(
                    f"{AF_BASE}/fixtures",
                    headers=_af_headers(),
                    params={"league": league_id, "season": season, "from": start_d, "to": end_d},
                    timeout=25
                )
                if resp.status_code != 200:
                    continue
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
            ts = fx.get("date")
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
    key = os.environ.get("ODDS_API_KEY")
    if fixtures_df.empty or not key:
        return pd.DataFrame(columns=["match_id","market","selection","price","book"])

    rows = []
    for lg, group in fixtures_df.groupby("league"):
        odds_key = LEAGUE_MAP.get(lg, {}).get("odds_key")
        if not odds_key:
            continue
        try:
            resp = requests.get(
                f"{ODDS_BASE}/sports/{odds_key}/odds",
                params={"apiKey": key, "regions": "eu", "oddsFormat": "decimal", "markets": "h2h,totals"},
                timeout=25
            )
            if resp.status_code != 200:
                continue
            events = resp.json()
        except Exception:
            continue

        def norm(s): return s.strip().lower() if isinstance(s, str) else s
        index = { (norm(r["home"]), norm(r["away"])): r["match_id"] for _, r in group.iterrows() }

        for ev in events:
            home = norm(ev.get("home_team"))
            away = norm(ev.get("away_team"))
            mid = index.get((home, away)) or index.get((away, home))
            if not mid:
                continue

            for bk in ev.get("bookmakers", []) or []:
                for mk in (bk.get("markets", []) or []):
                    keym = mk.get("key")
                    outs = mk.get("outcomes", []) or []
                    if keym == "h2h":
                        for o in outs:
                            name, price = o.get("name"), o.get("price")
                            if name and price:
                                rows.append({"match_id": str(mid), "market": "1X2",
                                             "selection": name, "price": float(price),
                                             "book": bk.get("title","")})
                    elif keym == "totals":
                        line = outs[0].get("point") if outs else None
                        for o in outs:
                            name, price = o.get("name"), o.get("price")
                            point = o.get("point", line)
                            if name and price and point is not None:
                                rows.append({"match_id": str(mid), "market": f"OU{point}",
                                             "selection": name, "price": float(price),
                                             "book": bk.get("title","")})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if book_preference:
        df["_pref"] = (df["book"].str.lower() == str(book_preference).lower()).astype(int)
        df = (df.sort_values(["match_id","market","selection","_pref"], ascending=[True,True,True,False])
                .drop_duplicates(subset=["match_id","market","selection"], keep="first")
                .drop(columns=["_pref"]))
    return df.reset_index(drop=True)
