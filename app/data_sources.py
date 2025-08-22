import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# -------------------------------
# Helpers / config
# -------------------------------

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "").strip()

# e.g. "EPL:39,LaLiga:140,SerieA:135,Bundesliga:78,Ligue1:61"
APIFOOTBALL_LEAGUES = os.getenv(
    "APIFOOTBALL_LEAGUES",
    "EPL:39,LaLiga:140,SerieA:135,Bundesliga:78,Ligue1:61"
)

BOOK_FILTER = os.getenv("BOOK_FILTER", "").strip()  # optional: filter to a single book name (e.g., "bet365")

# Which odds markets to extract
WANTED_MARKETS = {"1X2", "OU2.5", "OU3.5", "AH-0.5", "AH+0.5", "AH-1.0", "AH+1.0"}

# -------------------------------
# League id mapping
# -------------------------------

def parse_league_map():
    """
    Returns dict like {"EPL": "39", "LaLiga": "140", ...}
    Only includes leagues listed in LEAGUES env if provided.
    """
    raw = APIFOOTBALL_LEAGUES
    picks = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, lid = part.split(":", 1)
        name = name.strip()
        lid = lid.strip()
        picks[name] = lid

    # apply LEAGUES filter if set (comma-separated), otherwise keep all
    env_leagues = os.getenv("LEAGUES", "").strip()
    if env_leagues:
        wanted = {s.strip() for s in env_leagues.split(",") if s.strip()}
        picks = {k: v for k, v in picks.items() if k in wanted}

    return picks

# -------------------------------
# API-Football low-level calls
# -------------------------------

def af_headers():
    if not API_FOOTBALL_KEY:
        raise RuntimeError("API_FOOTBALL_KEY missing")
    return {"x-apisports-key": API_FOOTBALL_KEY}

def af_get(path: str, params: dict, sleep_sec: float = 0.0):
    """
    Simple GET wrapper with tiny throttle to be nice.
    Raises for HTTP errors; returns parsed json.
    """
    if sleep_sec > 0:
        time.sleep(sleep_sec)
    url = f"https://v3.football.api-sports.io{path}"
    r = requests.get(url, headers=af_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json(), r.headers

# -------------------------------
# Fixtures (API-Football)
# -------------------------------

def fetch_fixtures(leagues, hours_ahead=240):
    """
    leagues: list like ["EPL","LaLiga",...]
    Returns DataFrame columns: match_id, league, utc_kickoff, home, away
    """
    if not leagues:
        # derive from map if not provided
        leagues = list(parse_league_map().keys())

    league_map = parse_league_map()
    if not league_map:
        return pd.DataFrame(columns=["match_id","league","utc_kickoff","home","away"])

    now_utc = datetime.now(timezone.utc)
    to_utc   = now_utc + timedelta(hours=int(hours_ahead))
    date_from = now_utc.strftime("%Y-%m-%d")
    date_to   = to_utc.strftime("%Y-%m-%d")

    rows = []
    for lname in leagues:
        if lname not in league_map:
            continue
        lid = league_map[lname]
        params = {
            "league": lid,
            "season": now_utc.year,      # season year, e.g. 2025
            "from": date_from,
            "to": date_to,
            "timezone": "UTC",
        }
        try:
            data, _hdr = af_get("/fixtures", params, sleep_sec=0.15)
        except Exception:
            continue

        resp = data.get("response", []) or []
        for item in resp:
            fix = item.get("fixture", {})
            teams = item.get("teams", {})
            dt = fix.get("date")
            try:
                # normalize kickoff to pure UTC ISO Z
                ts = fix.get("timestamp")
                if ts:
                    kickoff = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    kickoff = dt
            except Exception:
                kickoff = dt

            rows.append({
                "match_id": str(fix.get("id")),
                "league": lname,
                "utc_kickoff": kickoff,
                "home": (teams.get("home") or {}).get("name"),
                "away": (teams.get("away") or {}).get("name"),
            })

    df = pd.DataFrame(rows, columns=["match_id","league","utc_kickoff","home","away"])
    # Drop any null ids
    df = df[df["match_id"].notna()]
    return df.reset_index(drop=True)

# -------------------------------
# Odds (API-Football)
# -------------------------------

def _map_bet_to_market_and_selection(bet_name: str, value: str):
    """
    Map API-Football 'bet' and 'value' into our (market, selection).
    Returns (market, selection) or (None, None) if not supported.
    """
    # 1X2
    if bet_name.lower() in ("match winner", "1x2"):
        v = value.strip().lower()
        if v in ("home", "1"):
            return ("1X2", "Home")
        if v in ("draw", "x"):
            return ("1X2", "Draw")
        if v in ("away", "2"):
            return ("1X2", "Away")
        return (None, None)

    # Over/Under: values like "Over 2.5" / "Under 3.5"
    if bet_name.lower() in ("over/under", "totals"):
        parts = value.strip().split()
        if len(parts) == 2:
            side, line = parts
            try:
                fline = float(line)
                market = f"OU{fline}"
                sel = "Over" if side.lower() == "over" else "Under"
                return (market, sel)
            except Exception:
                return (None, None)

    # Asian Handicap: values like "Home -0.5", "Away +1.0"
    if "asian" in bet_name.lower():
        parts = value.strip().split()
        if len(parts) == 2:
            side, line = parts
            try:
                fline = float(line)
                # normalize to signed with one decimal
                if fline >= 0:
                    m = f"AH+{fline:.1f}"
                else:
                    m = f"AH{fline:.1f}"
                sel = "Home" if side.lower() == "home" else "Away"
                return (m, sel)
            except Exception:
                return (None, None)

    return (None, None)

def fetch_odds(fixtures_df: pd.DataFrame, leagues, hours_ahead=240):
    """
    Pull odds from API-Football for the given fixtures.
    Returns DataFrame columns:
      match_id, league, utc_kickoff, market, selection, price, book, home, away
    """
    if fixtures_df is None or fixtures_df.empty:
        return pd.DataFrame(columns=["match_id","league","utc_kickoff","market","selection","price","book","home","away"])

    rows = []
    # Optional bookmaker filter by name (case-insensitive substring)
    want_book = BOOK_FILTER.lower() if BOOK_FILTER else ""

    # API-Football odds endpoint supports fixture param
    # We'll request per fixture (7500/day gives us plenty of headroom)
    fixture_groups = fixtures_df.groupby("match_id", as_index=False).first()

    for _, row in fixture_groups.iterrows():
        fid = row["match_id"]
        league = row["league"]
        kickoff = row["utc_kickoff"]
        home = row["home"]
        away = row["away"]

        params = {"fixture": fid}
        try:
            data, _hdr = af_get("/odds", params, sleep_sec=0.12)
        except Exception:
            continue

        resp = data.get("response", []) or []
        for item in resp:
            # item is typically per fixture -> per bookmaker list under 'bookmakers'
            for bm in (item.get("bookmakers") or []):
                book_title = bm.get("name") or bm.get("id") or "Book"
                if want_book and (want_book not in str(book_title).lower()):
                    continue

                for bet in (bm.get("bets") or []):
                    bet_name = bet.get("name") or ""
                    for val in (bet.get("values") or []):
                        price = val.get("odd")
                        vlabel = val.get("value")
                        if price is None or vlabel is None:
                            continue
                        try:
                            price_f = float(price)
                        except Exception:
                            # odds come as string — if cannot parse, skip
                            continue

                        market, selection = _map_bet_to_market_and_selection(bet_name, vlabel)
                        if market is None or selection is None:
                            continue
                        if market not in WANTED_MARKETS:
                            # keep only our selected set
                            continue

                        rows.append({
                            "match_id": str(fid),
                            "league": league,
                            "utc_kickoff": kickoff,
                            "market": market,
                            "selection": selection,
                            "price": price_f,
                            "book": str(book_title),
                            "home": home,
                            "away": away,
                        })

    if not rows:
        return pd.DataFrame(columns=["match_id","league","utc_kickoff","market","selection","price","book","home","away"])

    df = pd.DataFrame(rows, columns=["match_id","league","utc_kickoff","market","selection","price","book","home","away"])

    # Keep the best price per (match, market, selection)
    df.sort_values(["match_id","market","selection","price"], ascending=[True,True,True,False], inplace=True)
    df = df.groupby(["match_id","market","selection"], as_index=False).first()

    return df.reset_index(drop=True)

# -------------------------------
# Optional probe route support
# -------------------------------

def probe_apifootball(leagues, hours_ahead=240):
    """
    Lightweight probe to check headers/limits and that fixtures are returned.
    """
    league_map = parse_league_map()
    if not leagues:
        leagues = list(league_map.keys())

    now_utc = datetime.now(timezone.utc)
    to_utc   = now_utc + timedelta(hours=int(hours_ahead))
    date_from = now_utc.strftime("%Y-%m-%d")
    date_to   = to_utc.strftime("%Y-%m-%d")

    report = {}
    headers = {}
    tried = []

    for lname in leagues:
        lid = league_map.get(lname)
        if not lid:
            continue
        params = {"league": lid, "season": now_utc.year, "from": date_from, "to": date_to, "timezone": "UTC"}
        tried.append({lname: {"league_id": lid, "params": params}})
        try:
            data, hdr = af_get("/fixtures", params, sleep_sec=0.1)
            resp = data.get("response", []) or []
            first_item = (resp[0] if resp else {})
            headers[lname] = {
                "status": 200,
                "x-ratelimit-requests-limit": hdr.get("x-ratelimit-requests-limit"),
                "x-ratelimit-requests-remaining": hdr.get("x-ratelimit-requests-remaining"),
            }
            report[lname] = {
                "status": 200,
                "errors": data.get("errors", []),
                "results_count": data.get("results"),
                "first_item_sample": first_item,
                "paging": data.get("paging", {}),
            }
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", 0)
            headers[lname] = {"status": status}
            report[lname] = {"status": status, "errors": [str(e)]}
        except Exception as e:
            headers[lname] = {"status": 0}
            report[lname] = {"status": 0, "errors": [str(e)]}

    return {
        "ok": True,
        "window": {"from": date_from, "to": date_to},
        "leagues_tried": tried,
        "headers": headers,
        "report": report,
    }
