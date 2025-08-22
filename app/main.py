import os
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, render_template, request
import pandas as pd
import requests

# Local modules
from app.sheets import SheetClient
from app.data_sources import fetch_fixtures, fetch_odds
from app.model import predict_match, market_probs
from app.markets import find_value_bets
from app.staking import StakeSizer

app = Flask(__name__)

def get_env_list(name: str, default: str = ""):
    raw = os.getenv(name, default)
    return [s.strip() for s in raw.split(",") if s.strip()]

def parse_apifootball_leagues():
    """
    Env format: 'EPL:39,LaLiga:140,SerieA:135,Bundesliga:78,Ligue1:61'
    Returns dict like {'EPL': '39', ...}
    """
    mapping = {}
    raw = os.getenv("APIFOOTBALL_LEAGUES", "")
    for part in raw.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            mapping[k.strip()] = v.strip()
    return mapping

@app.route("/")
def health():
    return "OK", 200

@app.route("/diag")
def diag():
    cfg = {
        "sheet_name": os.getenv("SHEET_NAME", "Football Picks"),
        "provider": os.getenv("PROVIDER", "(not set)"),
        "leagues": os.getenv("LEAGUES", ""),
        "hours_ahead": os.getenv("HOURS_AHEAD", ""),
        "edge_threshold": os.getenv("EDGE_THRESHOLD", ""),
        "max_picks": os.getenv("MAX_PICKS", ""),
        "book_filter": os.getenv("BOOK_FILTER", ""),
        "has_odds_api_key": bool(os.getenv("ODDS_API_KEY")),
        "has_apifootball_key": bool(os.getenv("API_FOOTBALL_KEY")),
        "apifootball_leagues": os.getenv("APIFOOTBALL_LEAGUES", ""),
    }
    return jsonify(cfg)

@app.route("/probe_apifootball")
def probe_apifootball():
    """
    Calls API-Football fixtures for a small window and returns raw diagnostics.
    Verifies: key present, season=2025, league IDs, date window, and headers.
    """
    key = os.getenv("API_FOOTBALL_KEY")
    leagues_map = parse_apifootball_leagues()  # {'EPL':'39', ...}
    hours_ahead = int(os.getenv("HOURS_AHEAD", "240"))

    now_utc = datetime.now(timezone.utc)
    start = now_utc.strftime("%Y-%m-%d")
    end = (now_utc + timedelta(hours=hours_ahead)).strftime("%Y-%m-%d")

    if not key:
        return jsonify({"ok": False, "error": "API_FOOTBALL_KEY missing"}), 200
    if not leagues_map:
        return jsonify({"ok": False, "error": "APIFOOTBALL_LEAGUES missing/empty"}), 200

    tried = []
    results = {}
    headers_out = {}
    base = "https://v3.football.api-sports.io/fixtures"

    for lname, lid in leagues_map.items():
        params = {
            "league": lid,
            "season": "2025",
            "from": start,
            "to": end,
            "timezone": "UTC",
        }
        tried.append({lname: {"league_id": lid, "params": params}})
        try:
            resp = requests.get(base, params=params, headers={"x-apisports-key": key}, timeout=30)
            headers_out[lname] = {
                "status": resp.status_code,
                "x-ratelimit-requests-remaining": resp.headers.get("x-ratelimit-requests-remaining"),
                "x-ratelimit-requests-limit": resp.headers.get("x-ratelimit-requests-limit"),
            }
            js = resp.json()
            results[lname] = {
                "status": resp.status_code,
                "results_count": js.get("results"),
                "errors": js.get("errors"),
                "paging": js.get("paging"),
                "first_item_sample": (js.get("response") or [None])[0],
            }
        except Exception as e:
            results[lname] = {"status": "exception", "detail": str(e)}

    return jsonify({
        "ok": True,
        "window": {"from": start, "to": end},
        "leagues_tried": tried,
        "headers": headers_out,
        "report": results,
    }), 200

@app.route("/run")
def run():
    # Config
    sheet_name  = os.getenv("SHEET_NAME", "Football Picks")
    leagues     = get_env_list("LEAGUES", "EPL,LaLiga")
    hours_ahead = int(os.getenv("HOURS_AHEAD", "240"))
    book_pref   = os.getenv("BOOK_FILTER", "")
    edge_thresh = float(os.getenv("EDGE_THRESHOLD", "0.05"))
    bankroll    = float(os.getenv("BANKROLL_START", "500"))
    min_stake   = float(os.getenv("MIN_STAKE_PCT", "0.0025"))
    max_stake   = float(os.getenv("MAX_STAKE_PCT", "0.025"))
    max_picks   = int(os.getenv("MAX_PICKS", "50"))

    # 1) Fixtures
    fixtures_df = fetch_fixtures(leagues=leagues, hours_ahead=hours_ahead)
    if fixtures_df is None or fixtures_df.empty:
        return jsonify({
            "ok": True,
            "fixtures": 0,
            "odds_rows": 0,
            "model_rows": 0,
            "picks": 0,
            "detail": "No fixtures returned. Try /probe_apifootball for diagnostics."
        }), 200

    # 2) Odds (requires fixtures_df)
    odds_df = fetch_odds(fixtures_df=fixtures_df, leagues=leagues, hours_ahead=hours_ahead)
    if odds_df is None:
        odds_df = pd.DataFrame(columns=["match_id","league","utc_kickoff","market","selection","price","book","home","away"])

    # 3) Model probabilities per match (simple Poisson baseline)
    model_rows = []
    for _, m in fixtures_df.iterrows():
        lh, la = predict_match(m["home"], m["away"], m["league"])
        # Include OU 2.5 and 3.5 as defaults
        probs = market_probs(lh, la, ou_lines=(2.5, 3.5))
        model_rows.append({
            "match_id": m["match_id"],
            "league": m["league"],
            "utc_kickoff": m["utc_kickoff"],
            "home": m["home"],
            "away": m["away"],
            **probs
        })
    model_df = pd.DataFrame(model_rows)

    # 4) Find value picks
    picks_df = find_value_bets(
        odds_df=odds_df,
        model_df=model_df,
        league_filter=set(leagues),
        book_pref=book_pref,
        edge_threshold=edge_thresh
    )

    # Cap and sort
    if not picks_df.empty:
        picks_df = picks_df.sort_values(["utc_kickoff", "edge"], ascending=[True, False]).head(max_picks)

    # 5) Save to Google Sheets
    sc = SheetClient(sheet_name)
    sc.write_table("Fixtures", fixtures_df)
    sc.write_table("Odds", odds_df)
    sc.write_table("Model", model_df)
    sc.write_table("Picks", picks_df)
    sc.write_summary(
        {
            "fixtures": len(fixtures_df),
            "odds_rows": len(odds_df),
            "model_rows": len(model_df),
            "picks": len(picks_df)
        }
    )

    return jsonify({
        "ok": True,
        "fixtures": len(fixtures_df),
        "odds_rows": len(odds_df),
        "model_rows": len(model_df),
        "picks": len(picks_df)
    }), 200

@app.route("/view")
def view_picks():
    # Load from sheet and render
    sheet_name = os.getenv("SHEET_NAME", "Football Picks")
    sc = SheetClient(sheet_name)
    picks = sc.read_table("Picks")
    if picks is None or picks.empty:
        return render_template(
            "picks.html",
            rows=[],
            leagues=[],
            books=[],
            markets=[],
            qs={}
        )
    leagues = sorted(picks["league"].dropna().unique().tolist())
    books   = sorted(picks["book"].dropna().unique().tolist())
    markets = sorted(picks["market"].dropna().unique().tolist())

    # Filters from query
    qs = {
        "league": request.args.get("league", ""),
        "book": request.args.get("book", ""),
        "market": request.args.get("market", ""),
        "min_edge": request.args.get("min_edge", ""),
        "min_prob": request.args.get("min_prob", ""),
    }

    df = picks.copy()
    if qs["league"]:
        df = df[df["league"] == qs["league"]]
    if qs["book"]:
        df = df[df["book"] == qs["book"]]
    if qs["market"]:
        df = df[df["market"] == qs["market"]]
    if qs["min_edge"]:
        try:
            df = df[df["edge"] >= float(qs["min_edge"])]
        except Exception:
            pass
    if qs["min_prob"]:
        try:
            df = df[df["model_prob"] >= float(qs["min_prob"])]
        except Exception:
            pass

    rows = df.to_dict(orient="records")
    return render_template(
        "picks.html",
        rows=rows,
        leagues=leagues,
        books=books,
        markets=markets,
        qs=qs
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
