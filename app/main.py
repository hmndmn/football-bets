import os
from flask import Flask, request, render_template
import pandas as pd

from app.sheets import SheetClient
from app.data_sources import fetch_fixtures, fetch_odds
from app.model import predict_match, market_probs
from app.markets import find_value_bets
from app.staking import StakeSizer

app = Flask(__name__)

@app.route("/")
def health():
    return "OK", 200

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y")

@app.route("/run")
def run():
    sheet_name   = os.getenv("SHEET_NAME", "Football Picks")
    leagues      = [s.strip() for s in os.getenv("LEAGUES", "EPL,LaLiga").split(",") if s.strip()]
    hours_ahead  = int(os.getenv("HOURS_AHEAD", "240"))
    edge_thresh  = float(os.getenv("EDGE_THRESHOLD", "0.05"))
    bankroll     = float(os.getenv("BANKROLL_START", "500"))
    min_stake    = float(os.getenv("MIN_STAKE_PCT", "0.0025"))
    max_stake    = float(os.getenv("MAX_STAKE_PCT", "0.025"))
    max_picks    = int(os.getenv("MAX_PICKS", "50"))
    book_filter  = os.getenv("BOOK_FILTER", "").strip()

    # 1) Pull fixtures
    fixtures = fetch_fixtures(leagues=leagues, hours_ahead=hours_ahead)
    # 2) Pull odds
    odds = fetch_odds(leagues=leagues, hours_ahead=hours_ahead)

    # 3) Build model probabilities per match
    model_rows = []
    grouped = fixtures.groupby("match_id") if not fixtures.empty else []
    for match_id, rows in grouped:
        row = rows.iloc[0]
        league = row["league"]; home = row["home"]; away = row["away"]
        lh, la = predict_match(home, away, league)
        # collect OU lines seen for this match (from odds rows, if any)
        ou_lines = sorted(
            set(
                float(x)
                for x in odds.loc[odds["match_id"] == match_id, "market_param"].dropna().unique()
                if str(x).replace(".","",1).isdigit()
            )
        ) if not odds.empty else []
        probs = market_probs(lh, la, ou_lines=tuple(ou_lines) if ou_lines else (2.5, 3.5))
        model_rows.append({
            "match_id": match_id,
            "league": league,
            "home": home,
            "away": away,
            **probs
        })
    model = pd.DataFrame(model_rows) if model_rows else pd.DataFrame(columns=["match_id","league","home","away"])

    # 4) Find value bets
    picks = find_value_bets(fixtures, odds, model, edge_threshold=edge_thresh, max_picks=max_picks, book_filter=book_filter)

    # 5) Size stakes
    sizer = StakeSizer(bankroll, min_stake_pct=min_stake, max_stake_pct=max_stake)
    if not picks.empty:
        picks = sizer.apply(picks)

    # 6) Write to Google Sheets
    sc = SheetClient(sheet_name)
    sc.write_table("fixtures", fixtures)
    sc.write_table("odds", odds)
    sc.write_table("model", model)
    sc.write_table("picks", picks)

    return (
        f"OK fixtures={len(fixtures)} odds_rows={len(odds)} "
        f"model_rows={len(model)} picks={len(picks)}",
        200
    )

@app.route("/view")
def view_picks():
    sheet_name = os.getenv("SHEET_NAME", "Football Picks")
    sc = SheetClient(sheet_name)

    q_market  = request.args.get("market", "").strip()
    q_league  = request.args.get("league", "").strip()
    q_book    = request.args.get("book", "").strip()
    q_minedge = request.args.get("min_edge", "").strip()
    q_minprob = request.args.get("min_prob", "").strip()

    df = sc.read_table("picks")
    if df is None or df.empty:
        return render_template("picks.html", picks=[], leagues=[], books=[], markets=[], sheet_name=sheet_name)

    # Filters
    if q_market:
        df = df[df["market"] == q_market]
    if q_league:
        df = df[df["league"] == q_league]
    if q_book:
        df = df[df["book"] == q_book]
    if q_minedge:
        try:
            df = df[df["edge"] >= float(q_minedge)]
        except:
            pass
    if q_minprob:
        try:
            df = df[df["model_prob"] >= float(q_minprob)]
        except:
            pass

    # Sort by edge desc
    if not df.empty and "edge" in df.columns:
        df = df.sort_values(by="edge", ascending=False)

    leagues = sorted(set(sc.read_table("picks")["league"])) if sc.read_table("picks") is not None and not sc.read_table("picks").empty else []
    books   = sorted(set(sc.read_table("picks")["book"])) if sc.read_table("picks") is not None and not sc.read_table("picks").empty else []
    markets = sorted(set(sc.read_table("picks")["market"])) if sc.read_table("picks") is not None and not sc.read_table("picks").empty else []

    # Convert for template
    rows = df.to_dict(orient="records") if df is not None and not df.empty else []
    return render_template("picks.html",
                           picks=rows,
                           leagues=leagues,
                           books=books,
                           markets=markets,
                           sheet_name=sheet_name)

@app.route("/diag")
def diag():
    # Simple diagnostic without touching data_sources
    env = {
        "provider": os.getenv("ODDS_PROVIDER", "(not set)"),
        "leagues": os.getenv("LEAGUES", ""),
        "apifootball_leagues": os.getenv("APIFOOTBALL_LEAGUES", ""),
        "hours_ahead": os.getenv("HOURS_AHEAD", ""),
        "edge_threshold": os.getenv("EDGE_THRESHOLD", ""),
        "max_picks": os.getenv("MAX_PICKS", ""),
        "book_filter": os.getenv("BOOK_FILTER", ""),
        "sheet_name": os.getenv("SHEET_NAME", ""),
        "has_apifootball_key": bool(os.getenv("APIFOOTBALL_KEY")),
        "has_odds_api_key": bool(os.getenv("ODDS_API_KEY")),
    }
    return env, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
