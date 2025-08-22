# app/main.py
import os
from flask import Flask, request, jsonify, render_template
import pandas as pd

from app.sheets import SheetClient
from app.data_sources import fetch_fixtures, fetch_odds, probe_apifootball
from app.model import predict_match, market_probs
from app.markets import find_value_bets
from app.staking import StakeSizer

app = Flask(__name__)

def _env_leagues():
    return [s.strip() for s in os.getenv("LEAGUES", "EPL,LaLiga").split(",") if s.strip()]

@app.route("/")
def health():
    return "OK", 200

@app.route("/probe")
def probe():
    leagues = _env_leagues()
    hours_ahead = int(os.getenv("HOURS_AHEAD", "240"))
    provider = os.getenv("ODDS_PROVIDER", "apifootball")
    res = {
        "provider": provider,
        "leagues": leagues,
    }
    if provider == "apifootball":
        res.update(probe_apifootball(leagues, hours_ahead))
    return jsonify(res), 200

@app.route("/run")
def run():
    # Optional token gate
    want = os.getenv("RUN_TOKEN", "").strip()
    got = request.args.get("token", "").strip()
    if want and got != want:
        return jsonify({"ok": False, "error": "invalid_token"}), 403

    sheet_name  = os.getenv("SHEET_NAME", "Football Picks")
    leagues     = _env_leagues()
    hours_ahead = int(os.getenv("HOURS_AHEAD", "240"))
    edge_thresh = float(os.getenv("EDGE_THRESHOLD", "0.05"))
    max_picks   = int(os.getenv("MAX_PICKS", "50"))
    book_pref   = os.getenv("BOOK_FILTER", "").strip()

    # 1) Fixtures
    fixtures = fetch_fixtures(leagues, hours_ahead)

    # 2) Odds (API-Football now)
    odds = fetch_odds(fixtures, leagues, hours_ahead, book_filter=book_pref)

    # 3) Model probabilities
    model_rows = []
    # decide OU lines to compute: any seen in odds, otherwise default [2.5, 3.5]
    ou_lines_seen = set()
    if not odds.empty:
        for m in odds["market"].unique():
            if m.startswith("OU"):
                try:
                    ou_lines_seen.add(float(m[2:]))
                except Exception:
                    pass
    if not ou_lines_seen:
        ou_lines = (2.5, 3.5)
    else:
        ou_lines = tuple(sorted(ou_lines_seen))

    for _, row in fixtures.iterrows():
        h, a = str(row["home"]), str(row["away"])
        lh, la = predict_match(h, a, str(row["league"]))
        probs = market_probs(lh, la, ou_lines=ou_lines)
        out = {"match_id": row["match_id"], **probs}
        model_rows.append(out)

    model_probs_df = pd.DataFrame(model_rows) if model_rows else pd.DataFrame()

    # 4) Picks
    picks_df = find_value_bets(
        odds_df=odds,
        model_df=model_probs_df,
        edge_threshold=edge_thresh,
        max_picks=max_picks,
        book_filter=book_pref
    )

    # 5) Write to Google Sheets (best effort, no sys.exit)
    try:
        sc = SheetClient(sheet_name)
        sc.write_table("fixtures", fixtures, header=True)
        sc.write_table("odds", odds, header=True)
        sc.write_table("model", model_probs_df, header=True)
        sc.write_table("picks", picks_df, header=True)
        sc.write_summary({
            "fixtures": len(fixtures),
            "odds_rows": len(odds),
            "model_rows": len(model_probs_df),
            "picks": len(picks_df),
        })
    except Exception as e:
        # Do not crash the worker; just return diagnostic
        return jsonify({
            "ok": True,
            "fixtures": len(fixtures),
            "odds_rows": len(odds),
            "model_rows": len(model_probs_df),
            "picks": len(picks_df),
            "sheet_error": str(e)
        }), 200

    return jsonify({
        "ok": True,
        "fixtures": len(fixtures),
        "odds_rows": len(odds),
        "model_rows": len(model_probs_df),
        "picks": len(picks_df),
    }), 200

@app.route("/view")
def view_picks():
    sheet_name = os.getenv("SHEET_NAME", "Football Picks")
    # Filters from query
    qs = {
        "market": request.args.get("market","").strip(),
        "league": request.args.get("league","").strip(),
        "book": request.args.get("book","").strip(),
        "min_edge": request.args.get("min_edge","").strip(),
        "min_prob": request.args.get("min_prob","").strip(),
    }
    try:
        sc = SheetClient(sheet_name)
        picks = sc.read_table("picks")
    except Exception:
        picks = pd.DataFrame(columns=[
            "utc_kickoff","match","market","selection","price","book",
            "model_prob","implied","edge","stake_pct","stake_amt",
            "league","home","away","match_id"
        ])

    # Apply filters in memory
    df = picks.copy()
    if not df.empty:
        if qs["market"]:
            df = df[df["market"].astype(str).str.lower() == qs["market"].lower()]
        if qs["league"]:
            df = df[df["league"].astype(str).str.lower() == qs["league"].lower()]
        if qs["book"]:
            df = df[df["book"].astype(str).str.contains(qs["book"], case=False, na=False)]
        if qs["min_edge"]:
            try:
                m = float(qs["min_edge"])
                df = df[df["edge"] >= m]
            except Exception:
                pass
        if qs["min_prob"]:
            try:
                p = float(qs["min_prob"])
                df = df[df["model_prob"] >= p]
            except Exception:
                pass

    leagues = _env_leagues()
    return render_template(
        "picks.html",
        sheet_name=sheet_name,
        rows=df.to_dict(orient="records"),
        leagues=leagues
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
