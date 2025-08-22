import os
from flask import Flask, request, render_template, jsonify
import pandas as pd

from app.sheets import SheetClient
from app.data_sources import fetch_fixtures, fetch_odds
from app.model import predict_match, market_probs
from app.markets import find_value_bets
from app.staking import StakeSizer

app = Flask(__name__)

def _env(name, default=""):
    v = os.getenv(name)
    return v if v is not None else default

@app.route("/")
def health():
    return "OK", 200

@app.route("/diag")
def diag():
    info = {
        "sheet_name": _env("SHEET_NAME", "Football Picks"),
        "leagues": _env("LEAGUES", ""),
        "hours_ahead": _env("HOURS_AHEAD", "240"),
        "edge_threshold": _env("EDGE_THRESHOLD", "0.05"),
        "max_picks": _env("MAX_PICKS", "50"),
        "book_filter": _env("BOOK_FILTER", ""),
        "apifootball_leagues": _env("APIFOOTBALL_LEAGUES", ""),
        "provider": _env("ODDS_PROVIDER", "(not set)"),
        "has_apifootball_key": bool(os.getenv("APIFOOTBALL_KEY")),
        "has_odds_api_key": bool(os.getenv("ODDS_API_KEY")),
    }
    return jsonify(info), 200

@app.route("/run")
def run():
    sheet_name   = _env("SHEET_NAME", "Football Picks")
    leagues      = [s.strip() for s in _env("LEAGUES", "EPL,LaLiga").split(",") if s.strip()]
    hours_ahead  = int(_env("HOURS_AHEAD", "240"))
    book_pref    = _env("BOOK_FILTER", "")  # allow any book by default
    edge_thresh  = float(_env("EDGE_THRESHOLD", "0.05"))
    bankroll     = float(_env("BANKROLL_START", "500"))
    min_stake    = float(_env("MIN_STAKE_PCT", "0.0025"))
    max_stake    = float(_env("MAX_STAKE_PCT", "0.025"))
    max_picks    = int(_env("MAX_PICKS", "50"))

    # 1) Fixtures
    fixtures = fetch_fixtures(leagues=leagues, hours_ahead=hours_ahead)
    if fixtures.empty:
        # still write empty tabs so /view doesn’t 500
        try:
            sc = SheetClient(sheet_name)
            sc.write_table("Fixtures", fixtures)
            sc.write_table("Odds", pd.DataFrame(columns=["match_id","league","utc_kickoff","market","selection","price","book","home","away"]))
            sc.write_table("Model", pd.DataFrame(columns=["match_id","lh","la"]))
            sc.write_table("Picks", pd.DataFrame(columns=[
                "utc_kickoff","match","market","selection","price","book","model_prob","implied","edge",
                "stake_pct","stake_amt","league","home","away","match_id"
            ]))
        except Exception:
            pass
        return f"OK fixtures=0 odds_rows=0 model_rows=0 picks=0", 200

    # 2) Odds (PASS fixtures_df!)
    odds = fetch_odds(leagues=leagues, hours_ahead=hours_ahead, fixtures_df=fixtures)

    # 3) Simple model probs per match, then compute market probabilities and find value
    model_rows = []
    for _, m in fixtures.iterrows():
        lh, la = predict_match(m["home"], m["away"], m["league"])
        model_rows.append({"match_id": m["match_id"], "lh": lh, "la": la})
    model_df = pd.DataFrame(model_rows)

    # Compute market probabilities for each odds row (vectorized enough for our size)
    probs_rows = []
    for _, row in odds.iterrows():
        mid = row["match_id"]
        mm = model_df[model_df["match_id"] == mid]
        if mm.empty:
            continue
        lh = float(mm["lh"].iloc[0])
        la = float(mm["la"].iloc[0])

        # Gather OU lines seen for this match to feed into market_probs
        ou_lines = odds[(odds["match_id"] == mid) & (odds["market"].str.startswith("OU"))]["market"] \
                      .str.replace("OU","", regex=False) \
                      .str.replace(" AH","", regex=False) \
                      .tolist()
        try:
            ou_lines = tuple(float(x) for x in ou_lines if x)
        except Exception:
            ou_lines = tuple()

        mp = market_probs(lh, la, ou_lines=ou_lines)
        probs_rows.append({
            "match_id": mid,
            **mp
        })
    probs_df = pd.DataFrame(probs_rows).drop_duplicates(subset=["match_id"]) if probs_rows else pd.DataFrame(columns=["match_id"])

    # Merge model probs into odds (for value calculation)
    merged = odds.merge(probs_df, on="match_id", how="left")

    # 4) Picks: find value bets
    sizer = StakeSizer(bankroll=bankroll, min_pct=min_stake, max_pct=max_stake)
    picks = find_value_bets(merged, book_filter=book_pref, edge_threshold=edge_thresh, stake_sizer=sizer)
    if max_picks and len(picks) > max_picks:
        picks = picks.sort_values("edge", ascending=False).head(max_picks)

    # 5) Write to Google Sheets
    try:
        sc = SheetClient(sheet_name)
        sc.write_table("Fixtures", fixtures)
        sc.write_table("Odds", odds)
        sc.write_table("Model", model_df)
        sc.write_table("Picks", picks)
    except Exception as e:
        # still return counts so you can see status
        return f"OK fixtures={len(fixtures)} odds_rows={len(odds)} model_rows={len(model_df)} picks={len(picks)} (sheet write error: {e})", 200

    return f"OK fixtures={len(fixtures)} odds_rows={len(odds)} model_rows={len(model_df)} picks={len(picks)}", 200

@app.route("/view")
def view_picks():
    sheet_name = _env("SHEET_NAME", "Football Picks")
    market = request.args.get("market", "").strip()
    league = request.args.get("league", "").strip()
    book = request.args.get("book", "").strip()
    min_edge = float(request.args.get("min_edge", "0") or 0)
    min_prob = float(request.args.get("min_prob", "0") or 0)

    sc = SheetClient(sheet_name)
    picks = sc.read_table("Picks")

    if picks is None or picks.empty:
        return render_template("picks.html", sheet_name=sheet_name, rows=[], leagues=[], books=[], markets=[])

    # Filter
    df = picks.copy()
    if market:
        df = df[df["market"] == market]
    if league:
        df = df[df["league"] == league]
    if book:
        df = df[df["book"] == book]
    if min_edge:
        df = df[df["edge"] >= min_edge]
    if min_prob:
        df = df[df["model_prob"] >= min_prob]

    # Build dropdowns
    leagues = sorted(picks["league"].dropna().unique().tolist())
    books = sorted(picks["book"].dropna().unique().tolist())
    markets = sorted(picks["market"].dropna().unique().tolist())

    rows = df.sort_values(["utc_kickoff","edge"], ascending=[True, False]).to_dict(orient="records")
    return render_template("picks.html", sheet_name=sheet_name, rows=rows, leagues=leagues, books=books, markets=markets)
