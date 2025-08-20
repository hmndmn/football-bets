import os
import json
from pathlib import Path
from flask import Flask, request, render_template, jsonify
import pandas as pd

# Internal modules
from app.data_sources import fetch_fixtures, fetch_odds
from app.model import predict_match, market_probs
from app.markets import find_value_bets
from app.staking import StakeSizer

# Google Sheets is optional now
USE_SHEETS = os.getenv("USE_SHEETS", "false").lower() in ("1", "true", "yes")

if USE_SHEETS:
    try:
        from app.sheets import SheetClient  # will error if creds invalid
    except Exception:
        SheetClient = None
        USE_SHEETS = False

app = Flask(__name__, template_folder="templates", static_folder="static")

# Where we persist latest picks on the server (so /view works without Sheets)
DATA_DIR = Path("/opt/render/project/src/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
PICKS_JSON = DATA_DIR / "latest_picks.json"
SUMMARY_JSON = DATA_DIR / "latest_summary.json"

def df_to_json_records(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    # replace non-finite with None so JSON is valid
    safe = df.replace([float("inf"), float("-inf")], pd.NA).fillna(value=pd.NA)
    # convert to pure python types
    return json.loads(safe.to_json(orient="records", date_format="iso"))

@app.route("/")
def health():
    return "OK", 200

@app.route("/probe_odds")
def probe_odds():
    regions = os.getenv("ODDS_REGIONS", "uk,eu,us")
    leagues = [s.strip() for s in os.getenv("LEAGUES", "EPL,LaLiga").split(",") if s.strip()]
    out = {"ok": True, "regions": regions, "leagues": {}}
    for lg in leagues:
        status, sample, headers = fetch_odds(lg, limit_only=True)
        out["leagues"][lg] = {
            "status": status,
            "sample": sample,
            "x-requests-remaining": headers.get("x-requests-remaining"),
            "x-requests-used": headers.get("x-requests-used"),
            "events": sample.get("events") if isinstance(sample, dict) else None,
        }
    return jsonify(out), 200

@app.route("/run")
def run():
    # Config
    sheet_name   = os.getenv("SHEET_NAME", "Football Picks")
    leagues      = [s.strip() for s in os.getenv("LEAGUES", "EPL,LaLiga").split(",") if s.strip()]
    hours_ahead  = int(os.getenv("HOURS_AHEAD", "168"))
    edge_thresh  = float(os.getenv("EDGE_THRESHOLD", "0.05"))
    bankroll     = float(os.getenv("BANKROLL_START", "500"))
    min_stake    = float(os.getenv("MIN_STAKE_PCT", "0.0025"))
    max_stake    = float(os.getenv("MAX_STAKE_PCT", "0.025"))
    max_picks    = int(os.getenv("MAX_PICKS", "50"))
    book_pref    = os.getenv("BOOK_FILTER", "")  # optional filter by book name

    # Fetch fixtures & odds
    fixtures_df = fetch_fixtures(leagues=leagues, hours_ahead=hours_ahead)
    odds_df     = fetch_odds(leagues=leagues)

    fixtures_ct = len(fixtures_df) if fixtures_df is not None else 0
    odds_ct     = len(odds_df) if odds_df is not None else 0

    # Basic model: for each fixture, compute market probabilities
    model_rows = []
    if fixtures_ct > 0:
        for _, f in fixtures_df.iterrows():
            lh, la = predict_match(f["home"], f["away"], f["league"])
            # collect OU lines available from odds for this match (if any)
            ou_lines = []
            if odds_ct > 0:
                sub = odds_df[(odds_df["match_id"] == f["match_id"]) & (odds_df["market"].str.startswith("OU"))]
                if not sub.empty:
                    for L in sorted(set(sub["market"].str.extract(r"OU([0-9.]+)")[0].dropna().astype(float))):
                        ou_lines.append(float(L))
            probs = market_probs(lh, la, ou_lines=tuple(ou_lines) if ou_lines else (2.5, 3.5))
            model_rows.append({
                "match_id": f["match_id"],
                "league": f["league"],
                "home": f["home"],
                "away": f["away"],
                **probs
            })
    model_df = pd.DataFrame(model_rows)

    # Find value bets
    picks_df = find_value_bets(
        fixtures=fixtures_df,
        odds=odds_df,
        model_probs=model_df,
        edge_threshold=edge_thresh,
        book_filter=book_pref,
        max_picks=max_picks
    )

    # Staking suggestions
    sizer = StakeSizer(bankroll=bankroll, min_pct=min_stake, max_pct=max_stake)
    if not picks_df.empty:
        picks_df = sizer.apply(picks_df)

    # Save JSON for /view (so we don't need Google Sheets)
    picks_records = df_to_json_records(picks_df)
    summary = {
        "fixtures": fixtures_ct,
        "odds_rows": odds_ct,
        "model_rows": len(model_df),
        "picks": len(picks_records)
    }
    with open(PICKS_JSON, "w") as fp:
        json.dump(picks_records, fp)
    with open(SUMMARY_JSON, "w") as fp:
        json.dump(summary, fp)

    # Optional: also write to Google Sheets if enabled and available
    if USE_SHEETS and SheetClient is not None:
        try:
            sc = SheetClient(sheet_name)
            sc.write_table("fixtures", fixtures_df)
            sc.write_table("odds", odds_df)
            sc.write_table("model", model_df)
            sc.write_table("picks", picks_df)
        except Exception as e:
            # Don't crash run; just log to response string
            summary["sheets_error"] = str(e)

    return (
        f"OK fixtures={fixtures_ct} odds_rows={odds_ct} "
        f"model_rows={len(model_df)} picks={len(picks_records)}"
        + (f" sheets_error={summary.get('sheets_error')}" if summary.get("sheets_error") else ""),
        200
    )

@app.route("/view")
def view_picks():
    """
    Render picks from local JSON (no Sheets needed).
    Supports query filters via ?market=&league=&book=&min_edge=&min_prob=
    """
    # Load picks JSON
    rows = []
    if PICKS_JSON.exists():
        try:
            with open(PICKS_JSON, "r") as fp:
                rows = json.load(fp)
        except Exception:
            rows = []

    # Filters
    market   = request.args.get("market", "").strip()
    league   = request.args.get("league", "").strip()
    book     = request.args.get("book", "").strip()
    min_edge = float(request.args.get("min_edge", "0") or 0)
    min_prob = float(request.args.get("min_prob", "0") or 0)

    def keep(r):
        if market and r.get("market", "").lower() != market.lower():
            return False
        if league and r.get("league", "").lower() != league.lower():
            return False
        if book and r.get("book", "").lower() != book.lower():
            return False
        try:
            if float(r.get("edge", 0) or 0) < min_edge:
                return False
            if float(r.get("model_prob", 0) or 0) < min_prob:
                return False
        except Exception:
            return False
        return True

    filtered = [r for r in rows if keep(r)]

    # Sort by edge desc
    filtered.sort(key=lambda x: float(x.get("edge", 0) or 0), reverse=True)

    # Summary (if available)
    summary = {}
    if SUMMARY_JSON.exists():
        try:
            with open(SUMMARY_JSON, "r") as fp:
                summary = json.load(fp)
        except Exception:
            summary = {}

    return render_template(
        "picks.html",
        rows=filtered,
        summary=summary,
        # feed current filters back to the form
        market_value=market,
        league_value=league,
        book_value=book,
        min_edge_value=min_edge if min_edge else "",
        min_prob_value=min_prob if min_prob else ""
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
