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

# Sheets are optional now (can be re-enabled with USE_SHEETS=true)
USE_SHEETS = os.getenv("USE_SHEETS", "false").lower() in ("1", "true", "yes")
if USE_SHEETS:
    try:
        from app.sheets import SheetClient  # may fail if creds not present/valid
    except Exception:
        SheetClient = None
        USE_SHEETS = False

app = Flask(__name__, template_folder="templates", static_folder="static")

# Persisted data for /view (so we don't rely on Sheets)
DATA_DIR = Path("/opt/render/project/src/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
PICKS_JSON = DATA_DIR / "latest_picks.json"
SUMMARY_JSON = DATA_DIR / "latest_summary.json"

def df_to_json_records(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    safe = df.replace([float("inf"), float("-inf")], pd.NA).fillna(value=pd.NA)
    return json.loads(safe.to_json(orient="records", date_format="iso"))

@app.route("/")
def health():
    return "OK", 200

@app.route("/probe_odds")
def probe_odds():
    regions = os.getenv("ODDS_REGIONS", "uk,eu,us")
    leagues = [s.strip() for s in os.getenv("LEAGUES", "EPL,LaLiga").split(",") if s.strip()]
    out = {"ok": True, "regions": regions, "leagues": {}}
    status, sample, headers = fetch_odds(leagues=leagues, limit_only=True)
    out["leagues"] = sample if isinstance(sample, dict) else {"raw": sample}
    out["headers"] = {
        "x-requests-remaining": headers.get("x-requests-remaining"),
        "x-requests-used": headers.get("x-requests-used"),
    }
    out["status"] = status
    return jsonify(out), 200

@app.route("/run")
def run():
    try:
        # Config
        sheet_name   = os.getenv("SHEET_NAME", "Football Picks")
        leagues      = [s.strip() for s in os.getenv("LEAGUES", "EPL,LaLiga").split(",") if s.strip()]
        hours_ahead  = int(os.getenv("HOURS_AHEAD", "168"))
        edge_thresh  = float(os.getenv("EDGE_THRESHOLD", "0.05"))
        bankroll     = float(os.getenv("BANKROLL_START", "500"))
        min_stake    = float(os.getenv("MIN_STAKE_PCT", "0.0025"))
        max_stake    = float(os.getenv("MAX_STAKE_PCT", "0.025"))
        max_picks    = int(os.getenv("MAX_PICKS", "50"))
        book_pref    = os.getenv("BOOK_FILTER", "")  # optional book filter

        # Fetch fixtures & odds
        fixtures_df = fetch_fixtures(leagues=leagues, hours_ahead=hours_ahead)
        odds_df     = fetch_odds(leagues=leagues)

        fixtures_ct = len(fixtures_df) if fixtures_df is not None else 0
        odds_ct     = len(odds_df) if odds_df is not None else 0

        # Model probs
        model_rows = []
        if fixtures_ct > 0:
            for _, f in fixtures_df.iterrows():
                lh, la = predict_match(f["home"], f["away"], f["league"])
                # OU lines from odds for this match, if present
                ou_lines = []
                if odds_ct > 0:
                    sub = odds_df[
                        (odds_df["match_id"] == f["match_id"])
                        & (odds_df["market"].astype(str).str.startswith("OU"))
                    ]
                    if not sub.empty:
                        try:
                            vals = (
                                sub["market"]
                                .astype(str)
                                .str.extract(r"OU([0-9.]+)")[0]
                                .dropna()
                                .astype(float)
                                .tolist()
                            )
                            ou_lines = sorted(set(vals))
                        except Exception:
                            ou_lines = []
                probs = market_probs(lh, la, ou_lines=tuple(ou_lines) if ou_lines else (2.5, 3.5))
                model_rows.append({
                    "match_id": f.get("match_id"),
                    "league": f.get("league"),
                    "home": f.get("home"),
                    "away": f.get("away"),
                    **probs
                })
        model_df = pd.DataFrame(model_rows)

        # Value bets
        picks_df = find_value_bets(
            fixtures=fixtures_df,
            odds=odds_df,
            model_probs=model_df,
            edge_threshold=edge_thresh,
            book_filter=book_pref,
            max_picks=max_picks
        )

        # Staking
        sizer = StakeSizer(bankroll=bankroll, min_pct=min_stake, max_pct=max_stake)
        if picks_df is not None and not picks_df.empty:
            picks_df = sizer.apply(picks_df)

        # Save JSON for /view
        picks_records = df_to_json_records(picks_df if picks_df is not None else pd.DataFrame())
        summary = {
            "fixtures": fixtures_ct,
            "odds_rows": odds_ct,
            "model_rows": len(model_df),
            "picks": len(picks_records),
        }
        try:
            with open(PICKS_JSON, "w") as fp:
                json.dump(picks_records, fp)
            with open(SUMMARY_JSON, "w") as fp:
                json.dump(summary, fp)
        except Exception as e:
            return jsonify({"ok": True, "note": f"saved JSON failed: {e}", **summary}), 200

        # Optional Sheets write (non-fatal)
        if USE_SHEETS and 'SheetClient' in globals() and SheetClient is not None:
            try:
                sc = SheetClient(sheet_name)
                sc.write_table("fixtures", fixtures_df)
                sc.write_table("odds", odds_df)
                sc.write_table("model", model_df)
                sc.write_table("picks", picks_df)
            except Exception as e:
                summary["sheets_error"] = str(e)

        return jsonify({"ok": True, **summary, **({"sheets_error": summary["sheets_error"]} if "sheets_error" in summary else {})}), 200

    except Exception as e:
        # Return readable error instead of 500
        return jsonify({"ok": False, "error": "run_failed", "detail": str(e)}), 200

@app.route("/debug_json")
def debug_json():
    """Quick diagnostics for picks JSON existence and content length."""
    exists = PICKS_JSON.exists()
    size = 0
    rows = 0
    if exists:
        try:
            size = PICKS_JSON.stat().st_size
            with open(PICKS_JSON, "r") as fp:
                j = json.load(fp)
            rows = len(j) if isinstance(j, list) else 0
        except Exception as e:
            return jsonify({"ok": False, "error": f"read_error: {e}", "exists": True, "size": size}), 200
    return jsonify({"ok": True, "exists": exists, "size": size, "rows": rows}), 200

@app.route("/view")
def view_picks():
    """
    Render picks from local JSON (no Sheets needed).
    Supports ?market=&league=&book=&min_edge=&min_prob=
    """
    # Load picks JSON (safe)
    rows = []
    if PICKS_JSON.exists():
        try:
            with open(PICKS_JSON, "r") as fp:
                j = json.load(fp)
            rows = j if isinstance(j, list) else []
        except Exception:
            rows = []

    # Filters
    market   = request.args.get("market", "").strip()
    league   = request.args.get("league", "").strip()
    book     = request.args.get("book", "").strip()
    min_edge = float(request.args.get("min_edge", "0") or 0)
    min_prob = float(request.args.get("min_prob", "0") or 0)

    def keep(r):
        try:
            if market and str(r.get("market", "")).lower() != market.lower():
                return False
            if league and str(r.get("league", "")).lower() != league.lower():
                return False
            if book and str(r.get("book", "")).lower() != book.lower():
                return False
            if float(r.get("edge", 0) or 0) < min_edge:
                return False
            if float(r.get("model_prob", 0) or 0) < min_prob:
                return False
            return True
        except Exception:
            return False

    try:
        filtered = [r for r in rows if keep(r)]
        filtered.sort(key=lambda x: float(x.get("edge", 0) or 0), reverse=True)

        summary = {}
        if SUMMARY_JSON.exists():
            try:
                with open(SUMMARY_JSON, "r") as fp:
                    summary = json.load(fp) or {}
            except Exception:
                summary = {}

        return render_template(
            "picks.html",
            rows=filtered,
            summary=summary,
            market_value=market,
            league_value=league,
            book_value=book,
            min_edge_value=min_edge if min_edge else "",
            min_prob_value=min_prob if min_prob else ""
        )
    except Exception as e:
        return jsonify({"error": "render_failed", "detail": str(e), "rows_loaded": len(rows)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
