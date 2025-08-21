import os, json
from pathlib import Path
from flask import Flask, request, render_template, jsonify
import pandas as pd

# Project modules
from app.data_sources import fetch_fixtures, fetch_odds
from app.model import predict_match, market_probs
from app.markets import find_value_bets
from app.staking import StakeSizer

# Optional Sheets (disabled unless USE_SHEETS=true)
USE_SHEETS = os.getenv("USE_SHEETS", "false").lower() in ("1", "true", "yes")
if USE_SHEETS:
    try:
        from app.sheets import SheetClient
    except Exception:
        SheetClient = None
        USE_SHEETS = False

app = Flask(__name__, template_folder="templates", static_folder="static")

DATA_DIR = Path("/opt/render/project/src/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
PICKS_JSON = DATA_DIR / "latest_picks.json"
SUMMARY_JSON = DATA_DIR / "latest_summary.json"

REQ_FIXTURE = ["match_id", "league", "utc_kickoff", "home", "away"]
REQ_ODDS    = ["match_id", "league", "utc_kickoff", "market", "selection", "price", "book", "home", "away"]

def ensure_cols(df: pd.DataFrame, req_cols):
    if df is None:
        return pd.DataFrame({c: [] for c in req_cols})
    df = df.copy()
    for c in req_cols:
        if c not in df.columns:
            # sensible defaults
            if c in ("price",):
                df[c] = pd.NA
            else:
                df[c] = ""
    # normalize dtypes where possible
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df

def preview(df: pd.DataFrame, n=5):
    try:
        return {"len": int(len(df)), "cols": list(df.columns), "head": df.head(n).to_dict(orient="records")}
    except Exception:
        return {"len": "?", "cols": [], "head": []}

def to_records(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    safe = df.replace([float("inf"), float("-inf")], pd.NA).fillna(value=pd.NA)
    return json.loads(safe.to_json(orient="records", date_format="iso"))

@app.route("/")
def health():
    return "OK", 200

@app.route("/probe")
def probe():
    """Quick look at raw fetch outputs (columns only)."""
    leagues = [s.strip() for s in os.getenv("LEAGUES", "EPL,LaLiga").split(",") if s.strip()]
    hours_ahead = int(os.getenv("HOURS_AHEAD", "168"))
    fx = fetch_fixtures(leagues=leagues, hours_ahead=hours_ahead)
    od = fetch_odds(leagues=leagues)

    fx_info = preview(fx)
    od_info = preview(od)
    return jsonify({"ok": True, "fixtures": fx_info, "odds": od_info, "leagues": leagues}), 200

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
        book_pref    = os.getenv("BOOK_FILTER", "")

        # 1) Fetch raw
        fixtures_df = fetch_fixtures(leagues=leagues, hours_ahead=hours_ahead)
        odds_df     = fetch_odds(leagues=leagues)

        # 2) Guarantee schemas
        fixtures_df = ensure_cols(fixtures_df, REQ_FIXTURE)
        odds_df     = ensure_cols(odds_df, REQ_ODDS)

        # 3) If any essential columns still missing for some reason, abort with diagnostics
        missing_fx = [c for c in REQ_FIXTURE if c not in fixtures_df.columns]
        missing_od = [c for c in REQ_ODDS    if c not in odds_df.columns]
        if missing_fx or missing_od:
            return jsonify({
                "ok": False,
                "error": "schema_missing",
                "missing_fixtures_cols": missing_fx,
                "missing_odds_cols": missing_od,
                "fixtures": preview(fixtures_df),
                "odds": preview(odds_df),
            }), 200

        # 4) Build simple model probs for each fixture (SAFE ACCESS via .get)
        model_rows = []
        for _, f in fixtures_df.iterrows():
            home = f.get("home", "")
            away = f.get("away", "")
            league = f.get("league", "")
            mid = f.get("match_id", "")
            # predict & markets
            lh, la = predict_match(str(home), str(away), str(league))
            # collect OU lines seen for that match
            ou_lines = []
            sub = odds_df[odds_df["match_id"] == mid]
            if not sub.empty and "market" in sub.columns:
                try:
                    vals = (
                        sub["market"].astype(str)
                        .str.extract(r"OU([0-9.]+)")[0]
                        .dropna()
                        .astype(float)
                        .tolist()
                    )
                    ou_lines = sorted(set(vals))
                except Exception:
                    ou_lines = []
            probs = market_probs(lh, la, ou_lines=tuple(ou_lines) if ou_lines else (2.5, 3.5))
            row = {"match_id": mid, "league": league, "home": home, "away": away}
            row.update(probs)
            model_rows.append(row)

        model_df = pd.DataFrame(model_rows)

        # 5) Value picks (function itself now tolerant)
        picks_df = find_value_bets(
            fixtures=fixtures_df,
            odds=odds_df,
            model_probs=model_df,
            edge_threshold=edge_thresh,
            book_filter=book_pref,
            max_picks=max_picks,
        )

        # 6) Staking
        sizer = StakeSizer(bankroll=bankroll, min_pct=min_stake, max_pct=max_stake)
        if isinstance(picks_df, pd.DataFrame) and not picks_df.empty:
            picks_df = sizer.apply(picks_df)

        # 7) Persist JSON for /view
        picks_records = to_records(picks_df if isinstance(picks_df, pd.DataFrame) else pd.DataFrame())
        summary = {
            "fixtures": int(len(fixtures_df)),
            "odds_rows": int(len(odds_df)),
            "model_rows": int(len(model_df)),
            "picks": int(len(picks_records)),
        }
        with open(PICKS_JSON, "w") as fp:
            json.dump(picks_records, fp)
        with open(SUMMARY_JSON, "w") as fp:
            json.dump(summary, fp)

        # Optional Google Sheet sync (non-fatal)
        if USE_SHEETS and 'SheetClient' in globals() and SheetClient is not None:
            try:
                sc = SheetClient(sheet_name)
                sc.write_table("fixtures", fixtures_df)
                sc.write_table("odds", odds_df)
                sc.write_table("model", model_df)
                sc.write_table("picks", picks_df)
            except Exception as se:
                summary["sheets_error"] = str(se)

        return jsonify({"ok": True, **summary, **({"sheets_error": summary["sheets_error"]} if "sheets_error" in summary else {})}), 200

    except Exception as e:
        # Rich diagnostics so we can see exactly what's missing
        diag = {
            "error": str(e),
            "type": type(e).__name__,
        }
        try:
            if 'fixtures_df' in locals():
                diag["fixtures"] = preview(fixtures_df)
            if 'odds_df' in locals():
                diag["odds"] = preview(odds_df)
        except Exception:
            pass
        return jsonify({"ok": False, "error": "run_failed", "detail": diag}), 200

@app.route("/debug_json")
def debug_json():
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
    # Load picks JSON
    rows = []
    if PICKS_JSON.exists():
        try:
            with open(PICKS_JSON, "r") as fp:
                j = json.load(fp)
            rows = j if isinstance(j, list) else []
        except Exception:
            rows = []

    # Filters from query
    market   = request.args.get("market", "").strip()
    league   = request.args.get("league", "").strip()
    book     = request.args.get("book", "").strip()
    min_edge = float(request.args.get("min_edge", "0") or 0)
    min_prob = float(request.args.get("min_prob", "0") or 0)

    def keep(r):
        try:
            if market and str(r.get("market","")).lower() != market.lower(): return False
            if league and str(r.get("league","")).lower() != league.lower(): return False
            if book   and str(r.get("book","")).lower()   != book.lower():   return False
            if float(r.get("edge", 0) or 0) < min_edge: return False
            if float(r.get("model_prob", 0) or 0) < min_prob: return False
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
            min_prob_value=min_prob if min_prob else "",
        )
    except Exception as e:
        return jsonify({"error": "render_failed", "detail": str(e), "rows_loaded": len(rows)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
