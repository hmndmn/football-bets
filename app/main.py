import os
import requests
import pandas as pd
from zoneinfo import ZoneInfo
from flask import Flask, render_template, request, jsonify

from app.sheets import SheetClient
from app.data_sources import fetch_fixtures, fetch_odds, SPORT_KEYS
from app.model import predict_match, market_probs
from app.markets import find_value_bets
from app.staking import StakeSizer

app = Flask(__name__)

@app.route("/")
def health():
    return "OK", 200

# ---------- helpers ----------
def _add_local_time(df: pd.DataFrame, utc_col: str = "utc_kickoff", local_col: str = "kickoff_local"):
    if df.empty or utc_col not in df.columns:
        return df
    dt = pd.to_datetime(df[utc_col], utc=True, errors="coerce")
    local = dt.dt.tz_convert(ZoneInfo("America/Vancouver"))
    df[local_col] = local.dt.strftime("%Y-%m-%d %H:%M (%Z)")
    return df

def _selection_with_team(row):
    sel = str(row.get("selection",""))
    mkt = str(row.get("market",""))
    if mkt == "1X2":
        if sel.lower() == "home": return row.get("home", sel)
        if sel.lower() == "away": return row.get("away", sel)
        return "Draw"
    if mkt == "AH":
        team = row.get("selection"); line = row.get("line")
        try: return f"{team} {float(line):+g}"
        except Exception: return f"{team}"
    return sel

def _check_odds_quota():
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    regions = os.getenv("ODDS_REGIONS", "uk,eu,us")
    if not api_key:
        return {"ok": False, "status": None, "remaining": None, "used": None, "error_code": "MISSING_API_KEY"}
    key = SPORT_KEYS.get("EPL") or next(iter(SPORT_KEYS.values()))
    url = f"https://api.the-odds-api.com/v4/sports/{key}/odds"
    params = {"apiKey": api_key, "regions": regions, "oddsFormat": "decimal", "markets": "h2h", "dateFormat": "iso"}
    try:
        r = requests.get(url, params=params, timeout=10)
        try: j = r.json()
        except Exception: j = {}
        return {
            "ok": r.status_code == 200,
            "status": r.status_code,
            "remaining": r.headers.get("x-requests-remaining"),
            "used": r.headers.get("x-requests-used"),
            "error_code": (j.get("error_code") if isinstance(j, dict) else None),
        }
    except Exception as e:
        return {"ok": False, "status": None, "remaining": None, "used": None, "error_code": str(e)}

# ---------- run (guarded) ----------
@app.route("/run")
def run():
    sheet_name   = os.getenv("SHEET_NAME", "Football Picks")
    leagues_env  = os.getenv("LEAGUES", "EPL,LaLiga,SerieA,Bundesliga,Ligue1,UCL")
    leagues      = [s.strip() for s in leagues_env.split(",") if s.strip()]
    hours_ahead  = int(os.getenv("HOURS_AHEAD", "240"))
    edge_thresh  = float(os.getenv("EDGE_THRESHOLD", "0.05"))
    bankroll     = float(os.getenv("BANKROLL_START", "500"))
    min_stake    = float(os.getenv("MIN_STAKE_PCT", "0.0025"))
    max_stake    = float(os.getenv("MAX_STAKE_PCT", "0.025"))
    try: max_picks = int(os.getenv("MAX_PICKS", "0"))
    except Exception: max_picks = 0

    sc = SheetClient(sheet_name)

    # guard: if out of credits, DO NOT fetch; keep last data
    quota = _check_odds_quota()
    if not quota.get("ok", False):
        try: fixtures = sc.read_table("fixtures")
        except Exception: fixtures = pd.DataFrame()
        try: odds = sc.read_table("odds")
        except Exception: odds = pd.DataFrame()
        try: model_probs = sc.read_table("model_probs")
        except Exception: model_probs = pd.DataFrame()
        try: picks = sc.read_table("picks")
        except Exception: picks = pd.DataFrame()
        msg = (
            f"SKIP quota_exhausted status={quota.get('status')} "
            f"remaining={quota.get('remaining')} used={quota.get('used')} "
            f"(kept previous data) fixtures={len(fixtures)} odds_rows={len(odds)} "
            f"model_rows={len(model_probs)} picks={len(picks)}"
        )
        return msg, 200

    # fetch
    fixtures = fetch_fixtures(leagues, hours_ahead=hours_ahead)
    odds     = fetch_odds(fixtures, hours_ahead=hours_ahead)

    # model
    model_rows = []
    for _, m in fixtures.iterrows():
        lh, la = predict_match(m["home"], m["away"], m["league"])
        # collect OU/AH lines present in odds for this match
        ou_lines, ah_home_lines = [], []
        mo = odds[odds["match_id"] == m["match_id"]] if not odds.empty else pd.DataFrame()

        for L in mo.loc[mo["market"].astype(str).str.startswith("OU", na=False), "market"].unique():
            try: ou_lines.append(float(str(L)[2:]))
            except Exception: pass
        for _, r in mo[mo["market"] == "AH"].iterrows():
            side = str(r.get("side","")).lower()
            try: line = float(r.get("line"))
            except Exception: continue
            ah_home_lines.append(line if side == "home" else -line if side == "away" else 0)

        ou_lines = sorted(set([x for x in ou_lines if isinstance(x, float)])) or [2.5]
        ah_home_lines = sorted(set([x for x in ah_home_lines if isinstance(x, float)]))

        probs = market_probs(lh, la, ou_lines=tuple(ou_lines), ah_home_lines=tuple(ah_home_lines))
        model_rows.append({"match_id": m["match_id"], **probs})

    model_probs = pd.DataFrame(model_rows)

    # value + staking
    picks = find_value_bets(model_probs, odds, edge_threshold=edge_thresh)
    sizer = StakeSizer(bankroll, min_pct=min_stake, max_pct=max_stake)
    picks = sizer.apply(picks)

    # decorate + sort + cap
    if not picks.empty:
        meta = fixtures[["match_id","league","utc_kickoff","home","away"]].copy()
        picks = picks.merge(meta, on="match_id", how="left")
        picks = _add_local_time(picks, "utc_kickoff", "kickoff_local")
        picks["match"] = picks.apply(lambda r: f"{r.get('home','?')} vs {r.get('away','?')}", axis=1)
        picks["selection_name"] = picks.apply(_selection_with_team, axis=1)
        picks["stake_amt"] = picks["stake_amt"].fillna(0).round(0).astype(int)
        sort_cols = ["kickoff_local","edge"] if "kickoff_local" in picks.columns else ["utc_kickoff","edge"]
        picks = picks.sort_values(by=sort_cols, ascending=[True, False]).reset_index(drop=True)
        if max_picks and len(picks) > max_picks:
            picks = picks.head(max_picks).reset_index(drop=True)
        col_order = [
            "kickoff_local","utc_kickoff","league","match","market",
            "selection_name","selection","line","price","book",
            "model_prob","implied","edge","stake_pct","stake_amt",
            "home","away","match_id","side"
        ]
        picks = picks[[c for c in col_order if c in picks.columns]]

    summary = pd.DataFrame([{
        "num_picks": int(len(picks)),
        "total_stake": float(picks["stake_amt"].sum() if not picks.empty else 0.0),
        "avg_edge": float(picks["edge"].mean() if not picks.empty else 0.0),
        "leagues": ",".join(leagues),
    }])

    # write to Google Sheets (never write NaN/inf)
    try:
        fixtures = _add_local_time(fixtures, "utc_kickoff", "kickoff_local")
        if fixtures.empty:
            fixtures = pd.DataFrame(columns=["match_id","league","utc_kickoff","kickoff_local","home","away"])
        if odds.empty:
            odds = pd.DataFrame(columns=["match_id","market","selection","price","book","side","line"])
        if model_probs.empty:
            model_probs = pd.DataFrame(columns=["match_id"])
        if picks.empty:
            picks = pd.DataFrame(columns=[
                "kickoff_local","utc_kickoff","league","match","market",
                "selection_name","selection","line","price","book",
                "model_prob","implied","edge","stake_pct","stake_amt",
                "home","away","match_id","side"
            ])
        sc = SheetClient(sheet_name)
        sc.write_table("fixtures", fixtures)
        sc.write_table("odds", odds)
        sc.write_table("model_probs", model_probs)
        sc.write_table("picks", picks)
        sc.write_table("summary", summary)
    except Exception as e:
        return f"Error updating sheet: {e}", 500

    return (f"OK fixtures={len(fixtures)} odds_rows={len(odds)} model_rows={len(model_probs)} picks={len(picks)}", 200)

# ---------- view ----------
@app.route("/view")
def view_picks():
    sheet_name = os.getenv("SHEET_NAME", "Football Picks")
    sc = SheetClient(sheet_name)
    try:
        df = sc.read_table("picks")
    except Exception as e:
        return f"Error reading sheet: {e}", 500

    for c in ["price","model_prob","implied","edge","stake_amt","line"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    q_market   = (request.args.get("market") or "").strip()
    q_league   = (request.args.get("league") or "").strip()
    q_book     = (request.args.get("book") or "").strip()
    q_min_edge = request.args.get("min_edge")
    q_min_prob = request.args.get("min_prob")

    if q_market:
        if q_market.upper() == "OU": df = df[df["market"].str.startswith("OU", na=False)]
        else: df = df[df["market"].astype(str).str.upper() == q_market.upper()]
    if q_league: df = df[df["league"].astype(str).str.upper() == q_league.upper()]
    if q_book:   df = df[df["book"].astype(str).str.lower() == q_book.lower()]
    if q_min_edge:
        try: df = df[(df["edge"].fillna(0) >= float(q_min_edge))]
        except Exception: pass
    if q_min_prob:
        try: df = df[(df["model_prob"].fillna(0) >= float(q_min_prob))]
        except Exception: pass

    if not df.empty:
        sort_cols = ["kickoff_local","edge"] if "kickoff_local" in df.columns else ["utc_kickoff","edge"]
        df = df.sort_values(by=sort_cols, ascending=[True, False])

    cols = [c for c in [
        "kickoff_local","league","match","market",
        "selection_name","price","book","model_prob","implied","edge","stake_amt"
    ] if c in df.columns]
    df = df[cols] if cols else df

    return render_template("picks.html", rows=df.to_dict(orient="records"),
                           qs={"market": q_market, "league": q_league, "book": q_book,
                               "min_edge": q_min_edge or "", "min_prob": q_min_prob or ""})

# ---------- probe (debug) ----------
@app.route("/probe_odds")
def probe_odds():
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    regions = os.getenv("ODDS_REGIONS", "uk,eu,us")
    if not api_key:
        return jsonify({"ok": False, "error": "ODDS_API_KEY missing"}), 500

    sample_leagues = ["EPL", "LaLiga"]
    out = {"ok": True, "regions": regions, "leagues": {}}

    for lg in sample_leagues:
        key = SPORT_KEYS.get(lg)
        if not key:
            out["leagues"][lg] = {"error": "sport key missing"}
            continue
        url = f"https://api.the-odds-api.com/v4/sports/{key}/odds"
        params = {"apiKey": api_key, "regions": regions, "oddsFormat": "decimal", "markets": "h2h", "dateFormat": "iso"}
        try:
            r = requests.get(url, params=params, timeout=15)
            try_json = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
            out["leagues"][lg] = {
                "status": r.status_code,
                "x-requests-remaining": r.headers.get("x-requests-remaining"),
                "x-requests-used": r.headers.get("x-requests-used"),
                "events": len(try_json) if isinstance(try_json, list) else None,
                "sample": (try_json[0] if isinstance(try_json, list) and try_json else try_json) or {},
            }
        except Exception as e:
            out["leagues"][lg] = {"error": str(e)}
    return jsonify(out), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
