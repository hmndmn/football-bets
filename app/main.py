import os
import pandas as pd
from zoneinfo import ZoneInfo
from flask import Flask, render_template, request

from app.sheets import SheetClient
from app.data_sources import fetch_fixtures, fetch_odds
from app.model import predict_match, market_probs
from app.markets import find_value_bets
from app.staking import StakeSizer

app = Flask(__name__)

@app.route("/")
def health():
    return "OK", 200

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
        if sel.lower() == "home":
            return row.get("home", sel)
        if sel.lower() == "away":
            return row.get("away", sel)
        return "Draw"
    if mkt == "AH":
        # Show like "Real Madrid -1.0" or "Osasuna +1.0"
        team = row.get("selection")
        line = row.get("line")
        try:
            return f"{team} {float(line):+g}"
        except Exception:
            return f"{team}"
    return sel

@app.route("/run")
def run():
    # Settings
    sheet_name   = os.getenv("SHEET_NAME", "Football Picks")
    # New default leagues list includes all big ones; you can override via env
    leagues_env  = os.getenv("LEAGUES", "EPL,LaLiga,SerieA,Bundesliga,Ligue1,UCL")
    leagues      = [s.strip() for s in leagues_env.split(",") if s.strip()]
    hours_ahead  = int(os.getenv("HOURS_AHEAD", "240"))
    book_pref    = os.getenv("BOOK_FILTER", "bet365")
    edge_thresh  = float(os.getenv("EDGE_THRESHOLD", "0.05"))
    bankroll     = float(os.getenv("BANKROLL_START", "500"))
    min_stake    = float(os.getenv("MIN_STAKE_PCT", "0.0025"))
    max_stake    = float(os.getenv("MAX_STAKE_PCT", "0.025"))
    try:
        max_picks = int(os.getenv("MAX_PICKS", "0"))
    except Exception:
        max_picks = 0

    sc = SheetClient(sheet_name)

    # 1) Fixtures & Odds
    fixtures = fetch_fixtures(leagues, hours_ahead=hours_ahead)
    odds     = fetch_odds(fixtures, book_preference=book_pref)

    # 2) Model probs per match (collect OU lines and AH lines seen in odds)
    model_rows = []
    for _, m in fixtures.iterrows():
        lh, la = predict_match(m["home"], m["away"], m["league"])

        # OU lines present
        ou_lines = []
        # AH lines we normalize to *home-relative* numbers:
        # - If an odds row has side=='home' with +0.5 => home_line = +0.5
        # - If side=='away' with +0.5 => home_line = -0.5
        ah_home_lines = []

        if not odds.empty:
            mo = odds[odds["match_id"] == m["match_id"]]
            for L in mo.loc[mo["market"].astype(str).str.startswith("OU", na=False), "market"].unique():
                try:
                    ou_lines.append(float(str(L)[2:]))
                except Exception:
                    pass

            for _, r in mo[mo["market"] == "AH"].iterrows():
                side = str(r.get("side","")).lower()
                try:
                    line = float(r.get("line"))
                except Exception:
                    continue
                if side == "home":
                    ah_home_lines.append(line)
                elif side == "away":
                    ah_home_lines.append(-line)

        ou_lines = sorted(set(ou_lines)) if ou_lines else [2.5]
        ah_home_lines = sorted(set(ah_home_lines))

        probs = market_probs(lh, la, ou_lines=tuple(ou_lines), ah_home_lines=tuple(ah_home_lines))
        model_rows.append({"match_id": m["match_id"], **probs})

    model_probs = pd.DataFrame(model_rows)

    # 3) Value bets
    picks = find_value_bets(model_probs, odds, edge_threshold=edge_thresh)

    # 4) Staking
    sizer = StakeSizer(bankroll, min_pct=min_stake, max_pct=max_stake)
    picks = sizer.apply(picks)

    # 4b) Merge meta, local time, nice selection name, sort/cap
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
        "bankroll": bankroll,
        "edge_threshold": edge_thresh,
        "max_picks": max_picks,
        "leagues": ",".join(leagues),
    }])

    # 5) Write to Sheets
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

    return (
        f"OK fixtures={len(fixtures)} odds_rows={len(odds)} "
        f"model_rows={len(model_probs)} picks={len(picks)}",
        200
    )

@app.route("/view")
def view_picks():
    sheet_name = os.getenv("SHEET_NAME", "Football Picks")
    sc = SheetClient(sheet_name)
    try:
        df = sc.read_table("picks")
    except Exception as e:
        return f"Error reading sheet: {e}", 500

    for c in ["price","model_prob","implied","edge","stake_amt","line"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filters
    q_market   = (request.args.get("market") or "").strip()    # "1X2", "OU", "AH"
    q_league   = (request.args.get("league") or "").strip()
    q_book     = (request.args.get("book") or "").strip()
    q_min_edge = request.args.get("min_edge")
    q_min_prob = request.args.get("min_prob")

    if q_market:
        if q_market.upper() == "OU":
            df = df[df["market"].str.startswith("OU", na=False)]
        else:
            df = df[df["market"].astype(str).str.upper() == q_market.upper()]

    if q_league:
        df = df[df["league"].astype(str).str.upper() == q_league.upper()]

    if q_book:
        df = df[df["book"].astype(str).str.lower() == q_book.lower()]

    if q_min_edge:
        try:
            v = float(q_min_edge); df = df[(df["edge"].fillna(0) >= v)]
        except Exception:
            pass

    if q_min_prob:
        try:
            v = float(q_min_prob); df = df[(df["model_prob"].fillna(0) >= v)]
        except Exception:
            pass

    if not df.empty:
        sort_cols = ["kickoff_local","edge"] if "kickoff_local" in df.columns else ["utc_kickoff","edge"]
        df = df.sort_values(by=sort_cols, ascending=[True, False])

    cols = [c for c in [
        "kickoff_local","league","match","market",
        "selection_name","price","book","model_prob","implied","edge","stake_amt"
    ] if c in df.columns]
    df = df[cols] if cols else df

    return render_template(
        "picks.html",
        rows=df.to_dict(orient="records"),
        qs={
            "market": q_market,
            "league": q_league,
            "book": q_book,
            "min_edge": q_min_edge or "",
            "min_prob": q_min_prob or ""
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
