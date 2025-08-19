import os
from flask import Flask
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

@app.route("/run")
def run():
    # ---- Settings (env vars) ----
    sheet_name   = os.getenv("SHEET_NAME", "Football Picks")
    leagues      = [s.strip() for s in os.getenv("LEAGUES", "EPL,LaLiga").split(",")]
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

    # ---- 1) Fixtures + Odds ----
    fixtures = fetch_fixtures(leagues, hours_ahead=hours_ahead)
    odds     = fetch_odds(fixtures, book_preference=book_pref)

    # ---- 2) Model probabilities per match ----
    model_rows = []
    for _, m in fixtures.iterrows():
        lh, la = predict_match(m["home"], m["away"], m["league"])

        # Collect OU lines present in odds for this match (e.g., 2.5, 3.0)
        ou_lines = []
        if not odds.empty:
            lines = odds.loc[odds["match_id"] == m["match_id"], "market"].unique()
            for line in lines:
                s = str(line)
                if s.startswith("OU"):
                    try:
                        ou_lines.append(float(s[2:]))
                    except Exception:
                        pass
        ou_lines = sorted(set(ou_lines)) if ou_lines else [2.5]

        probs = market_probs(lh, la, ou_lines=tuple(ou_lines))
        model_rows.append({"match_id": m["match_id"], **probs})

    model_probs = pd.DataFrame(model_rows)

    # ---- 3) Value bets vs odds ----
    picks = find_value_bets(model_probs, odds, edge_threshold=edge_thresh)

    # ---- 4) Staking (Half-Kelly with caps) ----
    sizer = StakeSizer(bankroll, min_pct=min_stake, max_pct=max_stake)
    picks = sizer.apply(picks)

    # ---- 4b) Add readable match info; sort; cap ----
    if not picks.empty:
        # Merge kickoff/home/away
        meta = fixtures[["match_id", "league", "utc_kickoff", "home", "away"]].copy()
        picks = picks.merge(meta, on="match_id", how="left")

        # Human-friendly match name
        picks["match"] = picks.apply(
            lambda r: f"{r.get('home','?')} vs {r.get('away','?')}", axis=1
        )

        # Round stake_amt to whole dollars (keep pct as is)
        picks["stake_amt"] = picks["stake_amt"].fillna(0).round(0).astype(int)

        # Sort by kickoff (soonest first) then by edge (highest first)
        picks = picks.sort_values(
            by=["utc_kickoff", "edge"], ascending=[True, False]
        ).reset_index(drop=True)

        # Cap number of picks if MAX_PICKS > 0
        if max_picks and len(picks) > max_picks:
            picks = picks.head(max_picks).reset_index(drop=True)

        # Reorder columns for readability (keep only those that exist)
        col_order = [
            "utc_kickoff","match","market","selection","price","book",
            "model_prob","implied","edge","stake_pct","stake_amt",
            "league","home","away","match_id"
        ]
        picks = picks[[c for c in col_order if c in picks.columns]]

    # ---- 4c) Build a small summary table ----
    summary = pd.DataFrame([{
        "num_picks": int(len(picks)),
        "total_stake": float(picks["stake_amt"].sum() if not picks.empty else 0.0),
        "avg_edge": float(picks["edge"].mean() if not picks.empty else 0.0),
        "bankroll": bankroll,
        "edge_threshold": edge_thresh,
        "max_picks": max_picks
    }])

    # ---- 5) Write to Google Sheets ----
    try:
        if fixtures.empty:
            fixtures = pd.DataFrame(columns=["match_id","league","utc_kickoff","home","away"])
        if odds.empty:
            odds = pd.DataFrame(columns=["match_id","market","selection","price","book"])
        if model_probs.empty:
            model_probs = pd.DataFrame(columns=["match_id"])
        if picks.empty:
            picks = pd.DataFrame(columns=[
                "utc_kickoff","match","market","selection","price","book",
                "model_prob","implied","edge","stake_pct","stake_amt",
                "league","home","away","match_id"
            ])

        sc.write_table("fixtures", fixtures)
        sc.write_table("odds", odds)
        sc.write_table("model_probs", model_probs)
        sc.write_table("picks", picks)
        sc.write_table("summary", summary)
    except Exception as e:
        return f"Error updating sheet: {e}", 500

    return (
        f"OK fixtures={len(fixtures)} "
        f"odds_rows={len(odds)} "
        f"model_rows={len(model_probs)} "
        f"picks={len(picks)}",
        200
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
