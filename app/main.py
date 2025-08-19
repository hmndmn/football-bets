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
    sheet_name = os.getenv("SHEET_NAME", "Football Picks")
    leagues = [s.strip() for s in os.getenv("LEAGUES", "EPL,LaLiga").split(",")]
    hours_ahead = int(os.getenv("HOURS_AHEAD", "240"))
    book_pref = os.getenv("BOOK_FILTER", "bet365")
    edge_threshold = float(os.getenv("EDGE_THRESHOLD", "0.05"))
    bankroll = float(os.getenv("BANKROLL_START", "500"))
    min_stake = float(os.getenv("MIN_STAKE_PCT", "0.0025"))
    max_stake = float(os.getenv("MAX_STAKE_PCT", "0.025"))

    sc = SheetClient(sheet_name)

    # 1) Fixtures + Odds
    fixtures = fetch_fixtures(leagues, hours_ahead=hours_ahead)
    odds = fetch_odds(fixtures, book_preference=book_pref)

    # 2) Model probabilities per match
    model_rows = []
    for _, m in fixtures.iterrows():
        lh, la = predict_match(m["home"], m["away"], m["league"])
        # collect OU lines present in odds for this match (e.g., 2.5, 3.0) to compute both
        ou_lines = []
        if not odds.empty:
            for line in (odds[odds["match_id"] == m["match_id"]]["market"].unique() or []):
                if str(line).startswith("OU"):
                    try:
                        ou_lines.append(float(str(line)[2:]))
                    except Exception:
                        pass
        ou_lines = sorted(set(ou_lines)) or (2.5,)
        probs = market_probs(lh, la, ou_lines=tuple(ou_lines))
        model_rows.append({"match_id": m["match_id"], **probs})
    model_probs = pd.DataFrame(model_rows)

    # 3) Value bets vs odds
    picks = find_value_bets(model_probs, odds, edge_threshold=edge_threshold)

    # 4) Staking
    sizer = StakeSizer(bankroll, min_pct=min_stake, max_pct=max_stake)
    picks = sizer.apply(picks)

    # 5) Write to Google Sheets
    try:
        if fixtures.empty:
            fixtures = pd.DataFrame(columns=["match_id","league","utc_kickoff","home","away"])
        if odds.empty:
            odds = pd.DataFrame(columns=["match_id","market","selection","price","book"])
        if model_probs.empty:
            model_probs = pd.DataFrame(columns=["match_id"])
        if picks.empty:
            picks = pd.DataFrame(columns=["match_id","market","selection","price","book","model_prob","implied","edge","stake_pct","stake_amt"])

        sc.write_table("fixtures", fixtures)
        sc.write_table("odds", odds)
        sc.write_table("model_probs", model_probs)
        sc.write_table("picks", picks)
    except Exception as e:
        return f"Error updating sheet: {e}", 500

    return f"OK fixtures={len(fixtures)} odds_rows={len(odds)} model_rows={len(model_probs)} picks={len(picks)}", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
