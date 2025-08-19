import os, math
from flask import Flask, jsonify
import pandas as pd

from app.data_sources import fetch_fixtures, fetch_odds
from app.model import market_probs
from app.markets import find_value_bets
from app.staking import stake_plan
from app.sheets import SheetClient

app = Flask(__name__)

@app.route("/")
def root():
    return "OK", 200

@app.route("/probe")
def probe():
    leagues = ["EPL"]
    fixtures = fetch_fixtures(leagues)
    odds = fetch_odds(fixtures)
    return jsonify({
        "fixtures": len(fixtures),
        "odds_rows": len(odds)
    })

@app.route("/run")
def run():
    leagues = ["EPL"]
    fixtures = fetch_fixtures(leagues)
    odds = fetch_odds(fixtures)

    picks = []
    for _, f in fixtures.iterrows():
        mid = f["match_id"]
        home, away = f["home"], f["away"]
        match_odds = odds[odds.match_id == mid]

        # Model probabilities (only if both odds and teams exist)
        if match_odds.empty:
            continue

        lh, la = 1.4, 1.0  # placeholder lambda values
        ou_lines = [2.5]

        probs = market_probs(lh, la, ou_lines=tuple(ou_lines))
        values = find_value_bets(probs, match_odds)
        stakes = stake_plan(values)

        for row in stakes:
            picks.append({
                "match": f"{home} vs {away}",
                "market": row["market"],
                "selection": row["selection"],
                "odds": row["odds"],
                "model_prob": row["model_prob"],
                "stake": row["stake"],
            })

    df = pd.DataFrame(picks)

    # Write to Google Sheets if available
    if not df.empty and os.environ.get("GOOGLE_SA_JSON_BASE64"):
        try:
            sc = SheetClient("Football Bets")
            sc.write_table("Picks", df)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"count": len(df), "picks": df.to_dict(orient="records")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
