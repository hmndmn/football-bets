import os
from flask import Flask
import pandas as pd
from app.sheets import SheetClient
from app.data_sources import fetch_fixtures, fetch_odds

app = Flask(__name__)

@app.route("/")
def health():
    return "OK", 200

@app.route("/run")
def run():
    sheet_name = os.getenv("SHEET_NAME", "Football Picks")
    leagues = [s.strip() for s in os.getenv("LEAGUES", "EPL,LaLiga").split(",")]
    hours_ahead = int(os.getenv("HOURS_AHEAD", "48"))
    book_pref = os.getenv("BOOK_FILTER", "bet365")

    sc = SheetClient(sheet_name)

    # 1) Fixtures (next N hours)
    fixtures = fetch_fixtures(leagues, hours_ahead=hours_ahead)

    # 2) Odds for those fixtures
    odds = fetch_odds(fixtures, book_preference=book_pref)

    # 3) Write to sheets (no model yet)
    try:
        sc.write_table("fixtures", fixtures if not fixtures.empty else pd.DataFrame(columns=["match_id","league","utc_kickoff","home","away"]))
        sc.write_table("odds", odds if not odds.empty else pd.DataFrame(columns=["match_id","market","selection","price","book"]))
        # keep picks/model_probs as-is for now
    except Exception as e:
        return f"Error updating sheet: {e}", 500

    return f"OK fixtures={len(fixtures)} odds_rows={len(odds)}", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8000")))
