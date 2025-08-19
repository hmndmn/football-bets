import os
from flask import Flask
import pandas as pd
from app.sheets import SheetClient
from app.data_sources import fetch_fixtures, fetch_odds, check_af_status

app = Flask(__name__)

@app.route("/")
def health():
    return "OK", 200

@app.route("/debug")
def debug():
    ok, txt = check_af_status()
    return (f"API-Football status ok={ok}\n{txt[:1000]}", 200)

@app.route("/run")
def run():
    sheet_name = os.getenv("SHEET_NAME", "Football Picks")
    leagues = [s.strip() for s in os.getenv("LEAGUES", "EPL,LaLiga").split(",")]
    hours_ahead = int(os.getenv("HOURS_AHEAD", "240"))
    book_pref = os.getenv("BOOK_FILTER", "bet365")

    sc = SheetClient(sheet_name)

    fixtures = fetch_fixtures(leagues, hours_ahead=hours_ahead)
    odds = fetch_odds(fixtures, book_preference=book_pref)

    if fixtures.empty:
        fixtures = pd.DataFrame(columns=["match_id","league","utc_kickoff","home","away"])
    if odds.empty:
        odds = pd.DataFrame(columns=["match_id","market","selection","price","book"])

    try:
        sc.write_table("fixtures", fixtures)
        sc.write_table("odds", odds)
    except Exception as e:
        return f"Error updating sheet: {e}", 500

    return f"OK fixtures={len(fixtures)} odds_rows={len(odds)}", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8000")))
