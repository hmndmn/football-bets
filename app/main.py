import os
from flask import Flask
import pandas as pd
from sheets import SheetClient

app = Flask(__name__)

@app.route("/")
def health():
    return "OK", 200

@app.route("/run")
def run():
    # This will just prove we can write to your Google Sheet.
    sheet_name = os.getenv("SHEET_NAME", "Football Picks")
    sc = SheetClient(sheet_name)

    demo = pd.DataFrame([{"status": "connected", "note": "we will add real picks next"}])
    try:
        sc.write_table("picks", demo)
        return "Sheet updated (picks)", 200
    except Exception as e:
        return f"Error updating sheet: {e}", 500

if __name__ == "__main__":
    # Local testing
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))