import os
import json
import base64
import gspread
import pandas as pd
import numpy as np
from google.oauth2.service_account import Credentials


def _client():
    """
    Authenticate with Google Sheets API using base64-encoded service account JSON.
    """
    sa_b64 = os.environ["GOOGLE_SA_JSON_BASE64"]
    sa = json.loads(base64.b64decode(sa_b64))
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa, scopes=scopes)
    return gspread.authorize(creds)


class SheetClient:
    def __init__(self, sheet_name: str):
        self.gc = _client()
        self.sh = self.gc.open(sheet_name)

    def write_table(self, tab: str, df: pd.DataFrame):
        """
        Write a DataFrame into the given worksheet tab.
        Replaces NaN and inf values with empty strings before upload.
        """
        ws = self.sh.worksheet(tab)
        ws.clear()

        if df.empty:
            ws.update("A1", [["empty"]])
            return

        # --- Sanitize values ---
        clean = df.copy()
        clean.replace([np.inf, -np.inf], "", inplace=True)
        clean = clean.where(pd.notnull(clean), "")

        # Convert DataFrame to list of lists for gspread
        data = [clean.columns.tolist()] + clean.astype(object).values.tolist()

        ws.update(data)
