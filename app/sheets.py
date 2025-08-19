import os
import json
import base64
import gspread
import pandas as pd
import numpy as np
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound

def _client():
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

    def _get_or_create_ws(self, tab: str):
        try:
            return self.sh.worksheet(tab)
        except WorksheetNotFound:
            return self.sh.add_worksheet(title=tab, rows=200, cols=26)

    def write_table(self, tab: str, df: pd.DataFrame):
        ws = self._get_or_create_ws(tab)
        ws.clear()
        if df.empty:
            ws.update("A1", [["empty"]])
            return
        clean = df.copy()
        clean.replace([np.inf, -np.inf], "", inplace=True)
        clean = clean.where(pd.notnull(clean), "")
        data = [clean.columns.tolist()] + clean.astype(object).values.tolist()
        ws.update(data)

    def read_table(self, tab: str) -> pd.DataFrame:
        ws = self._get_or_create_ws(tab)
        values = ws.get_all_values()
        if not values:
            return pd.DataFrame()
        header, rows = values[0], values[1:]
        if not header:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=header)
        for c in ["price","model_prob","implied","edge","stake_amt"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
