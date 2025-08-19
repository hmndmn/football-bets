import os, json, base64, gspread, pandas as pd
from google.oauth2.service_account import Credentials

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

    def write_table(self, tab: str, df: pd.DataFrame):
        ws = self.sh.worksheet(tab)
        ws.clear()
        if df.empty:
            ws.update("A1", [["empty"]])
        else:
            ws.update([df.columns.tolist()] + df.values.tolist())