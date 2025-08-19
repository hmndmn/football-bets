import pandas as pd
import numpy as np

# Map odds rows -> which model prob key to use
def selection_key(market: str, selection: str):
    market = str(market)
    sel = str(selection)
    if market == "1X2":
        if sel.lower() == "home":
            return "p_H"
        if sel.lower() == "draw":
            return "p_D"
        if sel.lower() == "away":
            return "p_A"
    if market.upper().startswith("OU"):
        line = market[2:]  # e.g., "2.5"
        if sel.lower() == "over":
            return f"p_OU{line}_Over"
        if sel.lower() == "under":
            return f"p_OU{line}_Under"
    if market.upper() == "BTTS":
        if sel.lower() == "yes":
            return "p_BTTS_Yes"
        if sel.lower() == "no":
            return "p_BTTS_No"
    return None

def implied_prob_from_price(price: float):
    try:
        p = 1.0 / float(price)
        return p if p > 0 else np.nan
    except Exception:
        return np.nan

def find_value_bets(model_probs_df: pd.DataFrame, odds_df: pd.DataFrame, edge_threshold=0.05):
    """
    model_probs_df: rows per match_id with columns for p_* keys
    odds_df: rows per (match_id, market, selection, price, book)
    """
    if model_probs_df.empty or odds_df.empty:
        return pd.DataFrame(columns=["match_id","market","selection","price","book","model_prob","implied","edge"])

    mp = model_probs_df.set_index("match_id")
    rows = []

    for _, r in odds_df.iterrows():
        mid = r["match_id"]
        key = selection_key(r["market"], r["selection"])
        if not key or mid not in mp.index or key not in mp.columns:
            continue
        p_model = float(mp.loc[mid, key])
        p_implied = implied_prob_from_price(float(r["price"]))
        edge = p_model - p_implied
        if edge >= edge_threshold:
            rows.append({
                "match_id": mid,
                "market": r["market"],
                "selection": r["selection"],
                "price": float(r["price"]),
                "book": r.get("book",""),
                "model_prob": round(p_model, 4),
                "implied": round(p_implied, 4),
                "edge": round(edge, 4),
            })

    return pd.DataFrame(rows).sort_values("edge", ascending=False).reset_index(drop=True)