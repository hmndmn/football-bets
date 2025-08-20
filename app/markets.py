import pandas as pd
import numpy as np

def _key_for_row(row) -> str | None:
    """
    Map an odds row to a model probability key.
    - 1X2: p_H / p_D / p_A
    - OUx.x Over/Under: p_OUx.x_Over / p_OUx.x_Under
    - AH (with side + line): p_AH_home_{+/-L} or p_AH_away_{+/-L}
    """
    market = str(row.get("market",""))
    sel = str(row.get("selection",""))
    if market == "1X2":
        if sel.lower() == "home":
            return "p_H"
        if sel.lower() == "away":
            return "p_A"
        if sel.lower() == "draw":
            return "p_D"
        return None

    if market.startswith("OU"):
        # market looks like OU2.5
        L = market[2:]
        if sel.lower() == "over":
            return f"p_OU{L}_Over"
        if sel.lower() == "under":
            return f"p_OU{L}_Under"
        return None

    if market == "AH":
        side = str(row.get("side","")).lower()
        L    = row.get("line", None)
        if L is None or side not in ("home","away"):
            return None
        try:
            Lf = float(L)
        except Exception:
            return None
        label = f"{Lf:+g}" if side == "home" else f"{Lf:+g}"  # we’ll convert below
        if side == "home":
            return f"p_AH_home_{label}"
        else:
            # away line in model is stored as p_AH_away_{+/-L} with L as typed for away in odds
            return f"p_AH_away_{label}"

    return None

def implied_from_decimal(price: float) -> float:
    try:
        p = float(price)
        return 1.0 / p if p > 0 else np.nan
    except Exception:
        return np.nan

def find_value_bets(model_probs: pd.DataFrame, odds: pd.DataFrame, edge_threshold: float = 0.05) -> pd.DataFrame:
    """
    Compare model probabilities to book implied. Return rows where edge >= threshold.
    Output columns: match_id, market, selection, price, book, model_prob, implied, edge, stake_pct, stake_amt
    (Stake columns are filled later by StakeSizer)
    """
    if model_probs.empty or odds.empty:
        return pd.DataFrame(columns=["match_id","market","selection","price","book","model_prob","implied","edge"])

    # Model probs indexed by match_id for quick lookup
    mp = model_probs.set_index("match_id")

    out = []
    for _, r in odds.iterrows():
        mid = r.get("match_id")
        if mid not in mp.index:
            continue
        key = _key_for_row(r)
        if not key:
            continue
        prob = mp.at[mid, key] if key in mp.columns else np.nan
        implied = implied_from_decimal(r.get("price"))
        if pd.isna(prob) or pd.isna(implied):
            continue
        edge = float(prob) - float(implied)
        if edge >= edge_threshold:
            out.append({
                "match_id": mid,
                "market": r.get("market"),
                "selection": r.get("selection"),
                "price": float(r.get("price")),
                "book": r.get("book"),
                "model_prob": float(prob),
                "implied": float(implied),
                "edge": float(edge),
                # carry-through (helpful later)
                "side": r.get("side"),
                "line": r.get("line"),
            })

    if not out:
        return pd.DataFrame(columns=["match_id","market","selection","price","book","model_prob","implied","edge"])

    df = pd.DataFrame(out)
    return df.sort_values(["match_id","edge"], ascending=[True, False]).reset_index(drop=True)
