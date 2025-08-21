import math
import pandas as pd

# Expected schemas:
# fixtures: [match_id, league, utc_kickoff, home, away]
# odds:     [match_id, league, utc_kickoff, market, selection, price, book, home, away]
# model:    [match_id, league, home, away, p_H, p_D, p_A, p_BTTS_Yes, p_BTTS_No, p_OU{line}_Over, p_OU{line}_Under, ...]

REQUIRED_FIXTURE_COLS = ["match_id", "league", "utc_kickoff", "home", "away"]
REQUIRED_ODDS_COLS     = ["match_id", "league", "utc_kickoff", "market", "selection", "price", "book", "home", "away"]

def _ensure_cols(df: pd.DataFrame, req):
    if df is None:
        return pd.DataFrame({c: [] for c in req})
    for c in req:
        if c not in df.columns:
            df[c] = "" if c != "price" else pd.NA
    # coerce price to float when present
    if "price" in df.columns:
        try:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        except Exception:
            pass
    return df

def _implied_prob(price):
    try:
        p = 1.0 / float(price)
        return p if math.isfinite(p) else None
    except Exception:
        return None

def _edge(model_p, price):
    ip = _implied_prob(price)
    if model_p is None or ip is None:
        return None
    return float(model_p) - float(ip)

def _pick_row_common(frow, orow, market, selection, model_p, edge, price):
    return {
        "utc_kickoff": frow.get("utc_kickoff", "") or orow.get("utc_kickoff", ""),
        "match": f"{frow.get('home','')} vs {frow.get('away','')}",
        "market": market,
        "selection": selection,
        "price": price,
        "book": orow.get("book", ""),
        "model_prob": model_p if model_p is not None else "",
        "implied": _implied_prob(price) if price is not None else "",
        "edge": edge if edge is not None else "",
        "stake_pct": "",
        "stake_amt": "",
        "league": frow.get("league", "") or orow.get("league", ""),
        "home": frow.get("home", "") or orow.get("home", ""),
        "away": frow.get("away", "") or orow.get("away", ""),
        "match_id": frow.get("match_id", "") or orow.get("match_id", ""),
    }

def _extract_model_prob(model_row: pd.Series, market: str, selection: str):
    # 1X2
    if market == "1X2":
        if selection == "Home":
            return float(model_row.get("p_H", 0.0))
        if selection == "Draw":
            return float(model_row.get("p_D", 0.0))
        if selection == "Away":
            return float(model_row.get("p_A", 0.0))
        return None

    # Totals "OU{line}" where selection is "Over" or "Under"
    if market.startswith("OU"):
        try:
            line = market[2:]
            key = f"p_OU{line}_{selection}"
            val = model_row.get(key, None)
            return float(val) if val is not None else None
        except Exception:
            return None

    # Asian Handicap "AH{handicap}" — basic proxy (can refine later)
    if market.startswith("AH"):
        if selection == "Home":
            return float(model_row.get("p_H", 0.0))
        if selection == "Away":
            return float(model_row.get("p_A", 0.0))
        return None

    return None

def find_value_bets(
    fixtures: pd.DataFrame,
    odds: pd.DataFrame,
    model_probs: pd.DataFrame,
    edge_threshold: float = 0.05,
    book_filter: str = "",
    max_picks: int = 50,
) -> pd.DataFrame:
    fixtures = _ensure_cols(fixtures, REQUIRED_FIXTURE_COLS)
    odds     = _ensure_cols(odds, REQUIRED_ODDS_COLS)

    if odds is None or odds.empty or fixtures is None or fixtures.empty:
        return pd.DataFrame(columns=[
            "utc_kickoff", "match", "market", "selection", "price", "book",
            "model_prob", "implied", "edge", "stake_pct", "stake_amt",
            "league", "home", "away", "match_id"
        ])

    # Keep only odds that have a matching fixture (ensures league/home/away present)
    odds = odds.merge(
        fixtures[["match_id", "league", "utc_kickoff", "home", "away"]],
        on="match_id", how="inner", suffixes=("", "_fx")
    )

    # Optional book filter
    if book_filter:
        bl = book_filter.lower().strip()
        odds = odds[odds["book"].astype(str).str.lower().str.contains(bl, na=False)]

    # Best price per (match_id, market, selection)
    odds = (
        odds.sort_values("price", ascending=False)
            .drop_duplicates(subset=["match_id", "market", "selection"], keep="first")
            .reset_index(drop=True)
    )

    if model_probs is None or model_probs.empty:
        return pd.DataFrame(columns=[
            "utc_kickoff", "match", "market", "selection", "price", "book",
            "model_prob", "implied", "edge", "stake_pct", "stake_amt",
            "league", "home", "away", "match_id"
        ])

    # Fast lookup for model rows
    mp_index = {row["match_id"]: row for _, row in model_probs.iterrows() if "match_id" in row and pd.notna(row["match_id"])}

    picks = []
    for _, orow in odds.iterrows():
        mid = orow.get("match_id")
        frow = orow  # already merged
        mrow = mp_index.get(mid)
        if mrow is None:
            continue

        market = str(orow.get("market", ""))
        sel    = str(orow.get("selection", ""))
        price  = orow.get("price", None)

        mp = _extract_model_prob(mrow, market, sel)
        eg = _edge(mp, price)
        if eg is None or eg < float(edge_threshold):
            continue

        picks.append(_pick_row_common(frow, orow, market, sel, mp, eg, price))

    if not picks:
        return pd.DataFrame(columns=[
            "utc_kickoff", "match", "market", "selection", "price", "book",
            "model_prob", "implied", "edge", "stake_pct", "stake_amt",
            "league", "home", "away", "match_id"
        ])

    out = pd.DataFrame(picks).sort_values("edge", ascending=False).head(int(max_picks)).reset_index(drop=True)

    # Clean any non-finite numbers
    for col in ("model_prob", "implied", "edge"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([float("inf"), float("-inf")], pd.NA)

    return out
