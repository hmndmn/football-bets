import math
import numpy as np
import pandas as pd

# Starter model: neutral Poisson with small home advantage baked into baselines.
# We'll upgrade later with team strengths and injury adjustments.

BASE_HOME_XG = 1.45
BASE_AWAY_XG = 1.25

def predict_match(home: str, away: str, league: str):
    """
    Returns (lambda_home, lambda_away) expected goals.
    """
    lam_home = max(0.4, BASE_HOME_XG)
    lam_away = max(0.4, BASE_AWAY_XG)
    return float(lam_home), float(lam_away)

def poisson_pmf(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def score_matrix(lh, la, maxg=7):
    ph = np.array([poisson_pmf(lh, k) for k in range(maxg + 1)])
    pa = np.array([poisson_pmf(la, k) for k in range(maxg + 1)])
    return np.outer(ph, pa)  # P(home=i, away=j)

def market_probs(lh, la, maxg=7, ou_lines=(2.5,)):
    """
    Compute probabilities for 1X2, BTTS, and specified OU lines.
    Returns dict of probabilities.
    """
    M = score_matrix(lh, la, maxg=maxg)
    home = np.tril(M, -1).sum()
    draw = np.trace(M)
    away = np.triu(M, 1).sum()

    # BTTS
    p_home0 = M[0, :].sum()
    p_away0 = M[:, 0].sum()
    p00 = M[0, 0]
    btts_yes = 1 - p_home0 - p_away0 + p00

    out = {
        "p_H": float(home),
        "p_D": float(draw),
        "p_A": float(away),
        "p_BTTS_Yes": float(btts_yes),
        "p_BTTS_No": float(1 - btts_yes),
    }

    # Over/Under lines
    for L in ou_lines:
        try:
            Lf = float(L)
        except Exception:
            continue
        over = 0.0
        for i in range(maxg + 1):
            for j in range(maxg + 1):
                if (i + j) > Lf:  # e.g., > 2.5 means 3+
                    over += M[i, j]
        out[f"p_OU{Lf}_Over"] = float(over)
        out[f"p_OU{Lf}_Under"] = float(1 - over)

    return out
