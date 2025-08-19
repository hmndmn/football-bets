import numpy as np
import pandas as pd

# Very simple starter model:
# - Assume league-average expected goals with a small home advantage.
# - We'll improve this later with team strengths & historical data.

DEFAULT_HOME_ADV = 0.25   # log-scale home advantage
BASE_HOME_XG = 1.45       # baseline home expected goals
BASE_AWAY_XG = 1.25       # baseline away expected goals

def predict_match(home: str, away: str, league: str):
    """
    Returns (lambda_home, lambda_away) expected goals.
    """
    lam_home = max(0.4, BASE_HOME_XG)
    lam_away = max(0.4, BASE_AWAY_XG)
    return float(lam_home), float(lam_away)

def poisson_pmf(lam, k):
    return np.exp(-lam) * (lam ** k) / np.math.factorial(k)

def score_matrix(lh, la, maxg=7):
    ph = np.array([poisson_pmf(lh, k) for k in range(maxg + 1)])
    pa = np.array([poisson_pmf(la, k) for k in range(maxg + 1)])
    return np.outer(ph, pa)  # P(home=i, away=j)

def market_probs(lh, la, maxg=7, ou_lines=(2.5,)):
    """
    Compute probabilities for 1X2, BTTS, and a few OU lines.
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
        over = 0.0
        for i in range(maxg + 1):
            for j in range(maxg + 1):
                if (i + j) > L:  # e.g., > 2.5 means 3+
                    over += M[i, j]
        out[f"p_OU{L}_Over"] = float(over)
        out[f"p_OU{L}_Under"] = float(1 - over)

    return out