import math
import numpy as np

# Very simple baseline xG; you can later plug real team strengths
BASE_HOME_XG = 1.45
BASE_AWAY_XG = 1.25

def predict_match(home: str, away: str, league: str):
    """Return (lambda_home, lambda_away) expected goals (Poisson means)."""
    lam_home = max(0.4, BASE_HOME_XG)
    lam_away = max(0.4, BASE_AWAY_XG)
    return float(lam_home), float(lam_away)

def poisson_pmf(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def score_matrix(lh, la, maxg=7):
    ph = np.array([poisson_pmf(lh, k) for k in range(maxg + 1)])
    pa = np.array([poisson_pmf(la, k) for k in range(maxg + 1)])
    return np.outer(ph, pa)  # P(home=i, away=j)

def _prob_ou_over(M, L):
    over = 0.0
    maxg = M.shape[0] - 1
    for i in range(maxg + 1):
        for j in range(maxg + 1):
            if (i + j) > L:
                over += M[i, j]
    return float(over)

def _prob_diff(M, cmp):
    """Return P(diff relation holds). cmp is a callable taking (diff)->bool."""
    maxg = M.shape[0] - 1
    p = 0.0
    for i in range(maxg + 1):
        for j in range(maxg + 1):
            diff = i - j
            if cmp(diff):
                p += M[i, j]
    return float(p)

def _ah_cover_push_probs(M, home_line: float):
    """
    Compute cover & push probabilities for a *home* handicap line.
    For quarter lines (e.g. -0.25, -0.75), we apply half-stake splits:
      - L = -0.25  =>  0.5 of L=0 and 0.5 of L=-0.5
      - L = -0.75  =>  0.5 of L=-0.5 and 0.5 of L=-1.0
    Returns dict: { "cover": p_cover, "push": p_push }
    """
    # If it's a quarter line, average the two adjacent half/whole lines
    frac = abs(home_line - round(home_line*2)/2)  # distance to nearest half
    if abs(home_line % 0.5) > 1e-9:  # quarter-line
        # e.g., -0.75 -> average of -0.5 and -1.0
        upper = math.ceil(home_line*2)/2
        lower = math.floor(home_line*2)/2
        d1 = _ah_cover_push_probs(M, lower)
        d2 = _ah_cover_push_probs(M, upper)
        return {
            "cover": 0.5*(d1["cover"]+d2["cover"]),
            "push":  0.5*(d1["push"] +d2["push"]),
        }

    # Whole or half lines
    if abs(home_line - round(home_line)) < 1e-9:
        # whole number: push possible if diff == L
        L = int(round(home_line))
        p_push  = _prob_diff(M, lambda d: d ==  L)
        p_cover = _prob_diff(M, lambda d: d >   L)
    else:
        # half number: no push; cover if diff > L
        L = float(home_line)
        p_push  = 0.0
        p_cover = _prob_diff(M, lambda d: d >  L)

    return {"cover": p_cover, "push": p_push}

def market_probs(lh, la, maxg=7, ou_lines=(2.5,), ah_home_lines=()):
    """
    Compute probabilities for:
      - 1X2
      - OU for given lines
      - AH cover for a list of *home-relative* lines (we'll also provide away cover for opposite lines)
    Returns a dict of keys:
      - p_H, p_D, p_A
      - p_OU{L}_Over / p_OU{L}_Under
      - p_AH_home_{L}, p_AH_away_{+L} where away line is the opposite of home line
    """
    M = score_matrix(lh, la, maxg=maxg)

    # 1X2
    home = np.tril(M, -1).sum()
    draw = np.trace(M)
    away = np.triu(M,  1).sum()

    out = {
        "p_H": float(home),
        "p_D": float(draw),
        "p_A": float(away),
    }

    # OU lines
    for L in ou_lines:
        try:
            Lf = float(L)
        except Exception:
            continue
        over = _prob_ou_over(M, Lf)
        out[f"p_OU{Lf}_Over"]  = float(over)
        out[f"p_OU{Lf}_Under"] = float(1 - over)

    # AH lines (home-relative)
    for L in sorted(set(ah_home_lines)):
        try:
            Lf = float(L)
        except Exception:
            continue
        d = _ah_cover_push_probs(M, Lf)
        p_cover_home = d["cover"]      # probability home covers the line
        # For away, the equivalent line is -Lf (because diff = home - away)
        d2 = _ah_cover_push_probs(M, -Lf)
        p_cover_away = d2["cover"]

        # Store with signed labels
        label_home = f"{Lf:+g}"  # e.g., +0.5, -1, +1.25
        label_away = f"{-Lf:+g}"
        out[f"p_AH_home_{label_home}"] = float(p_cover_home)
        out[f"p_AH_away_{label_away}"] = float(p_cover_away)

    return out
