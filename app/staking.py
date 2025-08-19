import pandas as pd

class StakeSizer:
    def __init__(self, bankroll: float, min_pct=0.0025, max_pct=0.025):
        self.bankroll = float(bankroll)
        self.min_pct = float(min_pct)
        self.max_pct = float(max_pct)

    def _kelly_half(self, p: float, price: float):
        b = float(price) - 1.0
        if b <= 0:
            return 0.0
        f_star = (b * p - (1 - p)) / b
        f_star = max(0.0, f_star) * 0.5  # half-Kelly
        # clamp
        return min(max(f_star, self.min_pct), self.max_pct)

    def apply(self, picks_df: pd.DataFrame):
        if picks_df.empty:
            return picks_df
        out = picks_df.copy()
        out["stake_pct"] = out.apply(lambda r: self._kelly_half(r["model_prob"], r["price"]), axis=1)
        out["stake_amt"] = (self.bankroll * out["stake_pct"]).round(2)
        return out