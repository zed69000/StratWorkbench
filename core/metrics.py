from __future__ import annotations
import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = (equity / cummax) - 1.0
    return float(dd.min())

def sharpe(equity: pd.Series, periods_per_year: int = 252) -> float:
    rets = equity.pct_change().fillna(0.0)
    mu = rets.mean() * periods_per_year
    sigma = rets.std(ddof=0) * np.sqrt(periods_per_year)
    if sigma == 0:
        return 0.0
    return float(mu / sigma)

def growth_index(equity: pd.Series) -> float:
    perf = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    return float(50.0 + 50.0 * np.tanh(perf))

def stability_index(equity: pd.Series) -> float:
    s = sharpe(equity)
    dd = max_drawdown(equity)  # négatif
    score = 50.0 + 30.0 * np.tanh(s / 2.0) + 20.0 * (1.0 + dd)
    return float(np.clip(score, 0.0, 100.0))

def equity_from_position(df: pd.DataFrame,
                         position: pd.Series,
                         cash_start: float = 10_000.0,
                         fee_bps: float = 0.0,
                         spread_bps: float = 0.0,
                         slippage_bps: float = 0.0,
                         fee_on_sell_only: bool = False) -> pd.Series:
    """
    Backtest vectoriel simple avec coûts de transaction.
    - position: série [-1,0,1], appliquée en *t* sur le retour de *t->t+1*.
    - frais: appliqués lors des *changements* de position (turnover).
      coût = (fee + spread + slippage) * turnover * equity.
    - fee_on_sell_only: si True, commission uniquement quand delta<0 (réduction/vente).
    Notes:
      1 bps = 0.01%. spread_bps est one-way (moitié de l'écart).
    """
    pos = position.reindex(df.index).fillna(0.0).clip(-1, 1).astype(float)
    close = df["close"].astype(float)
    r = close.pct_change().fillna(0.0).to_numpy()

    one_way_cost = (float(fee_bps) + float(spread_bps) + float(slippage_bps)) / 10_000.0

    eq = np.empty(len(pos), dtype=float)
    eq[0] = float(cash_start)

    pos_vals = pos.to_numpy()
    for i in range(1, len(pos_vals)):
        pos_prev = pos_vals[i-1]
        pos_now  = pos_vals[i]

        # coût de changement de position au début de la barre i
        delta = pos_now - pos_prev
        if delta != 0.0:
            apply_cost = True
            if fee_on_sell_only:
                apply_cost = (delta < 0)  # réduction/vente nette
            if apply_cost:
                cost = abs(delta) * one_way_cost * eq[i-1]
                eq[i-1] -= cost
                if eq[i-1] < 0:
                    eq[i-1] = 0.0

        # performance de la barre i avec la position détenue sur la barre i-1
        eq[i] = eq[i-1] * (1.0 + pos_prev * r[i])

    return pd.Series(eq, index=pos.index, name="equity")
