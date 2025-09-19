from __future__ import annotations
import numpy as np, pandas as pd

def equity_from_position(df: pd.DataFrame, position: pd.Series, cash_start: float=10_000.0, fee_bps: float=0.0):
    price = df["close"]

    # 🔒 Sécurité si None
    if position is None:
        position = pd.Series(0.0, index=price.index, name="position")

    pos = position.reindex(price.index).fillna(0.0).clip(-1,1)
    ret = pos.shift(1).fillna(0.0) * price.pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(abs(pos.iloc[0]))
    fee = turnover * (fee_bps/10_000.0)
    eq = (1 + ret - fee).cumprod() * cash_start
    return eq.rename("equity")

def max_drawdown(equity: pd.Series):
    peak = equity.cummax(); dd = equity/peak - 1.0
    return dd.min()

def sharpe(equity: pd.Series, rf=0.0):
    rets = equity.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0: return 0.0
    return np.sqrt(252)*(rets.mean()-rf)/(rets.std()+1e-12)

def growth_index(equity: pd.Series):
    total = equity.iloc[-1]/equity.iloc[0] - 1.0
    return float(np.clip(50 + 50*np.tanh(total), 0, 100))

def stability_index(equity: pd.Series):
    s = sharpe(equity); dd = max_drawdown(equity)
    return float(np.clip(50 + 30*np.tanh(s/2.0) + 20*(1.0 + dd), 0, 100))
