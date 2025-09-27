from __future__ import annotations
import numpy as np, pandas as pd

def equity_from_position(
    df: pd.DataFrame,
    position: pd.Series,
    cash_start: float = 10_000.0,
    fee_bps: float = 0.0,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    fee_on_sell_only: bool = False,
    stop_loss_pct: float | None = None,
) -> pd.Series:
    price = df["close"].astype(float)
    pos = pd.Series(0.0, index=price.index, name="position")
    if position is not None:
        pos = position.reindex(price.index).fillna(0.0).clip(-1, 1).astype(float)

    # Stop loss discret à la clôture
    if stop_loss_pct is not None and stop_loss_pct > 0:
        pos_sl = pos.copy()
        state = 0.0
        entry = np.nan
        for i, px in enumerate(price.values):
            p = pos_sl.iloc[i]
            if p != state:
                state = p
                entry = px if state != 0 else np.nan
            elif state != 0 and entry == entry:
                pnl = (px / entry - 1.0) * np.sign(state)
                if pnl <= -float(stop_loss_pct):
                    pos_sl.iloc[i] = 0.0
                    state = 0.0
                    entry = np.nan
        pos = pos_sl

    # Rendement brut (exposition t-1)
    ret_gross = pos.shift(1).fillna(0.0) * price.pct_change().fillna(0.0)

    # Coûts: variation d'exposition
    delta = pos.diff().fillna(pos.iloc[0])
    buy  = delta.clip(lower=0.0)
    sell = (-delta).clip(lower=0.0)

    commission = (sell if fee_on_sell_only else (buy + sell)) * (fee_bps / 10_000.0)
    frictions  = (buy + sell) * ((spread_bps + slippage_bps) / 10_000.0)
    cost = commission + frictions

    eq = (1.0 + ret_gross - cost).cumprod() * cash_start
    return eq.rename("equity")


def max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    s = equity.astype(float)
    roll_max = s.cummax()
    dd = (s / roll_max - 1.0).fillna(0.0)
    return float(dd.min())


def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    if equity is None or len(equity) < 2:
        return 0.0
    rets = equity.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    vol = rets.std()
    if vol == 0:
        return 0.0
    return float(np.sqrt(252.0) * (rets.mean() - rf) / (vol + 1e-12))


def growth_index(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    return float(np.clip(50.0 + 50.0 * np.tanh(total), 0.0, 100.0))


def stability_index(equity: pd.Series) -> float:
    s = sharpe(equity)
    dd = max_drawdown(equity)
    return float(np.clip(50.0 + 30.0 * np.tanh(s / 2.0) + 20.0 * (1.0 + dd), 0.0, 100.0))
