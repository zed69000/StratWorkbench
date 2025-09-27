NAME = "ADX Gate"
PARAMS_SCHEMA = {
    "len": {"type": "int", "min": 3, "max": 100, "default": 14, "step": 1},
    "adx_min": {"type": "float", "min": 0.0, "max": 100.0, "default": 20.0, "step": 0.5},
    "force_exit": {"type": "int", "min": 0, "max": 1, "default": 0, "step": 1}
}

import numpy as np
import pandas as pd

def _tr(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)

def _rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0/n, adjust=False).mean()

def apply(df: pd.DataFrame, pos: pd.Series, p: dict) -> pd.Series:
    n = int(p.get("len", 14))
    adx_min = float(p.get("adx_min", 20.0))
    force_exit = int(p.get("force_exit", 0)) == 1

    up   = df["high"].diff()
    down = -df["low"].diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = _tr(df)
    atr = _rma(tr, n).replace(0, np.nan)
    plus_di  = 100.0 * _rma(pd.Series(plus_dm, index=df.index), n) / atr
    minus_di = 100.0 * _rma(pd.Series(minus_dm, index=df.index), n) / atr

    den = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / den) * 100.0
    adx = _rma(dx.fillna(0.0), n)

    ok = (adx >= adx_min) & (plus_di > minus_di)
    out = pos.copy()
    block = (out.shift(1) == 0) & (out == 1) & (~ok)
    out[block] = 0
    if force_exit:
        out[~ok] = 0
    return out
