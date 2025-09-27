NAME = "ATR% Band"
PARAMS_SCHEMA = {
    "len": {"type": "int", "min": 3, "max": 100, "default": 14, "step": 1},
    "atr_min_pct": {"type": "float", "min": 0.0, "max": 20.0, "default": 0.3, "step": 0.1},
    "atr_max_pct": {"type": "float", "min": 0.0, "max": 50.0, "default": 5.0, "step": 0.1},
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
    lo = float(p.get("atr_min_pct", 0.3))
    hi = float(p.get("atr_max_pct", 5.0))
    force_exit = int(p.get("force_exit", 0)) == 1

    atr = _rma(_tr(df), n)
    atr_pct = (atr / df["close"].replace(0, np.nan)) * 100.0
    ok = (atr_pct >= lo) & (atr_pct <= hi)

    out = pos.copy()
    block = (out.shift(1) == 0) & (out == 1) & (~ok)
    out[block] = 0
    if force_exit:
        out[~ok] = 0
    return out
