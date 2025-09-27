NAME = "CHOP Gate"
PARAMS_SCHEMA = {
    "len": {"type": "int", "min": 3, "max": 200, "default": 14, "step": 1},
    "chop_max": {"type": "float", "min": 0.0, "max": 100.0, "default": 55.0, "step": 0.5},
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

def apply(df: pd.DataFrame, pos: pd.Series, p: dict) -> pd.Series:
    n = int(p.get("len", 14))
    chop_max = float(p.get("chop_max", 55.0))
    force_exit = int(p.get("force_exit", 0)) == 1

    tr = _tr(df)
    sum_tr = tr.rolling(n).sum()
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    rng = (hh - ll).replace(0, np.nan)
    ratio = (sum_tr / rng).replace([np.inf, -np.inf], np.nan)
    chop = 100.0 * (np.log10(ratio)) / np.log10(n)
    ok = chop <= chop_max

    out = pos.copy()
    block = (out.shift(1) == 0) & (out == 1) & (~ok)
    out[block] = 0
    if force_exit:
        out[~ok] = 0
    return out
