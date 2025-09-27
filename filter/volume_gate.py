NAME = "Volume Gate"
PARAMS_SCHEMA = {
    "len": {"type": "int", "min": 2, "max": 200, "default": 20, "step": 1},
    "mult": {"type": "float", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
    "force_exit": {"type": "int", "min": 0, "max": 1, "default": 0, "step": 1}
}

import pandas as pd

def apply(df: pd.DataFrame, pos: pd.Series, p: dict) -> pd.Series:
    if "volume" not in df.columns:
        return pos
    n = int(p.get("len", 20))
    k = float(p.get("mult", 1.0))
    force_exit = int(p.get("force_exit", 0)) == 1

    vma = df["volume"].rolling(n).mean()
    ok = df["volume"] >= k * vma

    out = pos.copy()
    block = (out.shift(1) == 0) & (out == 1) & (~ok)
    out[block] = 0
    if force_exit:
        out[~ok] = 0
    return out
