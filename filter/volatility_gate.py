NAME = "Volatility gate (ATR)"
PARAMS_SCHEMA = {"window":{"type":"int","min":5,"max":100,"step":1,"default":14},
                 "min_atr_pct":{"type":"float","min":0.0,"max":5.0,"step":0.1,"default":0.5}}
import pandas as pd, np as _np
def _atr(df, w):
    tr = pd.concat([(df['high']-df['low']).abs(),
                   (df['high']-df['close'].shift(1)).abs(),
                   (df['low']-df['close'].shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(w).mean()
def apply(df, pos: pd.Series, params: dict):
    w = int(params.get("window",14))
    minp = float(params.get("min_atr_pct",0.5)) / 100.0
    atr = _atr(df, w)
    thresh = df["close"].rolling(w).mean() * minp
    gate = (atr > thresh).astype(float)
    p = (pos * gate).replace(0, _np.nan).ffill().fillna(0.0)
    return p.rename("position")
