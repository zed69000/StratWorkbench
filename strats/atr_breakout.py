NAME = "ATR Channel Breakout"
PARAMS_SCHEMA = {'window':{'type':'int','min':5,'max':100,'step':1,'default':14},
                 'mult':  {'type':'float','min':1.0,'max':5.0,'step':0.1,'default':2.0}}
import pandas as pd, numpy as np
def _atr(df, w):
    tr = pd.concat([(df['high']-df['low']).abs(),
                   (df['high']-df['close'].shift(1)).abs(),
                   (df['low']-df['close'].shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(w).mean()
def generate_signals(df: pd.DataFrame, params: dict):
    w = int(params.get('window',14)); m = float(params.get('mult',2.0))
    c = df['close']; atr = _atr(df, w)
    upper = c.rolling(w).mean() + m*atr
    lower = c.rolling(w).mean() - m*atr
    long_ = (c > upper).astype(float); short_ = (c < lower).astype(float)
    return (long_ - short_).rename('position')
