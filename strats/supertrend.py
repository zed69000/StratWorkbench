NAME = "Supertrend"
PARAMS_SCHEMA = {'window':{'type':'int','min':5,'max':50,'step':1,'default':10},
                 'mult':  {'type':'float','min':1.0,'max':5.0,'step':0.1,'default':3.0}}
import pandas as pd
def _atr(df, w):
    tr = pd.concat([(df['high']-df['low']).abs(),
                   (df['high']-df['close'].shift(1)).abs(),
                   (df['low']-df['close'].shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(w).mean()
def generate_signals(df: pd.DataFrame, params: dict):
    w = int(params.get('window',10)); m = float(params.get('mult',3.0))
    c = df['close']; hl2 = (df['high']+df['low'])/2.0; atr = _atr(df, w)
    upper = hl2 + m*atr; lower = hl2 - m*atr
    trend = pd.Series(1.0, index=c.index)
    trend[c < lower] = -1.0
    trend[c > upper] = 1.0
    return trend.rename('position')
