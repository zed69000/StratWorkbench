NAME = "RSI Trend"
PARAMS_SCHEMA = {'window':{'type':'int','min':5,'max':50,'step':1,'default':14},
                 'low':   {'type':'int','min':5,'max':50,'step':1,'default':30},
                 'high':  {'type':'int','min':50,'max':95,'step':1,'default':70}}
import pandas as pd, numpy as np
def _rsi(s, w):
    d = s.diff()
    u = d.clip(lower=0).rolling(w).mean()
    v = (-d.clip(upper=0)).rolling(w).mean()
    rs = u/(v.replace(0,1e-9))
    return 100 - (100/(1+rs))
def generate_signals(df: pd.DataFrame, params: dict):
    w = int(params.get('window',14)); lo = int(params.get('low',30)); hi=int(params.get('high',70))
    r = _rsi(df['close'], w)
    long_ = (r < lo).astype(float); short_ = (r > hi).astype(float)
    return (long_ - short_).rename('position')
