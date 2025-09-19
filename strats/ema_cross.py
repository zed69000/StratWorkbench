NAME = "EMA Cross"
PARAMS_SCHEMA = {'short':{'type':'int','min':3,'max':50,'step':1,'default':12},
                 'long': {'type':'int','min':10,'max':200,'step':1,'default':26},
                 'wait': {'type':'int','min':0,'max':10,'step':1,'default':1}}
import pandas as pd
def generate_signals(df: pd.DataFrame, params: dict):
    s = int(params.get('short',12)); l = int(params.get('long',26)); w = int(params.get('wait',1))
    if s >= l: l = s+1
    c = df['close']
    ema_s = c.ewm(span=s, adjust=False).mean()
    ema_l = c.ewm(span=l, adjust=False).mean()
    raw = (ema_s > ema_l).astype(float)*2 - 1
    return raw.shift(w).fillna(0.0).rename('position')
