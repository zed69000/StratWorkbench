NAME = "Bollinger Breakout"
PARAMS_SCHEMA = {'window':{'type':'int','min':10,'max':100,'step':1,'default':20},
                 'k':     {'type':'float','min':1.0,'max':3.0,'step':0.1,'default':2.0}}
import pandas as pd
def generate_signals(df: pd.DataFrame, params: dict):
    w = int(params.get('window',20)); k = float(params.get('k',2.0))
    c = df['close']; ma = c.rolling(w).mean(); sd = c.rolling(w).std().replace(0,1e-9)
    upper = ma + k*sd; lower = ma - k*sd
    long_ = (c > upper).astype(float); short_ = (c < lower).astype(float)
    return (long_ - short_).rename('position')
