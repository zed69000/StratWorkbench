NAME = "SMA Cross"
PARAMS_SCHEMA = {'short':{'type':'int','min':3,'max':80,'step':1,'default':20},
                 'long': {'type':'int','min':10,'max':250,'step':1,'default':50}}
import pandas as pd
def generate_signals(df: pd.DataFrame, params: dict):
    s = int(params.get('short',20)); l = int(params.get('long',50))
    if s >= l: l = s+1
    c = df['close']
    sma_s = c.rolling(s).mean()
    sma_l = c.rolling(l).mean()
    return ((sma_s > sma_l).astype(float)*2 - 1).rename('position')
