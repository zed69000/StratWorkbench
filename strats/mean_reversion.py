NAME = "Mean Reversion (Z)"
PARAMS_SCHEMA = {'window':{'type':'int','min':5,'max':120,'step':1,'default':20},
                 'k':     {'type':'float','min':0.5,'max':3.0,'step':0.1,'default':1.0}}
import pandas as pd
def generate_signals(df: pd.DataFrame, params: dict):
    w = int(params.get('window',20)); k = float(params.get('k',1.0))
    c = df['close']; m = c.rolling(w).mean(); s = c.rolling(w).std().replace(0,1e-9)
    z = (c-m)/s
    return (-z).clip(-k, k)/k
