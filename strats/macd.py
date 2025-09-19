NAME = "MACD"
PARAMS_SCHEMA = {'fast':{'type':'int','min':3,'max':30,'step':1,'default':12},
                 'slow':{'type':'int','min':10,'max':60,'step':1,'default':26},
                 'signal':{'type':'int','min':3,'max':20,'step':1,'default':9}}
import pandas as pd
def generate_signals(df: pd.DataFrame, params: dict):
    f = int(params.get('fast',12)); s = int(params.get('slow',26)); sig = int(params.get('signal',9))
    c = df['close']
    ema_f = c.ewm(span=f, adjust=False).mean()
    ema_s = c.ewm(span=s, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    return ((macd > signal).astype(float)*2 - 1).rename('position')
