NAME = "Donchian Breakout"
PARAMS_SCHEMA = {'window':{'type':'int','min':10,'max':200,'step':1,'default':55},
                 'hold':  {'type':'int','min':0,'max':10,'step':1,'default':1}}
import pandas as pd
def generate_signals(df: pd.DataFrame, params: dict):
    w = int(params.get('window',55)); h = int(params.get('hold',1))
    high = df['high'].rolling(w).max(); low = df['low'].rolling(w).min(); c = df['close']
    long_sig = (c > high.shift(1)).astype(float); short_sig = (c < low.shift(1)).astype(float)
    raw = long_sig - short_sig
    for _ in range(max(h,0)): raw = raw.where(raw != 0, raw.shift(1).fillna(0))
    return raw.rename('position')
