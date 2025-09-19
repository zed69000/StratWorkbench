NAME = "VWAP MeanRev"
PARAMS_SCHEMA = {'window':{'type':'int','min':10,'max':200,'step':5,'default':60},
                 'k':     {'type':'float','min':0.5,'max':3.0,'step':0.1,'default':1.0}}
import pandas as pd
def generate_signals(df: pd.DataFrame, params: dict):
    w = int(params.get('window',60)); k = float(params.get('k',1.0))
    tp = (df['high']+df['low']+df['close'])/3.0
    vol = df['volume'].rolling(w).sum().replace(0,1e-9)
    vwap = (tp*df['volume']).rolling(w).sum() / vol
    z = (df['close'] - vwap).rolling(w).apply(
    lambda x: (x[-1] - x.mean()) / (x.std() if x.std() else 1e-9),
    raw=True
)

