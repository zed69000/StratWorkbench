NAME = "Momentum ROC"
PARAMS_SCHEMA = {'window':{'type':'int','min':2,'max':120,'step':1,'default':20}}
import pandas as pd
def generate_signals(df: pd.DataFrame, params: dict):
    w = int(params.get('window',20))
    roc = df['close'].pct_change(w)
    return (roc.apply(lambda x: 1.0 if x>0 else (-1.0 if x<0 else 0.0))).rename('position')
