NAME = "Hold until opposite"
PARAMS_SCHEMA = {}
import numpy as np, pandas as pd
def apply(df, pos: pd.Series, params: dict):
    p = pos.replace(0, np.nan).ffill().fillna(0.0).clip(-1,1)
    return p.rename("position")
