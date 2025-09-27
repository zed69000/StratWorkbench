NAME = "Cooldown"
PARAMS_SCHEMA = {"min_hold": {"type":"int","min":1,"max":50,"step":1,"default":5}}
import pandas as pd
def apply(df, pos: pd.Series, params: dict):
    n = int(params.get("min_hold",5))
    p = pos.copy().astype(float).clip(-1,1).fillna(0.0)
    out = p.copy()
    last = 0.0
    hold = n
    for i in range(len(p)):
        v = p.iloc[i]
        if v != last and hold < n:
            out.iloc[i] = last
            hold += 1
        else:
            if v != last:
                last = v
                hold = 0
            else:
                hold = min(hold+1, n)
    return out.rename("position")
