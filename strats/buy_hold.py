NAME = "Buy & Hold"
PARAMS_SCHEMA = {}
import pandas as pd
def generate_signals(df: pd.DataFrame, params: dict):
    return pd.Series(1.0, index=df.index, name='position')
