from __future__ import annotations
import pandas as pd

def entries_exits(pos: pd.Series):
    """
    Détecte entrées/sorties pour signaux binaires {0,1}.
    Retourne (entries_index, exits_index, nb_trades).
    """
    p = pos.fillna(0).astype(float)
    prev = p.shift(1, fill_value=0)
    entries = p[(prev == 0) & (p == 1)].index
    exits   = p[(prev == 1) & (p == 0)].index
    changes = (p.diff().fillna(p.iloc[0]) != 0).sum()
    nb_trades = int(changes // 2)
    return entries, exits, nb_trades
