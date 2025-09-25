# Stop Loss pourcentage (overlay) avec option intrabar et cooldown
NAME = "Stop Loss %"
PARAMS_SCHEMA = {
    "percent": {"type": "float", "min": 0.5, "max": 50.0, "step": 0.5, "default": 5.0},
    # 0 = fixe depuis l'entrée ; 1 = trailing basé sur extrême favorable
    "trailing": {"type": "int", "min": 0, "max": 1, "step": 1, "default": 0},
    # 1 = utilise low/high intrabar pour déclenchement, 0 = close uniquement
    "use_hl": {"type": "int", "min": 0, "max": 1, "step": 1, "default": 1},
    # nombre de barres à rester flat après un stop (pour éviter ré-entrée immédiate)
    "cooldown_bars": {"type": "int", "min": 0, "max": 50, "step": 1, "default": 0},
}

import pandas as pd
import numpy as np

def apply(df: pd.DataFrame, position: pd.Series, params: dict) -> pd.Series:
    """
    Applique un stop loss en pourcentage sur la série de positions [-1,0,1].
    Trailing optionnel et déclenchement intrabar via low/high.
    Quand le stop est touché, on reste flat 'cooldown_bars' barres avant d'accepter une nouvelle entrée.
    """
    pct = float(params.get("percent", 5.0)) / 100.0
    trailing = int(params.get("trailing", 0)) == 1
    use_hl = int(params.get("use_hl", 1)) == 1
    cooldown = int(params.get("cooldown_bars", 0))

    pos_src = position.reindex(df.index).fillna(0.0).clip(-1, 1)

    c = df["close"].values
    lo = df["low"].values if "low" in df.columns else c
    hi = df["high"].values if "high" in df.columns else c

    out = np.zeros_like(c, dtype=float)

    in_pos = 0.0
    entry = np.nan
    trail_ref = np.nan
    cool = 0

    for i in range(len(c)):
        p_close = c[i]
        p_low = lo[i]
        p_high = hi[i]
        sig = pos_src.iloc[i]

        # Cooldown en cours -> pas de prise de position
        if cool > 0:
            out[i] = 0.0
            cool -= 1
            continue

        # Détection d'entrée/inversion autorisée uniquement si sig != 0
        if (in_pos == 0 and sig != 0) or (np.sign(sig) != np.sign(in_pos) and sig != 0):
            in_pos = float(np.sign(sig))
            entry = p_close
            trail_ref = p_close

        # Trailing
        if trailing and in_pos != 0:
            if in_pos > 0:
                trail_ref = max(trail_ref, p_high if use_hl else p_close)
            else:
                trail_ref = min(trail_ref, p_low if use_hl else p_close)

        # Calcul stop courant
        if in_pos > 0:
            base = trail_ref if trailing else entry
            stop = base * (1 - pct)
            hit = (p_low <= stop) if use_hl else (p_close <= stop)
        elif in_pos < 0:
            base = trail_ref if trailing else entry
            stop = base * (1 + pct)
            hit = (p_high >= stop) if use_hl else (p_close >= stop)
        else:
            hit = False

        # Sortie forcée si stop
        if in_pos != 0 and hit:
            in_pos = 0.0
            entry = np.nan
            trail_ref = np.nan
            cool = cooldown  # imposer un délai de ré-entrée

        # Si le signal d'origine est 0, rester flat
        if sig == 0:
            in_pos = 0.0
            entry = np.nan
            trail_ref = np.nan

        out[i] = in_pos

    return pd.Series(out, index=position.index, name="position")
