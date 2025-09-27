Déposer le contenu de ce zip dans votre dossier `filter/`.
Chaque fichier expose:
- NAME
- PARAMS_SCHEMA
- apply(df: DataFrame, pos: Series, params: dict) -> Series

Filtres inclus:
- adx_gate.py        : tendance forte (ADX, +DI > -DI)
- atr_band_gate.py   : fenêtre de volatilité via ATR%
- chop_gate.py       : évite le marché haché (Choppiness)
- volume_gate.py     : seuil de liquidité par rapport à la moyenne

Tous compatibles avec `discover_filters()`.
