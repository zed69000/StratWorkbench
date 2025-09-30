from __future__ import annotations
import json
import io
from typing import Tuple, Dict, Any
import pandas as pd

def try_cast(v: Any):
    """Cast robuste depuis string -> int/float/bool si possible, sinon retour brut."""
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        try:
            if "." in s:
                return float(s)
            return int(s)
        except Exception:
            try:
                return float(s)
            except Exception:
                return s
    return v

def _normalize_json_payload(data: dict) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Accepte:
      { "StratA": {"params": {...}}, "filters": {"[FILTER]X": {...}} }
    ou   { "StratA": {...}, "filters": {...} }
    Retourne (by_strat, filters)
    """
    by_strat: Dict[str, dict] = {}
    filters: Dict[str, dict] = {}
    if not isinstance(data, dict):
        return by_strat, filters
    # stratégies
    for k, v in data.items():
        if k == "filters":
            continue
        if isinstance(v, dict) and "params" in v and isinstance(v["params"], dict):
            by_strat[k] = {kk: try_cast(vv) for kk, vv in v["params"].items()}
        elif isinstance(v, dict):
            by_strat[k] = {kk: try_cast(vv) for kk, vv in v.items()}
    # filtres
    froot = data.get("filters", {})
    if isinstance(froot, dict):
        filters = {fk: {kk: try_cast(vv) for kk, vv in fv.items()} for fk, fv in froot.items() if isinstance(fv, dict)}
    return by_strat, filters

def _parse_csv(file_bytes: bytes, filename: str) -> Dict[str, dict]:
    """
    CSV long:  columns = strat,param,value
    CSV large: first column = strat (ou une colonne nommée 'strat'), autres = params
    """
    bio = io.BytesIO(file_bytes)
    dfp = pd.read_csv(bio)
    cols_lower = [c.lower() for c in dfp.columns]
    dfp.columns = cols_lower
    if {"strat", "param", "value"}.issubset(set(cols_lower)):  # format long
        pivot = dfp.pivot(index="strat", columns="param", values="value")
        by = {}
        for strat in pivot.index:
            row = pivot.loc[strat].to_dict()
            by[strat] = {k: try_cast(v) for k, v in row.items() if pd.notna(v)}
        return by
    # format large
    if "strat" in cols_lower:
        dfp = dfp.set_index("strat")
    else:
        dfp = dfp.set_index(dfp.columns[0])
    by = {}
    for strat, row in dfp.iterrows():
        params = {c: try_cast(row[c]) for c in dfp.columns if pd.notna(row[c])}
        by[strat] = params
    return by

def load_params_from_file(file_obj, filename: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Lit un fichier JSON ou CSV et retourne (by_strat, filters).
    - file_obj: objet type UploadedFile (Streamlit) ou file-like supportant .read()
    """
    name = filename.lower()
    content = file_obj.read()
    if name.endswith(".json"):
        data = json.loads(content.decode("utf-8"))
        return _normalize_json_payload(data)
    elif name.endswith(".csv"):
        by = _parse_csv(content, name)
        return by, {}
    else:
        raise ValueError("Format non supporté. Utiliser .json ou .csv")

def save_params_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
