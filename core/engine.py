from __future__ import annotations
import importlib.util, pathlib, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import pandas as pd

from .metrics import equity_from_position

@dataclass
class StrategyInfo:
    module_name: str
    ref: Any          # module exposing NAME, PARAMS_SCHEMA, generate_signals
    name: str
    params_schema: dict

def _module_has_strategy(mod) -> bool:
    return hasattr(mod, 'NAME') and hasattr(mod, 'PARAMS_SCHEMA') and hasattr(mod, 'generate_signals')

def discover_strategies(strats_dir: str) -> Dict[str, StrategyInfo]:
    path = pathlib.Path(strats_dir)
    if str(path.resolve()) not in sys.path:
        sys.path.insert(0, str(path.resolve()))
    found: Dict[str, StrategyInfo] = {}
    for pyfile in path.glob("*.py"):
        if pyfile.name.startswith("_"):  # ignore __init__ etc.
            continue
        spec = importlib.util.spec_from_file_location(pyfile.stem, pyfile)
        mod = importlib.util.module_from_spec(spec)
        try:
            assert spec.loader is not None
            spec.loader.exec_module(mod)  # type: ignore
        except Exception as e:
            print(f"[DISCOVER] Import error {pyfile.name}: {e}")
            continue
        if _module_has_strategy(mod):
            name = getattr(mod, 'NAME')
            schema = getattr(mod, 'PARAMS_SCHEMA')
            found[name] = StrategyInfo(module_name=mod.__name__, ref=mod, name=name, params_schema=schema)
            print(f"[DISCOVER] OK: {name}")
        else:
            print(f"[DISCOVER] Skip {pyfile.name}: no NAME/PARAMS_SCHEMA/generate_signals")
    return found

def run_strategy(df: pd.DataFrame, strat_ref, params: dict) -> pd.Series:
    pos = strat_ref.generate_signals(df, params)

    # 🔒 Sécurité si une stratégie renvoie None
    if pos is None:
        pos = pd.Series(0.0, index=df.index, name="position")

    elif not isinstance(pos, pd.Series):
        pos = pd.Series(pos, index=df.index, name="position")
    else:
        pos = pos.reindex(df.index)

    return pos.fillna(0.0).clip(-1, 1)

def backtest_portfolio(df: pd.DataFrame, active: List[Tuple[object, dict]], cash_start=10_000.0):
    if not active:
        eq = pd.Series([cash_start] * len(df), index=df.index, name='equity')
        return eq, {}, {}
    w = 1.0 / len(active)
    equities = []
    stats = {}
    per = {}
    for ref, params in active:
        pos = run_strategy(df, ref, params)
        eq = equity_from_position(df, pos, cash_start=cash_start * w)
        equities.append(eq)
        stats[getattr(ref, 'NAME', str(ref))] = {'final': float(eq.iloc[-1])}
        per[getattr(ref, 'NAME', str(ref))] = eq
    return sum(equities).rename('equity'), stats, per
