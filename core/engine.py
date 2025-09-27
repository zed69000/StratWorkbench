from __future__ import annotations
import importlib.util, pathlib, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import pandas as pd

from .metrics import equity_from_position

@dataclass
class StrategyInfo:
    module_name: str
    ref: Any
    name: str
    params_schema: dict

def _module_has_strategy(mod) -> bool:
    return all(hasattr(mod, a) for a in ("NAME", "PARAMS_SCHEMA", "generate_signals"))

def discover_strategies(strats_dir: str) -> Dict[str, StrategyInfo]:
    path = pathlib.Path(strats_dir)
    if str(path.resolve()) not in sys.path:
        sys.path.insert(0, str(path.resolve()))
    found: Dict[str, StrategyInfo] = {}
    for pyfile in path.glob("*.py"):
        if pyfile.name.startswith("_"):
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
            found[mod.NAME] = StrategyInfo(mod.__name__, mod, mod.NAME, getattr(mod, "PARAMS_SCHEMA", {}))
    return found

@dataclass
class FilterInfo:
    module_name: str
    ref: Any
    name: str
    params_schema: dict

def _module_has_filter(mod) -> bool:
    return all(hasattr(mod, a) for a in ("NAME", "PARAMS_SCHEMA", "apply"))

def discover_filters(filter_dir: str) -> Dict[str, FilterInfo]:
    path = pathlib.Path(filter_dir)
    if not path.exists():
        return {}
    if str(path.resolve()) not in sys.path:
        sys.path.insert(0, str(path.resolve()))
    found: Dict[str, FilterInfo] = {}
    for pyfile in path.glob("*.py"):
        if pyfile.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(pyfile.stem, pyfile)
        mod = importlib.util.module_from_spec(spec)
        try:
            assert spec.loader is not None
            spec.loader.exec_module(mod)  # type: ignore
        except Exception as e:
            print(f"[FILTER] Import error {pyfile.name}: {e}")
            continue
        if _module_has_filter(mod):
            found[mod.NAME] = FilterInfo(mod.__name__, mod, mod.NAME, getattr(mod, "PARAMS_SCHEMA", {}))
    return found

def run_strategy(df: pd.DataFrame, strat_ref, params: dict) -> pd.Series:
    pos = strat_ref.generate_signals(df, params or {})
    if not isinstance(pos, pd.Series):
        pos = pd.Series(pos, index=df.index)
    return pos.rename("position").clip(-1, 1).astype(float)

def backtest_portfolio(
    df: pd.DataFrame,
    active: List[Tuple[Any, dict]],
    cash_start: float = 10_000.0,
    filters: List[Tuple[Any, dict]] | None = None,
    fee_bps: float = 0.0,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    fee_on_sell_only: bool = False,
    stop_loss_pct: float | None = None,
):
    if not active:
        eq = pd.Series([cash_start] * len(df), index=df.index, name='equity')
        return eq, {}, {}
    w = 1.0 / len(active)
    equities = []
    stats = {}
    per = {}
    for ref, params in active:
        pos = run_strategy(df, ref, params)
        for fref, fparams in (filters or []):
            pos = fref.apply(df, pos, fparams or {})
        eq = equity_from_position(
            df, pos, cash_start=cash_start * w,
            fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps,
            fee_on_sell_only=fee_on_sell_only, stop_loss_pct=stop_loss_pct
        )
        equities.append(eq)
        stats[getattr(ref, 'NAME', str(ref))] = {'final': float(eq.iloc[-1])}
        per[getattr(ref, 'NAME', str(ref))] = eq
    return sum(equities).rename('equity'), stats, per
