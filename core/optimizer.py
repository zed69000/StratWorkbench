# core/optimizer.py
from __future__ import annotations
import itertools
import random
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from .engine import run_strategy
from .metrics import equity_from_position, sharpe, growth_index, stability_index, max_drawdown

# -------- helpers --------
def _values_from_spec(spec: Dict[str, Any]) -> List[Any]:
    t = spec.get("type")
    if t == "int":
        mn, mx, st = int(spec["min"]), int(spec["max"]), int(spec.get("step", 1))
        return list(range(mn, mx + 1, st))
    if t == "float":
        mn, mx, st = float(spec["min"]), float(spec["max"]), float(spec.get("step", 1.0))
        n = int(round((mx - mn) / st)) + 1
        return [mn + i * st for i in range(max(n, 1))]
    if t == "bool":
        return [False, True] if spec.get("both", True) else [bool(spec.get("default", False))]
    if t == "list":
        return list(spec.get("options", []))
    # fallback: single default
    return [spec.get("default")]

def _product_params(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not schema:
        return [{}]
    keys = list(schema.keys())
    grids = [_values_from_spec(schema[k] if isinstance(schema[k], dict) else {"default": schema[k]}) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*grids)]

def _score_factory(objective: str, weights: Optional[Dict[str, float]]):
    def _score(stats: Dict[str, float]) -> float:
        if objective == "sharpe":
            return stats["sharpe"]
        if objective == "final":
            return stats["final"]
        if objective == "dd":
            return -abs(stats["dd"])
        if objective == "growth":
            return stats["growth"]
        if objective == "stability":
            return stats["stab"]
        if objective == "composite":
            w = weights or {"final":1.0,"sharpe":1.0,"dd":1.0,"growth":0.0,"stability":0.0}
            return (
                w.get("final",0.0)*stats["final"]
              + w.get("sharpe",0.0)*stats["sharpe"]
              - w.get("dd",0.0)*abs(stats["dd"])
              + w.get("growth",0.0)*stats["growth"]
              + w.get("stability",0.0)*stats["stab"]
            )
        return stats["sharpe"]
    return _score

# -------- main --------
def optimize_strategy(
    df: pd.DataFrame,
    strat_info,
    cash_start: float = 10_000.0,
    max_combos: int = 2000,
    fee_bps: float = 0.0,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    fee_on_sell_only: bool = False,
    stop_loss_pct: float | None = None,
    filters: List[Tuple[Any, dict]] | None = None,
    optimize_filters: bool = False,
    objective: str = "sharpe",
    weights: Dict[str, float] | None = None,
):
    # Grille stratégie
    strat_schema = getattr(strat_info, "params_schema", {}) or {}
    strat_grid = _product_params(strat_schema)
    random.shuffle(strat_grid)

    # Grilles filtres
    filters = filters or []
    filters_grids: List[List[dict]] = []
    for fref, fparams in filters:
        if optimize_filters:
            schema = getattr(fref, "PARAMS_SCHEMA", {}) or {}
            grid = _product_params(schema)
        else:
            grid = [fparams or {}]
        random.shuffle(grid)
        filters_grids.append(grid)

    # Prépare combinateur de filtres
    filters_product = list(itertools.product(*filters_grids)) if filters_grids else [()]

    score = _score_factory(objective, weights)
    best_score = float("-inf")
    best: Optional[Dict[str, Any]] = None
    tried = 0

    for p_strat in strat_grid:
        for f_combo in filters_product:
            # run strategy
            pos = run_strategy(df, strat_info.ref, p_strat)

            # apply filters
            filt_params_list = []
            for (fref, _fixed), fp in zip(filters, f_combo):
                pos = fref.apply(df, pos, fp)
                filt_params_list.append(fp)

            # equity + stats
            eq = equity_from_position(
                df, pos, cash_start=cash_start,
                fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps,
                fee_on_sell_only=fee_on_sell_only, stop_loss_pct=stop_loss_pct
            )
            stats = {
                "params": p_strat,
                "final": float(eq.iloc[-1]) if len(eq) else 0.0,
                "sharpe": float(sharpe(eq)) if len(eq) else 0.0,
                "growth": float(growth_index(eq)) if len(eq) else 0.0,
                "stab":   float(stability_index(eq)) if len(eq) else 0.0,
                "dd":     float(max_drawdown(eq)) if len(eq) else 0.0,
                "filters_params": filt_params_list,
            }

            s = score(stats)
            tried += 1
            if s > best_score:
                best_score, best = s, stats

            if tried >= max_combos:
                return best or stats

    return best or {
        "params": {},
        "final": 0.0, "sharpe": 0.0, "growth": 0.0, "stab": 0.0, "dd": 0.0,
        "filters_params": [{} for _ in filters]
    }
