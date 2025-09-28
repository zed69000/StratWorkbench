# core/optimizer.py
from __future__ import annotations
import itertools
import random
from typing import Dict, Any, List, Tuple, Optional
import math
import pandas as pd

from .engine import run_strategy
from .metrics import equity_from_position, sharpe, growth_index, stability_index, max_drawdown

# -------- helpers: grilles --------
def _values_from_spec(spec: Dict[str, Any]) -> List[Any]:
    t = spec.get("type")
    if t == "int":
        mn, mx, st = int(spec["min"]), int(spec["max"]), int(spec.get("step", 1))
        return list(range(mn, mx + 1, st))
    if t == "float":
        mn, mx, st = float(spec["min"]), float(spec["max"]), float(spec.get("step", 1.0))
        n = int(round((mx - mn) / st)) + 1
        return [round(mn + i * st, 10) for i in range(max(n, 1))]
    if t == "bool":
        return [False, True] if spec.get("both", True) else [bool(spec.get("default", False))]
    if t == "list":
        return list(spec.get("options", []))
    return [spec.get("default")]

def _product_params(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not schema:
        return [{}]
    keys = list(schema.keys())
    grids = [_values_from_spec(schema[k] if isinstance(schema[k], dict) else {"default": schema[k]}) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*grids)]

def _snap_float(x: float, step: float, lo: float) -> float:
    return round(lo + round((x - lo) / step) * step, 10)

def _sampled_product_params(schema: Dict[str, Any], target: int) -> List[Dict[str, Any]]:
    """
    Sous-échantillonne une grille de paramètres numériques pour limiter l'explosion combinatoire.
    target = nombre approx. de combos visés pour CE filtre seul.
    """
    if not schema:
        return [{}]
    keys = list(schema.keys())
    # combien de paramètres numériques pour répartir "target"
    pnum = sum(1 for s in schema.values() if isinstance(s, dict) and s.get("type") in ("int", "float"))
    per_axis = max(1, int(round(target ** (1 / max(1, pnum)))))

    vals: List[List[Any]] = []
    for k in keys:
        s = schema[k] if isinstance(schema[k], dict) else {"default": schema[k]}
        t = s.get("type")
        if t == "int":
            lo, hi, st = int(s["min"]), int(s["max"]), int(s.get("step", 1))
            full = list(range(lo, hi + 1, st))
            if len(full) <= per_axis:
                vals.append(full)
            else:
                # prélève régulièrement sur l'axe
                step_idx = max(1, len(full) // per_axis)
                cand = full[::step_idx]
                if cand[-1] != full[-1]:
                    cand.append(full[-1])
                vals.append(cand[:per_axis])
        elif t == "float":
            lo, hi, st = float(s["min"]), float(s["max"]), float(s.get("step", 1.0))
            if per_axis == 1:
                v = [_snap_float((lo + hi) / 2.0, st, lo)]
            else:
                v = [_snap_float(lo + i * (hi - lo) / (per_axis - 1), st, lo) for i in range(per_axis)]
            vals.append(sorted(set(v)))
        else:
            vals.append([s.get("default")])

    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*vals):
        out.append(dict(zip(keys, combo)))
        if len(out) >= target:
            break
    return out if out else [{}]

# -------- helpers: scoring --------
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
            w = {"final": 1.0, "sharpe": 1.0, "dd": 1.0}
            if weights:
                w.update(weights)
            ret = stats["final"] / stats.get("_cash_start", 1.0) - 1.0
            return w["sharpe"] * stats["sharpe"] + w["final"] * ret - w["dd"] * abs(stats["dd"])
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
    # 1) Grille stratégie
    strat_schema = getattr(strat_info, "params_schema", {}) or {}
    strat_grid = _product_params(strat_schema)
    random.shuffle(strat_grid)

    # 2) Grilles filtres (échantillonnées si optimize_filters)
    filters = filters or []
    n_active_filters = len(filters) if optimize_filters else 0

    # budget par combo de filtres pour un set de paramètres strat
    # approx: on répartit max_combos sur les variations de stratégie
    budget_filters = max(1, max_combos // max(1, len(strat_grid)))
    per_filter_budget = max(1, int(round(budget_filters ** (1 / max(1, n_active_filters)))))

    filters_grids: List[List[dict]] = []
    for fref, fparams in filters:
        if optimize_filters:
            schema = getattr(fref, "PARAMS_SCHEMA", {}) or {}
            grid = _sampled_product_params(schema, per_filter_budget)
        else:
            grid = [fparams or {}]
        random.shuffle(grid)
        filters_grids.append(grid)

    score = _score_factory(objective, weights)
    best_score = float("-inf")
    best: Optional[Dict[str, Any]] = None
    tried = 0

    # 3) Boucle: on calcule une fois la stratégie, puis on applique les filtres
    for p_strat in strat_grid:
        base_pos = run_strategy(df, strat_info.ref, p_strat)

        # réinitialise le produit cartésien des filtres à chaque itération
        filters_product = itertools.product(*filters_grids) if filters_grids else [()]

        for f_combo in filters_product:
            # applique les filtres sur la position de base
            pos = base_pos
            filt_params_list = []
            for (fref, _fixed), fp in zip(filters, f_combo):
                pos = fref.apply(df, pos, fp)
                filt_params_list.append(fp)

            # équité + stats
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
                "_cash_start": cash_start,
            }

            s = score(stats)
            tried += 1
            if s > best_score:
                best_score, best = s, {k: v for k, v in stats.items() if not k.startswith("_")}

            if tried >= max_combos:
                return best if best is not None else {k: v for k, v in stats.items() if not k.startswith("_")}

    return best or {
        "params": {},
        "final": 0.0, "sharpe": 0.0, "growth": 0.0, "stab": 0.0, "dd": 0.0,
        "filters_params": [{} for _ in filters]
    }
