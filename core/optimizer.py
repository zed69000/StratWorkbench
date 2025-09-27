# optimizer.py
from __future__ import annotations
import itertools
from typing import Dict, Any, List, Tuple
import pandas as pd
from .engine import run_strategy
from .metrics import equity_from_position, sharpe, growth_index, stability_index, max_drawdown

def _grid_from_schema(schema: Dict[str, Any]):
    keys = list(schema.keys())
    values = []
    for k in keys:
        s = schema[k]
        if s.get("type") == "int":
            values.append(list(range(int(s["min"]), int(s["max"]) + 1, int(s["step"]))))
        elif s.get("type") == "float":
            mn, mx, st = float(s["min"]), float(s["max"]), float(s["step"])
            seq, x = [], mn
            while x <= mx + 1e-12:
                seq.append(round(x, 10))
                x = x + st
            values.append(seq)
        else:
            values.append([s.get("default")])
    return keys, values

def _product_params(schema: Dict[str, Any]):
    """Retourne une liste de dicts couvrant la grille du schema."""
    if not schema:
        return [{}]
    k, v = _grid_from_schema(schema)
    return [dict(zip(k, combo)) for combo in itertools.product(*v)]

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
    objective: str = "sharpe",      # "sharpe" | "final" | "dd" | "growth" | "stability" | "composite"
    weights: Dict[str, float] | None = None,  # pour "composite": {"final":1.0, "sharpe":1.0, "dd":1.0}
):
    # Grille des paramètres de la stratégie
    strat_grid = _product_params(getattr(strat_info, "params_schema", {}) or {})

    # Grille des filtres: paramètre unique fixé OU grille si optimize_filters=True
    filters = filters or []
    filters_grids: List[List[dict]] = []
    for fref, fparams in filters:
        if optimize_filters:
            schema = getattr(fref, "PARAMS_SCHEMA", {}) or {}
            filters_grids.append(_product_params(schema))
        else:
            filters_grids.append([fparams or {}])

    # Produit cartésien de toutes les combinaisons
    if filters_grids:
        filters_product = itertools.product(*filters_grids)
    else:
        filters_product = [()]

    best = None
    best_score = float("-inf")
    tried = 0

    def _score(stats: Dict[str, float]) -> float:
        if objective == "sharpe":
            return stats["sharpe"]
        if objective == "final":
            return stats["final"]
        if objective == "dd":
            # max_drawdown() renvoie un nombre négatif. Moins pire = plus grand.
            return stats["dd"]
        if objective == "growth":
            return stats["growth"]
        if objective == "stability":
            return stats["stab"]
        # composite: normalise le résultat en rendement et pénalise le DD
        w = {"final": 1.0, "sharpe": 1.0, "dd": 1.0}
        if weights:
            w.update(weights)
        ret = stats["final"] / cash_start - 1.0
        return w["sharpe"] * stats["sharpe"] + w["final"] * ret + w["dd"] * (-abs(stats["dd"]))

    for p_strat in strat_grid:
        # réinitialiser l’itérateur des filtres à chaque p_strat
        if filters_grids:
            filters_product = itertools.product(*filters_grids)
        else:
            filters_product = [()]

        for f_combo in filters_product:
            # Génère et applique
            pos = run_strategy(df, strat_info.ref, p_strat)
            filt_params_list = []
            for (fref, _fixed), fp in zip(filters, f_combo):
                pos = fref.apply(df, pos, fp)
                filt_params_list.append(fp)

            eq = equity_from_position(
                df, pos, cash_start=cash_start,
                fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps,
                fee_on_sell_only=fee_on_sell_only, stop_loss_pct=stop_loss_pct
            )

            stats = {
                "params": p_strat,
                "final": float(eq.iloc[-1]),
                "sharpe": float(sharpe(eq)),
                "growth": float(growth_index(eq)),
                "stab":   float(stability_index(eq)),
                "dd":     float(max_drawdown(eq)),
                "filters_params": filt_params_list,
            }
            score = _score(stats)
            tried += 1
            if score > best_score:
                best_score, best = score, stats

            if tried >= max_combos:
                return best or stats

    return best
