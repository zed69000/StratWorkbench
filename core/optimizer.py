from __future__ import annotations
import itertools
from typing import Dict, Any
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
            seq = []
            x = mn
            while x <= mx + 1e-12:
                seq.append(round(x, 10))
                x = x + st
            values.append(seq)
        else:
            values.append([s.get("default")])
    return keys, values

def optimize_strategy(
    df: pd.DataFrame, strat_info, cash_start: float = 10_000.0, max_combos: int = 200,
    fee_bps: float = 0.0, spread_bps: float = 0.0, slippage_bps: float = 0.0, fee_on_sell_only: bool = False,
    stop_loss_pct: float | None = None, filters=None,
):
    schema = getattr(strat_info, "params_schema", {}) or {}
    keys, values = _grid_from_schema(schema)
    results = []
    for combo in itertools.product(*values) if values else [()]:
        params = dict(zip(keys, combo)) if keys else {}
        pos = run_strategy(df, strat_info.ref, params)
        for fref, fparams in (filters or []):
            pos = fref.apply(df, pos, fparams or {})
        eq = equity_from_position(
            df, pos, cash_start=cash_start,
            fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps,
            fee_on_sell_only=fee_on_sell_only, stop_loss_pct=stop_loss_pct
        )
        stats = {
            "params": params,
            "final": float(eq.iloc[-1]),
            "sharpe": float(sharpe(eq)),
            "growth": float(growth_index(eq)),
            "stab": float(stability_index(eq)),
            "dd": float(max_drawdown(eq)),
        }
        results.append(stats)
        if len(results) >= max_combos:
            break
    if not results:
        return None
    return max(results, key=lambda r: r["sharpe"])
