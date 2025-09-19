import itertools, json
import pandas as pd
from .engine import backtest_portfolio, run_strategy
from .metrics import sharpe, growth_index, stability_index, max_drawdown

def optimize_strategy(df, strat_info, cash_start=10_000.0, max_combos=100):
    schema = strat_info.params_schema
    keys = list(schema.keys())

    # Construire grille de paramètres
    grids = []
    for k in keys:
        s = schema[k]
        if s["type"] == "int":
            vals = range(s["min"], s["max"]+1, s["step"])
        elif s["type"] == "float":
            n = int((s["max"]-s["min"])/s["step"])+1
            vals = [round(s["min"]+i*s["step"], 6) for i in range(n)]
        else:
            vals = [s.get("default")]
        grids.append(list(vals))

    results = []
    for combo in itertools.product(*grids):
        params = dict(zip(keys, combo))
        pos = run_strategy(df, strat_info.ref, params)
        eq = backtest_portfolio(df, [(strat_info.ref, params)], cash_start)[0]
        stats = {
            "params": params,
            "final": float(eq.iloc[-1]),
            "sharpe": float(sharpe(eq)),
            "growth": float(growth_index(eq)),
            "stab": float(stability_index(eq)),
            "dd": float(max_drawdown(eq)),
        }
        results.append(stats)
        if len(results) >= max_combos: break

    # Choisir le meilleur (ex: Sharpe max)
    best = max(results, key=lambda r: r["sharpe"])
    return best
