from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from core.metrics import equity_from_position, growth_index, stability_index, max_drawdown, sharpe, Fees
from core.portfolio import apply_time_filter

ActiveList = List[Tuple[object, dict]]
FiltersList = List[Tuple[object, dict]]

def _build_pos_single_strat(dfi: pd.DataFrame, ref, params: dict, filters: FiltersList) -> pd.Series:
    """Génère la position d'une stratégie avec application séquentielle des filtres."""
    pos = ref.generate_signals(dfi, params)
    for fref, fparams in filters:
        pos = fref.apply(dfi, pos, fparams)
    return pos.fillna(0).astype(float)

def run_benchmark(
    dfs: Dict[str, pd.DataFrame],
    active: ActiveList,
    filters: FiltersList,
    *,
    cash_start: float = 10_000.0,
    fee_bps: float = 0.0,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    fee_on_sell_only: bool = False,
    time_filter: str = "Tout",
) -> pd.DataFrame:
    """
    Évalue chaque stratégie individuellement sur chaque courbe.
    Retourne un DataFrame indexé par (courbe, strat) avec colonnes:
      final, growth_idx, stability_idx, dd_min, sharpe, trades, trades_per_day
    """
    rows = []
    for label, dfi in dfs.items():
        dfi_filtered = apply_time_filter(dfi, time_filter)
        for ref, params in active:
            pos = _build_pos_single_strat(dfi_filtered, ref, params, filters)

            fees = Fees(
                maker_bps=float(fee_bps),
                taker_bps=float(fee_bps),
                spread_bps=float(spread_bps),
                slippage_bps=float(slippage_bps),
                fee_on_sell_only=bool(fee_on_sell_only),
            )
            eqi = equity_from_position(dfi_filtered, pos, cash_start=cash_start, fees=fees)

            changes = (pos.diff().fillna(0) != 0).sum()
            nb_trades = int(changes // 2)
            if isinstance(dfi_filtered.index, pd.DatetimeIndex):
                n_days = max(1, (dfi_filtered.index[-1] - dfi_filtered.index[0]).days)
            else:
                n_days = max(1, len(dfi_filtered) // 100)
            trades_per_day = nb_trades / n_days

            rows.append({
                "courbe": label,
                "strat": getattr(ref, "NAME", str(ref)),
                "final": float(eqi.iloc[-1]),
                "growth_idx": growth_index(eqi),
                "stability_idx": stability_index(eqi),
                "dd_min": float(max_drawdown(eqi)),
                "sharpe": float(sharpe(eqi)),
                "trades": nb_trades,
                "trades_per_day": trades_per_day,
            })
    return pd.DataFrame(rows).set_index(["courbe", "strat"])

def rankings(res: pd.DataFrame):
    """
    Calcule:
      - best: meilleure stratégie par courbe sur 'final'
      - pivot_abs: nb de marchés où une strat > médiane globale
      - pivot_rel: nb de fois dans le Top-N par courbe (N= max(1, nb_strats//3))
      - pivot_all: fusion des deux pivots
      - seuil, topN: valeurs utilisées
    """
    # Meilleure stratégie par courbe
    best = (
        res["final"]
        .groupby(level=0).idxmax()
        .apply(lambda x: x[1])
        .rename("Meilleure stratégie (final)")
        .to_frame()
    )

    df_bench = res.reset_index()[["courbe", "strat", "final"]]
    seuil = df_bench["final"].median()
    pivot_abs = (df_bench
                 .assign(efficace=lambda d: d["final"] > seuil)
                 .groupby("strat")["efficace"].sum()
                 .sort_values(ascending=False)
                 .to_frame("nb_marches_efficaces"))

    nb_strats = df_bench["strat"].nunique()
    topN = max(1, nb_strats // 3)

    def _mark_topN(d, n=2):
        return d.sort_values("final", ascending=False).head(n).assign(topN=True)

    top_marks = (df_bench.groupby("courbe", group_keys=False).apply(_mark_topN, n=topN))
    pivot_rel = (top_marks.groupby("strat")["topN"].sum()
                 .sort_values(ascending=False)
                 .to_frame(f"nb_top{topN}"))

    pivot_all = pivot_abs.join(pivot_rel, how="outer").fillna(0).astype(int)

    return {"best": best, "pivot_abs": pivot_abs, "pivot_rel": pivot_rel,
            "pivot_all": pivot_all, "seuil": seuil, "topN": topN}
