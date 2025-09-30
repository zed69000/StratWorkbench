from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

from core.metrics import equity_from_position, Fees
from core.engine import backtest_portfolio

# ------------------ Time filter ------------------
def apply_time_filter(dfi: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Filtre temporel standardisé pour l'app.
    """
    if not isinstance(dfi.index, pd.DatetimeIndex):
        return dfi
    if mode == "1 Jour":
        return dfi.loc[dfi.index >= dfi.index.max() - pd.Timedelta(days=1)]
    elif mode == "1 Semaine":
        return dfi.loc[dfi.index >= dfi.index.max() - pd.Timedelta(weeks=1)]
    elif mode == "1 Mois":
        return dfi.loc[dfi.index >= dfi.index.max() - pd.Timedelta(days=30)]
    return dfi

# ------------------ Construction positions + équités ------------------
def _build_positions_and_individual_equities(
    dfi: pd.DataFrame,
    active_list: List[Tuple[object, dict]],
    filters: List[Tuple[object, dict]],
    cash_start: float,
    fee_bps: float,
    spread_bps: float,
    slippage_bps: float,
    fee_on_sell_only: bool,
) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """
    Retourne (pos_port, per_dict)
    pos_port : Series exposition de portefeuille agrégée (moyenne des positions)
    per_dict : {strat_name: equity_series} équité individuelle calculée avec cash_start fractionné
    """
    positions = []
    per: Dict[str, pd.Series] = {}
    if not active_list:
        return pd.Series(0, index=dfi.index), per

    for ref, params in active_list:
        pos = ref.generate_signals(dfi, params)
        for fref, fparams in filters:
            pos = fref.apply(dfi, pos, fparams)
        pos = pos.fillna(0).astype(float)
        positions.append(pos)

    if not positions:
        pos_port = pd.Series(0, index=dfi.index)
        return pos_port, per

    positions_df = pd.concat(positions, axis=1).fillna(0)
    pos_port = positions_df.mean(axis=1)  # exposition portefeuille partagée (moyenne simple)

    # équités individuelles pour affichage/debug (sur capital fractionné)
    w = 1.0 / max(1, len(positions))
    for i, (ref, _params) in enumerate(active_list):
        idx_name = getattr(ref, "NAME", str(ref))
        try:
            fees = Fees(maker_bps=fee_bps, taker_bps=fee_bps,
                        spread_bps=spread_bps, slippage_bps=slippage_bps,
                        fee_on_sell_only=fee_on_sell_only)
            eqi = equity_from_position(dfi, positions[i], cash_start=cash_start * w, fees=fees)
        except Exception:
            eqi = pd.Series(cash_start * w, index=dfi.index)
        per[idx_name] = eqi

    return pos_port, per


def compute_portfolio(
    dfi: pd.DataFrame,
    active_list: List[Tuple[object, dict]],
    filters: List[Tuple[object, dict]],
    mode: str,
    cash_start: float,
    fee_bps: float,
    spread_bps: float,
    slippage_bps: float,
    fee_on_sell_only: bool,
) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """
    Calcule l'équité portefeuille selon le mode choisi.
    Retourne (equity_series, per_dict)
    """
    if not active_list:
        return pd.Series(dtype=float), {}

    if mode.startswith("Capital partagé"):
        pos_port, per = _build_positions_and_individual_equities(
            dfi, active_list, filters, cash_start,
            fee_bps, spread_bps, slippage_bps, fee_on_sell_only
        )
        fees = Fees(maker_bps=fee_bps, taker_bps=fee_bps,
                    spread_bps=spread_bps, slippage_bps=slippage_bps,
                    fee_on_sell_only=fee_on_sell_only)
        equity = equity_from_position(dfi, pos_port, cash_start=cash_start, fees=fees)
        return equity, per
    else:
        equity, _stats, per = backtest_portfolio(
            dfi, active_list, cash_start=cash_start, filters=filters,
            fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps,
            fee_on_sell_only=fee_on_sell_only
        )
        return equity, per
