# core/state.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Optional
import streamlit as st
import pandas as pd

# Clés centralisées
SS = SimpleNamespace(
    PARAMS="__params",
    ACTIVE_FILTERS="__active_filters",
    BENCH="benchmark_results",
    LAST="last_config",
)

# ---------- Session ----------

def init_session() -> None:
    if SS.PARAMS not in st.session_state:
        st.session_state[SS.PARAMS] = {}
    if SS.ACTIVE_FILTERS not in st.session_state:
        st.session_state[SS.ACTIVE_FILTERS] = []
    if SS.BENCH not in st.session_state:
        st.session_state[SS.BENCH] = None
    if SS.LAST not in st.session_state:
        st.session_state[SS.LAST] = {}

# ---------- Params strat / filtres ----------

def get_params_all() -> Dict[str, dict]:
    return dict(st.session_state.get(SS.PARAMS, {}))

def set_params_for_strat(name: str, params: Dict[str, Any]) -> None:
    p = st.session_state.setdefault(SS.PARAMS, {})
    p[name] = dict(params)

def set_params_bulk(by_strat: Dict[str, dict]) -> None:
    p = st.session_state.setdefault(SS.PARAMS, {})
    for k, v in (by_strat or {}).items():
        p[k] = dict(v)

def get_filter_params_map() -> Dict[str, dict]:
    # clés de type "[FILTER]Name"
    return {k: v for k, v in get_params_all().items() if k.startswith("[FILTER]")}

def set_filter_params(name_with_tag: str, params: Dict[str, Any]) -> None:
    # name_with_tag = "[FILTER]X"
    p = st.session_state.setdefault(SS.PARAMS, {})
    p[name_with_tag] = dict(params)

def get_active_filter_names() -> List[str]:
    return list(st.session_state.get(SS.ACTIVE_FILTERS, []))

def add_active_filter(name: str) -> None:
    lst = st.session_state.setdefault(SS.ACTIVE_FILTERS, [])
    if name not in lst:
        lst.append(name)

def remove_active_filter(name: str) -> None:
    lst = st.session_state.setdefault(SS.ACTIVE_FILTERS, [])
    st.session_state[SS.ACTIVE_FILTERS] = [n for n in lst if n != name]

# ---------- Benchmark cache ----------

def get_benchmark() -> Optional[pd.DataFrame]:
    return st.session_state.get(SS.BENCH, None)

def set_benchmark(df: Optional[pd.DataFrame]) -> None:
    st.session_state[SS.BENCH] = df

def clear_benchmark() -> None:
    st.session_state[SS.BENCH] = None

# ---------- Config et invalidation ----------

@dataclass(frozen=True)
class AppConfig:
    # champs optionnels synthétiques
    seed: Optional[int] = None
    n_points: Optional[int] = None
    jitter: Optional[float] = None
    curve_kind: Optional[str] = None

    # commun
    strategies: Tuple[str, ...] = ()
    params: Dict[str, dict] = None
    filters: Dict[str, dict] = None
    fees: Dict[str, Any] = None
    time_filter: str = "Tout"
    port_mode: str = "Capital partagé (1 seul PnL)"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # normalisation pour comparaison stable
        d["strategies"] = list(self.strategies)
        d["params"] = dict(self.params or {})
        d["filters"] = dict(self.filters or {})
        d["fees"] = dict(self.fees or {})
        return d

def assemble_config(
    *,
    data_mode: str,
    synth_seed: Optional[int],
    n_points: Optional[int],
    jitter_pct: Optional[float],
    curve_kind: Optional[str],
    active_names: List[str],
    active_filters: List[Tuple[object, dict]],
    time_filter: str,
    port_mode: str,
    fees: Dict[str, Any],
) -> AppConfig:
    # map filtres -> dict { "[FILTER]Name": params }
    filt_map = {
        f"[FILTER]{getattr(ref, 'NAME', str(ref))}": dict(params or {})
        for ref, params in (active_filters or [])
    }
    return AppConfig(
        seed=(synth_seed if data_mode == "Synthétique" else None),
        n_points=(n_points if data_mode == "Synthétique" else None),
        jitter=(jitter_pct if data_mode == "Synthétique" else None),
        curve_kind=(curve_kind if data_mode == "Synthétique" else None),
        strategies=tuple(active_names or []),
        params={k: get_params_all().get(k, {}) for k in (active_names or [])},
        filters=filt_map,
        fees=dict(fees or {}),
        time_filter=time_filter,
        port_mode=port_mode,
    )

def get_last_config() -> Dict[str, Any]:
    return dict(st.session_state.get(SS.LAST, {}))

def set_last_config(cfg: AppConfig) -> None:
    st.session_state[SS.LAST] = cfg.to_dict()

def invalidate_if_changed(cfg: AppConfig) -> bool:
    """
    Retourne True si la config a changé et invalide le benchmark.
    """
    new_cfg = cfg.to_dict()
    if new_cfg != get_last_config():
        set_last_config(cfg)
        clear_benchmark()
        return True
    return False
