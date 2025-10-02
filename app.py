# app.py
from __future__ import annotations
import pathlib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json

from core.data import make_synth, load_multi_curves, load_csv
from core.engine import discover_strategies, backtest_portfolio, discover_filters
from core.metrics import (
    growth_index, stability_index, max_drawdown, sharpe, equity_from_position
)
from core.optimizer import optimize_strategy

# ------------------ Config Streamlit ------------------
st.set_page_config(page_title="StratWorkbench", layout="wide", page_icon="🧪")

STRATS_DIR = str(pathlib.Path(__file__).resolve().parent / "strats")
FILTER_DIR = str(pathlib.Path(__file__).resolve().parent / "filter")

if "__params" not in st.session_state:
    st.session_state["__params"] = {}
if "__active_filters" not in st.session_state:
    st.session_state["__active_filters"] = []
if "benchmark_results" not in st.session_state:
    st.session_state["benchmark_results"] = None
if "last_config" not in st.session_state:
    st.session_state["last_config"] = {}

# Couleurs
HIGHLIGHT = "#a9955ba6"
HIGHLIGHT2 = "#b6863f"

# ------------------ Helpers: synth + jitter déterministe ------------------
_SYNTH_KIND_SEED = {
    "sideways": 11,
    "slow_grind": 23,
    "trend_down": 37,
    "trend_up": 53,
    "volatile_whipsaw": 71,
}
def make_with_jitter(kind: str, n_points: int, seed: int, jitter_pct: float) -> pd.DataFrame:
    df = make_synth(n=n_points, kind=kind, seed=seed)
    if jitter_pct and jitter_pct > 0:
        base_seed = int(seed * 10007 + _SYNTH_KIND_SEED.get(kind, 97)) % (2**32)
        rng = np.random.default_rng(base_seed)
        noise = pd.Series(rng.normal(0, jitter_pct / 1000.0, size=len(df)), index=df.index)
        noise = noise.rolling(window=10, min_periods=1).mean()
        df["close"] *= (1 + noise)
        df["open"] = df["close"].shift(1).fillna(df["close"])
        absn = noise.abs()
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + absn / 2)
        df["low"]  = df[["open", "close"]].min(axis=1) * (1 - absn / 2)
    return df

# ------------------ Sidebar ------------------
st.sidebar.title("⚙️ Paramètres")
st.sidebar.caption("Auto-détection des stratégies dans /strats et des filtres dans /filter")

data_mode = st.sidebar.radio("Source des données", ["Synthétique", "Réelles (CSV)"])
dfs = {}  # {label: DataFrame}

if data_mode == "Synthétique":
    n_points = st.sidebar.slider("Taille série (points)", 500, 5000, 1500, 100)
    curve_kind = st.sidebar.selectbox(
        "Profil de marché (aperçu)",
        ["trend_up", "trend_down", "sideways", "volatile_whipsaw", "slow_grind"],
    )
    synth_seed = st.sidebar.number_input("Seed (0 = aléatoire)", value=123, min_value=0)
    jitter_pct = st.sidebar.slider("Jitter (%)", 0.0, 5.0, 0.0, 0.1)

    seed = np.random.randint(0, 1_000_000) if synth_seed == 0 else int(synth_seed)
    df_base = make_with_jitter(curve_kind, n_points, seed, jitter_pct)
    dfs = {curve_kind: df_base}

else:
    uploaded_files = st.sidebar.file_uploader(
        "Uploader un ou plusieurs fichiers CSV OHLCV",
        type=["csv"], accept_multiple_files=True
    )
    if uploaded_files:
        dfs = {f.name.replace(".csv", ""): load_csv(f) for f in uploaded_files}
        st.sidebar.success(f"✅ {len(uploaded_files)} fichiers chargés")
    else:
        st.sidebar.warning("⚠️ Charge au moins un CSV pour continuer")

# ------------------ Sélecteur de symboles ------------------
symbols_available = list(dfs.keys())
if symbols_available:
    selected_symbols = st.sidebar.multiselect(
        "Choisir les symboles à afficher",
        options=symbols_available,
        default=[symbols_available[0]]
    )
else:
    selected_symbols = []

# ------------------ Stratégies ------------------
st.sidebar.subheader("Stratégies détectées")
infos = discover_strategies(STRATS_DIR)
active_names: list[str] = []
if not infos:
    st.sidebar.error("Aucune stratégie trouvée dans /strats.")
else:
    for name, info in infos.items():
        if st.sidebar.checkbox(name, value=name in list(infos.keys())[:4]):
            active_names.append(name)

triangle_strategy = st.sidebar.selectbox(
    "Afficher les triangles pour une stratégie",
    ["Aucune"] + active_names
)

time_filter = st.sidebar.selectbox(
    "Filtre temporel",
    ["Tout", "1 Jour", "1 Semaine", "1 Mois"]
)

# --- NOUVEAU: sélecteur de mode portefeuille ---
port_mode = st.sidebar.radio(
    "Mode portefeuille",
    ["Capital partagé (1 seul PnL)", "Capital divisé (somme des PnL)"],
    index=0,
    help="Partagé: on moyenne les positions et on calcule une seule équité avec tout le capital.\nDivisé: on fractionne le capital entre les stratégies puis on somme leurs équités."
)

def apply_time_filter(dfi, mode):
    if not isinstance(dfi.index, pd.DatetimeIndex):
        return dfi
    if mode == "1 Jour":
        return dfi.loc[dfi.index >= dfi.index.max() - pd.Timedelta(days=1)]
    elif mode == "1 Semaine":
        return dfi.loc[dfi.index >= dfi.index.max() - pd.Timedelta(weeks=1)]
    elif mode == "1 Mois":
        return dfi.loc[dfi.index >= dfi.index.max() - pd.Timedelta(days=30)]
    return dfi

# ----- Nouveaux modes d'affichage + options -----
display_mode = st.sidebar.radio(
    "Type de courbe",
    [
        "Équité (portefeuille)",
        "Prix (actif)",
        "Équité par stratégie",
        "Chandelier (OHLC)",
        "Drawdown",
        "Rendement cumulé (%)",
        "Sharpe roulant",
    ],
)
log_scale = st.sidebar.checkbox("Échelle log (prix/ohlc)", value=False)
win_sharpe = st.sidebar.slider("Fenêtre Sharpe roulant", 20, 500, 252, 10)

# ------------------ Filtres de risque ------------------
st.sidebar.subheader("Filtres de risque")
filters_infos = discover_filters(FILTER_DIR)
active_filters: list[tuple[object, dict]] = []
if not filters_infos:
    st.sidebar.caption("Aucun filtre dans /filter")
else:
    for name, info in filters_infos.items():
        default_active = name in st.session_state.get("__active_filters", [])
        _chk = st.sidebar.checkbox(f"Filtre — {name}", value=default_active, key=f"filter_active::{name}")
        if _chk and not default_active:
            st.session_state["__active_filters"].append(name)
        if not _chk and default_active:
            st.session_state["__active_filters"] = [n for n in st.session_state["__active_filters"] if n != name]
        if _chk:
            with st.sidebar.expander(f"Paramètres — {name}", expanded=False):
                params = st.session_state["__params"].get(f"[FILTER]{name}", {})
                new_params: dict = {}
                for p, s in info.params_schema.items():
                    t = s.get("type")
                    if t == "int":
                        new_params[p] = st.slider(
                            p, int(s.get("min", 0)), int(s.get("max", 100)),
                            int(params.get(p, s.get("default", 0))),
                            int(s.get("step", 1)), key=f"param::F::{name}::{p}"
                        )
                    elif t == "float":
                        new_params[p] = st.slider(
                            p, float(s.get("min", 0.0)), float(s.get("max", 1.0)),
                            float(params.get(p, s.get("default", 0.0))),
                            float(s.get("step", 0.1)), key=f"param::F::{name}::{p}"
                        )
                    else:
                        new_params[p] = st.text_input(
                            p, value=str(params.get(p, s.get("default", ""))),
                            key=f"param::F::{name}::{p}"
                        )
                st.session_state["__params"][f"[FILTER]{name}"] = new_params
                active_filters.append((info.ref, new_params))

# ------------------ Coûts de transaction ------------------
st.sidebar.subheader("Coûts de transaction")
fee_bps = st.sidebar.number_input("Commission (bps, 1 = 0,01%)", min_value=0.0, max_value=200.0, value=10.0, step=0.5)
spread_bps = st.sidebar.number_input("Spread (bps, one-way)", min_value=0.0, max_value=200.0, value=5.0, step=0.5)
slippage_bps = st.sidebar.number_input("Slippage (bps, one-way)", min_value=0.0, max_value=200.0, value=2.0, step=0.5)
fee_on_sell_only = st.sidebar.checkbox("Commission à la vente uniquement", value=False)

# ------------------ Header ------------------
st.title("🧪 StratWorkbench — Combiner, paramétrer, benchmarker")
colKPI1, colKPI2, colKPI3 = st.columns(3)
kpi_port, kpi_growth, kpi_stab = colKPI1.empty(), colKPI2.empty(), colKPI3.empty()
st.markdown("---")

# ------------------ Paramètres ------------------
tabs = st.tabs(active_names if active_names else ["Aucune stratégie active"])
active: list[tuple[object, dict]] = []
for i, name in enumerate(active_names):
    info = infos[name]
    with tabs[i]:
        st.subheader(f"Paramètres — {name}")
        params = st.session_state["__params"].get(name, {})
        new_params: dict = {}
        for p, s in info.params_schema.items():
            t = s.get("type")
            if t == "int":
                new_params[p] = st.slider(
                    p, int(s.get("min", 0)), int(s.get("max", 100)),
                    int(params.get(p, s.get("default", 0))),
                    int(s.get("step", 1)), key=f"param::{name}::{p}"
                )
            elif t == "float":
                new_params[p] = st.slider(
                    p, float(s.get("min", 0.0)), float(s.get("max", 1.0)),
                    float(params.get(p, s.get("default", 0.0))),
                    float(s.get("step", 0.1)), key=f"param::{name}::{p}"
                )
            else:
                new_params[p] = st.text_input(
                    p, value=str(params.get(p, s.get("default", ""))),
                    key=f"param::{name}::{p}"
                )
        st.session_state["__params"][name] = new_params
        active.append((info.ref, new_params))
st.markdown("---")

# ------------------ Backtest ------------------
CASH_START = 10_000.0

def _build_positions_and_individual_equities(dfi, active_list, filters, fee_bps, spread_bps, slippage_bps, fee_on_sell_only):
    """
    Retourne (pos_port, per_dict)
    pos_port : Series exposition de portefeuille agrégée (moyenne des positions)
    per_dict : {strat_name: equity_series} équité individuelle calculée avec cash_start fractionné
    """
    positions = []
    per = {}
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
    else:
        positions_df = pd.concat(positions, axis=1).fillna(0)
        pos_port = positions_df.mean(axis=1)  # exposition portefeuille partagée (moyenne simple)

        # équités individuelles pour affichage/debug (sur capital fractionné)
        w = 1.0 / max(1, len(positions))
        for i, (ref, params) in enumerate(active_list):
            idx_name = getattr(ref, "NAME", str(ref))
            try:
                eqi = equity_from_position(
                    dfi, positions[i],
                    cash_start=CASH_START * w,
                    fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps, fee_on_sell_only=fee_on_sell_only
                )
            except Exception:
                eqi = pd.Series(CASH_START * w, index=dfi.index)
            per[idx_name] = eqi

    return pos_port, per

def compute_portfolio(dfi, active_list, filters, mode, cash_start, fee_bps, spread_bps, slippage_bps, fee_on_sell_only):
    """
    Calcule l'équité portefeuille selon le mode choisi.
    Retourne (equity_series, per_dict)
    """
    if not active_list:
        return pd.Series(dtype=float), {}

    if mode.startswith("Capital partagé"):
        pos_port, per = _build_positions_and_individual_equities(
            dfi, active_list, filters, fee_bps, spread_bps, slippage_bps, fee_on_sell_only
        )
        equity = equity_from_position(
            dfi, pos_port,
            cash_start=cash_start,
            fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps, fee_on_sell_only=fee_on_sell_only
        )
        return equity, per
    else:
        equity, stats, per = backtest_portfolio(
            dfi, active_list, cash_start=cash_start, filters=filters,
            fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps, fee_on_sell_only=fee_on_sell_only
        )
        return equity, per

if dfs and active:
    label, df0 = list(dfs.items())[0]
    try:
        equity, per = compute_portfolio(
            df0, active, active_filters, port_mode, CASH_START,
            fee_bps, spread_bps, slippage_bps, fee_on_sell_only
        )
    except Exception:
        equity, per = pd.Series(dtype=float), {}
    gi = growth_index(equity) if not equity.empty else 0.0
    si = stability_index(equity) if not equity.empty else 0.0
    kpi_port.metric("Portefeuille", f"{equity.iloc[-1]:,.0f} €" if not equity.empty else "—")
    kpi_growth.metric("Indice de croissance", f"{gi:.0f}")
    kpi_stab.metric("Indice de stabilité", f"{si:.0f}")
    st.caption(f"Mode: {port_mode} • Base: {label} • Stratégies actives: {len(active)} • Capital total: {CASH_START:,.0f} €")
else:
    equity, per = pd.Series(dtype=float), {}

# ------------------ Invalidation automatique benchmark ------------------
current_config = {
    "seed": (locals().get("synth_seed") if data_mode == "Synthétique" else None),
    "n_points": (locals().get("n_points") if data_mode == "Synthétique" else None),
    "jitter": (locals().get("jitter_pct") if data_mode == "Synthétique" else None),
    "curve_kind": (locals().get("curve_kind") if data_mode == "Synthétique" else None),
    "strategies": active_names,
    "params": {k: st.session_state["__params"].get(k, {}) for k in active_names},
    "filters": {f"[FILTER]{getattr(ref,'NAME',str(ref))}": params for ref, params in active_filters},
    "fees": dict(fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps, fee_on_sell_only=fee_on_sell_only),
    "time_filter": time_filter,
    "port_mode": port_mode,
}
if current_config != st.session_state["last_config"]:
    st.session_state["benchmark_results"] = None
    st.session_state["last_config"] = current_config

# ------------------ Graph principal ------------------
if dfs:
    fig = go.Figure()

    if display_mode == "Équité (portefeuille)" and isinstance(equity, pd.Series) and not equity.empty:
        for sym, dfi in dfs.items():
            dfi_filtered = apply_time_filter(dfi, time_filter)
            if sym in selected_symbols:
                try:
                    eq, _ = compute_portfolio(
                        dfi_filtered, active, active_filters, port_mode, CASH_START,
                        fee_bps, spread_bps, slippage_bps, fee_on_sell_only
                    )
                except Exception:
                    eq = pd.Series(dtype=float)
                fig.add_trace(go.Scatter(x=dfi_filtered.index, y=eq, mode="lines", name=f"Équité {sym}"))

    elif display_mode == "Prix (actif)":
        for sym, dfi in dfs.items():
            dfi_filtered = apply_time_filter(dfi, time_filter)
            if sym in selected_symbols:
                fig.add_trace(go.Scatter(x=dfi_filtered.index, y=dfi_filtered["close"], mode="lines", name=f"{sym} (close)"))
                # Triangles entrées/sorties
                if triangle_strategy != "Aucune":
                    for ref, params in active:
                        if getattr(ref, "NAME", "") == triangle_strategy:
                            pos = ref.generate_signals(dfi_filtered, params)
                            for fref, fparams in active_filters:
                                pos = fref.apply(dfi_filtered, pos, fparams)
                            entries = pos[(pos.shift(1) == 0) & (pos == 1)].index
                            exits   = pos[(pos.shift(1) == 1) & (pos == 0)].index
                            fig.add_trace(go.Scatter(
                                x=entries, y=dfi_filtered.loc[entries, "close"],
                                mode="markers", marker_symbol="triangle-up",
                                marker_color="green", marker_size=10,
                                name=f"Achat — {getattr(ref,'NAME','')}"
                            ))
                            fig.add_trace(go.Scatter(
                                x=exits, y=dfi_filtered.loc[exits, "close"],
                                mode="markers", marker_symbol="triangle-down",
                                marker_color="red", marker_size=10,
                                name=f"Vente — {getattr(ref,'NAME','')}"
                            ))
                            nb_trades = int(((pos.diff().fillna(0) != 0).sum()) // 2)
                            fig.add_annotation(
                                xref="paper", yref="paper", x=0.01, y=0.95,
                                text=f"Trades : {nb_trades}",
                                showarrow=False, font=dict(size=12, color="white"),
                                align="left", bordercolor="gray", borderwidth=1,
                                bgcolor="black", opacity=0.7
                            )

    elif display_mode == "Chandelier (OHLC)":
        for sym, dfi in dfs.items():
            dfi_filtered = apply_time_filter(dfi, time_filter)
            if sym in selected_symbols and {"open","high","low","close"}.issubset(dfi_filtered.columns):
                fig.add_trace(go.Candlestick(
                    x=dfi_filtered.index,
                    open=dfi_filtered["open"], high=dfi_filtered["high"],
                    low=dfi_filtered["low"], close=dfi_filtered["close"],
                    name=f"{sym} OHLC"
                ))
                # Triangles entrées/sorties sur OHLC
                if triangle_strategy != "Aucune":
                    for ref, params in active:
                        if getattr(ref, "NAME", "") == triangle_strategy:
                            pos = ref.generate_signals(dfi_filtered, params)
                            for fref, fparams in active_filters:
                                pos = fref.apply(dfi_filtered, pos, fparams)
                            entries = pos[(pos.shift(1) == 0) & (pos == 1)].index
                            exits   = pos[(pos.shift(1) == 1) & (pos == 0)].index
                            fig.add_trace(go.Scatter(
                                x=entries, y=dfi_filtered.loc[entries, "low"],
                                mode="markers", marker_symbol="triangle-up",
                                marker_color="green", marker_size=10,
                                name=f"Achat — {getattr(ref,'NAME','')}"
                            ))
                            fig.add_trace(go.Scatter(
                                x=exits, y=dfi_filtered.loc[exits, "high"],
                                mode="markers", marker_symbol="triangle-down",
                                marker_color="red", marker_size=10,
                                name=f"Vente — {getattr(ref,'NAME','')}"
                            ))

    elif display_mode == "Drawdown" and isinstance(equity, pd.Series) and not equity.empty:
        for sym, dfi in dfs.items():
            dfi_filtered = apply_time_filter(dfi, time_filter)
            if sym in selected_symbols:
                try:
                    eq_sym, _ = compute_portfolio(
                        dfi_filtered, active, active_filters, port_mode, CASH_START,
                        fee_bps, spread_bps, slippage_bps, fee_on_sell_only
                    )
                except Exception:
                    eq_sym = pd.Series(dtype=float)
                dd = (eq_sym / eq_sym.cummax() - 1.0) * 100.0 if not eq_sym.empty else pd.Series(dtype=float)
                fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name=f"DD {sym} (%)"))

    elif display_mode == "Rendement cumulé (%)":
        for sym, dfi in dfs.items():
            dfi_filtered = apply_time_filter(dfi, time_filter)
            if sym in selected_symbols:
                base = float(dfi_filtered["close"].iloc[0])
                rc = (dfi_filtered["close"] / base - 1.0) * 100.0
                fig.add_trace(go.Scatter(x=rc.index, y=rc, mode="lines", name=f"{sym} (%)"))

    elif display_mode == "Sharpe roulant":
        for sym, dfi in dfs.items():
            dfi_filtered = apply_time_filter(dfi, time_filter)
            if sym in selected_symbols:
                rets = dfi_filtered["close"].pct_change().fillna(0.0)
                mu = rets.rolling(win_sharpe).mean()
                sd = rets.rolling(win_sharpe).std().replace(0, 1e-12)
                sh = (mu / sd) * np.sqrt(252.0)
                fig.add_trace(go.Scatter(x=sh.index, y=sh, mode="lines", name=f"Sharpe {sym} (w={win_sharpe})"))

    else:
        # Équité par stratégie (affichage)
        if isinstance(per, dict) and per:
            for name, eq in per.items():
                fig.add_trace(go.Scatter(x=list(dfs.values())[0].index, y=eq, mode="lines", name=f"Équité — {name}"))

    # Échelle log pour prix/ohlc si demandé
    if log_scale and display_mode in ["Prix (actif)", "Chandelier (OHLC)"]:
        fig.update_yaxes(type="log")

    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Détail portefeuille pour compréhension ------------------
if isinstance(equity, pd.Series) and not equity.empty:
    with st.expander("🧮 Détail portefeuille", expanded=False):
        if per:
            df_per = pd.DataFrame({k: v for k, v in per.items()})
            last = df_per.iloc[-1].rename("final")
            n = len(per)
            init_per = (CASH_START / n) if n > 0 else 0.0
            pnl = (last - init_per).rename("PnL")
            tab = pd.concat([last, pnl], axis=1)
            st.dataframe(tab.style.format({"final":"{:,.0f}", "PnL":"{:,.0f}"}))
            st.caption(f"Somme des finales: {last.sum():,.0f} €  | Valeur KPI Portefeuille: {equity.iloc[-1]:,.0f} €")
            if port_mode.startswith("Capital partagé"):
                st.caption("En mode 'partagé', la somme ci-dessus est un repère. La vérité du PnL est la courbe portefeuille unique.")
            else:
                st.caption("En mode 'divisé', Somme(individuelles) = Portefeuille.")

# ------------------ Benchmark multi-courbes ------------------
with st.expander("🚀 Benchmark multi-courbes", expanded=False):
    if st.button("🔄 Recalculer benchmark"):
        if not active:
            st.warning("Active au moins une stratégie pour lancer le benchmark.")
        elif not dfs:
            st.warning("Charge des données pour lancer le benchmark.")
        else:
            if data_mode == "Synthétique":
                seed = int(locals().get("synth_seed", 123))
                seed = np.random.randint(0, 1_000_000) if seed == 0 else seed
                n_points = int(locals().get("n_points", 1500))
                jitter_pct = float(locals().get("jitter_pct", 0.0))
                kinds = ["sideways","slow_grind","trend_down","trend_up","volatile_whipsaw"]
                bench_dfs = {k: make_with_jitter(k, n_points, seed, jitter_pct) for k in kinds}
            else:
                bench_dfs = dfs

            rows = []
            for label, dfi in bench_dfs.items():
                dfi_filtered = apply_time_filter(dfi, time_filter)
                for ref, params in active:
                    pos = ref.generate_signals(dfi_filtered, params)
                    for fref, fparams in active_filters:
                        pos = fref.apply(dfi_filtered, pos, fparams)
                    eqi = equity_from_position(
                        dfi_filtered, pos, cash_start=10_000.0,
                        fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps, fee_on_sell_only=fee_on_sell_only
                    )

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
            st.session_state["benchmark_results"] = pd.DataFrame(rows).set_index(["courbe", "strat"])

    # Recalcul automatique si nécessaire
    if st.session_state["benchmark_results"] is None and dfs and active:
        if data_mode == "Synthétique":
            seed = int(locals().get("synth_seed", 123))
            seed = np.random.randint(0, 1_000_000) if seed == 0 else seed
            n_points = int(locals().get("n_points", 1500))
            jitter_pct = float(locals().get("jitter_pct", 0.0))
            kinds = ["sideways","slow_grind","trend_down","trend_up","volatile_whipsaw"]
            bench_dfs = {k: make_with_jitter(k, n_points, seed, jitter_pct) for k in kinds}
        else:
            bench_dfs = dfs

        rows = []
        for label, dfi in bench_dfs.items():
            dfi_filtered = apply_time_filter(dfi, time_filter)
            for ref, params in active:
                pos = ref.generate_signals(dfi_filtered, params)
                for fref, fparams in active_filters:
                    pos = fref.apply(dfi_filtered, pos, fparams)
                eqi = equity_from_position(
                    dfi_filtered, pos, cash_start=10_000.0,
                    fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps, fee_on_sell_only=fee_on_sell_only
                )

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
        st.session_state["benchmark_results"] = pd.DataFrame(rows).set_index(["courbe", "strat"])

    if st.session_state["benchmark_results"] is not None:
        res = st.session_state["benchmark_results"]

        st.subheader("Équité finale par marché × stratégie")
        st.dataframe(res["final"].unstack().style.format("{:,.0f}").highlight_max(axis=1, color=HIGHLIGHT))

        with st.expander("📊 Indicateurs complémentaires"):
            col1, col2 = st.columns(2)
            col1.caption("Sharpe (↑ mieux)")
            col1.dataframe(res["sharpe"].unstack().style.format("{:.2f}").highlight_max(axis=1, color=HIGHLIGHT2))
            col2.caption("Drawdown min (↑ moins pire)")
            col2.dataframe(res["dd_min"].unstack().style.format("{:.1%}").highlight_max(axis=1, color=HIGHLIGHT2))

            col3, col4 = st.columns(2)
            col3.caption("Indice de croissance (0–100)")
            col3.dataframe(res["growth_idx"].unstack().style.format("{:.0f}").highlight_max(axis=1, color=HIGHLIGHT2))
            col4.caption("Indice de stabilité (0–100)")
            col4.dataframe(res["stability_idx"].unstack().style.format("{:.0f}").highlight_max(axis=1, color=HIGHLIGHT2))

        # ------------------ Classements supplémentaires ------------------
        best = (
            res["final"]
            .groupby(level=0).idxmax()
            .apply(lambda x: x[1])
            .rename("Meilleure stratégie (final)")
            .to_frame()
        )
        st.subheader("🏆 Meilleure stratégie par type de marché")
        st.dataframe(best)

        st.subheader("📊 Stratégies efficaces sur plusieurs types de marché")
        df_bench = res.reset_index()[["courbe", "strat", "final"]]

        seuil = df_bench["final"].median()
        pivot_abs = (df_bench
                     .assign(efficace=lambda d: d["final"] > seuil)
                     .groupby("strat")["efficace"].sum()
                     .sort_values(ascending=False)
                     .to_frame("nb_marches_efficaces"))
        st.markdown(f"**Critère absolu** (perf > médiane globale = {seuil:,.0f}€)")
        st.dataframe(pivot_abs)

        nb_strats = df_bench["strat"].nunique()
        topN = max(1, nb_strats // 3)

        def mark_topN(d, n=2):
            return d.sort_values("final", ascending=False).head(n).assign(topN=True)

        top_marks = (df_bench.groupby("courbe", group_keys=False).apply(mark_topN, n=topN))
        pivot_rel = (top_marks.groupby("strat")["topN"].sum()
                     .sort_values(ascending=False)
                     .to_frame(f"nb_top{topN}"))
        st.markdown(f"**Classement relatif** (nombre de fois dans le Top-{topN})")
        st.dataframe(pivot_rel)

        pivot_all = pivot_abs.join(pivot_rel, how="outer").fillna(0).astype(int)
        st.markdown("**Vue combinée (absolu + relatif)**")
        st.dataframe(pivot_all)

# ------------------ Optimisation auto ------------------
st.markdown("### 🤖 Auto-calcul des meilleurs paramètres")

colA, colB, colC = st.columns(3)
objective = colA.selectbox(
    "Critère",
    ["Sharpe", "Équité finale", "Drawdown min", "Indice croissance", "Indice stabilité", "Composite"],
    index=0,
)
opt_filters = colB.checkbox("Optimiser aussi les filtres actifs", value=False)
max_combos = int(colC.number_input("Budget combos max", min_value=50, max_value=100_000, value=2000, step=50))

w_final = w_sharpe = w_dd = 1.0
if objective == "Composite":
    w_final  = st.slider("Poids — Équité finale", 0.0, 3.0, 1.0, 0.1)
    w_sharpe = st.slider("Poids — Sharpe",        0.0, 3.0, 1.0, 0.1)
    w_dd     = st.slider("Poids — |Drawdown|",    0.0, 3.0, 1.0, 0.1)


if st.button("🔍 Auto-calcul best values"):
    if not dfs or not active_names:
        st.warning("Charge des données et coche au moins une stratégie.")
    else:
        best_params = {}
        df_ref = list(dfs.values())[0]

        # Barre de progression par stratégie
        total = len(active_names)
        progress = st.progress(0)
        status = st.empty()

        for i, name in enumerate(active_names, start=1):
            info = infos[name]
            status.text(f"Optimisation {i}/{total} — {name}")
            with st.spinner(f"Optimisation '{name}' en cours"):
                best = optimize_strategy(
                    df_ref, info,
                    cash_start=CASH_START,
                    max_combos=max_combos,
                    fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps,
                    fee_on_sell_only=fee_on_sell_only,
                    filters=active_filters,
                    optimize_filters=opt_filters,
                    objective=(
                        "sharpe" if objective == "Sharpe" else
                        "final"  if objective == "Équité finale" else
                        "dd"     if objective == "Drawdown min" else
                        "growth" if objective == "Indice croissance" else
                        "stability" if objective == "Indice stabilité" else
                        "composite"
                    ),
                    weights={"final": w_final, "sharpe": w_sharpe, "dd": w_dd} if objective == "Composite" else None,
                )

            # Mémorise
            best_params[name] = {k: v for k, v in best.items() if k in ("params","final","sharpe","dd","growth","stab")}
            if "params" in best:
                st.session_state["__params"][name] = best["params"]

            progress.progress(int(i * 100 / max(1, total)))

        status.text("Optimisation terminée")

        # Applique paramètres de filtres si optimisés
        if opt_filters and 'best' in locals() and "filters_params" in best:
            for (fref, _), fparams in zip(active_filters, best["filters_params"]):
                _fname = getattr(fref, "NAME", str(fref))
                st.session_state["__params"][f"[FILTER]{_fname}"] = fparams
                if _fname not in st.session_state.get("__active_filters", []):
                    st.session_state["__active_filters"].append(_fname)

        st.json(best_params)
        _root_filters = {f"[FILTER]{n}": st.session_state["__params"].get(f"[FILTER]{n}", {}) for n in st.session_state.get("__active_filters", [])}
        _export_payload = {"filters": _root_filters, **best_params}
        with open("best_params.json", "w", encoding="utf-8") as f:
            json.dump(_export_payload, f, indent=2, ensure_ascii=False)
        st.success("✅ Paramètres appliqués et sauvegardés (best_params.json)")
        
# --- Charger des params optimisés (JSON ou CSV) ---
uploaded_best = st.file_uploader("Charger best_params (JSON ou CSV) — appliquer aux stratégies et filtres", type=["json","csv"])
def _try_cast(v):
    # convertit string -> int/float/bool si possible
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("true","false"):
            return s.lower() == "true"
        try:
            if "." in s:
                return float(s)
            return int(s)
        except Exception:
            try:
                return float(s)
            except Exception:
                return s
    return v

if uploaded_best is not None:
    try:
        if uploaded_best.name.lower().endswith(".json"):
            data = json.load(uploaded_best)
            # attendu : { "StratName": { "params": {...}, "final":..., ... }, ... }
            for strat_name, payload in data.items():
                params = payload.get("params") if isinstance(payload, dict) else payload
                if isinstance(params, dict):
                    st.session_state["__params"][strat_name] = {k: _try_cast(v) for k, v in params.items()}
            # cas où les filtres sont au niveau racine ou sous chaque stratégie
            for k, v in (data.get("filters", {}) if isinstance(data, dict) else {}).items():
                st.session_state["__params"][k] = {kk: _try_cast(vv) for kk, vv in v.items()}
            st.success("✅ Paramètres JSON appliqués aux stratégies et filtres")

        else:  # CSV
            dfp = pd.read_csv(uploaded_best)
            cols = set(dfp.columns.str.lower())
            # format long : strat,param,value
            if {"strat","param","value"}.issubset(cols):
                dfp.columns = [c.lower() for c in dfp.columns]
                pivot = dfp.pivot(index="strat", columns="param", values="value")
                for strat in pivot.index:
                    row = pivot.loc[strat].to_dict()
                    st.session_state["__params"][strat] = {k: _try_cast(v) for k, v in row.items() if pd.notna(v)}
                st.success("✅ CSV long appliqué")
            else:
                # format large : colonne 0 = strat (ou index), autres colonnes = params
                if "strat" in cols:
                    dfp = dfp.set_index([c for c in dfp.columns if c.lower()=="strat"][0])
                else:
                    dfp = dfp.set_index(dfp.columns[0])
                for strat, row in dfp.iterrows():
                    params = {c: _try_cast(row[c]) for c in dfp.columns if pd.notna(row[c])}
                    st.session_state["__params"][strat] = params
                st.success("✅ CSV large appliqué")

    except Exception as e:
        st.error(f"Erreur lors du chargement des paramètres : {e}")

# ------------------ Footer ------------------
st.caption(
    """Astuce : utilisez le menu **Type de courbe** (à gauche) pour passer de l'**Équité**
au **Prix**, au **Chandelier (OHLC)**, au **Drawdown**, au **Rendement cumulé (%)** ou au **Sharpe roulant**.
Ajoutez vos propres stratégies dans `/strats` et vos filtres de risque dans `/filter`."""
)
