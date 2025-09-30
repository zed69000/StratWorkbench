# app.py
from __future__ import annotations
import pathlib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from core.data import load_csv, make_with_jitter
from core.io import load_params_from_file, save_params_json
from core.engine import discover_strategies, discover_filters
from core.portfolio import compute_portfolio, apply_time_filter
from core.metrics import growth_index, stability_index
from core.benchmark import run_benchmark, rankings
from core.signals import entries_exits
from core.optimizer import optimize_strategy

from core.state import (
    init_session,
    assemble_config,
    invalidate_if_changed,
    get_benchmark,
    set_benchmark,
    set_params_for_strat,
    set_params_bulk,
    set_filter_params,
    get_params_all,
    get_active_filter_names,
    add_active_filter,
)

# ------------------ Config Streamlit ------------------
st.set_page_config(page_title="StratWorkbench", layout="wide", page_icon="🧪")

STRATS_DIR = str(pathlib.Path(__file__).resolve().parent / "strats")
FILTER_DIR = str(pathlib.Path(__file__).resolve().parent / "filter")

init_session()

# Couleurs
HIGHLIGHT = "#a9955ba6"
HIGHLIGHT2 = "#b6863f"


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

cfg = assemble_config(
    data_mode=data_mode,
    synth_seed=locals().get("synth_seed"),
    n_points=locals().get("n_points"),
    jitter_pct=locals().get("jitter_pct"),
    curve_kind=locals().get("curve_kind"),
    active_names=active_names,
    active_filters=active_filters,
    time_filter=time_filter,
    port_mode=port_mode,
    fees=dict(
        fee_bps=fee_bps,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        fee_on_sell_only=fee_on_sell_only,
    ),
)
invalidate_if_changed(cfg)

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
                            entries, exits, nb_trades = entries_exits(pos)
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
                            entries, exits, _nb_trades = entries_exits(pos)
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

            res = run_benchmark(
                bench_dfs, active, active_filters,
                cash_start=10_000.0,
                fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps,
                fee_on_sell_only=fee_on_sell_only, time_filter=time_filter
            )
            set_benchmark(res)

    # Recalcul automatique si nécessaire
    if get_benchmark() is None and dfs and active:
        if data_mode == "Synthétique":
            seed = int(locals().get("synth_seed", 123))
            seed = np.random.randint(0, 1_000_000) if seed == 0 else seed
            n_points = int(locals().get("n_points", 1500))
            jitter_pct = float(locals().get("jitter_pct", 0.0))
            kinds = ["sideways","slow_grind","trend_down","trend_up","volatile_whipsaw"]
            bench_dfs = {k: make_with_jitter(k, n_points, seed, jitter_pct) for k in kinds}
        else:
            bench_dfs = dfs

        res = run_benchmark(
            bench_dfs, active, active_filters,
            cash_start=10_000.0,
            fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps,
            fee_on_sell_only=fee_on_sell_only, time_filter=time_filter
        )
        set_benchmark(res)

    res = get_benchmark()
    if res is not None:
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
        rk = rankings(res)
        best = rk["best"]
        st.subheader("🏆 Meilleure stratégie par type de marché")
        st.dataframe(best)

        st.subheader("📊 Stratégies efficaces sur plusieurs types de marché")
        st.markdown(f"**Critère absolu** (perf > médiane globale = {rk['seuil']:,.0f}€)")
        st.dataframe(rk["pivot_abs"])

        st.markdown(f"**Classement relatif** (nombre de fois dans le Top-{rk['topN']})")
        st.dataframe(rk["pivot_rel"])

        pivot_all = rk["pivot_all"]
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
                set_params_for_strat(name, best["params"])

            progress.progress(int(i * 100 / max(1, total)))

        status.text("Optimisation terminée")

        # Applique paramètres de filtres si optimisés
        if opt_filters and 'best' in locals() and "filters_params" in best:
            for (fref, _), fparams in zip(active_filters, best["filters_params"]):
                _fname = getattr(fref, "NAME", str(fref))
                set_filter_params(f"[FILTER]{_fname}", fparams or {})
                add_active_filter(_fname)

        st.json(best_params)
        _root_filters = {f"[FILTER]{n}": get_params_all().get(f"[FILTER]{n}", {}) for n in get_active_filter_names()}
        _export_payload = {"filters": _root_filters, **best_params}
        save_params_json("best_params.json", _export_payload)
        st.success("✅ Paramètres appliqués et sauvegardés (best_params.json)")
        
# --- Charger des params optimisés (JSON ou CSV) ---
uploaded_best = st.file_uploader("Charger best_params (JSON ou CSV) — appliquer aux stratégies et filtres", type=["json","csv"])


if uploaded_best is not None:
    try:
        by_strat, filters = load_params_from_file(uploaded_best, uploaded_best.name)
        if by_strat:
            set_params_bulk(by_strat)
        for k, v in (filters or {}).items():
            set_filter_params(k, v)
        st.success("✅ Paramètres appliqués aux stratégies et filtres")

    except Exception as e:
        st.error(f"Erreur lors du chargement des paramètres : {e}")
        
# ------------------ Sauvegarde manuelle des paramètres ------------------
if st.button("💾 Sauvegarder stratégies + filtres"):
    _all = get_params_all()
    export_payload = {
        "strategies": active_names,
        "params": {k: _all.get(k, {}) for k in active_names},
        "filters": {k: v for k, v in _all.items() if k.startswith("[FILTER]")},
        "fees": dict(fee_bps=fee_bps, spread_bps=spread_bps, slippage_bps=slippage_bps, fee_on_sell_only=fee_on_sell_only),
        "port_mode": port_mode,
        "time_filter": time_filter,
    }
    save_params_json("saved_params.json", export_payload)
    st.success("✅ Stratégies et filtres sauvegardés dans saved_params.json")


# ------------------ Footer ------------------
st.caption(
    """Astuce : utilisez le menu **Type de courbe** (à gauche) pour passer de l'**Équité**
au **Prix**, au **Chandelier (OHLC)**, au **Drawdown**, au **Rendement cumulé (%)** ou au **Sharpe roulant**.
Ajoutez vos propres stratégies dans `/strats` et vos filtres de risque dans `/filter`."""
)
