# app.py
from __future__ import annotations
import pathlib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json

from core.data import make_synth, load_multi_curves, load_csv
from core.engine import discover_strategies, backtest_portfolio
from core.metrics import (
    growth_index, stability_index, max_drawdown, sharpe, equity_from_position
)
from core.optimizer import optimize_strategy

# ------------------ Config Streamlit ------------------
st.set_page_config(page_title="StratWorkbench", layout="wide", page_icon="🧪")

STRATS_DIR = str(pathlib.Path(__file__).resolve().parent / "strats")

if "__params" not in st.session_state:
    st.session_state["__params"] = {}

# Couleurs de surlignage
HIGHLIGHT = "#a9955ba6"
HIGHLIGHT2 = "#b6863f"

# ------------------ Sidebar ------------------
st.sidebar.title("⚙️ Paramètres")
st.sidebar.caption("Auto-détection des stratégies dans /strats")

data_mode = st.sidebar.radio("Source des données", ["Synthétique", "Réelles (CSV)"])

dfs = {}  # dictionnaire {label: DataFrame}
if data_mode == "Synthétique":
    n_points = st.sidebar.slider("Taille série (points)", 500, 5000, 1500, 100)
    curve_kind = st.sidebar.selectbox(
        "Profil de marché (aperçu)",
        ["trend_up", "trend_down", "sideways", "volatile_whipsaw", "slow_grind"],
    )
    dfs = {curve_kind: make_synth(n=n_points, kind=curve_kind, seed=123)}
else:
    uploaded_files = st.sidebar.file_uploader(
        "Uploader un ou plusieurs fichiers CSV OHLCV",
        type=["csv"],
        accept_multiple_files=True
    )
    if uploaded_files:
        dfs = {f.name.replace(".csv", ""): load_csv(f) for f in uploaded_files}
        st.sidebar.success(f"✅ {len(uploaded_files)} fichiers chargés")
    else:
        st.sidebar.warning("⚠️ Charge au moins un CSV pour continuer")

# Sélecteur de symboles
symbols_available = list(dfs.keys())
if symbols_available:
    selected_symbols = st.sidebar.multiselect(
        "Choisir les symboles à afficher",
        options=symbols_available,
        default=[symbols_available[0]]
    )
else:
    selected_symbols = []

# ------------------ Détection stratégies ------------------
st.sidebar.subheader("Stratégies détectées")
infos = discover_strategies(STRATS_DIR)
active_names: list[str] = []
if not infos:
    st.sidebar.error("Aucune stratégie trouvée dans /strats.")
else:
    for name, info in infos.items():
        on = st.sidebar.checkbox(name, value=name in list(infos.keys())[:4])
        if on:
            active_names.append(name)

display_mode = st.sidebar.radio(
    "Type de courbe",
    ["Équité (portefeuille)", "Prix (actif)", "Équité par stratégie"],
)

# ------------------ Header / KPIs ------------------
st.title("🧪 StratWorkbench — Combiner, paramétrer, benchmarker")
colKPI1, colKPI2, colKPI3 = st.columns(3)
kpi_port, kpi_growth, kpi_stab = colKPI1.empty(), colKPI2.empty(), colKPI3.empty()
st.markdown("---")

# ------------------ Paramètres par stratégie ------------------
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

# ------------------ Backtest portefeuille courant ------------------
if dfs and active:
    label, df0 = list(dfs.items())[0]  # premier symbole
    equity, stats, per = backtest_portfolio(df0, active, cash_start=10_000.0)
    gi, si = growth_index(equity), stability_index(equity)
    kpi_port.metric("Portefeuille", f"{equity.iloc[-1]:,.0f} €")
    kpi_growth.metric("Indice de croissance", f"{gi:.0f}")
    kpi_stab.metric("Indice de stabilité", f"{si:.0f}")
else:
    equity, per = pd.Series(dtype=float), {}

# ------------------ Graph principal ------------------
if dfs:
    fig = go.Figure()
    if display_mode == "Équité (portefeuille)" and not equity.empty:
        for sym, dfi in dfs.items():
            if sym in selected_symbols:
                eq, _, _ = backtest_portfolio(dfi, active, cash_start=10_000.0)
                fig.add_trace(go.Scatter(x=dfi.index, y=eq, mode="lines", name=f"Équité {sym}"))
    elif display_mode == "Prix (actif)":
        for sym, dfi in dfs.items():
            if sym in selected_symbols:
                fig.add_trace(go.Scatter(x=dfi.index, y=dfi["close"], mode="lines", name=f"{sym} (close)"))
    else:  # Équité par stratégie
        if per:
            for name, eq in per.items():
                fig.add_trace(go.Scatter(x=df0.index, y=eq, mode="lines", name=f"Équité — {name}"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Benchmark multi-courbes ------------------
st.markdown("### 🚀 Benchmark multi-courbes")
if st.button("Lancer benchmark"):
    if not active:
        st.warning("Active au moins une stratégie pour lancer le benchmark.")
    elif not dfs:
        st.warning("Charge des données pour lancer le benchmark.")
    else:
        if data_mode == "Synthétique":
            bench_dfs = load_multi_curves(n=1500, seed=999)
        else:
            bench_dfs = dfs

        rows = []
        for label, dfi in bench_dfs.items():
            for ref, params in active:
                pos = ref.generate_signals(dfi, params)
                eqi = equity_from_position(dfi, pos, cash_start=10_000.0)

                # ---- Comptage trades ----
                nb_trades = int((pos.diff().fillna(0) != 0).sum())
                if isinstance(dfi.index, pd.DatetimeIndex):
                    n_days = max(1, (dfi.index[-1] - dfi.index[0]).days)
                else:
                    # approximation pour les données synthétiques
                    n_days = max(1, len(dfi) // 100)  # 100 points ≈ 1 jour
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
        res = pd.DataFrame(rows).set_index(["courbe", "strat"])

        st.subheader("Équité finale par marché × stratégie (meilleur surligné)")
        st.dataframe(
            res["final"].unstack().style.format("{:,.0f}").highlight_max(axis=1, color=HIGHLIGHT)
        )

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

        with st.expander("📊 Activité de trading"):
            col1, col2 = st.columns(2)
            col1.caption("Nombre total de trades")
            col1.dataframe(res["trades"].unstack().style.format("{:,.0f}").highlight_max(axis=1, color=HIGHLIGHT2))
            col2.caption("Nombre moyen de trades par jour")
            col2.dataframe(res["trades_per_day"].unstack().style.format("{:.2f}").highlight_max(axis=1, color=HIGHLIGHT2))

        best = (
            res["final"].groupby(level=0).idxmax().apply(lambda x: x[1]).rename("Meilleure stratégie").to_frame()
        )
        st.subheader("🏆 Meilleure stratégie par type de marché")
        st.dataframe(best)

        # Robustesse multi-marchés
        st.subheader("📊 Stratégies efficaces sur plusieurs types de marché")
        df_bench = res.reset_index()[["courbe", "strat", "final"]]
        seuil = df_bench["final"].median()
        pivot_abs = (df_bench.assign(efficace=lambda d: d["final"] > seuil)
                     .groupby("strat")["efficace"].sum()
                     .sort_values(ascending=False)
                     .to_frame("nb_marches_efficaces"))
        st.dataframe(pivot_abs)

        nb_strats = df_bench["strat"].nunique()
        topN = max(1, nb_strats // 3)
        def mark_topN(d, n=2): return d.sort_values("final", ascending=False).head(n).assign(topN=True)
        top_marks = df_bench.groupby("courbe", group_keys=False).apply(mark_topN, n=topN)
        pivot_rel = (top_marks.groupby("strat")["topN"].sum()
                     .sort_values(ascending=False)
                     .to_frame(f"nb_top{topN}"))
        st.dataframe(pivot_rel)

        pivot_all = pivot_abs.join(pivot_rel, how="outer").fillna(0).astype(int)
        st.dataframe(pivot_all)

# ------------------ Optimisation auto ------------------
st.markdown("### 🤖 Auto-calcul des meilleurs paramètres")
if st.button("🔍 Auto-calcul best values"):
    best_params = {}
    for name, info in infos.items():
        st.write(f"Optimisation {name}...")
        best = optimize_strategy(list(dfs.values())[0], info) if dfs else {}
        best_params[name] = best
        if "params" in best:
            st.session_state["__params"][name] = best["params"]

    st.json(best_params)
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    st.success("✅ Paramètres appliqués et sauvegardés (best_params.json)")

# ------------------ Footer ------------------
st.caption(
    """Astuce : utilisez le menu **Type de courbe** (à gauche) pour passer de l'**Équité**
au **Prix** ou à l'**Équité par stratégie**. Ajoutez vos propres stratégies dans `/strats`."""
)
