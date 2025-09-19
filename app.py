# app.py
from __future__ import annotations
import pathlib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
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
if "benchmark_results" not in st.session_state:
    st.session_state["benchmark_results"] = None
if "last_config" not in st.session_state:
    st.session_state["last_config"] = {}

# Couleurs
HIGHLIGHT = "#a9955ba6"
HIGHLIGHT2 = "#b6863f"

# ------------------ Sidebar ------------------
st.sidebar.title("⚙️ Paramètres")
st.sidebar.caption("Auto-détection des stratégies dans /strats")

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

    seed = np.random.randint(0, 1_000_000) if synth_seed == 0 else synth_seed
    df_base = make_synth(n=n_points, kind=curve_kind, seed=seed)

    if jitter_pct > 0:
        noise = pd.Series(np.random.normal(0, jitter_pct / 1000, size=len(df_base)))
        noise = noise.rolling(window=10, min_periods=1).mean()
        df_base["close"] *= (1 + noise)
        df_base["open"] = df_base["close"].shift(1).fillna(df_base["close"])
        df_base["high"] = df_base[["open", "close"]].max(axis=1) * (1 + abs(noise) / 2)
        df_base["low"] = df_base[["open", "close"]].min(axis=1) * (1 - abs(noise) / 2)

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

display_mode = st.sidebar.radio(
    "Type de courbe",
    ["Équité (portefeuille)", "Prix (actif)", "Équité par stratégie"],
)

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
if dfs and active:
    label, df0 = list(dfs.items())[0]
    equity, stats, per = backtest_portfolio(df0, active, cash_start=10_000.0)
    gi, si = growth_index(equity), stability_index(equity)
    kpi_port.metric("Portefeuille", f"{equity.iloc[-1]:,.0f} €")
    kpi_growth.metric("Indice de croissance", f"{gi:.0f}")
    kpi_stab.metric("Indice de stabilité", f"{si:.0f}")
else:
    equity, per = pd.Series(dtype=float), {}

# ------------------ Invalidation automatique benchmark ------------------
current_config = {
    "seed": synth_seed if data_mode == "Synthétique" else None,
    "jitter": jitter_pct if data_mode == "Synthétique" else None,
    "strategies": active_names,
    "params": {k: st.session_state["__params"].get(k, {}) for k in active_names}
}
if current_config != st.session_state["last_config"]:
    st.session_state["benchmark_results"] = None
    st.session_state["last_config"] = current_config

# ------------------ Graph principal ------------------
if dfs:
    fig = go.Figure()
    if display_mode == "Équité (portefeuille)" and not equity.empty:
        for sym, dfi in dfs.items():
            dfi_filtered = apply_time_filter(dfi, time_filter)
            if sym in selected_symbols:
                eq, _, _ = backtest_portfolio(dfi_filtered, active, cash_start=10_000.0)
                fig.add_trace(go.Scatter(x=dfi_filtered.index, y=eq, mode="lines", name=f"Équité {sym}"))
    elif display_mode == "Prix (actif)":
        for sym, dfi in dfs.items():
            dfi_filtered = apply_time_filter(dfi, time_filter)
            if sym in selected_symbols:
                fig.add_trace(go.Scatter(x=dfi_filtered.index, y=dfi_filtered["close"], mode="lines", name=f"{sym} (close)"))

                if triangle_strategy != "Aucune":
                    for ref, params in active:
                        if ref.NAME == triangle_strategy:
                            pos = ref.generate_signals(dfi_filtered, params)
                            entries = pos[(pos.shift(1) == 0) & (pos == 1)].index
                            exits   = pos[(pos.shift(1) == 1) & (pos == 0)].index

                            fig.add_trace(go.Scatter(
                                x=entries, y=dfi_filtered.loc[entries, "close"],
                                mode="markers", marker_symbol="triangle-up",
                                marker_color="green", marker_size=10,
                                name=f"Achat — {ref.NAME}"
                            ))
                            fig.add_trace(go.Scatter(
                                x=exits, y=dfi_filtered.loc[exits, "close"],
                                mode="markers", marker_symbol="triangle-down",
                                marker_color="red", marker_size=10,
                                name=f"Vente — {ref.NAME}"
                            ))

                            nb_trades = int(((pos.diff().fillna(0) != 0).sum()) // 2)
                            fig.add_annotation(
                                xref="paper", yref="paper",
                                x=0.01, y=0.95,
                                text=f"Trades : {nb_trades}",
                                showarrow=False,
                                font=dict(size=12, color="white"),
                                align="left",
                                bordercolor="gray",
                                borderwidth=1,
                                bgcolor="black",
                                opacity=0.7
                            )
    else:
        if per:
            for name, eq in per.items():
                fig.add_trace(go.Scatter(x=df0.index, y=eq, mode="lines", name=f"Équité — {name}"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Benchmark multi-courbes ------------------
with st.expander("🚀 Benchmark multi-courbes", expanded=False):
    if st.button("🔄 Recalculer benchmark"):
        if not active:
            st.warning("Active au moins une stratégie pour lancer le benchmark.")
        elif not dfs:
            st.warning("Charge des données pour lancer le benchmark.")
        else:
            if data_mode == "Synthétique":
                seed = np.random.randint(0, 1_000_000) if synth_seed == 0 else synth_seed
                bench_dfs = load_multi_curves(n=1500, seed=seed)
            else:
                bench_dfs = dfs

            rows = []
            for label, dfi in bench_dfs.items():
                for ref, params in active:
                    pos = ref.generate_signals(dfi, params)
                    eqi = equity_from_position(dfi, pos, cash_start=10_000.0)

                    changes = (pos.diff().fillna(0) != 0).sum()
                    nb_trades = int(changes // 2)
                    if isinstance(dfi.index, pd.DatetimeIndex):
                        n_days = max(1, (dfi.index[-1] - dfi.index[0]).days)
                    else:
                        n_days = max(1, len(dfi) // 100)
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

        with st.expander("📊 Activité de trading"):
            col1, col2 = st.columns(2)
            col1.caption("Nombre total de trades (round-trips)")
            col1.dataframe(res["trades"].unstack().style.format("{:,.0f}").highlight_max(axis=1, color=HIGHLIGHT2))
            col2.caption("Nombre moyen de trades par jour")
            col2.dataframe(res["trades_per_day"].unstack().style.format("{:.2f}").highlight_max(axis=1, color=HIGHLIGHT2))

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
