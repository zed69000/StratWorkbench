import pandas as pd
import plotly.graph_objects as go

def add_trade_markers(fig: go.Figure, df: pd.DataFrame, position: pd.Series,
                      name_buy: str = "Achat", name_sell: str = "Vente",
                      pad: float = 0.005, eps: float = 1e-6) -> go.Figure:
    pos = position.reindex(df.index).fillna(0.0).clip(-1, 1).astype(float)
    delta = pos.diff().fillna(pos.iloc[0])

    def _jitter(idxs, micro):
        return [pd.to_datetime(t) + pd.Timedelta(microseconds=micro) for t in idxs]

    buy_idx  = delta[delta >  eps].index
    sell_idx = delta[delta < -eps].index

    fig.add_trace(go.Scatter(
        x=_jitter(buy_idx, 100),
        y=(df.loc[buy_idx, "low"]  * (1 - pad)).astype(float),
        mode="markers", name=name_buy,
        marker=dict(symbol="triangle-up", size=10, line=dict(width=1)),
        cliponaxis=False, hovertemplate="Achat %{x|%Y-%m-%d %H:%M}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=_jitter(sell_idx, 200),
        y=(df.loc[sell_idx, "high"] * (1 + pad)).astype(float),
        mode="markers", name=name_sell,
        marker=dict(symbol="triangle-down", size=10, line=dict(width=1)),
        cliponaxis=False, hovertemplate="Vente %{x|%Y-%m-%d %H:%M}<extra></extra>"
    ))
    return fig
