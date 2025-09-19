from __future__ import annotations
import numpy as np, pandas as pd

def _gbm(n, start=100.0, mu=0.08, sigma=0.2, dt=1/252, seed=None):
    rng = np.random.default_rng(seed)
    rets = rng.normal((mu-0.5*sigma*sigma)*dt, sigma*np.sqrt(dt), n)
    return start * np.exp(np.cumsum(rets))

def _to_ohlcv(close, seed=None):
    rng = np.random.default_rng(seed)
    close = np.asarray(close, float)
    n = close.size
    open_ = np.r_[close[0], close[:-1]]
    vol = np.maximum(1e-6, np.abs(pd.Series(close).pct_change().fillna(0.0).values))
    high = np.maximum(open_, close) * (1 + 0.002 + 0.01*vol)
    low  = np.minimum(open_, close) * (1 - 0.002 - 0.01*vol)
    volume = (1000 * (1 + 10*vol) * (1 + rng.random(n))).astype(float)
    idx = pd.RangeIndex(n, name="t")
    return pd.DataFrame({"open":open_, "high":high, "low":low, "close":close, "volume":volume}, index=idx)

def make_synth(n=1500, kind="trend_up", seed=42):
    if kind == "trend_up":
        c = _gbm(n, mu=0.25, sigma=0.25, seed=seed)
    elif kind == "trend_down":
        c = _gbm(n, mu=-0.25, sigma=0.25, seed=seed)
    elif kind == "sideways":
        c = _gbm(n, mu=0.0, sigma=0.12, seed=seed)
    elif kind == "volatile_whipsaw":
        c = _gbm(n, mu=0.0, sigma=0.5, seed=seed)
    elif kind == "slow_grind":
        c = _gbm(n, mu=0.08, sigma=0.08, seed=seed)
    else:
        c = _gbm(n, mu=0.0, sigma=0.2, seed=seed)
    return _to_ohlcv(c, seed=seed)

def load_multi_curves(n=1500, seed=123):
    kinds = ["trend_up","trend_down","sideways","volatile_whipsaw","slow_grind"]
    return {k: make_synth(n=n, kind=k, seed=seed+i) for i,k in enumerate(kinds)}

def load_csv(path: str) -> pd.DataFrame:
    """
    Charge un CSV de données OHLCV (comme Binance, Yahoo Finance, etc.).
    Doit contenir au minimum : time, open, high, low, close, volume.
    """
    df = pd.read_csv(path)
    # Normalise les colonnes (majuscules/minuscules)
    df.columns = [c.lower() for c in df.columns]

    required = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"Le CSV doit contenir {required}, colonnes trouvées = {list(df.columns)}")

    # Gestion de l’index temps
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.set_index("time")

    return df[required]

