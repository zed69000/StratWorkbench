# metrics.py
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

# ---------- Contraintes & frais (style Binance) ----------

@dataclass
class ExchangeConstraints:
    tick_size: float = 0.0
    step_size: float = 0.0
    min_qty: float = 0.0
    min_notional: float = 0.0
    def round_price(self, p: float) -> float:
        if self.tick_size <= 0: return float(p)
        return math.floor(float(p) / self.tick_size) * self.tick_size
    def round_qty(self, q: float) -> float:
        if self.step_size <= 0: return float(q)
        return math.floor(max(0.0, float(q)) / self.step_size) * self.step_size
    def valid_qty(self, q: float) -> bool:
        return q >= self.min_qty
    def valid_notional(self, price: float, q: float) -> bool:
        return (price * q) >= self.min_notional

@dataclass
class Fees:
    maker_bps: float = 0.0
    taker_bps: float = 10.0
    taker_ratio: float = 1.0      # 1 = tout au marché
    spread_bps: float = 0.0       # demi-spread one-way
    slippage_bps: float = 0.0     # one-way
    min_delta: float = 0.0        # filtre micro-trades sur l'expo (en fraction)
    fee_on_sell_only: bool = False
    def eff_trade_rate(self) -> float:
        mr = float(np.clip(self.taker_ratio, 0.0, 1.0))
        return (mr * self.taker_bps + (1.0 - mr) * self.maker_bps) / 10_000.0
    def eff_frictions_rate(self) -> float:
        return (self.spread_bps + self.slippage_bps) / 10_000.0

# ---------- Broker backtest simple mais réaliste (spot long-only) ----------

class SpotBroker:
    """Exécution spot réaliste: contraintes + frais. Pas de short."""
    def __init__(self, c: ExchangeConstraints, f: Fees, cash_start: float):
        self.c = c
        self.f = f
        self.cash = float(cash_start)   # quote (ex: USDT)
        self.qty  = 0.0                 # base  (ex: BTC)
        self.trades: list[dict] = []
        self._entry_price: float | None = None  # prix d'entrée courant pour SL

    def _prepare(self, side: str, price: float, qty_req: float):
        assert side in ("BUY", "SELL")
        px = self.c.round_price(float(price))
        q  = self.c.round_qty(float(qty_req))
        if side == "SELL":
            q = min(q, self.qty)  # pas de short
        if not self.c.valid_qty(q): return None
        if not self.c.valid_notional(px, q): return None
        if side == "BUY":
            # besoin en cash = notional * (1 + coûts one-way)
            rate = self.f.eff_trade_rate() + self.f.eff_frictions_rate()
            need = (px * q) * (1.0 + rate)
            if need > self.cash:
                q_max = self.c.round_qty(self.cash / ((1.0 + rate) * px))
                if not self.c.valid_qty(q_max) or not self.c.valid_notional(px, q_max):
                    return None
                q = q_max
        return px, q

    def market(self, side: str, price: float, qty_req: float, ts=None):
        pre = self._prepare(side, price, qty_req)
        if pre is None:
            return 0.0, 0.0
        px, q = pre
        notional = px * q
        # coût total one-way = commission + frictions
        fee_rate = self.f.eff_trade_rate()
        fric_rate = self.f.eff_frictions_rate()
        # option: commission sur ventes seulement
        if self.f.fee_on_sell_only:
            fee = (notional if side == "SELL" else 0.0) * fee_rate
        else:
            fee = notional * fee_rate
        fric = notional * fric_rate
        cost = fee + fric

        if side == "BUY":
            self.cash -= (notional + cost)
            self.qty  += q
            # set/average entry pour le stop loss discret
            self._entry_price = px if self.qty == q or self._entry_price is None \
                                else ((self._entry_price * (self.qty - q)) + notional) / self.qty
        else:
            self.cash += (notional - cost)
            self.qty  -= q
            if self.qty <= 0.0:
                self._entry_price = None

        self.trades.append({
            "ts": ts, "side": side, "price": px, "qty": q,
            "notional": notional, "fee": fee, "frictions": fric, "total_cost": cost,
            "cash_after": self.cash, "qty_after": self.qty
        })
        return q, px

    def equity(self, price: float) -> float:
        return self.cash + self.qty * float(price)

    def maybe_stop_out(self, price: float, stop_loss_pct: float | None, ts=None):
        if stop_loss_pct is None or stop_loss_pct <= 0: return
        if self.qty <= 0.0 or self._entry_price is None: return
        pnl = price / self._entry_price - 1.0
        if pnl <= -float(stop_loss_pct):
            # force vente de tout
            self.market("SELL", price, self.qty, ts=ts)

# ---------- Equity réaliste basé sur une cible d'exposition ----------

def equity_from_position(
    df: pd.DataFrame,
    position: pd.Series,                 # cible d'expo ∈ [0,1] (spot long-only)
    cash_start: float = 10_000.0,
    constraints: ExchangeConstraints | None = None,
    fees: Fees | None = None,
    stop_loss_pct: float | None = None,
    return_trades: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """
    Rebalance vers la cible d'exposition avec contraintes d'échange et coûts réels.
    """
    c = constraints or ExchangeConstraints()
    f = fees or Fees()
    price = df["close"].astype(float)
    sig = position.reindex(price.index).fillna(0.0).clip(0.0, 1.0).astype(float)

    broker = SpotBroker(c, f, cash_start)
    equity = []

    for t, px in zip(price.index, price.values):
        # stop loss discret à la clôture
        broker.maybe_stop_out(px, stop_loss_pct, ts=t)

        eq_before = broker.equity(px)
        # qty cible = expo_cible * equity / prix
        qty_target = (sig.loc[t] * eq_before) / px
        delta = qty_target - broker.qty

        # filtre micro-trades: par notional ET par delta absolu
        notional_delta = abs(delta) * px
        if notional_delta < max(c.min_notional, 0.0) * 0.5 and abs(delta) < max(f.min_delta, 0.0):
            equity.append(broker.equity(px))
            continue

        side = "BUY" if delta > 0 else "SELL"
        broker.market(side, px, abs(delta), ts=t)
        equity.append(broker.equity(px))

    eq = pd.Series(equity, index=price.index, name="equity")
    return (eq, pd.DataFrame(broker.trades)) if return_trades else eq

# ---------- Anciennes métriques, inchangées ----------

def max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    s = equity.astype(float)
    roll_max = s.cummax()
    dd = (s / roll_max - 1.0).fillna(0.0)
    return float(dd.min())

def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    if equity is None or len(equity) < 2:
        return 0.0
    rets = equity.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    vol = rets.std()
    if vol == 0:
        return 0.0
    return float(np.sqrt(252.0) * (rets.mean() - rf) / (vol + 1e-12))

def growth_index(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    return float(np.clip(50.0 + 50.0 * np.tanh(total), 0.0, 100.0))

def stability_index(equity: pd.Series) -> float:
    s = sharpe(equity)
    dd = max_drawdown(equity)
    return float(np.clip(50.0 + 30.0 * np.tanh(s / 2.0) + 20.0 * (1.0 + dd), 0.0, 100.0))
