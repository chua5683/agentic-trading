"""Leakage-free, bar-by-bar backtest replay.

Design rules, in priority order:

1. **No lookahead, structurally.**  The feature frame is computed exactly once
   for the whole series with the causal implementation in
   :mod:`agentic_trading.data.indicators`.  At bar ``i`` the strategy receives
   ``MarketContext(asof=bars[i].ts, bars=bars[: i + 1],
   features=latest_features(frame, i))`` -- a growing prefix whose newest bar is
   the decision bar.  There is no code path that hands the strategy a bar,
   feature or price from index ``> i``.  ``tests/test_backtest.py`` proves this
   with a recording strategy at every decision step.

2. **Explicit, boring execution.**  A decision made at bar ``i`` is filled at
   that bar's close, moved against us by ``slippage_bps``, with a
   ``fee_bps``-of-notional fee charged to cash.  Buys that cannot be funded are
   scaled down to available cash and the fill reason says so.  Positions are
   marked to market at every bar close, including during warm-up (flat cash),
   and the buy-and-hold benchmark pays the same costs.

3. **The strategy is a black box.**  The engine knows only
   :class:`agentic_trading.contracts.DecisionStrategy`, never mutates it, and
   executes ``decision.target_weight`` (the post-risk weight) as a target
   fraction of *pre-trade* equity.  It never interprets ``action`` or
   ``risk.vetoes``: the pipeline has already composed those into
   ``target_weight``.

Conventions worth knowing when reading a result:

* ``equity_curve`` and ``benchmark_curve`` have one point per input bar, in
  order.  During warm-up the book is flat, so those points equal
  ``initial_cash``.  The curve is marked *after* the bar's decision is filled,
  which means the first fill's costs are visible in the equity path.
* A trade is recorded only when a fill happens; ``|delta notional|`` below
  ``0.5%`` of equity is treated as HOLD to avoid churning on tiny weight
  changes.  One ``PortfolioDecision`` is recorded for every decided bar,
  whether or not it traded.
* ``started_at`` / ``finished_at`` are wall-clock UTC timestamps of the run,
  not market times; the traded period comes from the curves.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Sequence

from ..contracts import (
    BacktestConfig,
    BacktestResult,
    Bar,
    DecisionStrategy,
    MarketContext,
    PortfolioDecision,
    PortfolioState,
    Position,
    Trade,
)
from ..data.indicators import feature_frame, latest_features
from .metrics import compute_metrics

__all__ = ["BacktestEngine", "SUPPORTED_BENCHMARKS", "REBALANCE_EPSILON"]

#: Treat |delta notional| below this fraction of equity as HOLD (0.5%).
REBALANCE_EPSILON = 0.005

#: Benchmark modes accepted by :meth:`BacktestEngine.run`.
SUPPORTED_BENCHMARKS = ("buy_and_hold", "cash")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _normalise_bars(bars: Sequence[Bar]) -> list[Bar]:
    """Validate, sort ascending by ts and dedupe (keeping the last duplicate).

    The engine dedupes itself *before* building the feature frame so that row
    ``i`` of the frame always corresponds to ``bars[i]``.  A silently misaligned
    frame would be a lookahead bug hiding as a data-cleaning detail.
    """
    items = list(bars)
    for index, bar in enumerate(items):
        if not isinstance(bar, Bar):
            raise TypeError(f"bars[{index}] is {type(bar).__name__}, expected contracts.Bar")
        if not (math.isfinite(bar.close) and bar.close > 0):
            raise ValueError(
                f"bars[{index}] has non-finite or non-positive close {bar.close!r}; "
                "clean the data before backtesting"
            )
    if len(items) < 2:
        return items
    ordered = sorted(items, key=lambda bar: bar.ts)
    deduped: list[Bar] = []
    for bar in ordered:
        if deduped and deduped[-1].ts == bar.ts:
            deduped[-1] = bar
        else:
            deduped.append(bar)
    return deduped


def _update_avg_price(
    side: str,
    avg_price: float,
    old_qty: float,
    new_qty: float,
    filled_qty: float,
    fill_price: float,
) -> float:
    """Position cost basis after a fill (long supported fully; short best-effort)."""
    if abs(new_qty) <= 1e-12:
        return 0.0
    if abs(old_qty) <= 1e-12:
        return fill_price
    if (old_qty > 0) != (new_qty > 0):      # flipped through zero
        return fill_price
    adding = (side == "buy" and new_qty > old_qty) or (side == "sell" and new_qty < old_qty)
    if not adding:                           # reducing an existing position
        return avg_price
    return (avg_price * abs(old_qty) + filled_qty * fill_price) / abs(new_qty)


def _benchmark_mode(benchmark: str) -> str:
    name = (benchmark or "buy_and_hold").strip().lower().replace("-", "_")
    aliases = {"buy_and_hold": "buy_and_hold", "buyhold": "buy_and_hold",
               "bh": "buy_and_hold", "cash": "cash", "flat": "cash",
               "none": "cash", "no_benchmark": "cash"}
    if name not in aliases:
        raise ValueError(
            f"unknown benchmark {benchmark!r}; supported: {', '.join(SUPPORTED_BENCHMARKS)}"
        )
    return aliases[name]


def _buy_and_hold_curve(
    bars: Sequence[Bar],
    warmup_bars: int,
    initial_cash: float,
    fee_rate: float,
    slippage_rate: float,
) -> list[tuple[datetime, float]]:
    """Buy once at the first post-warmup close, paying costs, then hold."""
    curve: list[tuple[datetime, float]] = []
    cash = float(initial_cash)
    qty = 0.0
    bought = False
    for i, bar in enumerate(bars):
        if i < warmup_bars:
            curve.append((bar.ts, cash))
            continue
        if not bought:
            price = bar.close * (1.0 + slippage_rate)
            if price > 0 and cash > 0:
                notional = cash / (1.0 + fee_rate)
                qty = notional / price
                cash -= notional * (1.0 + fee_rate)
                if abs(cash) < 1e-9:
                    cash = 0.0
                bought = True
        curve.append((bar.ts, cash + qty * bar.close))
    return curve


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class BacktestEngine:
    """Replay bars through any :class:`~agentic_trading.contracts.DecisionStrategy`."""

    #: Overridable class attribute so callers can tune the HOLD dead-band.
    rebalance_epsilon = REBALANCE_EPSILON

    def __init__(self, cfg: BacktestConfig):
        if not isinstance(cfg, BacktestConfig):
            raise TypeError(f"cfg must be contracts.BacktestConfig, got {type(cfg).__name__}")
        self.cfg = cfg

    # -- public API --------------------------------------------------------- #
    def run(
        self,
        bars: Sequence[Bar],
        strategy: DecisionStrategy,
        symbol: str,
        timeframe: str = "1d",
        periods_per_year: float = 365.0,
        risk_free_rate: float = 0.04,
        benchmark: str = "buy_and_hold",
    ) -> BacktestResult:
        """Run the replay and return a fully populated :class:`BacktestResult`.

        Raises ``ValueError`` for an unsupported ``trade_at``/``benchmark`` or
        malformed bars; strategy exceptions propagate unchanged.
        """
        cfg = self.cfg
        if cfg.trade_at != "close":
            raise ValueError(
                f"unsupported trade_at={cfg.trade_at!r}; this engine fills at the decision bar's close"
            )
        if not symbol:
            raise ValueError("symbol must be a non-empty string")
        benchmark_mode = _benchmark_mode(benchmark)

        started_at = datetime.now(timezone.utc)
        norm = _normalise_bars(bars)
        fee_rate = cfg.fee_bps / 1e4
        slippage_rate = cfg.slippage_bps / 1e4

        # Computed ONCE for the whole series.  Every decision slices this frame;
        # nothing downstream ever sees a row > i.
        frame = feature_frame(norm, timeframe=timeframe) if norm else None
        if norm and (frame is None or len(frame) != len(norm)):
            # Row i of the frame must correspond to bars[i]; a length mismatch
            # would silently shift features across bars (a lookahead bug).  Fail
            # loudly instead of producing a backtest that cannot be trusted.
            raise ValueError(
                "feature frame length "
                f"({0 if frame is None else len(frame)}) does not match bar count "
                f"({len(norm)}); refusing to run a potentially misaligned backtest"
            )

        equity_curve: list[tuple[datetime, float]] = []
        trades: list[Trade] = []
        decisions: list[PortfolioDecision] = []

        cash = float(cfg.initial_cash)
        qty = 0.0
        avg_price = 0.0
        peak_equity = cash
        visible: list[Bar] = []          # grows to exactly bars[: i + 1]

        for i, bar in enumerate(norm):
            visible.append(bar)
            close = float(bar.close)

            # --- warm-up: no context is built, no decision is taken. -------- #
            if i < cfg.warmup_bars:
                equity_curve.append((bar.ts, cash))
                continue

            equity_before = cash + qty * close
            peak_equity = max(peak_equity, equity_before)

            positions = (
                {symbol: Position(symbol=symbol, qty=qty, avg_price=avg_price)}
                if abs(qty) > 0
                else {}
            )
            portfolio = PortfolioState(
                cash=cash,
                positions=positions,
                equity=equity_before,
                peak_equity=peak_equity,
                bar_index=i,
                last_prices={symbol: close},
            )

            # THE leakage boundary: tuple(visible) == tuple(bars[: i + 1]),
            # features are row i of the causal frame.  Nothing newer exists here.
            ctx = MarketContext(
                symbol=symbol,
                asof=bar.ts,
                bars=tuple(visible),
                features=latest_features(frame, i),
                timeframe=timeframe,
                provider="backtest",
                meta={"bar_index": i},
            )

            decision = strategy.decide(ctx, portfolio)
            if not isinstance(decision, PortfolioDecision):
                raise TypeError(
                    f"strategy {getattr(strategy, 'name', strategy)!r} returned "
                    f"{type(decision).__name__}, expected contracts.PortfolioDecision"
                )
            decisions.append(decision)

            target_weight = self._target_weight(decision)
            cash, qty, avg_price, trade = self._fill(
                symbol=symbol,
                ts=bar.ts,
                close=close,
                cash=cash,
                qty=qty,
                avg_price=avg_price,
                equity=equity_before,
                target_weight=target_weight,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )
            if trade is not None:
                trades.append(trade)

            equity_after = cash + qty * close
            peak_equity = max(peak_equity, equity_after)
            equity_curve.append((bar.ts, equity_after))

        if benchmark_mode == "buy_and_hold":
            benchmark_curve = _buy_and_hold_curve(
                norm, cfg.warmup_bars, cfg.initial_cash, fee_rate, slippage_rate
            )
        else:  # "cash": the strategy is measured against holding no risk at all.
            benchmark_curve = [(bar.ts, float(cfg.initial_cash)) for bar in norm]

        metrics = compute_metrics(
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            trades=trades,
            decisions=decisions,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            initial_cash=cfg.initial_cash,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            trades=trades,
            decisions=decisions,
            metrics=metrics,
            config=cfg,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )

    # -- internals ---------------------------------------------------------- #
    def _target_weight(self, decision: PortfolioDecision) -> float:
        """Final target weight, clamped to what the config allows."""
        try:
            weight = float(decision.target_weight)
        except (TypeError, ValueError):
            weight = 0.0
        if not math.isfinite(weight):
            weight = 0.0
        if self.cfg.allow_short:
            return max(-1.0, min(1.0, weight))
        return max(0.0, min(1.0, weight))

    def _fill(
        self,
        *,
        symbol: str,
        ts: datetime,
        close: float,
        cash: float,
        qty: float,
        avg_price: float,
        equity: float,
        target_weight: float,
        fee_rate: float,
        slippage_rate: float,
    ) -> tuple[float, float, float, Trade | None]:
        """Apply the target weight at this bar's close; return the new book + fill.

        Returns ``(cash, qty, avg_price, trade_or_None)``.  ``equity`` is the
        pre-trade mark used to size the order.
        """
        epsilon = max(self.rebalance_epsilon * max(equity, 0.0), 1e-9)
        target_notional = target_weight * max(equity, 0.0)
        delta = target_notional - qty * close
        if not math.isfinite(delta) or abs(delta) <= epsilon:
            return cash, qty, avg_price, None   # HOLD: too small to be worth costs

        side = "buy" if delta > 0 else "sell"
        fill_price = close * (1.0 + slippage_rate) if side == "buy" else close * (1.0 - slippage_rate)
        if not math.isfinite(fill_price) or fill_price <= 0:
            return cash, qty, avg_price, None
        reason = f"rebalance to {target_weight:.2%}"

        if side == "buy":
            filled_qty = delta / fill_price
            notional = filled_qty * fill_price
            max_notional = cash / (1.0 + fee_rate) if cash > 0 else 0.0
            if notional > max_notional:
                # Not enough cash for the full target: scale down and say so.
                notional = max(max_notional, 0.0)
                filled_qty = notional / fill_price
                reason += " (scaled to available cash)"
            if filled_qty <= 1e-12:
                return cash, qty, avg_price, None
            fee = notional * fee_rate
            cash -= notional + fee
            if -1e-6 < cash < 0.0:
                cash = 0.0
            new_qty = qty + filled_qty
        else:
            filled_qty = -delta / fill_price
            if not self.cfg.allow_short:
                filled_qty = min(filled_qty, max(qty, 0.0))
            if filled_qty <= 1e-12:
                return cash, qty, avg_price, None
            notional = filled_qty * fill_price
            fee = notional * fee_rate
            cash += notional - fee
            new_qty = qty - filled_qty
            if -1e-9 < new_qty <= 0.0:
                new_qty = 0.0
        if not math.isfinite(new_qty):
            return cash, qty, avg_price, None

        avg_price = _update_avg_price(side, avg_price, qty, new_qty, filled_qty, fill_price)
        equity_after = cash + new_qty * close
        trade = Trade(
            asof=ts,
            symbol=symbol,
            side=side,
            qty=filled_qty,
            price=fill_price,
            fee=fee,
            equity_after=equity_after,
            target_weight=target_weight,
            reason=reason,
        )
        return cash, new_qty, avg_price, trade
