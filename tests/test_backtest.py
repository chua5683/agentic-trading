"""Tests for the leakage-free backtest engine and its reports.

The centrepiece is :func:`test_leakage_proof_recording_strategy`, which asserts
at *every single decision* that the strategy saw exactly ``bars[: i + 1]``, that
the newest bar it saw is the decision bar, that its features equal row ``i`` of
the causal feature frame, and that the tuple it was handed never grew later
(no aliasing of a mutable bar list).

Everything else pins the money math: fills at the decision bar's close with
slippage and fees, cash-constrained order scaling, target-weight rebalancing,
the buy-and-hold benchmark, degenerate inputs, metrics/report contracts and
10k-bar performance.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agentic_trading.backtest.engine import REBALANCE_EPSILON, BacktestEngine
from agentic_trading.backtest.report import (
    print_summary,
    render_html,
    render_markdown,
    save_report,
)
from agentic_trading.config import BacktestRunConfig
from agentic_trading.contracts import (
    METRIC_KEYS,
    REQUIRED_FEATURES,
    Action,
    BacktestConfig,
    Bar,
    PortfolioDecision,
    RiskDecision,
)
from agentic_trading.data.indicators import compute_features, feature_frame, latest_features

START = datetime(2024, 6, 1, tzinfo=timezone.utc)
FEE_RATE = 10.0 / 1e4
SLIPPAGE = 5.0 / 1e4


# --------------------------------------------------------------------------- #
# Fixtures / helpers (deliberately local: tests/conftest.py belongs to another
# teammate, so nothing here may depend on shared fixtures)
# --------------------------------------------------------------------------- #
def make_bars(closes: list[float], *, volume: float = 1_000.0, start: datetime = START) -> list[Bar]:
    bars: list[Bar] = []
    for index, close in enumerate(closes):
        close = float(close)
        bars.append(
            Bar(
                ts=start + timedelta(days=index),
                open=close,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=volume,
            )
        )
    return bars


def make_config(**overrides: object) -> BacktestConfig:
    values: dict[str, object] = {
        "initial_cash": 1_000.0,
        "fee_bps": 10.0,
        "slippage_bps": 5.0,
        "allow_short": False,
        "warmup_bars": 0,
    }
    values.update(overrides)
    return BacktestConfig(**values)  # type: ignore[arg-type]


def decision_for(
    ctx,
    target_weight: float,
    *,
    action: Action | None = None,
    risk: RiskDecision | None = None,
) -> PortfolioDecision:
    return PortfolioDecision(
        asof=ctx.asof,
        symbol=ctx.symbol,
        action=action or (Action.BUY if target_weight > 0 else Action.HOLD),
        target_weight=target_weight,
        confidence=0.9,
        risk=risk,
        mode="deterministic",
    )


class ConstantWeightStrategy:
    """Always wants the same fraction of equity; records decision count."""

    name = "constant_weight"

    def __init__(self, weight: float) -> None:
        self.weight = weight
        self.calls = 0
        self.asofs: list[datetime] = []

    def decide(self, ctx, portfolio) -> PortfolioDecision:
        self.calls += 1
        self.asofs.append(ctx.asof)
        return decision_for(ctx, self.weight)


class ScriptedWeightStrategy:
    """Consumes ``weights`` one decision at a time, then holds flat."""

    name = "scripted"

    def __init__(self, weights: list[float]) -> None:
        self.weights = list(weights)
        self.calls = 0

    def decide(self, ctx, portfolio) -> PortfolioDecision:
        weight = self.weights[self.calls] if self.calls < len(self.weights) else 0.0
        self.calls += 1
        return decision_for(ctx, weight)


class RecordingStrategy:
    """Records exactly what the engine showed it at every decision.

    The assertions live in the test, not here, so a failure points at the engine.
    """

    name = "recording"

    def __init__(self, weight: float = 0.25) -> None:
        self.weight = weight
        self.records: list[dict[str, object]] = []

    def decide(self, ctx, portfolio) -> PortfolioDecision:
        assert isinstance(ctx.bars, tuple), "MarketContext.bars must be an immutable tuple"
        assert len(ctx.bars) >= 1
        assert ctx.bars[-1].ts == ctx.asof, "newest visible bar must be the decision bar"
        assert all(bar.ts <= ctx.asof for bar in ctx.bars), "a future bar was visible"
        self.records.append(
            {
                "asof": ctx.asof,
                "n_bars": len(ctx.bars),
                "bars_obj": ctx.bars,
                "first_ts": ctx.bars[0].ts,
                "last_ts": ctx.bars[-1].ts,
                "bar_index": ctx.meta.get("bar_index"),
                "features": dict(ctx.features),
                "portfolio_price": portfolio.price(ctx.symbol),
                "portfolio_equity": portfolio.equity,
                "portfolio_bar_index": portfolio.bar_index,
            }
        )
        return decision_for(ctx, self.weight)


class FeatureKeysStrategy:
    """Ignores values (they are mostly NaN in a 3-bar history) but checks the schema."""

    name = "feature_keys"

    def decide(self, ctx, portfolio) -> PortfolioDecision:
        assert set(ctx.features) == set(REQUIRED_FEATURES)
        assert all(isinstance(value, float) for value in ctx.features.values())
        assert math.isnan(ctx.features["sma_200"])
        return decision_for(ctx, 0.0)


class VetoingStrategy:
    """Vetoes every third decision; target weight stays flat."""

    name = "vetoing"

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, ctx, portfolio) -> PortfolioDecision:
        vetoed = self.calls % 3 == 0
        self.calls += 1
        risk = RiskDecision(
            approved=not vetoed,
            target_weight=0.0,
            vetoes=("drawdown_stop",) if vetoed else (),
        )
        return decision_for(ctx, 0.0, risk=risk)


def expected_max_affordable_entry(cash: float, close: float) -> tuple[float, float, float]:
    """Hand-computed (qty, cash_after, fill_price) for buying as much as cash allows."""
    fill_price = close * (1.0 + SLIPPAGE)
    notional = cash / (1.0 + FEE_RATE)
    qty = notional / fill_price
    cash_after = cash - notional * (1.0 + FEE_RATE)
    if abs(cash_after) < 1e-9:
        cash_after = 0.0
    return qty, cash_after, fill_price


# --------------------------------------------------------------------------- #
# (a) Deterministic equity math with fees and slippage
# --------------------------------------------------------------------------- #
def test_equity_math_matches_hand_computation() -> None:
    closes = [100.0, 110.0, 121.0]
    cfg = make_config(initial_cash=1_000.0)
    result = BacktestEngine(cfg).run(make_bars(closes), ConstantWeightStrategy(1.0), "TEST")

    qty0, cash0, fill0 = expected_max_affordable_entry(1_000.0, 100.0)
    notional0 = qty0 * fill0
    equity0 = cash0 + qty0 * 100.0

    # Bar 0: full-cash buy, scaled down so fee fits; marked at the raw close.
    assert result.equity_curve[0][1] == pytest.approx(equity0)
    # Bars 1-2: weight is already 1.0, so no rebalance (no churn), just marking.
    assert result.equity_curve[1][1] == pytest.approx(qty0 * 110.0)
    assert result.equity_curve[2][1] == pytest.approx(qty0 * 121.0)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "buy"
    assert trade.qty == pytest.approx(qty0)
    assert trade.price == pytest.approx(fill0)
    assert trade.fee == pytest.approx(notional0 * FEE_RATE)
    assert trade.equity_after == pytest.approx(equity0)
    assert trade.target_weight == pytest.approx(1.0)
    assert "scaled to available cash" in trade.reason
    assert trade.asof == result.decisions[0].asof

    # Slippage really cost us: a fee/slippage-free run would end at +21%.
    assert result.equity_curve[2][1] < 1_000.0 * 1.21


def test_zero_cost_buy_is_exactly_hand_computable() -> None:
    cfg = make_config(fee_bps=0.0, slippage_bps=0.0)
    result = BacktestEngine(cfg).run(
        make_bars([100.0, 110.0]), ConstantWeightStrategy(1.0), "TEST"
    )
    assert result.equity_curve[0][1] == pytest.approx(1_000.0)
    assert result.equity_curve[1][1] == pytest.approx(1_100.0)
    assert result.metrics["total_return"] == pytest.approx(0.10)
    trade = result.trades[0]
    assert trade.qty == pytest.approx(10.0)
    assert trade.price == pytest.approx(100.0)
    assert trade.fee == pytest.approx(0.0)


def test_sell_to_flat_math_includes_slippage_and_fee() -> None:
    closes = [100.0, 110.0]
    cfg = make_config()
    result = BacktestEngine(cfg).run(
        make_bars(closes), ScriptedWeightStrategy([1.0, 0.0]), "TEST"
    )
    qty0, _, _ = expected_max_affordable_entry(1_000.0, 100.0)

    assert len(result.trades) == 2
    buy, sell = result.trades
    assert buy.side == "buy" and sell.side == "sell"
    assert sell.qty == pytest.approx(qty0)          # exactly the whole position, no oversell
    sell_fill = 110.0 * (1.0 - SLIPPAGE)
    assert sell.price == pytest.approx(sell_fill)
    sell_notional = qty0 * sell_fill
    assert sell.fee == pytest.approx(sell_notional * FEE_RATE)
    assert result.equity_curve[-1][1] == pytest.approx(sell_notional * (1.0 - FEE_RATE))


def test_target_weight_rebalances_to_half_equity_with_zero_costs() -> None:
    closes = [100.0, 110.0, 121.0]
    cfg = make_config(fee_bps=0.0, slippage_bps=0.0)
    result = BacktestEngine(cfg).run(make_bars(closes), ConstantWeightStrategy(0.5), "TEST")

    # 50% invested: equity compounds at half the asset's 10% daily return.
    assert [value for _, value in result.equity_curve] == pytest.approx([1_000.0, 1_050.0, 1_102.5])
    # One buy, then two small sells to keep the weight at 50% after drift.
    assert [trade.side for trade in result.trades] == ["buy", "sell", "sell"]
    assert result.metrics["exposure"] == pytest.approx(1.0)


def test_hold_deadband_suppresses_churn() -> None:
    # |delta notional| = 4.0 <= epsilon * equity (5.0) -> no fill at all.
    tiny = BacktestEngine(make_config(fee_bps=0.0, slippage_bps=0.0)).run(
        make_bars([100.0, 100.0, 100.0]), ConstantWeightStrategy(0.004), "TEST"
    )
    assert tiny.trades == []
    assert [value for _, value in tiny.equity_curve] == pytest.approx([1_000.0, 1_000.0, 1_000.0])

    # 6.0 > 5.0 -> one fill, then flat prices hold the weight with no further churn.
    bigger = BacktestEngine(make_config(fee_bps=0.0, slippage_bps=0.0)).run(
        make_bars([100.0, 100.0, 100.0]), ConstantWeightStrategy(0.006), "TEST"
    )
    assert len(bigger.trades) == 1
    assert bigger.trades[0].qty == pytest.approx(0.06)
    assert REBALANCE_EPSILON == 0.005


# --------------------------------------------------------------------------- #
# (b) THE leakage proof
# --------------------------------------------------------------------------- #
def test_leakage_proof_recording_strategy() -> None:
    n, warmup = 40, 5
    closes = [100.0 + i for i in range(n)]
    bars = make_bars(closes)
    cfg = make_config(initial_cash=10_000.0, warmup_bars=warmup)
    strategy = RecordingStrategy(weight=0.2)

    result = BacktestEngine(cfg).run(bars, strategy, "LEAK", timeframe="1d")

    # One decision per post-warmup bar, and none during warm-up.
    assert len(strategy.records) == n - warmup
    assert len(result.decisions) == n - warmup

    # The reference frame computed once, exactly as the engine did.
    frame = feature_frame(bars)

    for k, record in enumerate(strategy.records):
        i = warmup + k
        assert record["bar_index"] == i
        # Exactly i + 1 bars were visible: a growing prefix, oldest -> newest.
        assert record["n_bars"] == i + 1
        assert record["asof"] == bars[i].ts
        assert record["first_ts"] == bars[0].ts
        # The newest visible bar IS the decision bar.
        assert record["last_ts"] == bars[i].ts

        # The tuple handed over must be an immutable snapshot: if the engine had
        # passed a growing list, earlier contexts would now show more bars.
        snapshot = record["bars_obj"]
        assert isinstance(snapshot, tuple)
        assert len(snapshot) == record["n_bars"]
        assert snapshot[-1].ts == record["last_ts"]

        # Features are row i of the single causal frame -- byte-for-byte.
        expected = latest_features(frame, i)
        actual = record["features"]
        assert set(actual) == set(REQUIRED_FEATURES)
        for name in REQUIRED_FEATURES:
            if math.isnan(expected[name]):
                assert math.isnan(actual[name]), f"{name} at i={i}: expected NaN"
            else:
                assert actual[name] == pytest.approx(expected[name], rel=1e-12, abs=1e-12), name

        # The 'close' feature is the decision bar's close, never a later one.
        assert actual["close"] == pytest.approx(closes[i])
        if i >= 1:
            assert actual["ret_1"] == pytest.approx(closes[i] / closes[i - 1] - 1.0)

        # PortfolioState handed in is also point-in-time: last price == this close.
        assert record["portfolio_price"] == pytest.approx(closes[i])
        assert record["portfolio_bar_index"] == i

    # Same guarantee expressed the way ARCHITECTURE.md words it: the engine's
    # frame slice is identical to recomputing features on each prefix.
    for i in (warmup, warmup + 7, n - 1):
        recomputed = compute_features(bars[: i + 1])
        sliced = latest_features(frame, i)
        for name in REQUIRED_FEATURES:
            if math.isnan(recomputed[name]):
                assert math.isnan(sliced[name])
            else:
                assert sliced[name] == pytest.approx(recomputed[name], rel=1e-12, abs=1e-12)

    # Decision timestamps line up with the decision bars, in order.
    assert [d.asof for d in result.decisions] == [bar.ts for bar in bars[warmup:]]


def test_warmup_bars_produce_no_decisions_and_flat_equity() -> None:
    n, warmup = 20, 6
    bars = make_bars([100.0 + i for i in range(n)])
    cfg = make_config(warmup_bars=warmup)
    strategy = ConstantWeightStrategy(1.0)
    result = BacktestEngine(cfg).run(bars, strategy, "WARM")

    assert strategy.calls == n - warmup
    assert [d.asof for d in result.decisions] == [bar.ts for bar in bars[warmup:]]
    for _, value in result.equity_curve[:warmup]:
        assert value == pytest.approx(1_000.0)
    # First fill happens on the first post-warmup bar.
    assert result.trades[0].asof == bars[warmup].ts


def test_equity_is_marked_on_every_bar() -> None:
    n = 25
    bars = make_bars([100.0 + (i % 5) for i in range(n)])
    cfg = make_config(warmup_bars=3)
    result = BacktestEngine(cfg).run(bars, ConstantWeightStrategy(0.5), "MARK")

    assert len(result.equity_curve) == n
    assert [ts for ts, _ in result.equity_curve] == [bar.ts for bar in bars]
    assert len(result.benchmark_curve) == n
    assert [ts for ts, _ in result.benchmark_curve] == [bar.ts for bar in bars]
    assert result.metrics["num_bars"] == float(n)


# --------------------------------------------------------------------------- #
# (d) Buy-and-hold benchmark with costs
# --------------------------------------------------------------------------- #
def test_buy_and_hold_benchmark_matches_hand_computation() -> None:
    closes = [100.0, 110.0, 121.0]
    cfg = make_config(warmup_bars=1)   # first bar is warm-up: flat cash baseline
    result = BacktestEngine(cfg).run(make_bars(closes), ConstantWeightStrategy(0.0), "BENCH")

    curve = result.benchmark_curve
    assert curve[0][1] == pytest.approx(1_000.0)   # warm-up: all cash

    fill_price = 110.0 * (1.0 + SLIPPAGE)
    notional = 1_000.0 / (1.0 + FEE_RATE)
    qty = notional / fill_price
    assert curve[1][1] == pytest.approx(qty * 110.0)
    assert curve[2][1] == pytest.approx(qty * 121.0)

    expected_total = qty * 121.0 / 1_000.0 - 1.0
    assert result.metrics["benchmark_total_return"] == pytest.approx(expected_total)

    # The strategy was flat: zero return, zero trades, nobody loses to costs.
    assert [value for _, value in result.equity_curve] == pytest.approx([1_000.0] * 3)
    assert result.metrics["total_return"] == pytest.approx(0.0)
    assert result.metrics["excess_return"] == pytest.approx(-expected_total)
    assert result.metrics["num_trades"] == 0.0


def test_cash_benchmark_holds_initial_cash() -> None:
    result = BacktestEngine(make_config(warmup_bars=1)).run(
        make_bars([100.0, 120.0, 140.0]), ConstantWeightStrategy(0.0), "CASH", benchmark="cash"
    )
    assert [value for _, value in result.benchmark_curve] == pytest.approx([1_000.0] * 3)
    assert result.metrics["benchmark_total_return"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Metrics contract from the engine
# --------------------------------------------------------------------------- #
def test_engine_metrics_have_exactly_the_contract_keys() -> None:
    bars = make_bars([100.0 + 5.0 * math.sin(i / 3.0) + 0.2 * i for i in range(60)])
    cfg = make_config(warmup_bars=10)
    result = BacktestEngine(cfg).run(bars, ConstantWeightStrategy(0.6), "KEYS")

    assert set(result.metrics) == set(METRIC_KEYS)
    assert len(result.metrics) == len(METRIC_KEYS)
    for key, value in result.metrics.items():
        assert isinstance(value, float), key
        assert math.isfinite(value), f"{key} is not finite: {value!r}"

    assert result.metrics["num_decisions"] == 50.0
    assert result.metrics["num_bars"] == 60.0
    assert result.metrics["num_trades"] == float(len(result.trades))
    assert result.initial_cash == cfg.initial_cash
    assert result.config is cfg
    assert result.started_at is not None and result.finished_at is not None
    assert result.finished_at >= result.started_at


def test_risk_vetoes_are_counted_from_decisions() -> None:
    bars = make_bars([100.0 + i for i in range(12)])
    cfg = make_config(warmup_bars=0)
    result = BacktestEngine(cfg).run(bars, VetoingStrategy(), "VETO")

    expected = sum(1 for call in range(12) if call % 3 == 0)
    assert result.metrics["risk_vetoes"] == float(expected)
    assert result.metrics["num_decisions"] == 12.0
    assert sum(bool(d.risk and d.risk.vetoes) for d in result.decisions) == expected


# --------------------------------------------------------------------------- #
# Degenerate inputs: never raise, always finite metrics
# --------------------------------------------------------------------------- #
def test_empty_bars_returns_empty_result_with_zero_metrics() -> None:
    result = BacktestEngine(make_config()).run([], ConstantWeightStrategy(1.0), "EMPTY")
    assert result.equity_curve == []
    assert result.benchmark_curve == []
    assert result.trades == []
    assert result.decisions == []
    assert set(result.metrics) == set(METRIC_KEYS)
    assert all(value == 0.0 for value in result.metrics.values())


def test_warmup_longer_than_history_takes_no_decisions() -> None:
    bars = make_bars([100.0, 101.0, 102.0])
    cfg = make_config(warmup_bars=60)
    strategy = ConstantWeightStrategy(1.0)
    result = BacktestEngine(cfg).run(bars, strategy, "SHORT")

    assert strategy.calls == 0
    assert result.decisions == []
    assert [value for _, value in result.equity_curve] == pytest.approx([1_000.0] * 3)
    assert [value for _, value in result.benchmark_curve] == pytest.approx([1_000.0] * 3)
    assert set(result.metrics) == set(METRIC_KEYS)
    assert all(math.isfinite(value) for value in result.metrics.values())


def test_single_bar_after_warmup() -> None:
    bars = make_bars([100.0, 105.0])
    result = BacktestEngine(make_config(warmup_bars=1)).run(bars, ConstantWeightStrategy(1.0), "ONE")

    assert len(result.decisions) == 1
    assert len(result.trades) == 1
    assert len(result.equity_curve) == 2
    for value in result.metrics.values():
        assert math.isfinite(value)
    assert result.metrics["num_bars"] == 2.0


def test_all_nan_features_do_not_break_the_run() -> None:
    bars = make_bars([100.0, 101.0, 102.0])
    result = BacktestEngine(make_config(warmup_bars=0)).run(bars, FeatureKeysStrategy(), "NAN")
    assert len(result.decisions) == 3
    assert set(result.metrics) == set(METRIC_KEYS)


def test_constant_prices_zero_volatility_metrics_are_zero_not_nan() -> None:
    bars = make_bars([100.0] * 40)
    result = BacktestEngine(make_config(warmup_bars=5)).run(bars, ConstantWeightStrategy(0.0), "FLAT")

    assert result.metrics["volatility"] == 0.0
    assert result.metrics["sharpe"] == 0.0
    # Sortino is finite (and legitimately negative: flat cash earns 0% against a
    # 4% risk-free target, so it has downside deviation but no return).  The
    # contract only requires "no ZeroDivisionError"; NaN/-inf are never allowed.
    assert math.isfinite(result.metrics["sortino"])
    assert abs(result.metrics["sortino"]) <= math.sqrt(365.0) + 1e-6
    assert result.metrics["max_drawdown"] == 0.0
    assert result.metrics["total_return"] == 0.0
    assert result.metrics["win_rate"] == 0.0
    assert all(math.isfinite(value) for value in result.metrics.values())


def test_constant_prices_with_position_stay_finite() -> None:
    bars = make_bars([100.0] * 30)
    result = BacktestEngine(make_config(warmup_bars=3)).run(bars, ConstantWeightStrategy(1.0), "FLAT2")
    assert len(result.trades) >= 1
    assert all(math.isfinite(value) for value in result.metrics.values())


def test_nan_target_weight_is_treated_as_flat() -> None:
    class NanWeightStrategy:
        name = "nan_weight"

        def decide(self, ctx, portfolio) -> PortfolioDecision:
            return decision_for(ctx, float("nan"))

    result = BacktestEngine(make_config(warmup_bars=0)).run(
        make_bars([100.0, 110.0, 120.0]), NanWeightStrategy(), "NANW"
    )
    assert result.trades == []
    assert [value for _, value in result.equity_curve] == pytest.approx([1_000.0] * 3)


def test_shorting_is_blocked_when_allow_short_is_false() -> None:
    closes = [100.0, 90.0, 80.0]
    blocked = BacktestEngine(make_config(allow_short=False, fee_bps=0.0, slippage_bps=0.0)).run(
        make_bars(closes), ConstantWeightStrategy(-0.5), "SHORT"
    )
    assert blocked.trades == []
    assert [value for _, value in blocked.equity_curve] == pytest.approx([1_000.0] * 3)

    # allow_short=True is the only way a negative weight may become a fill.  The
    # short is re-balanced as equity grows on the way down, so every fill is a sell.
    allowed = BacktestEngine(make_config(allow_short=True, fee_bps=0.0, slippage_bps=0.0)).run(
        make_bars(closes), ConstantWeightStrategy(-0.5), "SHORT"
    )
    assert allowed.trades, "a negative weight must be tradeable when allow_short=True"
    assert all(trade.side == "sell" for trade in allowed.trades)
    assert sum(trade.qty for trade in allowed.trades) > 0
    # Short 50% into a 20% decline -> roughly +10% on equity (rebalanced daily).
    assert allowed.equity_curve[-1][1] > 1_000.0


def test_unsorted_and_duplicate_bars_are_normalised() -> None:
    bars = make_bars([100.0, 101.0, 102.0])
    shuffled = [bars[2], bars[0], bars[1]]
    result = BacktestEngine(make_config(warmup_bars=1)).run(shuffled, ConstantWeightStrategy(0.5), "SORT")
    assert [ts for ts, _ in result.equity_curve] == [bar.ts for bar in bars]
    assert [d.asof for d in result.decisions] == [bars[1].ts, bars[2].ts]

    duplicate = Bar(ts=bars[1].ts, open=101.0, high=101.0, low=101.0, close=101.0, volume=1.0)
    deduped = BacktestEngine(make_config(warmup_bars=0)).run(
        bars + [duplicate], ConstantWeightStrategy(0.5), "DEDUPE"
    )
    stamps = [ts for ts, _ in deduped.equity_curve]
    assert stamps == [bar.ts for bar in bars]
    assert len(stamps) == len(set(stamps))


def test_invalid_configuration_and_inputs_raise_clear_errors() -> None:
    with pytest.raises(ValueError):
        BacktestEngine(BacktestConfig(trade_at="open")).run(
            make_bars([100.0, 101.0]), ConstantWeightStrategy(0.0), "BAD"
        )
    with pytest.raises(ValueError):
        BacktestEngine(make_config()).run(
            make_bars([100.0, 101.0]), ConstantWeightStrategy(0.0), "BAD", benchmark="magic"
        )
    with pytest.raises(TypeError):
        BacktestEngine(make_config()).run(
            [make_bars([100.0])[0], "not-a-bar"], ConstantWeightStrategy(0.0), "BAD"  # type: ignore[list-item]
        )
    with pytest.raises(ValueError):
        BacktestEngine(make_config()).run(
            make_bars([float("nan"), 100.0]), ConstantWeightStrategy(0.0), "BAD"
        )


def test_backtest_run_config_to_contract_integration() -> None:
    run_cfg = BacktestRunConfig(
        initial_cash=5_000.0, fee_bps=0.0, slippage_bps=0.0, warmup_bars=0, allow_short=False
    )
    result = BacktestEngine(run_cfg.to_contract()).run(
        make_bars([100.0, 110.0]), ConstantWeightStrategy(1.0), "CONFIG"
    )
    assert result.initial_cash == 5_000.0
    assert result.equity_curve[-1][1] == pytest.approx(5_500.0)
    assert result.metrics["total_return"] == pytest.approx(0.10)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _sample_result():
    closes = [100.0 + 5.0 * math.sin(i / 2.0) + 0.4 * i for i in range(40)]
    cfg = make_config(initial_cash=10_000.0, warmup_bars=8)
    return BacktestEngine(cfg).run(
        make_bars(closes), ScriptedWeightStrategy([0.6, 0.6, 0.2, 0.0]), "RPT"
    )


def test_markdown_report_contains_headline_metrics_and_trades() -> None:
    result = _sample_result()
    markdown = render_markdown(result)

    assert "Backtest report" in markdown
    assert result.symbol in markdown
    assert "| Metric | Strategy | Buy & hold |" in markdown
    for label in ("Total return", "Sharpe", "Max drawdown", "Excess return"):
        assert label in markdown
    assert "## Trades" in markdown
    assert "No fills" not in markdown


def test_html_report_is_self_contained_with_inline_svg_polyline() -> None:
    result = _sample_result()
    html = render_html(result)

    assert html.startswith("<!DOCTYPE html>")
    assert result.symbol in html
    assert "<style>" in html                      # inline CSS
    assert "<svg" in html and "</svg>" in html    # inline chart
    assert html.count("<polyline") == 2           # strategy + benchmark
    assert "#2563eb" in html and "#6b7280" in html
    assert "<script" not in html.lower()          # no JS
    assert "https://" not in html                 # no CDN / remote assets
    assert "<img" not in html.lower()
    # The only <link> is an inline data-URI favicon; nothing is fetched.
    assert '<link rel="icon" href="data:,"' in html
    assert html.lower().count("<link") == 1
    assert 'href="http' not in html and 'src="http' not in html
    # Valid SVG sizing (a bare height="auto" is an invalid attribute value).
    assert 'height="auto"' not in html
    assert 'viewBox="0 0 ' in html


def test_save_report_writes_md_html_json(tmp_path: Path) -> None:
    result = _sample_result()
    paths = save_report(result, str(tmp_path / "reports"))

    assert set(paths) == {"md", "html", "json"}
    for key in ("md", "html", "json"):
        assert Path(paths[key]).is_file(), paths[key]
    assert Path(paths["md"]).name == "RPT_1d.md"
    assert Path(paths["html"]).name == "RPT_1d.html"
    assert Path(paths["json"]).name == "RPT_1d.json"

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["symbol"] == "RPT"
    assert set(payload["metrics"]) == set(METRIC_KEYS)
    assert payload["report"]["engine"] == "agentic_trading.backtest.engine"


def test_save_report_honours_custom_stem_and_sanitises_it(tmp_path: Path) -> None:
    result = _sample_result()
    paths = save_report(result, str(tmp_path), stem="my report!/v2")
    assert Path(paths["html"]).name == "my_report_v2.html"
    assert Path(paths["md"]).is_file()


def test_print_summary_prints_and_returns_the_same_text(capsys: pytest.CaptureFixture[str]) -> None:
    result = _sample_result()
    summary = print_summary(result)
    captured = capsys.readouterr().out

    assert isinstance(summary, str)
    assert result.symbol in summary
    assert "Return" in summary and "buy & hold" in summary
    # It prints (the CLI calls it bare) and returns the identical text.
    assert captured.rstrip("\n") == summary
    assert summary in captured


def test_reports_work_on_empty_result() -> None:
    result = BacktestEngine(make_config()).run([], ConstantWeightStrategy(0.0), "NODATA")
    markdown = render_markdown(result)
    html = render_html(result)
    summary = print_summary(result)
    assert markdown and html and summary
    assert "No fills" in markdown
    assert "No equity data" in html or "no equity data" in html.lower()


# --------------------------------------------------------------------------- #
# 10k-bar performance (the task's explicit budget: "a few seconds")
# --------------------------------------------------------------------------- #
def test_10k_bars_complete_quickly() -> None:
    n = 10_000
    closes = [100.0 + 15.0 * math.sin(i / 37.0) + 0.002 * i for i in range(n)]
    bars = make_bars(closes)
    cfg = make_config(initial_cash=10_000.0, warmup_bars=60)

    started = time.perf_counter()
    result = BacktestEngine(cfg).run(bars, ConstantWeightStrategy(0.7), "PERF")
    elapsed = time.perf_counter() - started

    assert len(result.equity_curve) == n
    assert len(result.decisions) == n - 60
    assert elapsed < 5.0, f"10k bars took {elapsed:.2f}s (budget: a few seconds)"
