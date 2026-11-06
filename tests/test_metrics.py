"""Tests for :mod:`agentic_trading.backtest.metrics`.

The point of these tests is arithmetic honesty: every metric is pinned to a
hand-computed value (or to an independent recomputation) and every degenerate
input -- empty curve, one point, zero trades, zero volatility -- is asserted to
produce finite floats rather than an exception or ``nan``.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta, timezone

import pytest

from agentic_trading.backtest.metrics import (
    cagr,
    compute_metrics,
    max_drawdown,
    period_returns,
    sharpe,
    sortino,
)
from agentic_trading.contracts import (
    METRIC_KEYS,
    Action,
    PortfolioDecision,
    RiskDecision,
    Trade,
)

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def ts(index: int) -> datetime:
    return T0 + timedelta(days=index)


def curve(values: list[float]) -> list[tuple[datetime, float]]:
    return [(ts(i), float(value)) for i, value in enumerate(values)]


def make_trade(
    index: int,
    side: str,
    qty: float,
    price: float,
    fee: float = 0.0,
) -> Trade:
    return Trade(
        asof=ts(index),
        symbol="TEST",
        side=side,
        qty=qty,
        price=price,
        fee=fee,
        equity_after=1_000.0,
    )


def make_decision(
    index: int,
    *,
    vetoes: tuple[str, ...] = (),
    approved: bool = True,
    target_weight: float = 0.0,
    with_risk: bool = False,
) -> PortfolioDecision:
    risk = None
    if with_risk or vetoes or not approved:
        risk = RiskDecision(
            approved=approved,
            target_weight=target_weight,
            vetoes=vetoes,
        )
    return PortfolioDecision(
        asof=ts(index),
        symbol="TEST",
        action=Action.HOLD,
        target_weight=target_weight,
        confidence=0.9,
        risk=risk,
    )


# --------------------------------------------------------------------------- #
# Contract surface
# --------------------------------------------------------------------------- #
def test_metric_keys_are_exactly_the_contract_keys() -> None:
    metrics = compute_metrics(curve([100, 101, 102]), curve([100, 100.5, 101]), [], [])
    assert set(metrics) == set(METRIC_KEYS)
    assert len(metrics) == len(METRIC_KEYS)
    assert list(metrics) == list(METRIC_KEYS)


def test_metrics_are_finite_floats_for_degenerate_inputs() -> None:
    cases = [
        ([], [], [], []),
        (curve([100.0]), curve([100.0]), [], []),
        (curve([100.0, 100.0, 100.0]), curve([100.0, 100.0, 100.0]), [], []),
    ]
    for equity_curve, benchmark_curve, trades, decisions in cases:
        metrics = compute_metrics(equity_curve, benchmark_curve, trades, decisions)
        assert set(metrics) == set(METRIC_KEYS)
        for key, value in metrics.items():
            assert isinstance(value, float), key
            assert math.isfinite(value), f"{key} is not finite: {value!r}"


def test_all_metrics_finite_for_normal_curve_with_trades_and_vetoes() -> None:
    trades = [
        make_trade(0, "buy", 10, 100.0, fee=1.0),
        make_trade(2, "sell", 10, 110.0, fee=1.1),
    ]
    decisions = [
        make_decision(0, with_risk=True),
        make_decision(1, vetoes=("too_risky",), approved=False),
    ]
    metrics = compute_metrics(
        curve([1_000, 1_050, 1_100, 1_090]),
        curve([1_000, 1_010, 1_020, 1_030]),
        trades,
        decisions,
    )
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["num_trades"] == 2.0
    assert metrics["num_decisions"] == 2.0
    assert metrics["num_bars"] == 4.0
    assert metrics["risk_vetoes"] == 1.0


# --------------------------------------------------------------------------- #
# max_drawdown
# --------------------------------------------------------------------------- #
def test_max_drawdown_known_series() -> None:
    # peak 120 -> trough 60 == 50% drawdown, and it is reported as positive 0.5.
    assert max_drawdown([100, 120, 90, 110, 60]) == pytest.approx(0.5)
    # 100 -> 80 is a 20% drawdown, recovered to 95 then 55 is 45% from 100.
    assert max_drawdown([100, 80, 95, 55]) == pytest.approx(0.45)
    # Peak-to-trough inside the series, not just start-to-end.
    assert max_drawdown([100, 150, 120]) == pytest.approx(0.2)


def test_max_drawdown_is_never_negative() -> None:
    assert max_drawdown([100, 101, 102, 103]) == 0.0
    assert max_drawdown([100, 100, 100]) == 0.0


def test_max_drawdown_degenerate_inputs() -> None:
    assert max_drawdown([]) == 0.0
    assert max_drawdown([100]) == 0.0


# --------------------------------------------------------------------------- #
# sharpe / sortino
# --------------------------------------------------------------------------- #
def test_sharpe_zero_volatility_returns_zero() -> None:
    assert sharpe([0.01] * 10, 0.04, 365.0) == 0.0
    assert sharpe([0.0] * 10, 0.04, 365.0) == 0.0
    assert sharpe([], 0.04, 365.0) == 0.0
    assert sharpe([0.01], 0.04, 365.0) == 0.0


def test_sharpe_matches_hand_computation() -> None:
    returns = [0.01, -0.005, 0.02, 0.0]
    rf, periods = 0.04, 365.0
    rf_per_period = rf / periods
    excess = [r - rf_per_period for r in returns]
    expected = statistics.mean(excess) / statistics.stdev(excess) * math.sqrt(periods)
    assert sharpe(returns, rf, periods) == pytest.approx(expected)

    # Zero risk-free rate simplifies the same computation.
    expected_zero_rf = statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(365.0)
    assert sharpe(returns, 0.0, 365.0) == pytest.approx(expected_zero_rf)


def test_sortino_zero_downside_returns_zero() -> None:
    assert sortino([0.01] * 6, 0.04, 365.0) == 0.0
    assert sortino([], 0.04, 365.0) == 0.0
    assert sortino([0.0] * 6, 0.0, 365.0) == 0.0


def test_sortino_matches_hand_computation() -> None:
    returns = [0.02, -0.01, 0.03, -0.02, 0.0]
    rf, periods = 0.0, 365.0
    excess = [r - rf / periods for r in returns]
    downside = math.sqrt(sum(min(e, 0.0) ** 2 for e in excess) / len(excess))
    expected = statistics.mean(excess) / downside * math.sqrt(periods)
    assert sortino(returns, rf, periods) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Returns / CAGR
# --------------------------------------------------------------------------- #
def test_period_returns_basic() -> None:
    assert period_returns([100, 110, 99]) == pytest.approx([0.1, -0.1])
    assert period_returns([]) == []
    assert period_returns([100]) == []


def test_total_return_and_cagr_hand_computed() -> None:
    # Two points, one period, annualisation of 1 period/year -> 10% CAGR.
    metrics = compute_metrics(curve([100, 110]), curve([100, 100]), [], [], periods_per_year=1.0)
    assert metrics["total_return"] == pytest.approx(0.1)
    assert metrics["cagr"] == pytest.approx(0.1)
    assert metrics["best_day"] == pytest.approx(0.1)
    assert metrics["worst_day"] == pytest.approx(0.1)

    # 365 periods/year over one period compounds to 1.1**365 - 1 (finite).
    metrics_365 = compute_metrics(curve([100, 110]), curve([100, 100]), [], [])
    assert metrics_365["cagr"] == pytest.approx(1.1**365 - 1, rel=1e-9)


def test_cagr_degenerate_inputs() -> None:
    assert cagr([], 365.0) == 0.0
    assert cagr([100], 365.0) == 0.0
    assert cagr([100, 100, 100], 365.0) == 0.0
    assert cagr([100, 0], 365.0) == -1.0  # equity wiped out


# --------------------------------------------------------------------------- #
# Benchmark comparison identities
# --------------------------------------------------------------------------- #
def test_excess_return_and_alpha_are_derived_from_the_benchmark() -> None:
    equity = curve([1_000, 1_100, 1_210, 1_150])
    benchmark = curve([1_000, 1_020, 1_050, 1_040])
    metrics = compute_metrics(equity, benchmark, [], [])

    assert metrics["benchmark_total_return"] == pytest.approx(1_040 / 1_000 - 1)
    assert metrics["benchmark_max_drawdown"] == pytest.approx(10 / 1_050)
    assert metrics["excess_return"] == pytest.approx(
        metrics["total_return"] - metrics["benchmark_total_return"]
    )
    assert metrics["alpha_cagr"] == pytest.approx(metrics["cagr"] - metrics["benchmark_cagr"])
    assert metrics["benchmark_sharpe"] == pytest.approx(
        sharpe(period_returns([1_000, 1_020, 1_050, 1_040]), 0.04, 365.0)
    )


def test_calmar_is_cagr_over_drawdown_and_zero_without_drawdown() -> None:
    rising = compute_metrics(curve([100, 110, 121]), curve([100, 100, 100]), [], [])
    assert rising["max_drawdown"] == 0.0
    assert rising["calmar"] == 0.0

    with_drawdown = compute_metrics(curve([100, 120, 60]), curve([100, 100, 100]), [], [])
    assert with_drawdown["max_drawdown"] == pytest.approx(0.5)
    assert with_drawdown["calmar"] == pytest.approx(with_drawdown["cagr"] / 0.5)


# --------------------------------------------------------------------------- #
# Trade-derived metrics
# --------------------------------------------------------------------------- #
def test_zero_trades_metrics_are_zero() -> None:
    metrics = compute_metrics(curve([1_000, 1_000, 1_000]), curve([1_000, 1_000, 1_000]), [], [])
    assert metrics["num_trades"] == 0.0
    assert metrics["win_rate"] == 0.0
    assert metrics["turnover"] == 0.0
    assert metrics["exposure"] == 0.0


def test_win_rate_uses_closed_round_trips_including_fees() -> None:
    # One winner, one loser -> 0.5.
    trades = [
        make_trade(0, "buy", 10, 100.0),
        make_trade(1, "sell", 10, 110.0),
        make_trade(2, "buy", 10, 100.0),
        make_trade(3, "sell", 10, 90.0),
    ]
    metrics = compute_metrics(curve([1_000] * 5), curve([1_000] * 5), trades, [])
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["num_trades"] == 4.0

    # Fees turn a flat round trip into a loser.
    break_even_with_fees = [
        make_trade(0, "buy", 10, 100.0, fee=1.0),
        make_trade(1, "sell", 10, 100.0, fee=1.0),
    ]
    metrics = compute_metrics(curve([1_000] * 3), curve([1_000] * 3), break_even_with_fees, [])
    assert metrics["win_rate"] == 0.0


def test_turnover_and_exposure_are_reconstructed_from_fills() -> None:
    trades = [
        make_trade(0, "buy", 5, 100.0),    # 500 notional at bar 0
        make_trade(2, "sell", 5, 110.0),   # 550 notional at bar 2
    ]
    mean_equity = statistics.mean([1_000.0, 1_050.0, 1_100.0])
    metrics = compute_metrics(curve([1_000, 1_050, 1_100]), curve([1_000, 1_000, 1_000]), trades, [])
    assert metrics["turnover"] == pytest.approx((500 + 550) / mean_equity)
    # Position is held at bars 0, 1 and flat at bar 2 -> 2/3 of the curve.
    assert metrics["exposure"] == pytest.approx(2 / 3)


def test_best_and_worst_bar_are_extremes_of_period_returns() -> None:
    metrics = compute_metrics(curve([100, 110, 99, 105]), curve([100, 100, 100, 100]), [], [])
    returns = period_returns([100, 110, 99, 105])
    assert metrics["best_day"] == pytest.approx(max(returns))
    assert metrics["worst_day"] == pytest.approx(min(returns))


# --------------------------------------------------------------------------- #
# Decision-derived metrics
# --------------------------------------------------------------------------- #
def test_risk_vetoes_counts_only_non_empty_veto_tuples() -> None:
    decisions = [
        make_decision(0, vetoes=("max_position_weight",), approved=False),
        make_decision(1, approved=False),                 # rejected but no veto text
        make_decision(2, with_risk=True, approved=True),  # risk present, approved
        make_decision(3),                                 # no risk layer at all
    ]
    metrics = compute_metrics(curve([1_000, 1_000]), curve([1_000, 1_000]), [], decisions)
    assert metrics["risk_vetoes"] == 1.0
    assert metrics["num_decisions"] == 4.0
