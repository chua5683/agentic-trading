"""Tests for the deterministic risk authority (``agents/risk.py``).

The risk gate is the architectural keystone: an LLM (or any buggy proposer) may
put whatever it wants into a ``TradeProposal``, but ``evaluate`` is the only code
allowed to produce a ``RiskDecision`` and its ``target_weight`` always obeys the
``RiskConfig`` bounds.  These tests pin every documented rule:

* oversized weight -> clamped to ``max_position_weight``;
* low confidence -> veto -> HOLD (current exposure kept);
* shorting blocked unless ``allow_short``;
* drawdown stop blocks *new* exposure but still allows de-risking;
* stop-loss / take-profit / time stop close an existing position;
* cooldown after a stop-out;
* turnover cap;
* fuzz: the bounds can never be exceeded, whatever the input.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

try:  # works for both ``pytest`` and ``python -m pytest`` invocation styles
    from tests.conftest import BULLISH_FEATURES, make_context, make_portfolio
except ImportError:  # pragma: no cover - fallback for importlib import mode
    from conftest import BULLISH_FEATURES, make_context, make_portfolio

from agentic_trading.agents.risk import current_weight, evaluate
from agentic_trading.config import RiskConfig
from agentic_trading.contracts import (
    Action,
    MarketContext,
    PortfolioState,
    Position,
    RiskDecision,
    TradeProposal,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ctx(**kwargs):
    kwargs.setdefault("features", BULLISH_FEATURES)
    return make_context(**kwargs)


def _proposal(
    target_weight: float = 0.30,
    confidence: float = 0.90,
    action: Action = Action.BUY,
    symbol: str = "BTC-USD",
) -> TradeProposal:
    return TradeProposal(symbol=symbol, action=action, target_weight=target_weight,
                         confidence=confidence, thesis="test")


def _portfolio_with_weight(weight: float, price: float, *, equity: float = 10_000.0,
                           peak_equity: float | None = None) -> PortfolioState:
    qty = weight * equity / price if price > 0 else 0.0
    return make_portfolio(equity=equity, qty=qty, price=price, avg_price=price,
                          peak_equity=peak_equity)


# --------------------------------------------------------------------------- #
# Weight clamp
# --------------------------------------------------------------------------- #
def test_proposal_with_oversized_weight_is_clamped():
    ctx = _ctx()
    decision = evaluate(_proposal(target_weight=0.90), ctx, make_portfolio(), RiskConfig())
    assert isinstance(decision, RiskDecision)
    assert decision.approved is True
    assert decision.target_weight == pytest.approx(0.30)
    assert decision.was_modified is True
    assert any("max_position_weight" in text for text in decision.adjustments)
    assert any("90.00%" in text for text in decision.adjustments)


def test_clamp_respects_custom_max_position_weight():
    ctx = _ctx()
    cfg = RiskConfig(max_position_weight=0.15)
    decision = evaluate(_proposal(target_weight=0.90), ctx, make_portfolio(), cfg)
    assert decision.target_weight == pytest.approx(0.15)
    assert decision.target_weight <= cfg.max_position_weight


def test_negative_weight_is_floored_when_shorts_disabled():
    ctx = _ctx()
    decision = evaluate(_proposal(target_weight=-0.40, action=Action.SELL),
                        ctx, make_portfolio(), RiskConfig(allow_short=False))
    assert decision.target_weight >= 0.0
    assert decision.target_weight == pytest.approx(0.0)
    assert any("short" in text.lower() for text in decision.adjustments)


def test_shorts_are_bounded_when_allowed():
    ctx = _ctx()
    cfg = RiskConfig(allow_short=True)
    decision = evaluate(_proposal(target_weight=-0.90, action=Action.SELL),
                        ctx, make_portfolio(), cfg)
    assert decision.target_weight == pytest.approx(-cfg.max_position_weight)
    assert -cfg.max_position_weight - 1e-9 <= decision.target_weight <= cfg.max_position_weight


# --------------------------------------------------------------------------- #
# Confidence gate
# --------------------------------------------------------------------------- #
def test_low_confidence_is_vetoed_and_holds_flat():
    ctx = _ctx()
    decision = evaluate(_proposal(target_weight=0.30, confidence=0.30),
                        ctx, make_portfolio(), RiskConfig())
    assert decision.approved is False
    assert decision.target_weight == pytest.approx(0.0)
    assert decision.vetoes
    assert any("min_confidence" in text for text in decision.vetoes)


def test_low_confidence_holds_an_existing_position():
    ctx = _ctx()
    price = float(ctx.last_close)
    portfolio = _portfolio_with_weight(0.15, price)
    decision = evaluate(_proposal(target_weight=0.30, confidence=0.30),
                        ctx, portfolio, RiskConfig())
    assert decision.approved is False
    assert decision.target_weight == pytest.approx(0.15, abs=1e-9)


def test_confidence_gate_does_not_block_de_risking():
    ctx = _ctx()
    price = float(ctx.last_close)
    portfolio = _portfolio_with_weight(0.25, price)
    decision = evaluate(_proposal(target_weight=0.0, confidence=0.10, action=Action.SELL),
                        ctx, portfolio, RiskConfig())
    assert decision.approved is True
    assert decision.target_weight == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Drawdown stop
# --------------------------------------------------------------------------- #
def test_drawdown_stop_blocks_new_exposure():
    ctx = _ctx()
    underwater = make_portfolio(equity=7_000.0, cash=7_000.0, peak_equity=10_000.0)
    assert underwater.drawdown == pytest.approx(0.30)
    decision = evaluate(_proposal(target_weight=0.30, confidence=0.90),
                        ctx, underwater, RiskConfig(max_drawdown_stop=0.25))
    assert decision.approved is False
    assert decision.target_weight == pytest.approx(0.0)
    assert any("drawdown" in text.lower() for text in decision.vetoes)


def test_drawdown_stop_still_allows_de_risking():
    ctx = _ctx()
    price = float(ctx.last_close)
    portfolio = _portfolio_with_weight(0.20, price, equity=7_000.0, peak_equity=10_000.0)
    decision = evaluate(_proposal(target_weight=0.10, confidence=0.90, action=Action.SELL),
                        ctx, portfolio, RiskConfig(max_drawdown_stop=0.25))
    assert decision.approved is True
    assert decision.target_weight == pytest.approx(0.10, abs=1e-9)


# --------------------------------------------------------------------------- #
# Position-protection exits
# --------------------------------------------------------------------------- #
def _long_position(price: float, pnl: float, *, equity: float = 10_000.0) -> PortfolioState:
    """Portfolio holding one long whose unrealised P&L is ``pnl`` (fraction)."""
    avg_price = price / (1.0 + pnl)
    return make_portfolio(equity=equity, qty=10.0, price=price, avg_price=avg_price)


def test_stop_loss_closes_existing_position():
    ctx = _ctx()
    price = float(ctx.last_close)
    portfolio = _long_position(price, -0.15)
    decision = evaluate(_proposal(target_weight=0.30, confidence=0.90),
                        ctx, portfolio, RiskConfig(stop_loss_pct=0.08))
    assert decision.approved is True  # the exit itself is approved
    assert decision.target_weight == pytest.approx(0.0)
    assert any("stop-loss" in text.lower() for text in decision.adjustments)


def test_take_profit_closes_existing_position():
    ctx = _ctx()
    price = float(ctx.last_close)
    portfolio = _long_position(price, 0.30)
    decision = evaluate(_proposal(target_weight=0.30, confidence=0.90),
                        ctx, portfolio, RiskConfig(take_profit_pct=0.25))
    assert decision.approved is True
    assert decision.target_weight == pytest.approx(0.0)
    assert any("take-profit" in text.lower() for text in decision.adjustments)


def test_protective_exit_beats_low_confidence():
    ctx = _ctx()
    price = float(ctx.last_close)
    portfolio = _long_position(price, -0.20)
    decision = evaluate(_proposal(target_weight=0.30, confidence=0.05),
                        ctx, portfolio, RiskConfig(stop_loss_pct=0.08, min_confidence=0.55))
    assert decision.target_weight == pytest.approx(0.0)
    assert any("stop-loss" in text.lower() for text in decision.adjustments)
    # The confidence veto is irrelevant once a stop-out is mandated.
    assert not any("min_confidence" in text for text in decision.vetoes)


def test_no_forced_exit_inside_thresholds():
    ctx = _ctx()
    price = float(ctx.last_close)
    portfolio = _long_position(price, 0.05)  # +5%: above -8%, below +25%
    decision = evaluate(_proposal(target_weight=0.20, confidence=0.90),
                        ctx, portfolio, RiskConfig())
    assert decision.approved is True
    assert decision.target_weight == pytest.approx(0.20, abs=1e-9)


def test_time_stop_closes_position():
    ctx = _ctx(meta={"bars_held": 6})
    price = float(ctx.last_close)
    portfolio = _long_position(price, 0.0)
    decision = evaluate(_proposal(target_weight=0.20, confidence=0.90),
                        ctx, portfolio, RiskConfig(max_bars_hold=5))
    assert decision.approved is True
    assert decision.target_weight == pytest.approx(0.0)
    assert any("time stop" in text.lower() for text in decision.adjustments)


# --------------------------------------------------------------------------- #
# Cooldown
# --------------------------------------------------------------------------- #
def test_cooldown_blocks_new_entry():
    ctx = _ctx(meta={"bars_since_stop": 1})
    decision = evaluate(_proposal(target_weight=0.20, confidence=0.90),
                        ctx, make_portfolio(), RiskConfig(cooldown_bars_after_stop=2))
    assert decision.approved is False
    assert decision.target_weight == pytest.approx(0.0)
    assert any("cooldown" in text.lower() for text in decision.vetoes)


def test_cooldown_expires():
    ctx = _ctx(meta={"bars_since_stop": 3})
    decision = evaluate(_proposal(target_weight=0.20, confidence=0.90),
                        ctx, make_portfolio(), RiskConfig(cooldown_bars_after_stop=2))
    assert decision.approved is True
    assert decision.target_weight == pytest.approx(0.20)


def test_cooldown_accepts_stopped_out_flag():
    ctx = _ctx(meta={"stopped_out": True})
    decision = evaluate(_proposal(target_weight=0.20, confidence=0.90),
                        ctx, make_portfolio(), RiskConfig(cooldown_bars_after_stop=1))
    assert decision.approved is False
    assert any("cooldown" in text.lower() for text in decision.vetoes)


# --------------------------------------------------------------------------- #
# Turnover cap
# --------------------------------------------------------------------------- #
def test_turnover_cap_limits_increase():
    ctx = _ctx()
    cfg = RiskConfig(max_daily_turnover=0.10)
    decision = evaluate(_proposal(target_weight=0.30, confidence=0.90),
                        ctx, make_portfolio(), cfg)
    assert decision.approved is True
    assert decision.target_weight == pytest.approx(0.10)
    assert any("turnover" in text.lower() for text in decision.adjustments)


def test_turnover_cap_limits_reduction():
    ctx = _ctx()
    price = float(ctx.last_close)
    portfolio = _portfolio_with_weight(0.30, price)
    decision = evaluate(_proposal(target_weight=0.0, confidence=0.90, action=Action.SELL),
                        ctx, portfolio, RiskConfig(max_daily_turnover=0.20))
    assert decision.approved is True
    assert decision.target_weight == pytest.approx(0.10, abs=1e-9)


# --------------------------------------------------------------------------- #
# Stop / take-profit levels
# --------------------------------------------------------------------------- #
def test_stop_and_take_profit_levels_are_attached():
    ctx = _ctx()
    price = float(ctx.last_close)
    cfg = RiskConfig(stop_loss_pct=0.08, take_profit_pct=0.25)
    decision = evaluate(_proposal(target_weight=0.20), ctx, make_portfolio(), cfg)
    assert decision.stop_loss == pytest.approx(price * (1.0 - 0.08))
    assert decision.take_profit == pytest.approx(price * (1.0 + 0.25))


def test_exit_has_no_stop_levels():
    ctx = _ctx()
    price = float(ctx.last_close)
    portfolio = _portfolio_with_weight(0.20, price)
    decision = evaluate(_proposal(target_weight=0.0, action=Action.SELL),
                        ctx, portfolio, RiskConfig())
    assert decision.target_weight == pytest.approx(0.0)
    assert decision.stop_loss is None
    assert decision.take_profit is None


# --------------------------------------------------------------------------- #
# Helpers / edge cases
# --------------------------------------------------------------------------- #
def test_current_weight_prefers_the_decision_bar_price():
    price = 200.0
    portfolio = PortfolioState(
        cash=0.0,
        positions={"BTC-USD": Position(symbol="BTC-USD", qty=10.0, avg_price=100.0)},
        equity=10_000.0,
        peak_equity=10_000.0,
        last_prices={},  # deliberately empty -> must fall back to the given price
    )
    assert current_weight(portfolio, "BTC-USD", price) == pytest.approx(10.0 * 200.0 / 10_000.0)
    assert current_weight(portfolio, "BTC-USD") == pytest.approx(0.0)
    assert current_weight(portfolio, "MISSING", price) == pytest.approx(0.0)


def test_missing_market_data_is_vetoed_not_raised():
    empty_ctx = MarketContext(symbol="BTC-USD", asof=datetime.now(timezone.utc),
                              bars=(), features={})
    decision = evaluate(_proposal(), empty_ctx, make_portfolio(), RiskConfig())
    assert decision.approved is False
    assert decision.target_weight == 0.0
    assert decision.vetoes


def test_zero_equity_is_vetoed_not_raised():
    ctx = _ctx()
    decision = evaluate(_proposal(), ctx, make_portfolio(equity=0.0, cash=0.0), RiskConfig())
    assert decision.approved is False
    assert decision.target_weight == 0.0
    assert decision.vetoes


def test_risk_is_deterministic():
    ctx = _ctx()
    portfolio = make_portfolio()
    first = evaluate(_proposal(target_weight=0.42), ctx, portfolio, RiskConfig())
    second = evaluate(_proposal(target_weight=0.42), ctx, portfolio, RiskConfig())
    assert first == second


def test_was_modified_tracks_adjustments():
    ctx = _ctx()
    clamped = evaluate(_proposal(target_weight=0.90), ctx, make_portfolio(), RiskConfig())
    assert clamped.was_modified is True
    clean = evaluate(_proposal(target_weight=0.20), ctx, make_portfolio(), RiskConfig())
    assert clean.adjustments == ()
    assert clean.was_modified is False


# --------------------------------------------------------------------------- #
# Bounds invariant fuzz -- the core architectural proof
# --------------------------------------------------------------------------- #
def test_bounds_hold_for_fuzz_inputs():
    ctx = _ctx()
    price = float(ctx.last_close)
    portfolios = [
        make_portfolio(),
        make_portfolio(qty=40.0, price=price, avg_price=price),
        _long_position(price, -0.20),
        make_portfolio(equity=0.0, cash=0.0),
    ]
    configs = [
        RiskConfig(),
        RiskConfig(max_position_weight=0.15),
        RiskConfig(allow_short=True, max_daily_turnover=0.05),
        RiskConfig(max_daily_turnover=0.02, max_drawdown_stop=0.0,
                   cooldown_bars_after_stop=0, allow_short=False),
    ]
    weights = [-1.5, -0.4, -0.01, 0.0, 0.1, 0.3, 0.9, 2.0, float("nan")]
    confidences = [0.0, 0.3, 0.55, 0.9]
    actions = [Action.BUY, Action.SELL, Action.HOLD]

    for cfg in configs:
        for portfolio in portfolios:
            for weight in weights:
                for confidence in confidences:
                    for action in actions:
                        decision = evaluate(
                            _proposal(target_weight=weight, confidence=confidence,
                                      action=action),
                            ctx, portfolio, cfg)
                        assert isinstance(decision, RiskDecision)
                        assert isinstance(decision.approved, bool)
                        assert math.isfinite(decision.target_weight)
                        assert decision.target_weight <= cfg.max_position_weight + 1e-9
                        if not cfg.allow_short:
                            assert decision.target_weight >= -1e-9
                        else:
                            assert decision.target_weight >= -cfg.max_position_weight - 1e-9
                        assert all(isinstance(text, str) and text
                                   for text in decision.vetoes)
                        assert all(isinstance(text, str) and text
                                   for text in decision.adjustments)
                        assert all(isinstance(text, str) and text
                                   for text in decision.notes)


def test_never_above_max_even_with_zero_turnover_and_huge_confidence():
    ctx = _ctx()
    cfg = RiskConfig(max_position_weight=0.30, max_daily_turnover=0.0)
    decision = evaluate(_proposal(target_weight=2.0, confidence=1.0),
                        ctx, make_portfolio(), cfg)
    assert decision.target_weight <= 0.30 + 1e-9
    assert decision.target_weight >= 0.0
