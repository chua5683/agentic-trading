"""Tests for the agent pipeline: analysts -> debate -> trader -> deterministic risk.

Coverage map (required by the task):

* pipeline returns a valid ``PortfolioDecision`` for a synthetic MarketContext;
* deterministic mode provably never touches the network;
* LLM failure / hostile LLM output degrades gracefully and cannot bypass limits;
* every stance label is valid, including during NaN warmup;
* analysts are genuinely directional (not stubs);
* the debate's pooling is deterministic and the LLM cannot change its numbers.
"""

from __future__ import annotations

import inspect
import json
import math
import socket

import pytest

try:  # works for both ``pytest`` and ``python -m pytest`` invocation styles
    from tests.conftest import (
        BEARISH_FEATURES,
        BULLISH_FEATURES,
        CALM_FEATURES,
        NAN_FEATURES,
        STRESSED_FEATURES,
        ExplodingLLM,
        ScriptedLLM,
        make_bars,
        make_context,
        make_portfolio,
    )
except ImportError:  # pragma: no cover - fallback for importlib import mode
    from conftest import (
        BEARISH_FEATURES,
        BULLISH_FEATURES,
        CALM_FEATURES,
        NAN_FEATURES,
        STRESSED_FEATURES,
        ExplodingLLM,
        ScriptedLLM,
        make_bars,
        make_context,
        make_portfolio,
    )

from agentic_trading.agents import analysts as analysts_module
from agentic_trading.agents import debate as debate_module
from agentic_trading.agents import llm as llm_module
from agentic_trading.agents import pipeline as pipeline_module
from agentic_trading.agents import risk as risk_module
from agentic_trading.agents import trader as trader_module
from agentic_trading.agents.analysts import (
    ANALYST_WEIGHTS,
    EVIDENCE_WEIGHTS,
    MeanReversionAnalyst,
    MomentumAnalyst,
    RiskAnalyst,
    TechnicalAnalyst,
    run_analysts,
)
from agentic_trading.agents.debate import (
    CONSENSUS_THRESHOLD,
    consensus_for,
    pooled_net_score,
    run_debate,
    stance_weights,
    trend_regime,
)
from agentic_trading.agents.llm import (
    BudgetedLLM,
    LLMClient,
    NullLLM,
    extract_json,
    make_llm,
)
from agentic_trading.agents.pipeline import (
    DEFAULT_MIN_HOLD_BARS,
    MAX_LLM_CALLS_PER_DECISION,
    AgentStrategy,
    SmoothedConsensusPolicy,
    build_strategy,
)
from agentic_trading.agents.trader import annual_vol, propose
from agentic_trading.config import AppConfig, LLMConfig
from agentic_trading.contracts import (
    Action,
    DebateResult,
    DecisionStrategy,
    MarketContext,
    PortfolioDecision,
    REQUIRED_FEATURES,
    Stance,
)

VALID_LABELS = {"bullish", "bearish", "neutral"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _deterministic_cfg(**risk_overrides) -> AppConfig:
    cfg = AppConfig()
    cfg.llm.enabled = "off"
    for key, value in risk_overrides.items():
        setattr(cfg.risk, key, value)
    return cfg


def _stance_map(stances) -> dict[str, Stance]:
    return {stance.agent: stance for stance in stances}


def _debate_with_score(score: float) -> DebateResult:
    """Minimal DebateResult carrying a chosen raw net_score (policy testing)."""
    return DebateResult(
        bull=Stance("bull_researcher", "bullish", max(0.0, score), "bull case"),
        bear=Stance("bear_researcher", "bearish", max(0.0, -score), "bear case"),
        rounds=1,
        consensus=consensus_for(score),
        net_score=score,
    )


# --------------------------------------------------------------------------- #
# LLM client: offline by default, never raises
# --------------------------------------------------------------------------- #
def test_make_llm_without_key_is_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = make_llm(LLMConfig())
    assert isinstance(client, NullLLM)
    assert client.available is False
    assert client.model == LLMConfig().model
    assert client.complete_text("s", "u") == ""
    assert client.complete_json("s", "u", "{}") is None
    assert client.calls == 0


def test_make_llm_disabled_even_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    cfg = LLMConfig(enabled="off")
    client = make_llm(cfg)
    assert client.available is False
    assert "disabled" in client.reason


def test_make_llm_with_key_is_available(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    client = make_llm(LLMConfig(enabled="auto"))
    assert client.available is True
    assert client.model == "gpt-4o-mini"


def test_null_llm_never_raises_or_calls():
    null = NullLLM()
    assert null.available is False
    assert null.complete_text("a", "b") == ""
    assert null.complete_json("a", "b", "{}") is None
    assert null.calls == 0


def test_extract_json_handles_fences_and_prose():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Sure! Here it is: {"a": 3} hope that helps') == {"a": 3}
    assert extract_json("not json at all") is None
    assert extract_json(None) is None
    assert extract_json("") is None
    assert extract_json("[1, 2]") is None  # dicts only


def test_llm_client_parses_json_via_transport_seam():
    class Canned(LLMClient):
        def __init__(self, content):
            super().__init__(LLMConfig(enabled="on"), available=True)
            self._content = content

        def _chat(self, system, user, *, require_json=False):
            self.calls += 1
            return self._content

    fenced = Canned('```json\n{"label": "bullish", "confidence": 0.6}\n```')
    assert fenced.complete_json("s", "u", "{}") == {"label": "bullish", "confidence": 0.6}

    broken = Canned("definitely not json")
    assert broken.complete_json("s", "u", "{}") is None
    assert broken.failures >= 1

    text = Canned("hello trader")
    assert text.complete_text("s", "u") == "hello trader"


def test_llm_client_transport_error_never_raises(monkeypatch):
    import requests

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(requests, "post", boom)
    client = LLMClient(LLMConfig(enabled="on", max_retries=1), available=True)
    assert client.complete_text("s", "u") == ""
    assert client.complete_json("s", "u", "{}") is None
    assert client.calls >= 1
    assert client.failures >= 1
    assert "network down" in str(client.last_error)


def test_budgeted_llm_stops_after_budget():
    inner = ScriptedLLM([{"label": "bullish", "confidence": 0.6, "rationale": "x"}])
    budgeted = BudgetedLLM(inner, budget=2)
    assert budgeted.available is True
    expected = {"label": "bullish", "confidence": 0.6, "rationale": "x"}
    assert budgeted.complete_json("s", "u", "{}") == expected
    assert budgeted.complete_json("s", "u", "{}") == expected
    assert budgeted.available is False
    assert budgeted.complete_json("s", "u", "{}") is None
    assert inner.calls == 2


# --------------------------------------------------------------------------- #
# Strategy construction / modes
# --------------------------------------------------------------------------- #
def test_build_strategy_is_deterministic_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    strategy = build_strategy(AppConfig())
    assert isinstance(strategy, AgentStrategy)
    assert isinstance(strategy, DecisionStrategy)
    assert strategy.name
    assert strategy.mode == "deterministic"
    assert strategy._llm.available is False


def test_strategy_mode_is_hybrid_when_llm_injected():
    strategy = AgentStrategy(_deterministic_cfg(), llm=ScriptedLLM([]))
    assert strategy.mode == "hybrid"


def test_agents_use_llm_flag_disables_llm():
    cfg = _deterministic_cfg()
    cfg.agents.use_llm = False
    strategy = AgentStrategy(cfg, llm=ScriptedLLM([{"label": "bullish", "confidence": 1.0}]))
    assert strategy.mode == "deterministic"
    strategy.decide(make_context(features=BULLISH_FEATURES), make_portfolio())
    # llm was forced off even though an available client was injected.


# --------------------------------------------------------------------------- #
# Main pipeline contract
# --------------------------------------------------------------------------- #
def test_pipeline_returns_valid_portfolio_decision(cfg):
    ctx = make_context(features=BULLISH_FEATURES)
    portfolio = make_portfolio()
    decision = build_strategy(cfg).decide(ctx, portfolio)

    assert isinstance(decision, PortfolioDecision)
    assert decision.asof == ctx.asof
    assert decision.symbol == "BTC-USD"
    assert decision.action is Action.BUY
    assert 0.0 <= decision.target_weight <= cfg.risk.max_position_weight + 1e-9
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.thesis

    names = [stance.agent for stance in decision.analysis]
    assert names == ["technical", "momentum", "mean_reversion", "risk"]
    for stance in decision.analysis:
        assert stance.label in VALID_LABELS
        assert 0.0 <= stance.confidence <= 1.0
        assert stance.source in {"deterministic", "llm"}
        assert isinstance(stance.rationale, str) and stance.rationale

    assert decision.debate is not None
    assert decision.debate.consensus == "bullish"
    assert decision.debate.net_score > 0
    assert -1.0 <= decision.debate.net_score <= 1.0
    assert decision.proposal is not None
    assert decision.proposal.symbol == "BTC-USD"
    assert decision.proposal.action in (Action.BUY, Action.HOLD, Action.SELL)
    assert 0.0 <= decision.proposal.target_weight <= 1.0
    assert decision.risk is not None
    assert decision.risk.approved is True
    assert 0.0 <= decision.risk.target_weight <= cfg.risk.max_position_weight + 1e-9
    assert decision.mode == "deterministic"

    # The whole record must be journal-ready (JSON serialisable).
    payload = decision.to_dict()
    assert json.loads(json.dumps(payload))["action"] == "buy"
    assert payload["risk"]["approved"] is True


def test_decision_target_never_exceeds_risk_limit(cfg):
    contexts = [
        make_context(features=BULLISH_FEATURES),
        make_context(features=STRESSED_FEATURES),
        make_context(features=BEARISH_FEATURES),
        make_context(n=260, trend=0.01, vol=0.03, seed=5),
        make_context(n=90, trend=0.02, vol=0.01, seed=2),
    ]
    portfolios = [make_portfolio(), make_portfolio(qty=30.0, price=100.0)]
    for ctx in contexts:
        for portfolio in portfolios:
            decision = build_strategy(cfg).decide(ctx, portfolio)
            assert math.isfinite(decision.target_weight)
            assert -1e-9 <= decision.target_weight <= cfg.risk.max_position_weight + 1e-9
            assert 0.0 <= decision.confidence <= 1.0
            assert decision.risk is not None
            assert decision.risk.target_weight <= cfg.risk.max_position_weight + 1e-9
            assert decision.risk.target_weight >= -1e-9  # long-only shipped config


def test_decision_is_deterministic_across_runs(cfg):
    ctx = make_context(features=BULLISH_FEATURES)
    portfolio = make_portfolio()
    first = build_strategy(cfg).decide(ctx, portfolio).to_dict()
    second = build_strategy(cfg).decide(ctx, portfolio).to_dict()
    assert first == second


def test_deterministic_mode_never_touches_network(monkeypatch):
    import requests

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_TRADING_LLM", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("network access attempted in deterministic mode")

    monkeypatch.setattr(requests, "post", forbidden)
    monkeypatch.setattr(requests, "get", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)

    strategy = build_strategy(AppConfig())
    assert strategy.mode == "deterministic"
    decision = strategy.decide(make_context(features=BULLISH_FEATURES), make_portfolio())
    assert isinstance(decision, PortfolioDecision)
    assert decision.mode == "deterministic"


# --------------------------------------------------------------------------- #
# LLM failure / hostility
# --------------------------------------------------------------------------- #
def test_llm_transport_failure_degrades_to_deterministic(cfg):
    strategy = AgentStrategy(cfg, llm=ExplodingLLM())
    assert strategy.mode == "hybrid"
    decision = strategy.decide(make_context(features=BULLISH_FEATURES), make_portfolio())
    assert isinstance(decision, PortfolioDecision)
    assert decision.mode == "deterministic"
    assert all(stance.source == "deterministic" for stance in decision.analysis)
    assert 0.0 <= decision.target_weight <= cfg.risk.max_position_weight + 1e-9


def test_llm_none_responses_degrade_gracefully(cfg):
    strategy = AgentStrategy(cfg, llm=ScriptedLLM([], fail=True))
    decision = strategy.decide(make_context(features=BULLISH_FEATURES), make_portfolio())
    assert decision.mode == "deterministic"
    assert all(stance.source == "deterministic" for stance in decision.analysis)
    assert isinstance(decision.to_dict(), dict)


def test_llm_refinement_marks_hybrid(cfg):
    fake = ScriptedLLM([{"label": "bullish", "confidence": 0.9, "rationale": "LLM agrees"}])
    strategy = AgentStrategy(cfg, llm=fake)
    decision = strategy.decide(make_context(features=BULLISH_FEATURES), make_portfolio())
    assert decision.mode == "hybrid"
    assert any(stance.source == "llm" for stance in decision.analysis)
    assert fake.calls > 0
    # Budget is respected per bar.
    assert fake.calls <= MAX_LLM_CALLS_PER_DECISION


def test_hostile_llm_weight_suggestion_is_clamped_by_risk(cfg):
    cfg.llm.enabled = "on"
    cfg.llm.allow_weight_override = True
    hostile = ScriptedLLM([{
        "label": "bullish", "confidence": 1.0, "rationale": "to the moon",
        "action": "buy", "target_weight": 0.99, "thesis": "hostile thesis",
        "bull": "bull", "bear": "bear",
    }])
    strategy = AgentStrategy(cfg, llm=hostile)
    decision = strategy.decide(make_context(features=BULLISH_FEATURES), make_portfolio())

    max_weight = cfg.risk.max_position_weight
    # The LLM got its 99% suggestion into the proposal...
    assert decision.proposal is not None
    assert decision.proposal.target_weight > max_weight
    # ...and deterministic risk threw it out.
    assert decision.target_weight <= max_weight + 1e-9
    assert decision.risk is not None and decision.risk.target_weight <= max_weight + 1e-9
    assert any("max_position_weight" in text for text in decision.risk.adjustments)
    assert decision.mode == "hybrid"


def test_llm_cannot_size_without_allow_weight_override(cfg):
    hostile_response = {"action": "buy", "target_weight": 0.99, "thesis": "size me up"}
    cfg.llm.enabled = "on"
    cfg.llm.allow_weight_override = False
    baseline = build_strategy(cfg).decide(make_context(features=BULLISH_FEATURES), make_portfolio())
    with_llm = AgentStrategy(cfg, llm=ScriptedLLM([hostile_response])).decide(
        make_context(features=BULLISH_FEATURES), make_portfolio())
    assert with_llm.proposal is not None and baseline.proposal is not None
    assert with_llm.proposal.target_weight == pytest.approx(baseline.proposal.target_weight)
    assert with_llm.target_weight == pytest.approx(baseline.target_weight)


def test_llm_may_only_tighten_the_action(cfg):
    cfg.llm.enabled = "on"
    bullish_ctx = make_context(features=BULLISH_FEATURES)

    # 1) LLM "hold" is allowed: it de-risks the deterministic BUY.
    hold_llm = ScriptedLLM([{"action": "hold", "thesis": "wait for volume"}])
    held = AgentStrategy(cfg, llm=hold_llm).decide(bullish_ctx, make_portfolio())
    assert held.proposal is not None
    assert held.proposal.action is Action.HOLD
    assert held.action is Action.HOLD
    assert held.target_weight == 0.0

    # 2) LLM "sell" is NOT allowed against a deterministic BUY.
    sell_llm = ScriptedLLM([{"action": "sell", "thesis": "flip it"}])
    kept = AgentStrategy(cfg, llm=sell_llm).decide(bullish_ctx, make_portfolio())
    assert kept.proposal is not None
    assert kept.proposal.action is Action.BUY
    assert kept.action is Action.BUY


# --------------------------------------------------------------------------- #
# Warmup / NaN robustness
# --------------------------------------------------------------------------- #
def test_warmup_short_history_is_hold(cfg):
    ctx = make_context(n=10, features=NAN_FEATURES)
    decision = build_strategy(cfg).decide(ctx, make_portfolio())
    assert decision.action is Action.HOLD
    assert decision.target_weight == 0.0
    assert len(decision.analysis) == 4
    assert all(stance.label == "neutral" for stance in decision.analysis)
    assert decision.diagnostics["warmup"] is True


def test_nan_features_never_raise(cfg):
    ctx = make_context(n=260, features=NAN_FEATURES)
    decision = build_strategy(cfg).decide(ctx, make_portfolio())
    assert isinstance(decision, PortfolioDecision)
    assert decision.action is Action.HOLD
    assert all(stance.label in VALID_LABELS for stance in decision.analysis)
    assert all(stance.label == "neutral" for stance in decision.analysis)
    json.dumps(decision.to_dict())


def test_analysts_survive_inf_and_garbage_features(cfg):
    hostile = {key: float("inf") for key in REQUIRED_FEATURES}
    hostile["rsi_14"] = float("nan")
    hostile["close"] = "not-a-number"  # type: ignore[assignment]
    bars = make_bars(n=120)
    ctx = MarketContext(symbol="BTC-USD", asof=bars[-1].ts, bars=bars, features=hostile)
    stances = run_analysts(ctx, cfg.agents)
    assert len(stances) == 4
    for stance in stances:
        assert stance.label in VALID_LABELS
        assert 0.0 <= stance.confidence <= 1.0


def test_every_stance_label_is_valid_across_regimes(cfg):
    contexts = [
        make_context(features=BULLISH_FEATURES),
        make_context(features=BEARISH_FEATURES),
        make_context(features=CALM_FEATURES),
        make_context(features=STRESSED_FEATURES),
        make_context(features=NAN_FEATURES),
        make_context(n=10),
        make_context(n=80, trend=0.0, vol=0.03, seed=4),
    ]
    for ctx in contexts:
        for stance in run_analysts(ctx, cfg.agents):
            assert stance.label in VALID_LABELS
            assert 0.0 <= stance.confidence <= 1.0


# --------------------------------------------------------------------------- #
# Analysts are genuinely useful, not stubs
# --------------------------------------------------------------------------- #
def test_technical_and_momentum_are_directional(cfg):
    bullish = _stance_map(run_analysts(make_context(features=BULLISH_FEATURES), cfg.agents))
    bearish = _stance_map(run_analysts(make_context(features=BEARISH_FEATURES), cfg.agents))
    assert bullish["technical"].label == "bullish"
    assert bearish["technical"].label == "bearish"
    assert bullish["momentum"].label == "bullish"
    assert bearish["momentum"].label == "bearish"
    # Scores should be meaningfully strong, not 0.11 noise.
    assert bullish["technical"].confidence > 0.6
    assert bearish["momentum"].confidence > 0.6


def test_mean_reversion_fades_extremes(cfg):
    oversold = dict(BEARISH_FEATURES)
    oversold.update({"rsi_14": 20.0, "bb_pctb": 0.02, "dist_from_high_60": -0.25,
                     "dist_from_low_60": 0.01})
    overbought = dict(BULLISH_FEATURES)
    overbought.update({"rsi_14": 80.0, "bb_pctb": 1.05, "dist_from_high_60": -0.005,
                       "dist_from_low_60": 0.60})
    oversold_stance = _stance_map(run_analysts(
        make_context(features=oversold), cfg.agents))["mean_reversion"]
    overbought_stance = _stance_map(run_analysts(
        make_context(features=overbought), cfg.agents))["mean_reversion"]
    assert oversold_stance.label == "bullish"
    assert overbought_stance.label == "bearish"


def test_risk_analyst_tracks_the_volatility_regime(cfg):
    calm = _stance_map(run_analysts(make_context(features=CALM_FEATURES), cfg.agents))["risk"]
    stressed = _stance_map(run_analysts(make_context(features=STRESSED_FEATURES), cfg.agents))["risk"]
    assert calm.label == "bullish"
    assert stressed.label == "bearish"
    assert stressed.score < calm.score


def test_analyst_classes_expose_weights_summing_to_one():
    classes = [TechnicalAnalyst, MomentumAnalyst, MeanReversionAnalyst, RiskAnalyst]
    assert [cls.name for cls in classes] == ["technical", "momentum", "mean_reversion", "risk"]
    assert sum(ANALYST_WEIGHTS.values()) == pytest.approx(1.0)
    assert EVIDENCE_WEIGHTS is ANALYST_WEIGHTS
    for cls in classes:
        assert ANALYST_WEIGHTS[cls.name] == pytest.approx(cls.weight)
        assert cls.description


def test_run_analysts_respects_config_names(cfg):
    cfg.agents.analyst_names = ["risk", "technical"]
    stances = run_analysts(make_context(features=BULLISH_FEATURES), cfg.agents)
    assert [stance.agent for stance in stances] == ["risk", "technical"]

    cfg.agents.analyst_names = ["does_not_exist"]
    stances = run_analysts(make_context(features=BULLISH_FEATURES), cfg.agents)
    assert [stance.agent for stance in stances] == ["technical", "momentum",
                                                    "mean_reversion", "risk"]


def test_run_analysts_skips_llm_when_unavailable(cfg):
    fake = ScriptedLLM([{"label": "bullish", "confidence": 1.0}], available=False)
    stances = run_analysts(make_context(features=BULLISH_FEATURES), cfg.agents, fake)
    assert fake.calls == 0
    assert all(stance.source == "deterministic" for stance in stances)


def test_llm_refinement_is_bounded_and_validated():
    analyst = TechnicalAnalyst()
    ctx = make_context(features=BEARISH_FEATURES)
    base = analyst.assess(ctx)
    assert base.label == "bearish"

    # Confidence can move but only by LLM_CONFIDENCE_DELTA at most.
    overly_bold = ScriptedLLM([{"label": "bullish", "confidence": 1.0, "rationale": "x"}])
    refined = analyst.refine_with_llm(ctx, base, overly_bold)
    assert refined.source == "llm"
    assert refined.label == "bullish"
    assert refined.confidence <= base.confidence + analysts_module.LLM_CONFIDENCE_DELTA + 1e-9

    # Invalid label -> deterministic fallback.
    bad_label = ScriptedLLM([{"label": "STRONG BUY", "confidence": 0.9}])
    assert analyst.refine_with_llm(ctx, base, bad_label) == base

    # Invalid confidence -> deterministic fallback.
    bad_conf = ScriptedLLM([{"label": "bullish", "confidence": "a lot"}])
    assert analyst.refine_with_llm(ctx, base, bad_conf) == base

    # Exploding client -> deterministic fallback.
    assert analyst.refine_with_llm(ctx, base, ExplodingLLM()) == base


# --------------------------------------------------------------------------- #
# Debate: deterministic pooling, LLM only phrases
# --------------------------------------------------------------------------- #
def test_debate_pooling_is_evidence_weighted():
    stances = [
        Stance("technical", "bullish", 0.80, "trend up"),
        Stance("momentum", "bearish", 0.40, "momentum down"),
        Stance("mean_reversion", "neutral", 0.30, "no extreme"),
        Stance("risk", "bullish", 0.50, "calm vol"),
    ]
    # Range regime (all-NaN features) uses the documented base weights.
    range_ctx = make_context(features=NAN_FEATURES)
    expected_base = (0.30 * 0.80 + 0.25 * -0.40 + 0.20 * 0.0 + 0.25 * 0.50) / 1.0
    assert pooled_net_score(stances, range_ctx) == pytest.approx(expected_base)

    # Trend regime reweights the panel; the pooled number must match exactly.
    trend_ctx = make_context(features=BULLISH_FEATURES)
    expected_trend = pooled_net_score(stances, trend_ctx)
    cfg = AppConfig()
    cfg.agents.debate_rounds = 3
    debate = run_debate(trend_ctx, stances, cfg.agents)
    assert debate.net_score == pytest.approx(expected_trend)
    assert debate.consensus == consensus_for(expected_trend)
    assert debate.consensus == "bullish"
    assert debate.rounds == 3
    assert debate.bull.label == "bullish" and debate.bull.confidence > 0
    assert debate.bear.label == "bearish" and debate.bear.confidence > 0


def test_trend_regime_reweights_the_panel():
    names = ["technical", "momentum", "mean_reversion", "risk"]
    stances = [Stance(name, "neutral", 0.30, "flat panel") for name in names]
    range_ctx = make_context(features=NAN_FEATURES)
    trend_ctx = make_context(features=BULLISH_FEATURES)
    bear_ctx = make_context(features=BEARISH_FEATURES)

    assert trend_regime(range_ctx.features) == "range"
    assert trend_regime(trend_ctx.features) == "trend"
    assert trend_regime(bear_ctx.features) == "trend"  # downtrend is a trend too

    range_weights = dict(zip(names, stance_weights(stances, range_ctx)))
    trend_weights = dict(zip(names, stance_weights(stances, trend_ctx)))

    # In a trend the fade analyst is down-weighted, the trend analysts up-weighted.
    assert trend_weights["mean_reversion"] < range_weights["mean_reversion"]
    assert trend_weights["technical"] > range_weights["technical"]
    assert trend_weights["momentum"] > range_weights["momentum"]
    # No ctx -> base weights (keeps direct/stateless callers stable).
    assert stance_weights(stances) == stance_weights(stances, range_ctx)


def test_debate_consensus_thresholds():
    assert consensus_for(CONSENSUS_THRESHOLD) == "bullish"
    assert consensus_for(-CONSENSUS_THRESHOLD) == "bearish"
    assert consensus_for(CONSENSUS_THRESHOLD * 0.5) == "neutral"
    assert consensus_for(1.0) == "bullish"
    assert consensus_for(-1.0) == "bearish"

    bullish = run_debate(make_context(features=BULLISH_FEATURES),
                         run_analysts(make_context(features=BULLISH_FEATURES), AppConfig().agents),
                         AppConfig().agents)
    bearish = run_debate(make_context(features=BEARISH_FEATURES),
                         run_analysts(make_context(features=BEARISH_FEATURES), AppConfig().agents),
                         AppConfig().agents)
    assert bullish.consensus == "bullish" and bullish.net_score > 0
    assert bearish.consensus == "bearish" and bearish.net_score < 0


def test_debate_llm_cannot_change_the_numbers():
    stances = [
        Stance("technical", "bullish", 0.80, "trend up"),
        Stance("momentum", "bearish", 0.40, "momentum down"),
        Stance("mean_reversion", "neutral", 0.30, "no extreme"),
        Stance("risk", "bullish", 0.50, "calm vol"),
    ]
    cfg = AppConfig().agents
    baseline = run_debate(make_context(features=BULLISH_FEATURES), stances, cfg)
    fake = ScriptedLLM([{"bull": "THE BULL CASE", "bear": "THE BEAR CASE"}])
    phrased = run_debate(make_context(features=BULLISH_FEATURES), stances, cfg, fake)

    assert phrased.net_score == pytest.approx(baseline.net_score)
    assert phrased.consensus == baseline.consensus
    assert phrased.bull.rationale == "THE BULL CASE"
    assert phrased.bear.rationale == "THE BEAR CASE"
    assert phrased.bull.source == "llm"
    assert fake.calls == 1


# --------------------------------------------------------------------------- #
# Trader: vol-target sizing and long-only book
# --------------------------------------------------------------------------- #
def test_trader_sizes_by_volatility_budget(cfg):
    low_vol = dict(BULLISH_FEATURES, vol_20=0.20, vol_60=0.20, atr_pct=0.010)
    high_vol = dict(BULLISH_FEATURES, vol_20=0.90, vol_60=0.85, atr_pct=0.050)
    portfolio = make_portfolio()

    # Same debate for both contexts: only volatility differs, so any size
    # difference is the vol-target budget at work.
    stances = [
        Stance("technical", "bullish", 0.80, "trend up"),
        Stance("momentum", "neutral", 0.30, "neutral"),
        Stance("mean_reversion", "neutral", 0.30, "neutral"),
        Stance("risk", "neutral", 0.30, "neutral"),
    ]
    debate = DebateResult(bull=stances[0], bear=Stance("bear_researcher", "bearish", 0.0, "none"),
                          rounds=1, consensus="bullish", net_score=0.50)

    proposals = {}
    for name, features in (("low", low_vol), ("high", high_vol)):
        ctx = make_context(features=features)
        proposals[name] = propose(ctx, debate, list(stances), portfolio, cfg)

    assert proposals["low"].target_weight > proposals["high"].target_weight
    for proposal in proposals.values():
        assert 0.0 <= proposal.target_weight <= 1.0
        assert proposal.action is Action.BUY
        assert proposal.confidence > 0.5
    assert annual_vol(low_vol) == pytest.approx(0.20)
    assert annual_vol(NAN_FEATURES) is None


def test_trader_is_long_only_and_flat_without_signal(cfg):
    ctx = make_context(features=NAN_FEATURES)
    stances = run_analysts(ctx, cfg.agents)
    debate = run_debate(ctx, stances, cfg.agents)
    proposal = propose(ctx, debate, stances, make_portfolio(), cfg)
    assert proposal.action is Action.HOLD
    assert proposal.target_weight == 0.0


def test_bearish_consensus_exits_a_long_position(cfg):
    ctx = make_context(features=BEARISH_FEATURES)
    portfolio = make_portfolio(qty=20.0, price=float(ctx.last_close), avg_price=float(ctx.last_close))
    decision = build_strategy(cfg).decide(ctx, portfolio)
    assert decision.debate is not None and decision.debate.consensus == "bearish"
    assert decision.action is Action.SELL
    assert decision.target_weight == 0.0
    assert decision.risk is not None and decision.risk.approved


def test_bearish_consensus_does_not_short_a_flat_book(cfg):
    decision = build_strategy(cfg).decide(make_context(features=BEARISH_FEATURES), make_portfolio())
    assert decision.action is Action.HOLD
    assert decision.target_weight == 0.0


# --------------------------------------------------------------------------- #
# Deterministic (non-LLM) overflow is clamped by risk too
# --------------------------------------------------------------------------- #
def test_rule_based_oversized_proposal_is_clamped_by_risk(cfg):
    """A low-vol context makes the vol-target sizer propose ~100% exposure."""
    low_vol = dict(BULLISH_FEATURES, vol_20=0.05, vol_60=0.05, atr_pct=0.005)
    decision = build_strategy(cfg).decide(make_context(features=low_vol), make_portfolio())
    assert decision.proposal is not None
    assert decision.proposal.target_weight > cfg.risk.max_position_weight
    assert decision.target_weight <= cfg.risk.max_position_weight + 1e-9
    assert decision.risk is not None
    assert any("max_position_weight" in text for text in decision.risk.adjustments)


def test_pipeline_stop_loss_exits_an_existing_position(cfg):
    ctx = make_context(features=BULLISH_FEATURES)
    price = float(ctx.last_close)
    portfolio = make_portfolio(equity=10_000.0, qty=10.0, price=price,
                               avg_price=price / 0.85)  # ~-15% unrealised
    decision = build_strategy(cfg).decide(ctx, portfolio)
    assert decision.action is Action.SELL
    assert decision.target_weight == 0.0
    assert decision.risk is not None and decision.risk.approved
    assert any("stop-loss" in text.lower() for text in decision.risk.adjustments)


def test_pipeline_cooldown_blocks_reentry_after_stop(cfg):
    strategy = build_strategy(cfg)
    ctx = make_context(features=BULLISH_FEATURES)
    price = float(ctx.last_close)
    losing = make_portfolio(equity=10_000.0, qty=10.0, price=price,
                            avg_price=price / 0.85)
    flat = make_portfolio(price=price)

    first = strategy.decide(ctx, losing)
    assert first.action is Action.SELL
    assert strategy._bars_since_stop[ctx.symbol] == 0

    # Two cooldown bars must refuse fresh exposure...
    for expected_bars_since_stop in (0, 1):
        decision = strategy.decide(ctx, flat)
        assert decision.action is Action.HOLD
        assert decision.target_weight == 0.0
        assert decision.risk is not None and decision.risk.approved is False
        assert any("cooldown" in text.lower() for text in decision.risk.vetoes)
        assert decision.diagnostics["bars_since_stop"] == expected_bars_since_stop

    # ...and the third bar may re-enter.
    reentry = strategy.decide(ctx, flat)
    assert reentry.action is Action.BUY
    assert reentry.target_weight > 0.0


def test_min_bars_for_signals_config_is_honoured(cfg):
    cfg.agents.min_bars_for_signals = 300
    decision = build_strategy(cfg).decide(make_context(features=BULLISH_FEATURES), make_portfolio())
    assert decision.action is Action.HOLD
    assert decision.diagnostics["warmup"] is True
    assert all(stance.label == "neutral" for stance in decision.analysis)


# --------------------------------------------------------------------------- #
# OpenAI-compatible wire format
# --------------------------------------------------------------------------- #
def test_llm_client_uses_openai_compatible_payload(monkeypatch):
    import requests

    monkeypatch.setenv("OPENAI_API_KEY", "sk-wire-test")
    captured: dict = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, payload=json, headers=headers, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient(LLMConfig(enabled="on", require_json=True), available=True)
    assert client.complete_json("system prompt", "user prompt", "{}") == {"ok": True}

    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-wire-test"
    assert captured["payload"]["model"] == client.model
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert captured["payload"]["messages"][1]["role"] == "user"
    assert captured["timeout"] == pytest.approx(client.cfg.timeout_s)


# --------------------------------------------------------------------------- #
# Anti-whipsaw policy: causal smoothing, min holding period, dead-band
# --------------------------------------------------------------------------- #
def test_policy_smoothing_is_causal():
    ctx = make_context(features=BULLISH_FEATURES)
    policy = SmoothedConsensusPolicy(alpha=0.4)
    prefix = SmoothedConsensusPolicy(alpha=0.4)

    first, smoothed_first = policy.update(ctx, _debate_with_score(0.50))
    prefix.update(ctx, _debate_with_score(0.50))
    _, smoothed_second = policy.update(ctx, _debate_with_score(-0.20))
    _, prefix_second = prefix.update(ctx, _debate_with_score(-0.20))

    # First observation seeds the EMA with the raw score.
    assert smoothed_first == pytest.approx(0.50)
    # EMA recurrence, using only current and past bars (no lookahead).
    assert smoothed_second == pytest.approx(0.4 * -0.20 + 0.6 * 0.50)
    assert prefix_second == pytest.approx(smoothed_second)

    # Extending the sequence must not change earlier smoothed values.
    _, smoothed_third = policy.update(ctx, _debate_with_score(0.40))
    assert smoothed_third == pytest.approx(0.4 * 0.40 + 0.6 * smoothed_second)
    assert policy.state(ctx.symbol).smoothed == pytest.approx(smoothed_third)


def test_min_hold_bars_is_respected():
    policy = SmoothedConsensusPolicy(alpha=0.35)
    up_ctx = make_context(features=BULLISH_FEATURES)
    down_ctx = make_context(features=BEARISH_FEATURES)

    info, _ = policy.update(up_ctx, _debate_with_score(0.50), min_hold_bars=10)
    assert info["policy_direction"] == 1

    exit_bar = None
    for bar in range(1, 30):
        info, _ = policy.update(down_ctx, _debate_with_score(-0.20), min_hold_bars=10)
        if info["policy_direction"] == 0:
            exit_bar = bar
            break
        # Regime is broken, but the minimum holding period keeps the position on.
        assert info["regime_break"] is True
        assert info["bars_in_direction"] == bar
    assert exit_bar == 10


def test_hard_exit_overrides_min_hold():
    policy = SmoothedConsensusPolicy(alpha=0.35)
    up_ctx = make_context(features=BULLISH_FEATURES)
    down_ctx = make_context(features=BEARISH_FEATURES)
    policy.update(up_ctx, _debate_with_score(0.50), min_hold_bars=10)

    exit_bar = None
    for bar in range(1, 10):
        info, _ = policy.update(down_ctx, _debate_with_score(-0.90), min_hold_bars=10)
        if info["policy_direction"] == 0:
            exit_bar = bar
            break
    assert exit_bar is not None and exit_bar < 10
    assert info["policy_hard_exit"] is True


def test_single_mean_reversion_day_cannot_flip_a_confirmed_uptrend(cfg):
    strategy = build_strategy(cfg)
    up_ctx = make_context(features=BULLISH_FEATURES)
    price = float(up_ctx.last_close)
    portfolio = make_portfolio(price=price)

    first = strategy.decide(up_ctx, portfolio)
    assert first.action is Action.BUY
    assert first.diagnostics["policy_state"] == "long"
    portfolio = make_portfolio(qty=20.0, price=price, avg_price=price)
    for _ in range(3):  # establish the long regime across several bars
        decision = strategy.decide(up_ctx, portfolio)
        assert decision.diagnostics["policy_state"] == "long"

    # One strongly bearish (mean-reversion-ish) day must NOT flip the confirmed
    # uptrend: the smoothed score stays positive, the policy stays long, and risk
    # has no stop to force.  A partial trim is allowed; a flip to flat is not.
    down_ctx = make_context(features=BEARISH_FEATURES)
    down_price = float(down_ctx.last_close)
    portfolio = make_portfolio(qty=20.0, price=down_price, avg_price=down_price)
    decision = strategy.decide(down_ctx, portfolio)
    assert decision.diagnostics["policy_state"] == "long"
    assert decision.target_weight > 0.0
    assert decision.risk is not None and decision.risk.target_weight > 0.0


def test_policy_state_is_per_symbol():
    policy = SmoothedConsensusPolicy()
    btc = make_context(symbol="BTC-USD", features=BULLISH_FEATURES)
    eth = make_context(symbol="ETH-USD", features=BEARISH_FEATURES)
    policy.update(btc, _debate_with_score(0.50))
    info_eth, _ = policy.update(eth, _debate_with_score(-0.50))
    assert policy.state("BTC-USD").direction == 1
    assert info_eth["policy_direction"] == 0


def test_min_hold_bars_config_override(cfg):
    cfg.agents.min_hold_bars = 12  # Lead-owned field once added; getattr keeps this safe
    decision = build_strategy(cfg).decide(make_context(features=BULLISH_FEATURES), make_portfolio())
    assert decision.diagnostics["min_hold_bars"] == 12
    assert decision.diagnostics["min_hold_bars"] != DEFAULT_MIN_HOLD_BARS


def test_deadband_ignores_tiny_rebalances(cfg):
    from agentic_trading.agents.risk import current_weight
    from agentic_trading.agents.trader import propose as trader_propose

    ctx = make_context(features=BULLISH_FEATURES, meta={
        "policy_direction": 1, "policy_net_score": 0.20,
    })
    price = float(ctx.last_close)
    equity = 10_000.0
    portfolio = make_portfolio(equity=equity, qty=0.10 * equity / price, price=price,
                               avg_price=price)
    current = current_weight(portfolio, ctx.symbol, price)
    stances = run_analysts(ctx, cfg.agents)
    debate = _debate_with_score(0.20)
    proposal = trader_propose(ctx, debate, stances, portfolio, cfg)
    assert proposal.action is Action.HOLD
    assert proposal.target_weight == pytest.approx(current)


def test_llm_cannot_soften_a_deterministic_sell(cfg):
    ctx = make_context(features=BEARISH_FEATURES, meta={
        "policy_direction": 0, "policy_net_score": -0.10,
    })
    price = float(ctx.last_close)
    portfolio = make_portfolio(equity=10_000.0, qty=20.0, price=price, avg_price=price)
    from agentic_trading.agents.trader import propose as trader_propose
    stances = run_analysts(ctx, cfg.agents)
    debate = _debate_with_score(-0.10)
    fake = ScriptedLLM([{"action": "hold", "thesis": "stay in"}])
    proposal = trader_propose(ctx, debate, stances, portfolio, cfg, fake)
    # A HOLD response to a SELL would add risk, so the one-way ratchet refuses it.
    assert proposal.action is Action.SELL
    assert proposal.target_weight == 0.0


# --------------------------------------------------------------------------- #
# Architecture guards (the rule is only real if it is enforced in code)
# --------------------------------------------------------------------------- #
def test_risk_gate_takes_no_llm_input():
    parameters = list(inspect.signature(risk_module.evaluate).parameters)
    assert parameters == ["proposal", "ctx", "portfolio", "cfg"]
    source = inspect.getsource(risk_module)
    assert "complete_json" not in source
    assert "LLMClient" not in source


def test_agents_do_not_compute_indicators():
    for module in (analysts_module, debate_module, trader_module, risk_module,
                   pipeline_module, llm_module):
        source = inspect.getsource(module)
        assert "pct_change" not in source
        assert "rolling(" not in source
        assert "ewm(" not in source
        assert "import pandas" not in source


def test_pipeline_decision_survives_extreme_portfolio_states(cfg):
    ctx = make_context(features=BULLISH_FEATURES)
    portfolios = [
        make_portfolio(equity=0.0, cash=0.0),
        make_portfolio(equity=-100.0, cash=0.0),
        make_portfolio(equity=1.0, qty=10_000.0, price=100.0),
        make_portfolio(equity=10_000.0, qty=-5.0, price=100.0),
    ]
    for portfolio in portfolios:
        decision = build_strategy(cfg).decide(ctx, portfolio)
        assert isinstance(decision, PortfolioDecision)
        assert math.isfinite(decision.target_weight)
        assert decision.target_weight >= 0.0
        assert decision.target_weight <= cfg.risk.max_position_weight + 1e-9
