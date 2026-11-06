"""Bull/bear research debate: deterministic pooling, optional LLM phrasing.

The debate is the *aggregation step*.  Its numbers are closed-form and
deterministic:

    net_score = sum_i(weight_i * stance_i.score) / sum_i(weight_i)

with per-analyst evidence weights (``analysts.EVIDENCE_WEIGHTS``).  ``Stance.score``
is the signed conviction in ``[-1, 1]`` (neutral == 0), so disagreement between
analysts mechanically shrinks the net score -- a house divided cannot trade a
full-size position.

Consensus is a threshold on ``net_score``: ``|net_score| < CONSENSUS_THRESHOLD``
is neutral.  The LLM, when available, may only *phrase* the bull and bear
arguments (rationale text); it can never change ``net_score``, ``consensus`` or
the derived ``DebateResult.stance``.  That separation is deliberate: research
chatter is not allowed to leak into sizing inputs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from ..config import AgentConfig
from ..contracts import DebateResult, MarketContext, Stance
from .analysts import EVIDENCE_WEIGHTS, _blend, _feature_notes, _finite

__all__ = [
    "CONSENSUS_THRESHOLD",
    "DEFAULT_STANCE_WEIGHT",
    "EVIDENCE_WEIGHTS",
    "REGIME_WEIGHT_MULTIPLIERS",
    "TREND_NEUTRAL_BAND",
    "consensus_for",
    "pooled_net_score",
    "run_debate",
    "stance_weights",
    "trend_regime",
]

#: |net_score| at or above this is a directional consensus.  Chosen so that a
#: single strong analyst (e.g. a 0.9-confidence technical call weighted 0.30)
#: cannot by itself declare consensus, while a trend confirmed by momentum can.
CONSENSUS_THRESHOLD = 0.15

#: Weight for an analyst the debate does not know about (defensive default).
DEFAULT_STANCE_WEIGHT = 0.15

#: ``|price_vs_sma200|`` below this counts as "range", above it as a trend.
TREND_NEUTRAL_BAND = 0.02

#: Regime-aware evidence multipliers (normalisation happens implicitly in the
#: weighted mean).  Inside a trend -- up *or* down -- the regime-aligned analysts
#: carry the argument and the counter-trend mean-reversion view is deliberately
#: halved; in a range the base weights apply.  This is what stops a mean-reversion
#: day from drowning out a confirmed trend, while keeping the fade analyst
#: relevant exactly where fading is the edge.
REGIME_WEIGHT_MULTIPLIERS: dict[str, dict[str, float]] = {
    "trend": {"technical": 1.30, "momentum": 1.20, "mean_reversion": 0.50, "risk": 1.00},
    "range": {"technical": 1.00, "momentum": 1.00, "mean_reversion": 1.00, "risk": 1.00},
}


def trend_regime(features: Any) -> str:
    """Classify the current bar as ``"trend"`` or ``"range"``, causally.

    A trend is "confirmed" when either the SMA stack is fully aligned
    (``trend_up``) or price sits at least ``TREND_NEUTRAL_BAND`` away from a
    consistent side of SMA200.  Missing/NaN features -> ``"range"`` (base
    weights), which is the safe default during warmup.
    """
    if not features:
        return "range"
    trend_up = _finite(features.get("trend_up"))
    above200 = _finite(features.get("above_sma200"))
    price_vs_200 = _finite(features.get("price_vs_sma200"))
    if trend_up is not None and trend_up >= 0.5:
        return "trend"
    if above200 is not None and price_vs_200 is not None:
        if above200 >= 0.5 and price_vs_200 >= TREND_NEUTRAL_BAND:
            return "trend"
        if above200 < 0.5 and price_vs_200 <= -TREND_NEUTRAL_BAND:
            return "trend"
    return "range"

_LLM_TEXT_MAX = 500
_LLM_EVIDENCE_MAX = 500


def stance_weights(stances: Sequence[Stance], ctx: MarketContext | None = None) -> list[float]:
    """Evidence weights aligned with ``stances`` (unknown agents get a default).

    When ``ctx`` is supplied the base weights are multiplied by the current
    regime's multipliers (see :data:`REGIME_WEIGHT_MULTIPLIERS`).  The weights
    are relative; the pooled mean divides by their sum.
    """
    multipliers = REGIME_WEIGHT_MULTIPLIERS[trend_regime(ctx.features) if ctx is not None else "range"]
    return [
        float(EVIDENCE_WEIGHTS.get(stance.agent, DEFAULT_STANCE_WEIGHT))
        * float(multipliers.get(stance.agent, 1.0))
        for stance in stances
    ]


def pooled_net_score(stances: Sequence[Stance], ctx: MarketContext | None = None) -> float:
    """Weighted mean of signed stance scores, clipped to ``[-1, 1]``."""
    pairs = [(float(stance.score), weight)
             for stance, weight in zip(stances, stance_weights(stances, ctx))]
    pooled = _blend(pairs)
    if pooled is None:
        return 0.0
    return max(-1.0, min(1.0, pooled))


def consensus_for(net_score: float, threshold: float = CONSENSUS_THRESHOLD) -> str:
    """Map a pooled score to the frozen ``"bullish" | "bearish" | "neutral"`` label."""
    if net_score >= threshold:
        return "bullish"
    if net_score <= -threshold:
        return "bearish"
    return "neutral"


def _case(
    agent: str,
    label: str,
    stances: Sequence[Stance],
    direction: float,
    weights: Sequence[float],
) -> Stance:
    """Aggregate all analysts arguing for one side into a single case Stance."""
    selected = [(stance, weight) for stance, weight in zip(stances, weights)
                if stance.score * direction > 0]
    if not selected:
        return Stance(
            agent=agent,
            label=label,
            confidence=0.0,
            rationale=f"no {label} evidence from the analyst panel",
            evidence=(),
            source="deterministic",
        )
    total_weight = sum(weight for _, weight in selected)
    confidence = sum(weight * abs(stance.score) for stance, weight in selected) / total_weight
    confidence = max(0.0, min(1.0, confidence))
    rationale = " | ".join(
        f"{stance.agent}: {stance.rationale}" for stance, _ in selected
    )[:_LLM_EVIDENCE_MAX]
    evidence = tuple(f"{stance.agent} ({stance.confidence:.2f})" for stance, _ in selected)
    return Stance(
        agent=agent,
        label=label,
        confidence=confidence,
        rationale=rationale,
        evidence=evidence,
        source="deterministic",
    )


def _debate_prompt(ctx: MarketContext, stances: Sequence[Stance], bull: Stance, bear: Stance) -> str:
    panel = "\n".join(
        f"- {stance.agent}: {stance.label} (confidence {stance.confidence:.2f}) "
        f"-- {stance.rationale}"
        for stance in stances
    ) or "- (no analyst stances)"
    features = _feature_notes(ctx.features, (
        "close", "ret_20", "rsi_14", "atr_pct", "vol_20", "trend_up",
    ))
    return (
        f"Symbol {ctx.symbol} on {ctx.timeframe} bars, asof {ctx.asof.isoformat()}.\n"
        f"Key features: {features}\n"
        f"Analyst panel:\n{panel}\n"
        f"Deterministic bull case (strength {bull.confidence:.2f}): {bull.rationale}\n"
        f"Deterministic bear case (strength {bear.confidence:.2f}): {bear.rationale}\n"
        "Sharpen the two arguments. Do not change the stance labels or confidences "
        "and do not invent indicators or news."
    )


def _llm_phrase(
    ctx: MarketContext,
    stances: Sequence[Stance],
    bull: Stance,
    bear: Stance,
    cfg: AgentConfig,
    llm: Any,
    rounds: int,
) -> tuple[Stance, Stance]:
    """Optional per-round LLM phrasing of the bull/bear arguments."""
    if llm is None or not bool(getattr(llm, "available", False)):
        return bull, bear
    system = (
        "You are the research-desk moderator of an automated trading system. "
        "You phrase the bull and bear arguments; you do not decide trades. "
        "Answer with ONE JSON object and nothing else."
    )
    schema = '{"bull": "<the strongest long argument>", "bear": "<the strongest short argument>"}'
    for _ in range(max(1, int(rounds))):
        try:
            data = llm.complete_json(system, _debate_prompt(ctx, stances, bull, bear), schema)
        except Exception:
            break
        if not isinstance(data, dict):
            break
        changed = False
        new_bull_text = data.get("bull")
        if isinstance(new_bull_text, str) and new_bull_text.strip():
            bull = replace(bull, rationale=new_bull_text.strip()[:_LLM_TEXT_MAX], source="llm")
            changed = True
        new_bear_text = data.get("bear")
        if isinstance(new_bear_text, str) and new_bear_text.strip():
            bear = replace(bear, rationale=new_bear_text.strip()[:_LLM_TEXT_MAX], source="llm")
            changed = True
        if not changed:
            break
    return bull, bear


def run_debate(
    ctx: MarketContext,
    stances: Sequence[Stance],
    cfg: AgentConfig,
    llm: Any = None,
) -> DebateResult:
    """Pool the analyst panel into a :class:`DebateResult`.

    Deterministic guarantees (independent of any LLM):

    * ``net_score`` is the evidence-weighted mean of ``stance.score``;
    * ``consensus`` is a threshold on that score;
    * ``DebateResult.stance`` confidence is ``|net_score|``.

    The optional LLM only rewrites the bull/bear ``rationale`` text.
    """
    panel = list(stances or ())
    weights = stance_weights(panel, ctx)
    net_score = pooled_net_score(panel, ctx)
    consensus = consensus_for(net_score)
    rounds = max(1, int(getattr(cfg, "debate_rounds", 1) or 1))

    bull = _case("bull_researcher", "bullish", panel, direction=1.0, weights=weights)
    bear = _case("bear_researcher", "bearish", panel, direction=-1.0, weights=weights)
    bull, bear = _llm_phrase(ctx, panel, bull, bear, cfg, llm, rounds)

    return DebateResult(
        bull=bull,
        bear=bear,
        rounds=rounds,
        consensus=consensus,
        net_score=net_score,
    )
