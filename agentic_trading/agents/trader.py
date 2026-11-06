"""Trader agent: turn the research debate into a *sized* trade proposal.

Sizing logic (deterministic, vol-targeted)
------------------------------------------
The trader runs after the debate and before risk.  It owns *direction* and
*size*, never authority: its output is a :class:`TradeProposal` which
``agents/risk.py`` may clamp, veto or override.

1. **Direction** comes from the debate consensus (a threshold on ``net_score``).
2. **Size** comes from a volatility budget.  A position's risk contribution is
   roughly ``weight * realised_vol``; we target
   ``max_position_weight * vol_target_annual`` of equity-at-risk and solve for
   weight::

       raw_weight = (max_position_weight * vol_target_annual) / annualised_vol

   so a 20%-vol asset gets a larger weight than an 80%-vol asset for the same
   risk budget.  ``annualised_vol`` is read from ``ctx.features['vol_20']``
   (falling back to ``vol_60``); it is already annualised by the data layer.
3. **Conviction scaling** shrinks size when the debate is lukewarm:
   ``scale = clip(|net_score| / 0.55, 0.20, 1.0)``.
4. The result is clamped to ``[0, 1]`` (the ``TradeProposal`` contract).  It may
   still exceed ``max_position_weight`` -- risk.py is what enforces that; the
   journal records both numbers so the clamp is visible.
5. A **bearish** consensus moves a long-only book to ``target_weight = 0``.
   The action is derived from the *change*: buy if the target is above current
   exposure, sell if below, hold inside a 1% dead-band.

LLM refinement
--------------
When an available LLM is supplied it may only (a) switch the deterministic
action to ``hold`` (a timing veto that de-risks), and (b) rewrite the thesis /
invalidation text.  It can never initiate, reverse or size a trade: the
deterministic debate owns direction and the volatility budget owns size.  If
``cfg.llm.allow_weight_override`` is explicitly enabled the model may *suggest* a
long target weight, but it is clamped to ``[0, 1]`` here and to
``max_position_weight`` by risk.py -- the limit is unavoidable either way.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..config import AppConfig
from ..contracts import (
    Action,
    DebateResult,
    MarketContext,
    PortfolioState,
    Stance,
    TradeProposal,
)
from .analysts import ANALYST_WEIGHTS
from .debate import CONSENSUS_THRESHOLD
from .risk import current_weight

__all__ = [
    "CONVICTION_FULL_NET",
    "annual_vol",
    "conviction_scale",
    "propose",
]

#: |net_score| at which conviction scaling reaches 1.0 (full size for the vol budget).
CONVICTION_FULL_NET = 0.55
#: A modest but persistent edge must still buy a real position: even the weakest
#: actionable conviction gets CONVICTION_FLOOR of the volatility budget (risk.py
#: still caps the result at ``max_position_weight``).
CONVICTION_FLOOR = 0.35
_ACTION_DEADBAND = 0.01
#: Deliberate rebalance dead-band: a target change smaller than this fraction of
#: the current/target weight is a HOLD.  0.30 is the no-trade band from the
#: regime-window evaluation: on crypto-scale daily vol it absorbs 3-5pp of
#: weight noise (a full position only resizes on a genuine regime change) and
#: cut round-trips roughly in half versus the engine's 0.5%-of-equity epsilon
#: alone, without raising drawdown.  The engine adds its own execution epsilon.
_REBALANCE_BAND_REL = 0.30
_MIN_REBALANCE_BAND = 0.01
#: Optional target-weight grid (fraction of equity).  0 disables.  Bucketing
#: turns a continuously-moving target into a few stable levels, which is a
#: standard way to avoid re-trading a position on every score wobble.
TARGET_WEIGHT_STEP = 0.0
_THESIS_MAX = 700


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def annual_vol(features: Mapping[str, float] | None) -> float | None:
    """Realised annualised volatility from features (``vol_20`` then ``vol_60``)."""
    if not features:
        return None
    for key in ("vol_20", "vol_60"):
        value = _finite(features.get(key))
        if value is not None and value > 0:
            return value
    return None


def conviction_scale(net_score: float) -> float:
    """Map debate conviction to a position multiplier in ``[0, 1]``."""
    conviction = abs(_finite(net_score) or 0.0)
    if conviction <= 0:
        return 0.0
    return _clamp(conviction / CONVICTION_FULL_NET, CONVICTION_FLOOR, 1.0)


def _weighted_agreement(stances: Sequence[Stance], direction: int) -> float:
    """Fraction of evidence weight that agrees with ``direction``."""
    total = 0.0
    agree = 0.0
    for stance in stances:
        weight = float(ANALYST_WEIGHTS.get(stance.agent, 0.15))
        total += weight
        if direction > 0 and stance.score > 0:
            agree += weight
        elif direction < 0 and stance.score < 0:
            agree += weight
    if total <= 0:
        return 0.5
    return _clamp(agree / total, 0.0, 1.0)


def _confidence(net_score: float, stances: Sequence[Stance], direction: int) -> float:
    """Trader confidence fed to the risk gate.

    The confidence gate in risk.py uses this number, so it is deliberately a
    *deterministic* function: 30% base + 50% debate conviction + 20% panel
    agreement.  A lukewarm, split debate therefore cannot clear
    ``min_confidence`` and no trade happens.
    """
    conviction = abs(_finite(net_score) or 0.0)
    agreement = 0.5 if direction == 0 else _weighted_agreement(stances, direction)
    return _clamp(0.30 + 0.50 * conviction + 0.20 * agreement, 0.0, 0.95)


def _price(ctx: MarketContext, portfolio: PortfolioState) -> float:
    try:
        if ctx.bars:
            value = float(ctx.bars[-1].close)
            if math.isfinite(value) and value > 0:
                return value
    except Exception:
        pass
    try:
        value = float(portfolio.price(ctx.symbol))
        if math.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    return 0.0


def _vol_target_weight(ctx: MarketContext, risk_cfg: Any) -> float:
    """Raw (pre-conviction) weight that spends the configured volatility budget."""
    max_weight = float(risk_cfg.max_position_weight)
    budget = max_weight * float(risk_cfg.vol_target_annual)
    realised = annual_vol(ctx.features)
    if realised is None or realised <= 0:
        # Unknown volatility (early warmup): half the maximum, risk still clamps.
        return 0.5 * max_weight
    return budget / max(realised, 0.05)


def _thesis(
    debate: DebateResult,
    stances: Sequence[Stance],
    target: float,
    current: float,
) -> str:
    parts = [
        f"{debate.consensus} consensus (net {debate.net_score:+.2f}); "
        f"target {target:.0%} vs current {current:.0%}"
    ]
    ranked = sorted(stances, key=lambda stance: abs(stance.score), reverse=True)[:2]
    for stance in ranked:
        if stance.rationale:
            parts.append(
                f"{stance.agent} {stance.label} {stance.confidence:.2f}: {stance.rationale}"
            )
    return " | ".join(parts)[:_THESIS_MAX]


def _invalidation(ctx: MarketContext, action: Action, price: float) -> str:
    atr = _finite(ctx.features.get("atr_14")) if ctx.features else None
    if atr is not None and atr > 0 and price > 0:
        if action == Action.BUY:
            return f"close below {price - 2.0 * atr:.2f} (~2x ATR-14) invalidates the entry"
        if action == Action.SELL:
            return f"close above {price + 2.0 * atr:.2f} (~2x ATR-14) invalidates the exit"
    return "thesis invalidated if the debate net score flips sign"


def _allowed_llm_actions(action: Action) -> set[str]:
    """The LLM may only tighten: keep or cancel a buy; never soften a sell.

    A ``hold`` response to a BUY is a timing veto (less risk).  A ``hold``
    response to a SELL would *keep* exposure -- more risk -- so it is refused.
    """
    if action == Action.BUY:
        return {Action.BUY.value, Action.HOLD.value}
    if action == Action.SELL:
        return {Action.SELL.value}
    return {Action.HOLD.value}


def _refine_with_llm(
    ctx: MarketContext,
    debate: DebateResult,
    stances: Sequence[Stance],
    cfg: AppConfig,
    llm: Any,
    action: Action,
    target: float,
    current: float,
    confidence: float,
    thesis: str,
    invalidation: str,
    equity: float,
) -> tuple[Action, float, str, str, bool]:
    """One bounded LLM refinement.  Returns the (possibly) updated proposal fields."""
    allowed = _allowed_llm_actions(action)
    system = (
        "You are the trader agent of a deterministic-guarded trading system. A "
        "deterministic engine has already chosen the direction, the size and the "
        "confidence; you may only voice a timing veto (choose 'hold') and rewrite "
        "the thesis/invalidation. Never invent prices or indicators. Answer with "
        "ONE JSON object and nothing else."
    )
    ranked = sorted(stances, key=lambda stance: abs(stance.score), reverse=True)[:3]
    evidence = "\n".join(
        f"- {stance.agent}: {stance.label} (confidence {stance.confidence:.2f}) -- {stance.rationale}"
        for stance in ranked
    ) or "- (no stances)"
    schema = '{"action": "buy|sell|hold", "thesis": "<one paragraph>", "invalidation": "<price or condition>"}'
    if bool(getattr(cfg.llm, "allow_weight_override", False)):
        schema = ('{"action": "buy|sell|hold", "target_weight": <float 0..1>, '
                  '"thesis": "<one paragraph>", "invalidation": "<price or condition>"}')
    user = (
        f"Symbol {ctx.symbol} on {ctx.timeframe} bars, asof {ctx.asof.isoformat()}. "
        f"Last close {_finite(ctx.bars[-1].close) if ctx.bars else None}.\n"
        f"Debate: {debate.consensus} consensus, net score {debate.net_score:+.2f}.\n"
        f"Portfolio: current weight {current:.2%}, equity {equity:.2f}.\n"
        f"Deterministic proposal: action={action.value}, target_weight={target:.2%}, "
        f"confidence={confidence:.2f}.\n"
        f"Allowed actions for you: {sorted(allowed)}.\n"
        f"Top evidence:\n{evidence}\n"
        f"Current thesis: {thesis}\n"
        "You may only de-risk (choose hold) or improve the wording. Do not reverse "
        "or increase exposure."
    )
    try:
        data = llm.complete_json(system, user, schema)
    except Exception:
        return action, target, thesis, invalidation, False
    if not isinstance(data, dict):
        return action, target, thesis, invalidation, False

    new_action = action
    new_target = target
    new_thesis = thesis
    new_invalidation = invalidation
    changed = False

    raw_action = data.get("action")
    if isinstance(raw_action, str):
        try:
            parsed = Action.parse(raw_action)
        except Exception:
            parsed = None
        if parsed is not None and parsed.value in allowed:
            if parsed != action:
                changed = True
            new_action = parsed

    if new_action == Action.HOLD:
        new_target = current
    elif (bool(getattr(cfg.llm, "allow_weight_override", False))
          and new_action == Action.BUY and target > 0):
        suggested = _finite(data.get("target_weight"))
        if suggested is not None:
            clamped = _clamp(suggested, 0.0, 1.0)
            if abs(clamped - target) > 1e-9:
                changed = True
            new_target = clamped

    candidate = data.get("thesis")
    if isinstance(candidate, str) and candidate.strip():
        new_thesis = candidate.strip()[:_THESIS_MAX]
        changed = True
    candidate = data.get("invalidation")
    if isinstance(candidate, str) and candidate.strip():
        new_invalidation = candidate.strip()[:300]
        changed = True

    return new_action, new_target, new_thesis, new_invalidation, changed


def propose(
    ctx: MarketContext,
    debate: DebateResult,
    stances: Sequence[Stance],
    portfolio: PortfolioState,
    cfg: AppConfig,
    llm: Any = None,
) -> TradeProposal:
    """Build the sized trade proposal for one bar.

    Deterministic first; the optional LLM may only tighten the action or improve
    the narrative.  The caller (pipeline) always hands this to ``risk.evaluate``.
    """
    risk_cfg = cfg.risk
    symbol = ctx.symbol
    price = _price(ctx, portfolio)
    current = current_weight(portfolio, symbol, price)

    net_score = _clamp(_finite(debate.net_score) or 0.0, -1.0, 1.0)
    consensus = debate.consensus if debate.consensus in (
        "bullish", "bearish", "neutral") else "neutral"
    direction = 1 if consensus == "bullish" else (-1 if consensus == "bearish" else 0)

    # The pipeline may inject its stateful policy view (causal EMA + hysteresis)
    # through ctx.meta.  When present it owns the entry/exit decision; the trader
    # only sizes.  When absent (direct calls / tests) the raw debate is used.
    policy_direction = _finite(ctx.meta.get("policy_direction")) if ctx.meta else None
    policy_score = _finite(ctx.meta.get("policy_net_score")) if ctx.meta else None

    target = current
    if policy_direction is not None:
        direction = 1 if policy_direction > 0 else 0
        score = net_score if policy_score is None else _clamp(policy_score, -1.0, 1.0)
        if direction == 0:
            # Long-only policy says flat; this also exits an existing long.
            target = 0.0
        elif score >= CONSENSUS_THRESHOLD:
            target = _clamp(_vol_target_weight(ctx, risk_cfg) * conviction_scale(score), 0.0, 1.0)
        else:
            # Long policy with a lukewarm/negative score: keep the position and
            # do not add.  (A negative smoothed score inside an intact primary
            # uptrend is noise -- the regime gate decides the exit.)
            target = current
    else:
        if direction > 0 and abs(net_score) >= CONSENSUS_THRESHOLD:
            scale = conviction_scale(net_score)
            target = _clamp(_vol_target_weight(ctx, risk_cfg) * scale, 0.0, 1.0)
        elif direction < 0:
            # Long-only book: a bearish consensus means flat, never short.
            target = 0.0
        score = net_score

    if TARGET_WEIGHT_STEP > 0 and target > 0:
        target = round(target / TARGET_WEIGHT_STEP) * TARGET_WEIGHT_STEP
    # Deliberate rebalance dead-band: ignore changes too small to be worth costs.
    band = max(_MIN_REBALANCE_BAND, _REBALANCE_BAND_REL * max(abs(target), abs(current)))
    if abs(target - current) <= band:
        target = current

    delta = target - current
    if delta > _ACTION_DEADBAND:
        action = Action.BUY
    elif delta < -_ACTION_DEADBAND:
        action = Action.SELL
    else:
        action = Action.HOLD
        target = current

    confidence = _confidence(score, stances, direction)
    thesis = _thesis(debate, stances, target, current)
    if policy_direction is not None:
        state = "long" if direction > 0 else "flat"
        thesis = f"policy {state} (smoothed {score:+.2f}); {thesis}"[:_THESIS_MAX]
    invalidation = _invalidation(ctx, action, price)
    proposed_by = "trader"

    if llm is not None and bool(getattr(llm, "available", False)):
        refined = _refine_with_llm(
            ctx, debate, stances, cfg, llm,
            action, target, current, confidence, thesis, invalidation,
            float(portfolio.equity or 0.0),
        )
        if refined is not None:
            action, target, thesis, invalidation, changed = refined
            if changed:
                proposed_by = "trader+llm"

    return TradeProposal(
        symbol=symbol,
        action=action,
        target_weight=_clamp(target, 0.0, 1.0),
        confidence=confidence,
        thesis=thesis,
        invalidation=invalidation,
        horizon_days=5,
        proposed_by=proposed_by,
    )
