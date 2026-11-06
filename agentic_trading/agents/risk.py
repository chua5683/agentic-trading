"""Deterministic risk gate -- the only component allowed to dispose of a trade.

Architectural invariant
-----------------------
``evaluate`` is a **pure function** of ``(proposal, ctx, portfolio, RiskConfig)``.
It has no LLM parameter, it never imports the LLM client, and it reads no global
state: no language-model output can influence *whether* a limit is enforced.
Whatever the trader (or a future strategy, or an over-eager LLM) proposes, the
returned ``target_weight`` obeys the configured bounds.  This is what makes the
"LLM may propose, deterministic code may dispose" claim testable.

Rules, applied in this order
----------------------------
1. **Data sanity** -- no closed bars / invalid price / non-positive equity:
   veto and stay flat.
2. **Position-protection exits** (highest priority, even against a HOLD
   proposal): stop-loss, take-profit, time stop.  These force
   ``target_weight = 0`` and are approved so the engine closes the position.
3. **Confidence gate** -- a proposal that would *add* exposure needs
   ``proposal.confidence >= min_confidence``.  De-risking is always allowed
   (you never need confidence to sell).
4. **Drawdown stop** -- while ``portfolio.drawdown >= max_drawdown_stop`` no new
   exposure may be added; reducing risk is still allowed.
5. **Cooldown** -- after a stop-out, new exposure is blocked for
   ``cooldown_bars_after_stop`` bars.  The bar count comes from
   ``ctx.meta['bars_since_stop']`` (or ``ctx.meta['stopped_out'] is True``); the
   pipeline fills it in from its own counter when the engine does not provide it.
6. **Hard clamps** -- ``max_position_weight`` ceiling always; the floor is 0
   unless ``allow_short`` is true (in which case ``-max_position_weight``).
   With the shipped config (``allow_short=False``) the result is always in
   ``[0, max_position_weight]``.
7. **Turnover cap** -- ``|target_weight - current_weight| <= max_daily_turnover``
   per bar.
8. **Stops attached** -- for an approved non-flat target, absolute stop-loss and
   take-profit prices are derived from the entry price (existing position's
   ``avg_price``) or the current close for a new entry.

Every rule that fires appends a human-readable string to ``vetoes`` (hard block)
or ``adjustments`` (soft change), so the decision journal explains itself.
"""

from __future__ import annotations

import math
from typing import Any

from ..config import RiskConfig
from ..contracts import Action, MarketContext, PortfolioState, RiskDecision, TradeProposal

__all__ = ["current_weight", "evaluate"]

_EPS = 1e-9


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


def _increases(target: float, current: float) -> bool:
    """True when ``target`` adds risk relative to ``current`` (signed exposure)."""
    return abs(target) > abs(current) + _EPS


def current_weight(portfolio: PortfolioState, symbol: str, price: float | None = None) -> float:
    """Current exposure of ``symbol`` as a fraction of equity.

    ``portfolio.weight()`` relies on ``last_prices``; this helper prefers the
    authoritative decision-bar price (``ctx.last_close``) so agents, the trader
    and the risk gate all compute exposure identically.
    """
    try:
        equity = float(portfolio.equity or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(equity) or equity <= 0:
        return 0.0
    try:
        qty = float(portfolio.qty(symbol))
    except Exception:
        qty = 0.0
    px = _finite(price)
    if px is None or px <= 0:
        try:
            px = _finite(portfolio.price(symbol))
        except Exception:
            px = None
    if px is None or px <= 0:
        return 0.0
    weight = qty * px / equity
    if not math.isfinite(weight):
        return 0.0
    return _clamp(weight, -1.0, 1.0)


def _forced_exit_reason(
    ctx: MarketContext,
    portfolio: PortfolioState,
    symbol: str,
    price: float,
    cfg: RiskConfig,
) -> str | None:
    """Position-protection rules that fire even when the proposal says HOLD."""
    position = portfolio.positions.get(symbol) if portfolio.positions else None
    if position is None:
        return None
    try:
        qty = float(position.qty)
        avg_price = float(position.avg_price)
    except (TypeError, ValueError):
        return None
    if abs(qty) < 1e-12 or not math.isfinite(avg_price) or avg_price <= 0:
        return None
    pnl = position.unrealised_pct(price)
    long_position = qty > 0

    if long_position and pnl <= -float(cfg.stop_loss_pct):
        return (f"stop-loss triggered: unrealised {pnl:+.2%} breaches "
                f"-{cfg.stop_loss_pct:.2%}; target forced to 0")
    if (not long_position) and pnl >= float(cfg.stop_loss_pct):
        return (f"stop-loss triggered on short: unrealised {pnl:+.2%} breaches "
                f"+{cfg.stop_loss_pct:.2%}; target forced to 0")
    if long_position and pnl >= float(cfg.take_profit_pct):
        return (f"take-profit triggered: unrealised {pnl:+.2%} reaches "
                f"+{cfg.take_profit_pct:.2%}; target forced to 0")
    if (not long_position) and pnl <= -float(cfg.take_profit_pct):
        return (f"take-profit triggered on short: unrealised {pnl:+.2%} reaches "
                f"-{cfg.take_profit_pct:.2%}; target forced to 0")

    if int(cfg.max_bars_hold) > 0 and ctx.meta:
        bars_held = _finite(ctx.meta.get("bars_held"))
        if bars_held is not None and bars_held >= int(cfg.max_bars_hold):
            return (f"time stop: held {bars_held:.0f} bar(s) >= max_bars_hold "
                    f"{cfg.max_bars_hold}; target forced to 0")
    return None


def evaluate(
    proposal: TradeProposal,
    ctx: MarketContext,
    portfolio: PortfolioState,
    cfg: RiskConfig,
) -> RiskDecision:
    """Validate ``proposal`` and return the *only* executable verdict.

    The returned ``target_weight`` is guaranteed to satisfy
    ``lo <= target_weight <= max_position_weight`` with ``lo = 0`` unless
    ``cfg.allow_short`` is true.  This function never raises on malformed
    inputs; it vetoes and stays flat instead.
    """
    vetoes: list[str] = []
    adjustments: list[str] = []
    notes: list[str] = []

    max_weight = float(cfg.max_position_weight)
    floor = -max_weight if bool(cfg.allow_short) else 0.0

    # 1. Data sanity ----------------------------------------------------- #
    bars = tuple(ctx.bars or ())
    if not bars:
        return RiskDecision(approved=False, target_weight=0.0,
                            vetoes=("no market data: ctx.bars is empty",))
    price = _finite(bars[-1].close)
    if price is None or price <= 0:
        return RiskDecision(approved=False, target_weight=0.0,
                            vetoes=(f"invalid last price {bars[-1].close!r}",))
    try:
        equity = float(portfolio.equity or 0.0)
    except (TypeError, ValueError):
        equity = 0.0
    if not math.isfinite(equity) or equity <= 0:
        return RiskDecision(approved=False, target_weight=0.0,
                            vetoes=(f"invalid portfolio equity {portfolio.equity!r}",))

    symbol = proposal.symbol
    current = current_weight(portfolio, symbol, price)
    position = portfolio.positions.get(symbol) if portfolio.positions else None

    # 2. Position-protection exits (before anything else) ---------------- #
    forced = _forced_exit_reason(ctx, portfolio, symbol, price, cfg)
    if forced is not None:
        adjustments.append(forced)
        notes.append(f"risk-mandated exit: current exposure {current:.2%}, "
                     f"drawdown {portfolio.drawdown:.2%}")
        return RiskDecision(approved=True, target_weight=0.0,
                            vetoes=tuple(vetoes), adjustments=tuple(adjustments),
                            notes=tuple(notes), stop_loss=None, take_profit=None)

    # Raw target -> hard clamps (rule 6, applied early so all later rules
    # reason about an already-bounded target). --------------------------- #
    raw = _finite(proposal.target_weight)
    if raw is None:
        adjustments.append(f"non-finite target weight {proposal.target_weight!r} "
                           f"ignored; using current exposure")
        raw = current
    target = _clamp(raw, floor, max_weight)
    if raw > max_weight + _EPS:
        adjustments.append(f"target weight {raw:.2%} exceeds max_position_weight "
                           f"{max_weight:.2%}; clamped")
    elif raw < floor - _EPS:
        if bool(cfg.allow_short) and floor < 0:
            adjustments.append(f"target weight {raw:.2%} below max short exposure "
                               f"{floor:+.2%}; clamped")
        else:
            adjustments.append(f"shorting disabled: negative target {raw:+.2%} "
                               f"clamped to 0.00")

    changes = abs(target - current) > _EPS
    blocked = False

    # 3. Confidence gate -- only for trades that ADD risk ---------------- #
    if changes and _increases(target, current) and \
            float(proposal.confidence) < float(cfg.min_confidence):
        vetoes.append(f"confidence {proposal.confidence:.2f} below min_confidence "
                      f"{cfg.min_confidence:.2f}; new exposure refused and position held")
        blocked = True

    # 4. Drawdown stop -- halt NEW risk, allow de-risking ----------------- #
    drawdown = float(portfolio.drawdown)
    if _increases(target, current) and drawdown >= float(cfg.max_drawdown_stop) - _EPS:
        vetoes.append(f"drawdown {drawdown:.2%} >= max_drawdown_stop "
                      f"{cfg.max_drawdown_stop:.2%}; new exposure blocked")
        blocked = True

    # 5. Cooldown after a stop-out --------------------------------------- #
    bars_since_stop: float | None = None
    if ctx.meta:
        bars_since_stop = _finite(ctx.meta.get("bars_since_stop"))
        if bars_since_stop is None and bool(ctx.meta.get("stopped_out", False)):
            bars_since_stop = 0.0
    cooldown = int(cfg.cooldown_bars_after_stop)
    if (cooldown > 0 and bars_since_stop is not None
            and bars_since_stop < cooldown and _increases(target, current)):
        vetoes.append(f"cooldown: {bars_since_stop:.0f} bar(s) since stop-out "
                      f"< cooldown_bars_after_stop {cooldown}; new exposure blocked")
        blocked = True

    if blocked:
        held = _clamp(current, floor, max_weight)
        notes.append(f"risk veto -> holding current exposure {held:.2%} "
                     f"(drawdown {drawdown:.2%})")
        return RiskDecision(approved=False, target_weight=held,
                            vetoes=tuple(vetoes), adjustments=tuple(adjustments),
                            notes=tuple(notes), stop_loss=None, take_profit=None)

    # 7. Turnover cap ---------------------------------------------------- #
    max_turnover = float(cfg.max_daily_turnover)
    delta = target - current
    if max_turnover >= 0.0 and abs(delta) > max_turnover + _EPS:
        capped = current + math.copysign(max_turnover, delta)
        capped = _clamp(capped, floor, max_weight)
        adjustments.append(f"turnover cap {max_turnover:.0%}: change {current:.2%} -> "
                           f"{target:.2%} limited to {capped:.2%}")
        target = capped

    # 6 (final, defensive). Hard clamps again ---------------------------- #
    target = _clamp(target, floor, max_weight)

    # 8. Attach stops for approved non-flat targets ---------------------- #
    stop_loss: float | None = None
    take_profit: float | None = None
    if abs(target) > _EPS:
        entry = price
        if position is not None:
            candidate = _finite(position.avg_price)
            if candidate is not None and candidate > 0:
                entry = candidate
        if target > 0:
            stop_loss = entry * (1.0 - float(cfg.stop_loss_pct))
            take_profit = entry * (1.0 + float(cfg.take_profit_pct))
        else:
            stop_loss = entry * (1.0 + float(cfg.stop_loss_pct))
            take_profit = entry * (1.0 - float(cfg.take_profit_pct))
        notes.append(f"target {target:.2%} (current {current:.2%}), entry reference "
                     f"{entry:.4f}, drawdown {drawdown:.2%}")
    else:
        notes.append(f"target 0.00 (current {current:.2%}), flat, drawdown {drawdown:.2%}")

    action = proposal.action
    if action == Action.BUY and target < current - _EPS:
        notes.append("action/weight mismatch: BUY but the target is below current exposure")
    if action == Action.SELL and target > current + _EPS:
        notes.append("action/weight mismatch: SELL but the target is above current exposure")
    if abs(target - current) > _EPS:
        notes.append(f"approved change {target - current:+.2%} of equity")
    else:
        notes.append("no exposure change approved")

    return RiskDecision(
        approved=True,
        target_weight=target,
        vetoes=tuple(vetoes),
        adjustments=tuple(adjustments),
        notes=tuple(notes),
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
