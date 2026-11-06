"""End-to-end agent strategy: analysts -> debate -> trader -> deterministic risk.

``AgentStrategy`` is the object the backtest engine and the live loop see.  It
implements ``contracts.DecisionStrategy`` (``name`` + ``decide``) and produces an
auditable :class:`PortfolioDecision` for every closed bar.

Mode
----
* ``"deterministic"`` -- no key / LLM disabled / ``agents.use_llm=False``.
  This path provably never touches the network: the strategy holds a
  :class:`~agentic_trading.agents.llm.NullLLM` and passes ``llm=None`` down.
* ``"hybrid"`` -- an available LLM refines analyst stances, phrases the debate
  and may voice a timing veto in the trader.  **Risk stays deterministic.**
  Per-decision LLM fan-out is capped by ``MAX_LLM_CALLS_PER_DECISION`` so a
  backtest cannot accidentally issue thousands of API calls.

Policy overlay (anti-whipsaw)
-----------------------------
The raw debate net score is noisy: on a daily crypto series it can oscillate
around the consensus threshold, flipping the book in and out every few bars and
paying costs for adverse timing.  ``SmoothedConsensusPolicy`` adds a *causal,
per-symbol* overlay between the debate and the trader:

* the net score is EMA-smoothed (``DEFAULT_SMOOTHING_ALPHA``) so one noisy day
  cannot flip the book;
* entry and exit are asymmetric -- enter on moderate conviction
  (``POLICY_ENTRY_SCORE``), stay long while the primary trend is intact
  (price above SMA200), and exit only on a primary-trend break after
  ``min_hold_bars`` or instantly on a genuinely strong opposite signal
  (``POLICY_HARD_EXIT_SCORE``).  A merely lukewarm score inside an uptrend is
  noise, not an exit signal -- that asymmetry is the core of the fix;
* ``min_hold_bars`` (``AgentConfig``) is the minimum number of bars a long
  position survives before a soft exit may fire.  Risk-mandated exits
  (stop-loss / take-profit / time stop) always bypass the policy -- risk.py
  remains the only authority;
* a risk-forced exit resets the policy to flat, so re-entry needs fresh
  confirmation instead of immediately buying back the dip.

Everything is causal (only the current and previous bars are used) and is
recorded in ``diagnostics`` for audit.

Decision record
---------------
Every ``PortfolioDecision`` field is populated: the individual stances, the
debate, the raw proposal, the final risk verdict and a JSON-safe ``diagnostics``
mapping with the numbers a reviewer needs to reconstruct the decision
(net score, consensus, volatility budget, current vs target weight, vetoes,
LLM call counts).

Robustness
----------
``decide`` never raises: a bug anywhere in the pipeline degrades to a HOLD
decision carrying the error in ``diagnostics``.  In a live loop a crash at one
bar would otherwise abort the whole run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from ..config import AppConfig
from ..contracts import (
    Action,
    DebateResult,
    MarketContext,
    PortfolioDecision,
    PortfolioState,
    RiskDecision,
)
from .analysts import run_analysts
from .debate import CONSENSUS_THRESHOLD, run_debate
from .llm import BudgetedLLM, NullLLM, make_llm
from .risk import current_weight, evaluate as evaluate_risk
from .trader import annual_vol, propose

__all__ = [
    "DEFAULT_MIN_HOLD_BARS",
    "DEFAULT_SMOOTHING_ALPHA",
    "MAX_LLM_CALLS_PER_DECISION",
    "POLICY_ENTRY_SCORE",
    "POLICY_HARD_EXIT_SCORE",
    "POLICY_REGIME_BREAK",
    "AgentStrategy",
    "SmoothedConsensusPolicy",
    "build_strategy",
]

#: Hard cap on LLM calls spent on a single bar (4 analysts + debate rounds + trader).
MAX_LLM_CALLS_PER_DECISION = 8

#: Causal EMA weight on the newest bar.  0.35 keeps ~2-3 bars of memory: enough
#: to absorb a single mean-reversion day, small enough to still react to a turn.
DEFAULT_SMOOTHING_ALPHA = 0.35
#: Smoothed-score level required to open a long.  Slightly above the debate
#: consensus threshold so entry needs persistence, not a one-bar spike.
POLICY_ENTRY_SCORE = 0.18
#: Hard-exit level: a genuinely strong opposite signal may cut even inside the
#: minimum holding period.
POLICY_HARD_EXIT_SCORE = -0.30
#: Price below SMA200 by this much is a primary-trend break (policy exit).
POLICY_REGIME_BREAK = 0.0
#: Default minimum holding period (bars).  ``AgentConfig.min_hold_bars`` overrides
#: this when the Lead adds the field; 5 daily bars = one trading week.
DEFAULT_MIN_HOLD_BARS = 5


@dataclass
class _PolicyState:
    """Per-symbol state of the smoothed-consensus policy."""

    smoothed: float | None = None
    direction: int = 0            # +1 = long policy, 0 = flat policy
    bars_in_direction: int = 0


class SmoothedConsensusPolicy:
    """Causal, per-symbol EMA smoothing + asymmetric hysteresis over net_score.

    Long-only by construction: ``direction`` is ``+1`` (long) or ``0`` (flat),
    matching the long-only trader.  The policy never sizes or exits a position
    by itself -- it produces a directional view the trader consults, and risk.py
    remains the only authority on what is executed.
    """

    def __init__(
        self,
        alpha: float | None = None,
        entry_score: float | None = None,
        hard_exit_score: float | None = None,
    ) -> None:
        self.alpha = DEFAULT_SMOOTHING_ALPHA if alpha is None else float(alpha)
        self.entry_score = POLICY_ENTRY_SCORE if entry_score is None else float(entry_score)
        self.hard_exit_score = POLICY_HARD_EXIT_SCORE if hard_exit_score is None else float(hard_exit_score)
        self._states: dict[str, _PolicyState] = {}

    def reset(self) -> None:
        self._states.clear()

    def state(self, symbol: str) -> _PolicyState | None:
        return self._states.get(symbol)

    @staticmethod
    def regime_down(features: Any) -> bool:
        """Primary-trend gate: True when price has broken below SMA200.

        This is the "only exit on a strong opposite signal" leg of the policy.
        Missing (warmup) features are treated as *not* broken -- the policy then
        relies on the smoothed score, which is the safe default.
        """
        if not features:
            return False
        above200 = features.get("above_sma200")
        price_vs_200 = features.get("price_vs_sma200")
        try:
            above200_f = None if above200 is None else float(above200)
            price_vs_200_f = None if price_vs_200 is None else float(price_vs_200)
        except (TypeError, ValueError):
            return False
        if above200_f is not None and math.isfinite(above200_f) and above200_f < 0.5:
            return True
        if price_vs_200_f is not None and math.isfinite(price_vs_200_f) and price_vs_200_f <= -POLICY_REGIME_BREAK:
            return True
        return False

    def force_flat(self, symbol: str) -> None:
        """Clear the long regime (used after a risk-mandated exit)."""
        state = self._states.get(symbol)
        if state is not None:
            state.direction = 0
            state.bars_in_direction = 0

    def update(
        self,
        ctx: MarketContext,
        debate: DebateResult,
        min_hold_bars: int = DEFAULT_MIN_HOLD_BARS,
    ) -> tuple[dict[str, Any], float]:
        """Advance the policy one bar and return ``(info, smoothed_score)``.

        Pure with respect to the current bar: ``smoothed_t = alpha * raw_t +
        (1 - alpha) * smoothed_{t-1}``.  The first observation seeds the EMA with
        the raw score so a single-shot ``analyze`` call still produces a signal.
        """
        raw = debate.net_score
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raw = 0.0
        if not math.isfinite(raw):
            raw = 0.0
        raw = max(-1.0, min(1.0, raw))
        hold_bars = max(0, int(min_hold_bars))

        state = self._states.setdefault(ctx.symbol, _PolicyState())
        if state.smoothed is None:
            smoothed = raw                      # first observation: seed, no lookback
        else:
            smoothed = self.alpha * raw + (1.0 - self.alpha) * state.smoothed
        smoothed = max(-1.0, min(1.0, smoothed))

        hard_exit = False
        regime_break = self.regime_down(ctx.features)
        if state.direction == 0:
            state.bars_in_direction = 0
            if not regime_break and smoothed >= self.entry_score:
                state.direction = 1
                state.bars_in_direction = 0
        else:
            state.bars_in_direction += 1
            if smoothed <= self.hard_exit_score:
                # Strong opposite signal: may cut even inside the min-hold window.
                state.direction = 0
                state.bars_in_direction = 0
                hard_exit = True
            elif regime_break and state.bars_in_direction >= hold_bars:
                # The primary trend broke: an exit is warranted, but still
                # respect the minimum holding period.
                state.direction = 0
                state.bars_in_direction = 0

        state.smoothed = smoothed
        info = {
            "raw_net_score": raw,
            "smoothed_net_score": smoothed,
            "policy_direction": state.direction,
            "bars_in_direction": state.bars_in_direction,
            "min_hold_bars": hold_bars,
            "policy_hard_exit": hard_exit,
            "regime_break": regime_break,
        }
        return info, smoothed


def _price_of(ctx: MarketContext, portfolio: PortfolioState) -> float:
    """Last closed price, preferring the decision bar over the portfolio view."""
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


class AgentStrategy:
    """The multi-agent decision pipeline, LLM-optional."""

    name: str = "agentic_multi_agent"

    def __init__(self, cfg: AppConfig, llm: Any = None) -> None:
        self.cfg = cfg
        self._llm = self._resolve_llm(cfg, llm)
        #: "hybrid" when an LLM *will* be attempted, else "deterministic".
        self.mode: str = "hybrid" if bool(getattr(self._llm, "available", False)) else "deterministic"
        # Cooldown bookkeeping: kept here because the frozen PortfolioState has no
        # "bars since stop" field.  The risk gate consumes it through ctx.meta.
        # Keyed by symbol: one strategy instance drives every symbol in the live
        # loop, and a stop in BTC must not cooldown ETH.
        self._bars_since_stop: dict[str, int] = {}
        #: Causal smoothing + hysteresis overlay between debate and trader.
        self._policy = SmoothedConsensusPolicy()
        self.decisions: int = 0
        self.risk_vetoes: int = 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_llm(cfg: AppConfig, llm: Any) -> Any:
        if not bool(cfg.agents.use_llm):
            return NullLLM(cfg.llm, "agents.use_llm is False")
        if llm is not None:
            return llm
        return make_llm(cfg.llm)

    # ------------------------------------------------------------------ #
    def decide(self, ctx: MarketContext, portfolio: PortfolioState) -> PortfolioDecision:
        """Produce the auditable decision for one closed bar.  Never raises."""
        try:
            return self._decide(ctx, portfolio)
        except Exception as exc:  # pragma: no cover - defensive, exercised by tests
            return self._error_decision(ctx, portfolio, exc)

    # ------------------------------------------------------------------ #
    def _decide(self, ctx: MarketContext, portfolio: PortfolioState) -> PortfolioDecision:
        cfg = self.cfg
        llm: Any = None
        if self.mode == "hybrid":
            llm = BudgetedLLM(self._llm, MAX_LLM_CALLS_PER_DECISION)

        # 1) analysts -- deterministic first, each may refine with one LLM call.
        stances = run_analysts(ctx, cfg.agents, llm)
        # 2) debate -- deterministic pooling; LLM may phrase the arguments.
        debate = run_debate(ctx, stances, cfg.agents, llm)

        # 2b) policy overlay -- causal smoothing + asymmetric hysteresis.
        #     This is the anti-whipsaw layer: the raw net score is noisy, so the
        #     book must not flip on a single bar.  The policy only produces a
        #     directional view; it never sizes and never overrides risk.
        configured_min_hold = getattr(cfg.agents, "min_hold_bars", DEFAULT_MIN_HOLD_BARS)
        min_hold_bars = (DEFAULT_MIN_HOLD_BARS if configured_min_hold is None
                         else int(configured_min_hold))
        policy_info, smoothed_score = self._policy.update(ctx, debate, min_hold_bars)

        meta = dict(ctx.meta or {})
        meta["policy_direction"] = int(policy_info["policy_direction"])
        meta["policy_net_score"] = float(smoothed_score)
        stored_bars_since_stop = self._bars_since_stop.get(ctx.symbol)
        effective_bars_since_stop: Any = meta.get("bars_since_stop", stored_bars_since_stop)
        if effective_bars_since_stop is not None:
            meta["bars_since_stop"] = effective_bars_since_stop
        bar_ctx = replace(ctx, meta=meta)

        # 3) trader -- deterministic vol-targeted sizing; LLM may only de-risk.
        proposal = propose(bar_ctx, debate, stances, portfolio, cfg, llm)

        price = _price_of(ctx, portfolio)
        current = current_weight(portfolio, ctx.symbol, price)

        # 4) risk -- the deterministic authority.  Feed it the cooldown state.
        risk = evaluate_risk(proposal, bar_ctx, portfolio, cfg.risk)

        # 5) cooldown bookkeeping (only our own stops reset it).
        forced_exit = any(
            adjustment.startswith(("stop-loss", "take-profit", "time stop"))
            for adjustment in risk.adjustments
        )
        if forced_exit:
            self._bars_since_stop[ctx.symbol] = 0
            # A risk-mandated exit invalidates the long thesis: require fresh
            # confirmation before re-entering rather than buying the dip.
            self._policy.force_flat(ctx.symbol)
        elif effective_bars_since_stop is not None:
            try:
                self._bars_since_stop[ctx.symbol] = int(effective_bars_since_stop) + 1
            except (TypeError, ValueError):
                self._bars_since_stop.pop(ctx.symbol, None)

        final_weight = float(risk.target_weight)
        delta = final_weight - current
        if not risk.approved:
            action = Action.HOLD
        elif delta > 1e-6:
            action = Action.BUY
        elif delta < -1e-6:
            action = Action.SELL
        else:
            action = Action.HOLD

        used_llm = (
            any(stance.source == "llm" for stance in stances)
            or debate.bull.source == "llm"
            or debate.bear.source == "llm"
            or proposal.proposed_by.endswith("+llm")
        )
        decision_mode = "hybrid" if used_llm else "deterministic"
        confidence = float(proposal.confidence) if risk.approved else 0.0

        thesis = proposal.thesis
        if risk.vetoes:
            thesis = f"{thesis} | risk veto: {'; '.join(risk.vetoes)}"
        elif risk.adjustments:
            thesis = f"{thesis} | risk: {'; '.join(risk.adjustments)}"
        thesis = thesis[:900]

        diagnostics: dict[str, Any] = {
            "strategy_mode": self.mode,
            "mode_used": decision_mode,
            "consensus": debate.consensus,
            "net_score": round(float(debate.net_score), 6),
            "raw_consensus": debate.consensus,
            "raw_net_score": round(float(policy_info["raw_net_score"]), 6),
            "smoothed_net_score": round(float(smoothed_score), 6),
            "policy_direction": int(policy_info["policy_direction"]),
            "policy_state": "long" if int(policy_info["policy_direction"]) > 0 else "flat",
            "bars_in_direction": int(policy_info["bars_in_direction"]),
            "min_hold_bars": int(policy_info["min_hold_bars"]),
            "policy_hard_exit": bool(policy_info["policy_hard_exit"]),
            "regime_break": bool(policy_info.get("regime_break", False)),
            "consensus_threshold": CONSENSUS_THRESHOLD,
            "analysts": {
                stance.agent: {
                    "label": stance.label,
                    "confidence": round(float(stance.confidence), 4),
                    "source": stance.source,
                }
                for stance in stances
            },
            "annual_vol": _round(annual_vol(ctx.features)),
            "proposal_weight": round(float(proposal.target_weight), 6),
            "proposal_confidence": round(float(proposal.confidence), 4),
            "current_weight": round(float(current), 6),
            "final_weight": round(final_weight, 6),
            "risk_approved": bool(risk.approved),
            "risk_vetoes": len(risk.vetoes),
            "risk_adjustments": len(risk.adjustments),
            "stop_loss": risk.stop_loss,
            "take_profit": risk.take_profit,
            "bars": len(ctx.bars or ()),
            "warmup": len(ctx.bars or ()) < int(cfg.agents.min_bars_for_signals),
            "bars_since_stop": (None if effective_bars_since_stop is None
                                else _safe_int(effective_bars_since_stop)),
            "llm_available": self.mode == "hybrid",
            "llm_calls": int(getattr(self._llm, "calls", 0)),
            "llm_failures": int(getattr(self._llm, "failures", 0)),
            "llm_used": used_llm,
        }

        self.decisions += 1
        if risk.vetoes:
            self.risk_vetoes += 1

        return PortfolioDecision(
            asof=ctx.asof,
            symbol=ctx.symbol,
            action=action,
            target_weight=final_weight,
            confidence=confidence,
            thesis=thesis,
            analysis=tuple(stances),
            debate=debate,
            proposal=proposal,
            risk=risk,
            mode=decision_mode,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------ #
    def _error_decision(
        self,
        ctx: MarketContext,
        portfolio: PortfolioState,
        exc: Exception,
    ) -> PortfolioDecision:
        """HOLD with the error recorded -- a live loop must survive one bad bar."""
        try:
            asof = ctx.asof
            symbol = ctx.symbol
            price = _price_of(ctx, portfolio)
            current = current_weight(portfolio, symbol, price)
            bars = len(ctx.bars or ())
        except Exception:
            asof = datetime.now(timezone.utc)
            symbol = ""
            current = 0.0
            bars = 0
        message = f"{type(exc).__name__}: {exc}"
        risk = RiskDecision(
            approved=False,
            target_weight=current,
            vetoes=(f"internal pipeline error: {message}",),
        )
        return PortfolioDecision(
            asof=asof,
            symbol=symbol,
            action=Action.HOLD,
            target_weight=current,
            confidence=0.0,
            thesis=f"pipeline error, holding: {message}",
            analysis=(),
            debate=None,
            proposal=None,
            risk=risk,
            mode="deterministic",
            diagnostics={
                "error": message,
                "strategy_mode": self.mode,
                "bars": bars,
            },
        )


# --------------------------------------------------------------------------- #
# Small JSON-safe helpers for diagnostics
# --------------------------------------------------------------------------- #
def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _safe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def build_strategy(cfg: AppConfig) -> AgentStrategy:
    """Factory used by the CLI, the runner and the tests."""
    return AgentStrategy(cfg)
