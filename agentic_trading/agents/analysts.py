"""Rule-based analyst agents, optionally refined by an LLM.

Design contract
---------------
* A stance is an *opinion*, never an order.  Analysts cannot size or execute.
* Every number comes from ``ctx.features`` (keys in
  ``contracts.REQUIRED_FEATURES``) and ``ctx.bars`` -- this module never computes
  an indicator.
* NaN/inf features are a normal warmup state: a component with no data is
  dropped and the remaining weights are renormalised.  If nothing is usable the
  analyst returns a neutral stance.  Analysts never raise.
* ``assess()`` is the deterministic opinion and is always computed first.
  ``refine_with_llm()`` may relabel or re-case confidence by at most
  ``LLM_CONFIDENCE_DELTA`` and rewrite the rationale; it can never touch
  position size (that is ``agents/risk.py``'s job).
* Four analysts are provided, and they look at genuinely different evidence:
  trend, momentum, mean reversion and volatility risk.  Their evidence weights
  (``ANALYST_WEIGHTS``) are what the debate uses to pool them.

Every analyst turns its evidence into a signed score in ``[-1, 1]``
(positive = pro-long), then a :class:`~agentic_trading.contracts.Stance` is
derived: ``|score| < NEUTRAL_BAND`` -> neutral, otherwise bullish/bearish with
confidence ``0.35 + 0.60 * |score|`` (capped at 0.95).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..config import AgentConfig
from ..contracts import MarketContext, Stance

__all__ = [
    "ANALYSTS",
    "ANALYST_WEIGHTS",
    "DEFAULT_ANALYST_ORDER",
    "EVIDENCE_WEIGHTS",
    "LLM_CONFIDENCE_DELTA",
    "NEUTRAL_BAND",
    "NEUTRAL_CONFIDENCE",
    "Analyst",
    "MeanReversionAnalyst",
    "MomentumAnalyst",
    "RiskAnalyst",
    "TechnicalAnalyst",
    "run_analysts",
]

DEFAULT_ANALYST_ORDER: tuple[str, ...] = ("technical", "momentum", "mean_reversion", "risk")

#: Evidence weights used by the debate (sum to 1.0).  Rationale:
#: trend is the primary edge in this system, momentum confirms it, mean
#: reversion is deliberately the smallest weight because fading a trend is the
#: riskiest of the four views, and the risk analyst is a quarter-weight
#: volatility-regime overlay that can temper enthusiasm.
ANALYST_WEIGHTS: dict[str, float] = {
    "technical": 0.30,
    "momentum": 0.25,
    "mean_reversion": 0.20,
    "risk": 0.25,
}
#: Alias used by ``agents/debate.py`` (and by the Lead when auditing weights).
EVIDENCE_WEIGHTS = ANALYST_WEIGHTS

NEUTRAL_BAND = 0.12          # |score| below this -> neutral label
NEUTRAL_CONFIDENCE = 0.20    # confidence attached to a "no edge" call
LLM_CONFIDENCE_DELTA = 0.20  # LLM may move confidence at most this far
LLM_RATIONALE_MAX = 320

_PERIODS_PER_YEAR: dict[str, float] = {"1d": 365.0, "1h": 24.0 * 365.0, "15m": 96.0 * 365.0}


# --------------------------------------------------------------------------- #
# Small, NaN-safe numeric helpers
# --------------------------------------------------------------------------- #
def _finite(value: Any) -> float | None:
    """Coerce to a finite float or return ``None`` (NaN/inf/bad types -> None)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _feature(features: Mapping[str, float] | None, key: str) -> float | None:
    if not features:
        return None
    return _finite(features.get(key))


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _tanh(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return math.tanh(value)


def _blend(parts: Sequence[tuple[float | None, float]]) -> float | None:
    """Weighted mean of the available components (weights renormalised).

    Missing components simply drop out, which is how warmup NaNs degrade
    gracefully instead of poisoning the score.
    """
    usable = [(value, weight) for value, weight in parts
              if value is not None and math.isfinite(value) and weight > 0]
    total = sum(weight for _, weight in usable)
    if total <= 0:
        return None
    return _clip(sum(value * weight for value, weight in usable) / total)


def _fmt(value: float | None, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def _annualisation(timeframe: str) -> float:
    return math.sqrt(_PERIODS_PER_YEAR.get(str(timeframe or "1d"), 365.0))


def _daily_vol(annual_vol: float | None, timeframe: str) -> float | None:
    if annual_vol is None or annual_vol <= 0:
        return None
    return annual_vol / _annualisation(timeframe)


def _feature_notes(
    features: Mapping[str, float] | None,
    keys: Sequence[str],
) -> str:
    """Compact ``key=value`` string for LLM prompts (NaN rendered as ``n/a``)."""
    chunks: list[str] = []
    for key in keys:
        value = _feature(features, key)
        if value is None:
            chunks.append(f"{key}=n/a")
        elif key.startswith(("ret", "price_vs", "dist_from", "vol", "atr_pct", "bb_width", "mom")):
            chunks.append(f"{key}={value:+.4f}")
        else:
            chunks.append(f"{key}={value:.4f}")
    return ", ".join(chunks)


# --------------------------------------------------------------------------- #
# Base analyst
# --------------------------------------------------------------------------- #
class Analyst:
    """Base class: deterministic assessment + bounded LLM refinement."""

    name: str = "analyst"
    weight: float = 0.15
    description: str = ""
    LLM_FEATURE_KEYS: tuple[str, ...] = ("close",)

    # -- deterministic path ------------------------------------------------- #
    def _evaluate(self, ctx: MarketContext) -> tuple[float, list[str], str]:
        """Return ``(score, evidence, rationale)``; score is signed conviction."""
        raise NotImplementedError

    def to_stance(self, score: float, rationale: str, evidence: Sequence[str]) -> Stance:
        """Map a signed score to the frozen :class:`Stance` shape."""
        score = _clip(score)
        if abs(score) < NEUTRAL_BAND:
            return Stance(
                agent=self.name,
                label="neutral",
                confidence=NEUTRAL_CONFIDENCE,
                rationale=rationale or "no directional edge",
                evidence=tuple(evidence),
                source="deterministic",
            )
        label = "bullish" if score > 0 else "bearish"
        confidence = _clip(0.35 + 0.60 * min(1.0, abs(score)), 0.0, 0.95)
        return Stance(
            agent=self.name,
            label=label,
            confidence=confidence,
            rationale=rationale,
            evidence=tuple(evidence),
            source="deterministic",
        )

    def assess(self, ctx: MarketContext) -> Stance:
        """Run the deterministic rules.  Never raises, even on garbage input."""
        try:
            score, evidence, rationale = self._evaluate(ctx)
        except Exception as exc:  # a broken feature mapping must not kill a bar
            return Stance(
                agent=self.name,
                label="neutral",
                confidence=0.0,
                rationale=f"{self.name} analyst unavailable ({type(exc).__name__})",
                evidence=(),
                source="deterministic",
            )
        return self.to_stance(score, rationale, evidence)

    # -- optional LLM refinement ------------------------------------------- #
    def _llm_prompt(self, ctx: MarketContext, base: Stance) -> str:
        return (
            f"Symbol {ctx.symbol} on {ctx.timeframe} bars, asof {ctx.asof.isoformat()}.\n"
            f"Deterministic features: {_feature_notes(ctx.features, self.LLM_FEATURE_KEYS)}\n"
            f"Deterministic {self.name} stance: {base.label} "
            f"(confidence {base.confidence:.2f}).\n"
            f"Deterministic rationale: {base.rationale}\n"
            "Refine the stance only if the feature evidence justifies it. "
            "Do not invent indicators, prices or news."
        )

    def refine_with_llm(self, ctx: MarketContext, base: Stance, llm: Any) -> Stance:
        """Bounded refinement: label / confidence (+-DELTA) / rationale only."""
        if llm is None or not bool(getattr(llm, "available", False)):
            return base
        system = (
            f"You are the {self.name} analyst inside an automated trading system. "
            "You may only refine the deterministic stance you are given; you can never "
            "size positions or place orders. Answer with ONE JSON object and nothing else."
        )
        schema = (
            '{"label": "bullish|bearish|neutral", '
            '"confidence": <float 0..1>, '
            '"rationale": "<one or two sentences citing the features>"}'
        )
        try:
            data = llm.complete_json(system, self._llm_prompt(ctx, base), schema)
        except Exception:
            return base
        if not isinstance(data, dict):
            return base

        label = str(data.get("label", "")).strip().lower()
        if label not in ("bullish", "bearish", "neutral"):
            return base
        confidence = _finite(data.get("confidence"))
        if confidence is None:
            return base
        low = max(0.0, base.confidence - LLM_CONFIDENCE_DELTA)
        high = min(1.0, base.confidence + LLM_CONFIDENCE_DELTA)
        confidence = _clip(confidence, low, high)
        rationale = data.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            rationale = rationale.strip()[:LLM_RATIONALE_MAX]
        else:
            rationale = base.rationale
        return Stance(
            agent=self.name,
            label=label,
            confidence=confidence,
            rationale=rationale,
            evidence=base.evidence,
            source="llm",
        )


# --------------------------------------------------------------------------- #
# 1. Technical -- trend regime, MACD, price vs SMA
# --------------------------------------------------------------------------- #
class TechnicalAnalyst(Analyst):
    """Trend-following read of the SMA stack, MACD and price-vs-SMA stretch.

    Trading logic
    -------------
    * **SMA stack** (SMA20 > SMA50 > SMA200) is the regime filter.  A clean
      stack is the strongest bullish evidence; a fully inverted stack the
      strongest bearish evidence.  The binary ``trend_up`` feature plus
      ``above_sma50`` / ``above_sma200`` encode it.
    * **MACD histogram** measures trend *acceleration*.  It is normalised by
      ATR% so the signal is scale-free across symbols and volatility regimes.
    * **Price vs SMA50/200** measures how stretched the trend is; the
      contribution is capped (via ``tanh``) so a blow-off move cannot dominate
      the verdict.  SMA50 matters more than SMA200 for timing.
    * Components with missing data (warmup, no SMA200 yet) renormalise, so the
      analyst still works on 60-80 bar histories.
    """

    name = "technical"
    weight = ANALYST_WEIGHTS["technical"]
    description = "SMA-stack trend regime + MACD acceleration + stretch vs SMA50/200"
    LLM_FEATURE_KEYS = (
        "close", "trend_up", "above_sma50", "above_sma200",
        "macd_hist", "price_vs_sma50", "price_vs_sma200", "atr_pct",
    )

    def _evaluate(self, ctx: MarketContext) -> tuple[float, list[str], str]:
        features = ctx.features
        atr_pct = _feature(features, "atr_pct")
        close = _feature(features, "close")

        trend_up = _feature(features, "trend_up")
        above50 = _feature(features, "above_sma50")
        above200 = _feature(features, "above_sma200")

        stack = _blend([
            (None if trend_up is None else (1.0 if trend_up >= 0.5 else -1.0), 0.50),
            (None if above50 is None else (2.0 * above50 - 1.0), 0.25),
            (None if above200 is None else (2.0 * above200 - 1.0), 0.25),
        ])

        macd_hist = _feature(features, "macd_hist")
        macd_component = None
        if macd_hist is not None and close is not None and atr_pct:
            macd_component = _tanh((macd_hist / close) / max(abs(atr_pct), 1e-4))

        p50 = _feature(features, "price_vs_sma50")
        p50_component = _tanh(p50 / (2.0 * max(abs(atr_pct), 1e-4))) if p50 is not None and atr_pct else None

        p200 = _feature(features, "price_vs_sma200")
        p200_component = _tanh(p200 / (3.0 * max(abs(atr_pct), 1e-4))) if p200 is not None and atr_pct else None

        score = _blend([
            (stack, 0.45),
            (macd_component, 0.25),
            (p50_component, 0.20),
            (p200_component, 0.10),
        ])
        score = 0.0 if score is None else score

        evidence: list[str] = []
        if stack is not None:
            regime = "up (20>50>200)" if stack > 0.5 else (
                "down (20<50<200)" if stack < -0.5 else "mixed")
            evidence.append(f"SMA stack {regime} ({stack:+.2f})")
        if macd_hist is not None:
            evidence.append(f"MACD histogram {macd_hist:+.4f} "
                            f"({_fmt(macd_component, '+.2f')} ATR-normalised)")
        if p50 is not None:
            evidence.append(f"price {p50:+.1%} vs SMA50")
        if p200 is not None:
            evidence.append(f"price {p200:+.1%} vs SMA200")
        if not evidence:
            evidence.append("no trend features available")

        rationale = (
            f"SMA stack {_fmt(stack, '+.2f')}, MACD {_fmt(macd_component, '+.2f')}, "
            f"price vs SMA50 {_fmt_pct(p50)} vs SMA200 {_fmt_pct(p200)}"
        )
        return score, evidence, rationale


# --------------------------------------------------------------------------- #
# 2. Momentum -- rate of change at 5/20/60 bars + volume confirmation
# --------------------------------------------------------------------------- #
class MomentumAnalyst(Analyst):
    """Cross-horizon rate-of-change with volume confirmation.

    Trading logic
    -------------
    * Each horizon (5, 20, 60 bars) is converted to a **volatility-normalised
      z-score** and squashed with ``tanh``: a 10% 20-day move means something
      very different at 20% vs 80% annualised vol.  This makes the signal
      comparable across regimes and symbols.
    * **Acceleration**: the recent 5-bar rate minus the 20-bar average rate.
      Positive acceleration supports a trend, deceleration warns about it.
    * **Volume confirmation**: a 20-day move is trusted more when volume is
      above its 20-bar mean (``volume_z``); weak volume dampens the read.
    * Weighting: 20-bar (0.30) and 60-bar (0.25) dominate because they are less
      noisy; 5-bar (0.25) adds timing; acceleration and volume get 0.10 each.
    """

    name = "momentum"
    weight = ANALYST_WEIGHTS["momentum"]
    description = "5/20/60-bar rate of change (vol-normalised) + volume confirmation"
    LLM_FEATURE_KEYS = (
        "close", "ret_5", "ret_20", "ret_60", "vol_20", "vol_60",
        "volume_z", "mom_20",
    )

    def _evaluate(self, ctx: MarketContext) -> tuple[float, list[str], str]:
        features = ctx.features
        ret5 = _feature(features, "ret_5")
        ret20 = _feature(features, "ret_20")
        ret60 = _feature(features, "ret_60")
        vol20 = _feature(features, "vol_20")
        vol60 = _feature(features, "vol_60")
        volume_z = _feature(features, "volume_z")

        daily_vol = _daily_vol(vol20, ctx.timeframe)
        daily_vol60 = _daily_vol(vol60, ctx.timeframe)

        def z_component(ret: float | None, bars: int, vol: float | None) -> float | None:
            if ret is None or vol is None or vol <= 0:
                return None
            return _tanh(ret / (vol * math.sqrt(bars)))

        c5 = z_component(ret5, 5, daily_vol)
        c20 = z_component(ret20, 20, daily_vol)
        c60 = z_component(ret60, 60, daily_vol60 or daily_vol)

        acceleration = None
        reference_vol = daily_vol or daily_vol60
        if ret5 is not None and ret20 is not None and reference_vol:
            acceleration = _tanh((ret5 / 5.0 - ret20 / 20.0) / max(reference_vol, 1e-4))

        volume_confirmation = None
        if volume_z is not None and c20 is not None:
            direction = 1.0 if c20 >= 0 else -1.0
            volume_confirmation = direction * _clip(volume_z / 2.0, -1.0, 1.0)

        score = _blend([
            (c5, 0.25),
            (c20, 0.30),
            (c60, 0.25),
            (acceleration, 0.10),
            (volume_confirmation, 0.10),
        ])
        score = 0.0 if score is None else score

        evidence: list[str] = []
        if ret5 is not None:
            evidence.append(f"5-bar rate of change {ret5:+.2%} ({_fmt(c5, '+.2f')})")
        if ret20 is not None:
            evidence.append(f"20-bar return {ret20:+.2%} ({_fmt(c20, '+.2f')})")
        if ret60 is not None:
            evidence.append(f"60-bar return {ret60:+.2%} ({_fmt(c60, '+.2f')})")
        if acceleration is not None:
            evidence.append(f"acceleration {_fmt(acceleration, '+.2f')}")
        if volume_z is not None:
            confirm = "confirms" if (volume_confirmation or 0.0) > 0 else "does not confirm"
            evidence.append(f"volume z-score {volume_z:+.2f} {confirm}")
        if not evidence:
            evidence.append("no momentum features available")

        rationale = (
            f"20-bar {_fmt_pct(ret20)} ({_fmt(c20, '+.2f')} vol-adjusted), "
            f"60-bar {_fmt_pct(ret60)} ({_fmt(c60, '+.2f')}), "
            f"volume confirmation {_fmt(volume_confirmation, '+.2f')}"
        )
        return score, evidence, rationale


# --------------------------------------------------------------------------- #
# 3. Mean reversion -- RSI, Bollinger %B, 60-bar range position
# --------------------------------------------------------------------------- #
class MeanReversionAnalyst(Analyst):
    """Fade short-term extremes against the 60-bar range.

    Trading logic
    -------------
    * **RSI-14** below 30 is oversold (buy signal), above 70 is overbought
      (sell signal); the score is linear in ``(50 - RSI) / 30`` with a floor.
    * **Bollinger %B** locates the close inside the 20-bar band: below 0 means
      the price is under the lower band (stretched down -> reversion up), above
      1 means above the upper band.  ``(0.5 - %B) / 0.5`` is capped at +-1.
    * **60-bar range position** derived from ``dist_from_high_60`` and
      ``dist_from_low_60``: near the 60-bar low is a reversion buy, near the high
      a reversion sell.
    * This analyst is deliberately counter-trend and gets the smallest debate
      weight (0.20): it provides liquidity/mean-reversion alpha but must not
      overpower a real trend.
    """

    name = "mean_reversion"
    weight = ANALYST_WEIGHTS["mean_reversion"]
    description = "RSI-14 extremes + Bollinger %B + position in the 60-bar range"
    LLM_FEATURE_KEYS = (
        "close", "rsi_14", "bb_pctb", "bb_upper", "bb_lower",
        "dist_from_high_60", "dist_from_low_60", "bb_width",
    )

    def _evaluate(self, ctx: MarketContext) -> tuple[float, list[str], str]:
        features = ctx.features
        rsi = _feature(features, "rsi_14")
        pctb = _feature(features, "bb_pctb")
        dist_high = _feature(features, "dist_from_high_60")
        dist_low = _feature(features, "dist_from_low_60")
        bb_width = _feature(features, "bb_width")

        rsi_component = None
        if rsi is not None:
            rsi_component = _clip((50.0 - rsi) / 30.0, -1.0, 1.0)

        pctb_component = None
        if pctb is not None:
            pctb_component = _clip((0.5 - pctb) / 0.5, -1.0, 1.0)

        range_component = None
        range_position = None
        if dist_high is not None and dist_low is not None:
            span = dist_low - dist_high  # == (hi - lo) / lo, positive when hi > lo
            if span > 1e-9:
                range_position = _clip(dist_low / span, 0.0, 1.0)
                range_component = _clip((0.5 - range_position) * 2.0, -1.0, 1.0)

        score = _blend([
            (rsi_component, 0.45),
            (pctb_component, 0.35),
            (range_component, 0.20),
        ])
        score = 0.0 if score is None else score

        evidence: list[str] = []
        if rsi is not None:
            zone = "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral zone")
            evidence.append(f"RSI-14 {rsi:.1f} ({zone})")
        if pctb is not None:
            evidence.append(f"Bollinger %B {pctb:.2f}")
        if range_position is not None:
            evidence.append(f"at {range_position:.0%} of the 60-bar range")
        if bb_width is not None:
            evidence.append(f"band width {bb_width:.1%} of price")
        if not evidence:
            evidence.append("no mean-reversion features available")

        rationale = (
            f"RSI-14 {_fmt(rsi, '.1f')} -> {_fmt(rsi_component, '+.2f')}, "
            f"%B {_fmt(pctb, '.2f')} -> {_fmt(pctb_component, '+.2f')}, "
            f"60-bar range position {_fmt(range_position, '.0%')}"
        )
        return score, evidence, rationale


# --------------------------------------------------------------------------- #
# 4. Risk -- volatility, ATR, drawdown proximity, vol regime
# --------------------------------------------------------------------------- #
class RiskAnalyst(Analyst):
    """Risk-appetite overlay: calm markets support exposure, stressed ones do not.

    Trading logic
    -------------
    The score is *pro-risk* (positive = environment supports long exposure,
    negative = risk-off).  It is not a price forecast; it is the probability
    weighting on the other analysts' opinions.

    * **ATR%**: average true range as a fraction of price.  Above ~2.5% per bar
      is hostile for a 40% annual vol target; below ~1% is benign.
    * **Annualised vol** (``vol_20``): compared against a ~45% reference; high
      realised vol shrinks the risk budget, low vol expands it.
    * **Drawdown proximity** via ``dist_from_high_60``: sitting at the 60-bar
      high is a low-drawdown, healthy state; 25%+ below it is a risk-off state.
    * **Vol regime** = ``vol_20 / vol_60``: expanding volatility (ratio > 1) is
      bearish for risk assets, contracting volatility is supportive.
    """

    name = "risk"
    weight = ANALYST_WEIGHTS["risk"]
    description = "ATR%, annualised vol, drawdown proximity and vol regime overlay"
    LLM_FEATURE_KEYS = (
        "close", "atr_pct", "vol_20", "vol_60",
        "dist_from_high_60", "dist_from_low_60", "bb_width",
    )

    def _evaluate(self, ctx: MarketContext) -> tuple[float, list[str], str]:
        features = ctx.features
        atr_pct = _feature(features, "atr_pct")
        vol20 = _feature(features, "vol_20")
        vol60 = _feature(features, "vol_60")
        dist_high = _feature(features, "dist_from_high_60")

        atr_component = None
        if atr_pct is not None:
            atr_component = _clip((0.025 - abs(atr_pct)) / 0.025, -1.0, 1.0)

        vol_component = None
        if vol20 is not None:
            vol_component = _clip((0.45 - abs(vol20)) / 0.45, -1.0, 1.0)

        drawdown_component = None
        if dist_high is not None:
            # 0% below the 60-bar high -> +1; 25% below -> 0; 50%+ below -> -1.
            drawdown_component = _clip(1.0 + dist_high / 0.25, -1.0, 1.0)

        regime_component = None
        regime_ratio = None
        if vol20 is not None and vol60 is not None and vol60 > 1e-9:
            regime_ratio = abs(vol20) / abs(vol60)
            regime_component = _clip((1.0 - regime_ratio) * 2.0, -1.0, 1.0)

        score = _blend([
            (atr_component, 0.30),
            (vol_component, 0.30),
            (drawdown_component, 0.25),
            (regime_component, 0.15),
        ])
        score = 0.0 if score is None else score

        evidence: list[str] = []
        if atr_pct is not None:
            state = "stressed" if atr_pct > 0.04 else ("elevated" if atr_pct > 0.025 else "calm")
            evidence.append(f"ATR% {atr_pct:.2%} ({state})")
        if vol20 is not None:
            evidence.append(f"annualised 20-bar vol {vol20:.0%}")
        if vol60 is not None:
            evidence.append(f"annualised 60-bar vol {vol60:.0%}")
        if dist_high is not None:
            evidence.append(f"{dist_high:+.1%} vs 60-bar high")
        if regime_ratio is not None:
            regime = "expanding" if regime_ratio > 1.1 else (
                "contracting" if regime_ratio < 0.9 else "stable")
            evidence.append(f"vol regime {regime} ({regime_ratio:.2f}x)")
        if not evidence:
            evidence.append("no risk features available")

        rationale = (
            f"ATR% {_fmt(atr_pct, '.2%')} -> {_fmt(atr_component, '+.2f')}, "
            f"annual vol {_fmt(vol20, '.0%')} -> {_fmt(vol_component, '+.2f')}, "
            f"drawdown proximity {_fmt(dist_high, '+.1%')}, "
            f"vol regime {_fmt(regime_ratio, '.2f')}x"
        )
        return score, evidence, rationale


# --------------------------------------------------------------------------- #
# Registry + runner
# --------------------------------------------------------------------------- #
ANALYSTS: dict[str, type[Analyst]] = {
    TechnicalAnalyst.name: TechnicalAnalyst,
    MomentumAnalyst.name: MomentumAnalyst,
    MeanReversionAnalyst.name: MeanReversionAnalyst,
    RiskAnalyst.name: RiskAnalyst,
}


def run_analysts(ctx: MarketContext, cfg: AgentConfig, llm: Any = None) -> list[Stance]:
    """Run the configured analysts for one bar, deterministic first.

    * Unknown analyst names are ignored; an empty/invalid configuration falls
      back to :data:`DEFAULT_ANALYST_ORDER`.
    * During warmup (fewer closed bars than ``cfg.min_bars_for_signals``) every
      analyst returns a neutral, low-confidence stance: the cheapest correct
      answer when indicators are not yet defined.
    * When ``llm`` is available each analyst gets at most **one** compact JSON
      call to refine label/confidence/rationale; any problem falls back to the
      deterministic stance (``source="deterministic"``).
    """
    requested = list(cfg.analyst_names or DEFAULT_ANALYST_ORDER)
    names = list(dict.fromkeys(name for name in requested if name in ANALYSTS))
    if not names:
        names = list(DEFAULT_ANALYST_ORDER)

    bars = tuple(ctx.bars or ())
    min_bars = max(0, int(cfg.min_bars_for_signals))
    if len(bars) < min_bars:
        return [
            Stance(
                agent=name,
                label="neutral",
                confidence=0.10,
                rationale=(f"warmup: {len(bars)} closed bar(s) < "
                           f"{min_bars} required for {name} signals"),
                evidence=(f"bars={len(bars)}",),
                source="deterministic",
            )
            for name in names
        ]

    use_llm = llm is not None and bool(getattr(llm, "available", False))
    stances: list[Stance] = []
    for name in names:
        analyst = ANALYSTS[name]()
        stance = analyst.assess(ctx)
        if use_llm:
            try:
                stance = analyst.refine_with_llm(ctx, stance, llm)
            except Exception:
                pass  # keep the deterministic stance
        stances.append(stance)
    return stances
