"""Shared deterministic test fixtures and synthetic data builders.

Everything here is offline and seeded, so no test ever needs the network or an
API key.  The helpers are plain module-level functions (importable as
``from tests.conftest import make_bars, make_context, ...``) *and* exposed as
pytest fixtures for teammate test modules that prefer fixtures.

``make_context`` uses the real data-layer indicator code
(``agentic_trading.data.indicators.compute_features``) so agent tests exercise
exactly the feature contract that production uses.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

import numpy as np
import pytest

from agentic_trading.config import AppConfig
from agentic_trading.contracts import (
    REQUIRED_FEATURES,
    Bar,
    MarketContext,
    PortfolioState,
    Position,
)
from agentic_trading.data.indicators import compute_features

START = datetime(2024, 1, 1, tzinfo=timezone.utc)  # fixed, timezone-aware UTC
_DAY = timedelta(days=1)
_STEP = {"1d": _DAY, "1h": timedelta(hours=1), "15m": timedelta(minutes=15)}


# --------------------------------------------------------------------------- #
# Synthetic OHLCV + MarketContext
# --------------------------------------------------------------------------- #
def make_bars(
    n: int = 260,
    *,
    trend: float = 0.001,
    vol: float = 0.02,
    seed: int = 7,
    start_price: float = 100.0,
    start: datetime = START,
    timeframe: str = "1d",
    volume_base: float = 1_000.0,
) -> tuple[Bar, ...]:
    """Deterministic geometric random walk with valid OHLC relationships.

    ``trend`` is the per-bar drift and ``vol`` the per-bar relative shock, so
    ``trend=0.004, vol=0.02`` is a clean daily uptrend and ``trend=-0.004`` a
    clean downtrend.  ``high``/``low`` always bracket open/close and volumes are
    strictly positive, which is what real vendors guarantee.
    """
    if n <= 0:
        return ()
    rng = np.random.default_rng(seed)
    step = _STEP.get(str(timeframe), _DAY)
    bars: list[Bar] = []
    price = float(start_price)
    for i in range(int(n)):
        open_price = price
        close_price = max(1e-6, price * (1.0 + trend + vol * float(rng.normal())))
        high = max(open_price, close_price) * (1.0 + abs(float(rng.normal(0.0, vol / 3.0))))
        low = min(open_price, close_price) * (1.0 - abs(float(rng.normal(0.0, vol / 3.0))))
        low = max(low, 1e-9)
        volume = max(1.0, volume_base * (1.0 + 0.3 * float(rng.normal())))
        bars.append(Bar(ts=start + i * step, open=open_price, high=high,
                        low=low, close=close_price, volume=volume))
        price = close_price
    return tuple(bars)


def make_context(
    n: int = 260,
    *,
    trend: float = 0.001,
    vol: float = 0.02,
    seed: int = 7,
    start_price: float = 100.0,
    symbol: str = "BTC-USD",
    timeframe: str = "1d",
    features: Mapping[str, float] | None = None,
    meta: Mapping[str, Any] | None = None,
    bars: Iterable[Bar] | None = None,
) -> MarketContext:
    """Build a :class:`MarketContext` whose features come from the real indicator code.

    ``features`` overrides individual feature values after computation, which is
    how tests construct a precise, regime-independent signal while keeping the
    bar history realistic.
    """
    bar_tuple = tuple(bars) if bars is not None else make_bars(
        n=n, trend=trend, vol=vol, seed=seed, start_price=start_price, timeframe=timeframe)
    if not bar_tuple:
        raise ValueError("make_context needs at least one bar")
    computed = compute_features(bar_tuple, timeframe=timeframe)
    if features:
        for key, value in features.items():
            try:
                computed[key] = float("nan") if value is None else float(value)
            except (TypeError, ValueError):
                computed[key] = float("nan")
    # Guarantee the frozen contract: every REQUIRED_FEATURES key exists.
    for key in REQUIRED_FEATURES:
        computed.setdefault(key, float("nan"))
    return MarketContext(
        symbol=symbol,
        asof=bar_tuple[-1].ts,
        bars=bar_tuple,
        features=computed,
        timeframe=timeframe,
        provider="synthetic",
        meta=dict(meta or {}),
    )


def make_portfolio(
    *,
    symbol: str = "BTC-USD",
    equity: float = 10_000.0,
    cash: float | None = None,
    qty: float = 0.0,
    avg_price: float | None = None,
    price: float = 100.0,
    peak_equity: float | None = None,
    bar_index: int = 0,
) -> PortfolioState:
    """Build a :class:`PortfolioState`; ``qty != 0`` implies an open position."""
    price = float(price)
    positions: dict[str, Position] = {}
    if abs(qty) > 1e-12:
        positions[symbol] = Position(
            symbol=symbol, qty=float(qty),
            avg_price=float(avg_price if avg_price is not None else price))
    equity = float(equity)
    cash_value = float(cash) if cash is not None else equity - qty * price
    return PortfolioState(
        cash=cash_value,
        positions=positions,
        equity=equity,
        peak_equity=float(peak_equity if peak_equity is not None else equity),
        bar_index=int(bar_index),
        last_prices={symbol: price},
    )


# --------------------------------------------------------------------------- #
# Hand-crafted, complete feature sets (all REQUIRED_FEATURES keys present)
# --------------------------------------------------------------------------- #
BULLISH_FEATURES: dict[str, float] = {
    "close": 120.0, "ret_1": 0.010, "ret_5": 0.040, "ret_20": 0.150, "ret_60": 0.350,
    "sma_20": 118.0, "sma_50": 112.0, "sma_200": 102.0, "ema_12": 118.5, "ema_26": 114.0,
    "rsi_14": 66.0, "macd": 2.40, "macd_signal": 1.70, "macd_hist": 0.70,
    "atr_14": 1.80, "atr_pct": 0.015,
    "bb_upper": 125.0, "bb_lower": 111.0, "bb_pctb": 0.68, "bb_width": 0.120,
    "vol_20": 0.350, "vol_60": 0.320, "volume_z": 1.20, "mom_20": 0.150,
    "price_vs_sma50": 0.0714, "price_vs_sma200": 0.1765,
    "trend_up": 1.0, "above_sma50": 1.0, "above_sma200": 1.0,
    "dist_from_high_60": -0.020, "dist_from_low_60": 0.400,
}

BEARISH_FEATURES: dict[str, float] = {
    "close": 80.0, "ret_1": -0.015, "ret_5": -0.050, "ret_20": -0.180, "ret_60": -0.350,
    "sma_20": 84.0, "sma_50": 90.0, "sma_200": 100.0, "ema_12": 82.0, "ema_26": 86.0,
    "rsi_14": 28.0, "macd": -2.50, "macd_signal": -1.80, "macd_hist": -0.70,
    "atr_14": 2.20, "atr_pct": 0.0275,
    "bb_upper": 92.0, "bb_lower": 76.0, "bb_pctb": 0.22, "bb_width": 0.190,
    "vol_20": 0.650, "vol_60": 0.500, "volume_z": 1.50, "mom_20": -0.180,
    "price_vs_sma50": -0.111, "price_vs_sma200": -0.200,
    "trend_up": 0.0, "above_sma50": 0.0, "above_sma200": 0.0,
    "dist_from_high_60": -0.300, "dist_from_low_60": 0.050,
}

CALM_FEATURES: dict[str, float] = {
    "close": 100.0, "ret_1": 0.001, "ret_5": 0.003, "ret_20": 0.006, "ret_60": 0.010,
    "sma_20": 99.8, "sma_50": 99.5, "sma_200": 99.0, "ema_12": 99.9, "ema_26": 99.6,
    "rsi_14": 52.0, "macd": 0.10, "macd_signal": 0.08, "macd_hist": 0.02,
    "atr_14": 0.80, "atr_pct": 0.008,
    "bb_upper": 102.0, "bb_lower": 98.0, "bb_pctb": 0.50, "bb_width": 0.040,
    "vol_20": 0.180, "vol_60": 0.200, "volume_z": 0.10, "mom_20": 0.006,
    "price_vs_sma50": 0.0050, "price_vs_sma200": 0.0101,
    "trend_up": 0.0, "above_sma50": 1.0, "above_sma200": 1.0,
    "dist_from_high_60": -0.010, "dist_from_low_60": 0.020,
}

STRESSED_FEATURES: dict[str, float] = {
    **BEARISH_FEATURES,
    "atr_pct": 0.060, "vol_20": 0.900, "vol_60": 0.500,
}

NAN_FEATURES: dict[str, float] = {name: float("nan") for name in REQUIRED_FEATURES}


# --------------------------------------------------------------------------- #
# Scripted LLM stand-ins (duck-typed against agents/llm.LLMClient)
# --------------------------------------------------------------------------- #
class ScriptedLLM:
    """Deterministic fake LLM: replays scripted responses, counts calls."""

    def __init__(
        self,
        responses: Iterable[Any] | None = None,
        *,
        available: bool = True,
        fail: bool = False,
        model: str = "scripted",
        raises: bool = False,
    ) -> None:
        self.responses = list(responses or [])
        self.available = bool(available)
        self.model = model
        self.fail = bool(fail)
        self.raises = bool(raises)
        self.calls = 0
        self.failures = 0
        self.last_error: str | None = None
        self.prompts: list[tuple[str, str]] = []

    def _next(self) -> Any:
        if self.raises:
            raise RuntimeError("scripted LLM transport failure")
        if self.fail or not self.responses:
            return None
        return self.responses[(self.calls - 1) % len(self.responses)]

    def complete_text(self, system: str, user: str) -> str:
        self.calls += 1
        self.prompts.append((system, user))
        value = self._next()
        if isinstance(value, str):
            return value
        self.failures += 1
        return ""

    def complete_json(self, system: str, user: str, schema_hint: str = "{}"):
        self.calls += 1
        self.prompts.append((system, user))
        value = self._next()
        if isinstance(value, dict):
            return dict(value)
        self.failures += 1
        return None


class ExplodingLLM:
    """Available-looking client whose every call raises; proves graceful fallback."""

    available = True
    model = "exploding"
    calls = 0
    failures = 0
    last_error: str | None = None

    def complete_text(self, system: str, user: str) -> str:
        raise RuntimeError("boom")

    def complete_json(self, system: str, user: str, schema_hint: str = "{}"):
        raise RuntimeError("boom")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def cfg() -> AppConfig:
    """Default app config with the LLM explicitly disabled (pure deterministic)."""
    config = AppConfig()
    config.llm.enabled = "off"
    return config


@pytest.fixture
def flat_portfolio() -> PortfolioState:
    return make_portfolio()


@pytest.fixture
def bullish_ctx() -> MarketContext:
    return make_context(features=BULLISH_FEATURES)


@pytest.fixture
def bearish_ctx() -> MarketContext:
    return make_context(features=BEARISH_FEATURES)


@pytest.fixture
def calm_ctx() -> MarketContext:
    return make_context(features=CALM_FEATURES)


@pytest.fixture
def long_portfolio() -> PortfolioState:
    price = 120.0
    return make_portfolio(qty=20.0, price=price, avg_price=price)
