"""Deterministic offline synthetic OHLCV provider.

The whole system must run with **zero API keys and zero network**.  This
provider fabricates a plausible price history so every downstream component
(feature frame, agents, backtest, reports) is exercised identically offline.

Design guarantees:

* **Deterministic**: the RNG seed is ``sha256(symbol|timeframe)``; the same
  request always yields byte-identical bars.
* **Stable history**: the random walk starts from a *fixed per-timeframe
  anchor* and is generated with independent, prefix-stable RNG streams, so a
  bar's OHLCV does not change when ``end`` (or the requested window) moves --
  offline backtests therefore do not get rewritten histories.
* **Crypto vs equities**: symbols ending in ``-USD`` are 24/7; anything else
  is treated as an equity and gets weekday-only bars.
* **Always available**: never raises ``DataUnavailableError``.
* **Contract-safe**: ascending, tz-aware UTC, no bar whose close is after
  ``end``, and at least ``max(400, requested span)`` bars when the anchor
  allows it.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from ..config import DataConfig
from ..contracts import Bar
from ..errors import DataValidationError
from .base import ensure_utc, last_closed_bar_start, timeframe_seconds

__all__ = ["SyntheticProvider"]

_STREAM_XOR = {
    "returns": 0x9E3779B97F4A7C15,
    "high": 0xC2B2AE3D27D4EB4F,
    "low": 0x165667B19E3779F9,
    "volume": 0x85EBCA77C2B2AE63,
}

# Fixed generation anchors keep historical bars stable across calls.
_FIXED_ANCHORS: dict[str, datetime] = {
    "1d": datetime(2000, 1, 1, tzinfo=timezone.utc),
    "1h": datetime(2018, 1, 1, tzinfo=timezone.utc),
    "15m": datetime(2024, 1, 1, tzinfo=timezone.utc),
}
_MAX_SLOTS = 750_000  # memory/perf guard for absurd historical requests
_MIN_BARS = 400


class SyntheticProvider:
    """Fully offline, reproducible GBM price generator."""

    name = "synthetic"

    def __init__(self, cfg: DataConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else DataConfig()
        # (symbol, timeframe, anchor_epoch) -> generated slot arrays
        self._paths: dict[tuple[str, str, int], dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Bar]:
        timeframe = str(timeframe or getattr(self.cfg, "timeframe", "1d"))
        step = timeframe_seconds(timeframe)
        step_delta = timedelta(seconds=step)

        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if end_utc < start_utc:
            raise DataValidationError("synthetic: end precedes start")

        symbol_key = str(symbol or "").strip().upper()
        if not symbol_key:
            raise DataValidationError("synthetic: empty symbol")
        is_crypto = symbol_key.endswith("-USD")

        last_ts = last_closed_bar_start(end_utc, step)
        if not is_crypto:
            while last_ts.weekday() >= 5:  # Sat/Sun -> previous Friday
                last_ts -= step_delta

        anchor = self._anchor_for(timeframe, start_utc, last_ts)
        anchor_epoch = int(anchor.timestamp())
        total_slots = int((last_ts.timestamp() - anchor_epoch) // step) + 1
        if total_slots > _MAX_SLOTS:
            # Degrade gracefully: keep the most recent _MAX_SLOTS bars.
            anchor_epoch = int(last_ts.timestamp()) - (step * (_MAX_SLOTS - 1))
            total_slots = _MAX_SLOTS
        total_slots = max(1, total_slots)

        path = self._path(symbol_key, timeframe, anchor_epoch, total_slots, step, is_crypto)
        slots: np.ndarray = path["slots"][:total_slots]
        opens: np.ndarray = path["open"][:total_slots]
        highs: np.ndarray = path["high"][:total_slots]
        lows: np.ndarray = path["low"][:total_slots]
        closes: np.ndarray = path["close"][:total_slots]
        volumes: np.ndarray = path["volume"][:total_slots]
        mask: np.ndarray | None = path["mask"]
        if mask is not None:
            valid_idx = np.flatnonzero(mask[:total_slots])
        else:
            valid_idx = np.arange(total_slots, dtype=np.int64)

        if len(valid_idx) == 0:
            raise DataValidationError("synthetic: no bars generated")

        valid_slots = slots[valid_idx]
        start_epoch = int(start_utc.timestamp())
        first = int(np.searchsorted(valid_slots, start_epoch, side="left"))
        span = int(len(valid_slots) - first)
        n_bars = max(_MIN_BARS, span)
        n_bars = min(n_bars, len(valid_slots))
        selected = valid_idx[len(valid_idx) - n_bars:]

        bars = [
            Bar(
                ts=datetime.fromtimestamp(int(slots[i]), tz=timezone.utc),
                open=float(opens[i]),
                high=float(highs[i]),
                low=float(lows[i]),
                close=float(closes[i]),
                volume=float(volumes[i]),
            )
            for i in selected
        ]
        return bars

    # ------------------------------------------------------------------ #
    @staticmethod
    def _anchor_for(timeframe: str, start: datetime, last_ts: datetime) -> datetime:
        anchor = _FIXED_ANCHORS.get(timeframe, _FIXED_ANCHORS["1d"])
        if start < anchor:
            anchor = datetime(start.year - 1, 1, 1, tzinfo=timezone.utc)
        if last_ts < anchor:
            anchor = datetime(last_ts.year - 1, 1, 1, tzinfo=timezone.utc)
        return anchor

    def _path(
        self,
        symbol: str,
        timeframe: str,
        anchor_epoch: int,
        total_slots: int,
        step: int,
        is_crypto: bool,
    ) -> dict[str, Any]:
        key = (symbol, timeframe, anchor_epoch)
        cached = self._paths.get(key)
        if cached is not None and int(cached["total"]) >= total_slots:
            return cached
        path = self._generate(symbol, timeframe, anchor_epoch, total_slots, step, is_crypto)
        self._paths[key] = path
        return path

    @staticmethod
    def _generate(
        symbol: str,
        timeframe: str,
        anchor_epoch: int,
        total_slots: int,
        step: int,
        is_crypto: bool,
    ) -> dict[str, Any]:
        seed = _seed(symbol, timeframe)
        meta = np.random.default_rng(seed)
        sigma_ann = float(np.clip(0.20 + 0.55 * meta.random(), 0.15, 0.85))
        if not is_crypto:
            sigma_ann = float(np.clip(sigma_ann * 0.45, 0.10, 0.45))
        mu_ann = float(-0.02 + 0.14 * meta.random())
        base_price = float(10.0 ** (1.0 + 2.0 * meta.random()))
        base_volume = float(10.0 ** (5.0 + 1.5 * meta.random()))

        returns_rng = np.random.default_rng(seed ^ _STREAM_XOR["returns"])
        high_rng = np.random.default_rng(seed ^ _STREAM_XOR["high"])
        low_rng = np.random.default_rng(seed ^ _STREAM_XOR["low"])
        volume_rng = np.random.default_rng(seed ^ _STREAM_XOR["volume"])

        z = returns_rng.standard_normal(total_slots)
        u_high = high_rng.random(total_slots)
        u_low = low_rng.random(total_slots)
        u_volume = volume_rng.random(total_slots)

        dt_years = step / (365.0 * 86400.0)
        log_drift = (mu_ann - 0.5 * sigma_ann * sigma_ann) * dt_years
        log_vol = sigma_ann * math.sqrt(dt_years)
        log_returns = log_drift + log_vol * z
        log_close = math.log(base_price) + np.cumsum(log_returns)
        close = np.exp(log_close)

        open_ = np.empty(total_slots, dtype=np.float64)
        open_[0] = base_price
        open_[1:] = close[:-1]

        intrabar = log_vol * 0.5
        high = np.maximum(open_, close) * np.exp(intrabar * u_high)
        low = np.minimum(open_, close) * np.exp(-intrabar * u_low)
        volume = base_volume * np.exp(0.5 * (u_volume - 0.5)) * (1.0 + 0.5 * np.abs(z))

        slots = anchor_epoch + np.arange(total_slots, dtype=np.int64) * step
        mask: np.ndarray | None = None
        if not is_crypto:
            # 1970-01-01 was a Thursday (Python weekday 3); Mon..Fri -> 0..4.
            day_index = slots // 86400
            mask = (((day_index + 3) % 7) < 5)

        return {
            "total": total_slots,
            "slots": slots,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "mask": mask,
        }


def _seed(symbol: str, timeframe: str) -> int:
    digest = hashlib.sha256(f"{symbol}|{timeframe}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)
