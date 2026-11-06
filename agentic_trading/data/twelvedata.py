"""Twelve Data adapter (requires an API key, equities and crypto).

The key is read from the environment variable named by
``cfg.twelvedata_api_key_env`` (default ``TWELVEDATA_API_KEY``).  No key means
``DataUnavailableError`` -- the chain then falls through.  A vendor-side error
(e.g. the demo key refusing a symbol) is also surfaced as
``DataUnavailableError``, never as a crash.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from ..config import DataConfig
from ..contracts import Bar
from ..errors import DataUnavailableError, DataValidationError
from .base import ensure_utc, normalise_bars, request_json

__all__ = ["TwelveDataProvider"]

_TIME_SERIES_URL = "https://api.twelvedata.com/time_series"
_INTERVALS = {"1d": "1day", "1h": "1h", "15m": "15min"}


class TwelveDataProvider:
    """Twelve Data time-series adapter; silently unavailable without a key."""

    name = "twelvedata"

    def __init__(self, cfg: DataConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else DataConfig()

    # ------------------------------------------------------------------ #
    def api_key(self) -> str | None:
        env_name = str(
            getattr(self.cfg, "twelvedata_api_key_env", "TWELVEDATA_API_KEY")
            or "TWELVEDATA_API_KEY"
        )
        key = os.environ.get(env_name)
        return key or None

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Bar]:
        timeframe = str(timeframe or getattr(self.cfg, "timeframe", "1d"))
        interval = _INTERVALS.get(timeframe)
        if interval is None:
            raise DataUnavailableError(
                f"twelvedata: unsupported timeframe {timeframe!r}"
            )

        key = self.api_key()
        if not key:
            env_name = getattr(self.cfg, "twelvedata_api_key_env", "TWELVEDATA_API_KEY")
            raise DataUnavailableError(f"twelvedata: no API key in ${env_name}")

        text = str(symbol or "").strip().upper()
        if not text:
            raise DataUnavailableError("twelvedata: empty symbol")

        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if end_utc < start_utc:
            raise DataValidationError("twelvedata: end precedes start")
        now = datetime.now(timezone.utc)
        upper = min(end_utc, now)
        if upper < start_utc:
            raise DataUnavailableError("twelvedata: empty requested window")

        fmt = "%Y-%m-%d" if timeframe == "1d" else "%Y-%m-%d %H:%M:%S"
        payload: Any = request_json(
            self.cfg,
            _TIME_SERIES_URL,
            params={
                "symbol": text,
                "interval": interval,
                "start_date": start_utc.strftime(fmt),
                "end_date": upper.strftime(fmt),
                "outputsize": 5000,
                "timezone": "UTC",
                "apikey": key,
            },
            provider="twelvedata",
        )

        if not isinstance(payload, dict):
            raise DataValidationError("twelvedata: unexpected response shape")
        if str(payload.get("status", "")).lower() == "error":
            raise DataUnavailableError(f"twelvedata: {payload.get('message', 'error')}")
        values = payload.get("values")
        if not values:
            message = payload.get("message") or "no values returned"
            raise DataUnavailableError(f"twelvedata: {message}")
        if not isinstance(values, list):
            raise DataValidationError("twelvedata: 'values' is not a list")

        bars: list[Bar] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            try:
                ts = _parse_datetime(item["datetime"])
                open_ = float(item["open"])
                high = float(item["high"])
                low = float(item["low"])
                close = float(item["close"])
                volume = float(item.get("volume", 0.0) or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            bars.append(
                Bar(ts=ts, open=open_, high=high, low=low, close=close, volume=volume)
            )

        usable = normalise_bars(
            bars, start=start_utc, end=end_utc, timeframe=timeframe
        )
        if not usable:
            raise DataUnavailableError(
                f"twelvedata: no usable candles for {text} "
                f"({start_utc.isoformat()}..{end_utc.isoformat()})"
            )
        return usable


def _parse_datetime(raw: Any) -> datetime:
    text = str(raw).strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
