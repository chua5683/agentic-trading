"""Coinbase Exchange public market-data adapter (keyless).

Endpoint::

    GET https://api.exchange.coinbase.com/products/{product}/candles
        ?granularity={seconds}&start={ISO8601}&end={ISO8601}

Documented quirks this adapter handles:

* rows are ``[time, low, high, open, close, volume]`` -- note the unusual order;
* rows come back **newest first**;
* at most **300 candles per request**, and the API rejects a range that spans
  more than 300 candles, so we paginate backwards;
* bounds are inclusive, and a still-forming candle *is* returned when ``end``
  reaches into the current bar -- we drop it via the closed-bar rule.

Only USD-quoted pairs (e.g. ``BTC-USD``) are served; equities raise
``DataUnavailableError`` so the chain moves on to an equity vendor.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import DataConfig
from ..contracts import Bar
from ..errors import DataUnavailableError, DataValidationError
from .base import (
    ensure_utc,
    normalise_bars,
    request_json,
)

__all__ = ["CoinbaseProvider"]

_API_ROOT = "https://api.exchange.coinbase.com/products/{product}/candles"


class CoinbaseProvider:
    """Keyless Coinbase Exchange candle adapter for USD crypto pairs."""

    name = "coinbase"

    GRANULARITY: dict[str, int] = {"1d": 86400, "1h": 3600, "15m": 900}
    MAX_CANDLES = 300
    MAX_TOTAL_BARS = 20_000  # hard safety stop for absurd requests

    def __init__(self, cfg: DataConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else DataConfig()

    # ------------------------------------------------------------------ #
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Bar]:
        timeframe = str(timeframe or getattr(self.cfg, "timeframe", "1d"))
        product = self._product(symbol)
        try:
            granularity = self.GRANULARITY[timeframe]
        except KeyError:
            raise DataUnavailableError(
                f"coinbase: unsupported timeframe {timeframe!r}"
            ) from None

        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if end_utc < start_utc:
            raise DataValidationError("coinbase: end precedes start")

        now = datetime.now(timezone.utc)
        upper = min(end_utc, now)
        if upper <= start_utc:
            raise DataUnavailableError(
                f"coinbase: requested window {start_utc.isoformat()}..{end_utc.isoformat()} "
                "contains no elapsed time"
            )

        step = granularity
        step_delta = timedelta(seconds=step)
        span = step * (self.MAX_CANDLES - 1)
        url = _API_ROOT.format(product=product)

        rows_by_time: dict[int, tuple[float, float, float, float, float]] = {}
        cursor_end = upper
        guard = 0
        while cursor_end > start_utc and len(rows_by_time) < self.MAX_TOTAL_BARS:
            guard += 1
            if guard > 4 * self.MAX_TOTAL_BARS // self.MAX_CANDLES + 64:
                break
            chunk_start = max(start_utc, cursor_end - timedelta(seconds=span))
            payload: Any = request_json(
                self.cfg,
                url,
                params={
                    "granularity": granularity,
                    "start": _iso(chunk_start),
                    "end": _iso(cursor_end),
                },
                provider="coinbase",
            )
            if not isinstance(payload, list):
                raise DataValidationError(
                    f"coinbase: expected a JSON list of candles, got {type(payload).__name__}"
                )
            if not payload:
                if not rows_by_time:
                    raise DataUnavailableError(
                        f"coinbase: no candles for {product} in "
                        f"{start_utc.isoformat()}..{end_utc.isoformat()}"
                    )
                break

            earliest: int | None = None
            for row in payload:
                parsed = _parse_row(row)
                if parsed is None:
                    continue
                time_s, candle = parsed
                rows_by_time[time_s] = candle
                earliest = time_s if earliest is None else min(earliest, time_s)

            if earliest is None:
                raise DataValidationError("coinbase: every candle row was malformed")
            earliest_dt = datetime.fromtimestamp(earliest, tz=timezone.utc)
            if earliest_dt <= start_utc:
                break
            new_cursor = earliest_dt - step_delta
            if new_cursor >= cursor_end:  # no progress -> stop, never loop forever
                break
            cursor_end = new_cursor

        bars: list[Bar] = []
        for time_s in sorted(rows_by_time):
            low, high, open_, close, volume = rows_by_time[time_s]
            bars.append(
                Bar(
                    ts=datetime.fromtimestamp(time_s, tz=timezone.utc),
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
            )

        usable = normalise_bars(
            bars, start=start_utc, end=end_utc, timeframe=timeframe
        )
        if not usable:
            raise DataUnavailableError(
                f"coinbase: no closed candles for {product} in "
                f"{start_utc.isoformat()}..{end_utc.isoformat()}"
            )
        return usable

    # ------------------------------------------------------------------ #
    @staticmethod
    def _product(symbol: str) -> str:
        """Validate/normalise the product id (``btc-usd`` -> ``BTC-USD``)."""
        text = str(symbol or "").strip().upper()
        if not text.endswith("-USD") or text in ("-USD",):
            raise DataUnavailableError(
                f"coinbase: only USD crypto pairs are supported (got {symbol!r})"
            )
        base = text[: -len("-USD")]
        if not base or not all(ch.isalnum() for ch in base):
            raise DataUnavailableError(f"coinbase: invalid product id {symbol!r}")
        return text


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_row(row: Any) -> tuple[int, tuple[float, float, float, float, float]] | None:
    """``[time, low, high, open, close, volume]`` -> ``(epoch, candle)``."""
    if not isinstance(row, (list, tuple)) or len(row) < 6:
        return None
    try:
        time_s = int(float(row[0]))
        low = float(row[1])
        high = float(row[2])
        open_ = float(row[3])
        close = float(row[4])
        volume = float(row[5])
    except (TypeError, ValueError):
        return None
    return time_s, (low, high, open_, close, volume)
