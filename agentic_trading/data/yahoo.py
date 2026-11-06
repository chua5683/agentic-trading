"""Yahoo Finance chart adapter (keyless, best-effort).

Yahoo's undocumented ``v8/finance/chart`` endpoint is useful for equities
without an API key, but it is aggressively rate-limited and can return HTTP
429 / 401 at any time.  The contract for this adapter is therefore:
**any non-200 becomes ``DataUnavailableError``** so ``ProviderChain`` can fall
through to the next vendor instead of crashing the run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..config import DataConfig
from ..contracts import Bar
from ..errors import DataUnavailableError, DataValidationError
from .base import ensure_utc, normalise_bars, request_json

__all__ = ["YahooProvider"]

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_INTERVALS = {"1d": "1d", "1h": "1h", "15m": "15m"}


class YahooProvider:
    """Keyless Yahoo Finance adapter; raises ``DataUnavailableError`` on any failure."""

    name = "yahoo"

    def __init__(self, cfg: DataConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else DataConfig()

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
            raise DataUnavailableError(f"yahoo: unsupported timeframe {timeframe!r}")

        text = str(symbol or "").strip()
        if not text:
            raise DataUnavailableError("yahoo: empty symbol")

        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if end_utc < start_utc:
            raise DataValidationError("yahoo: end precedes start")

        now = datetime.now(timezone.utc)
        upper = min(end_utc, now)
        period1 = int(start_utc.timestamp())
        period2 = int(upper.timestamp())
        if period2 <= period1:
            raise DataUnavailableError("yahoo: empty requested window")

        payload = request_json(
            self.cfg,
            _CHART_URL.format(symbol=text),
            params={
                "interval": interval,
                "period1": period1,
                "period2": period2,
                "includePrePost": "false",
                "events": "div,splits",
            },
            headers={"User-Agent": _BROWSER_UA, "Accept": "application/json"},
            provider="yahoo",
        )
        if not isinstance(payload, dict):
            raise DataValidationError("yahoo: unexpected response shape")

        chart = payload.get("chart")
        if not isinstance(chart, dict):
            raise DataUnavailableError("yahoo: response missing 'chart'")
        if chart.get("error"):
            raise DataUnavailableError(f"yahoo: {chart['error']}")
        results = chart.get("result")
        if not results:
            raise DataUnavailableError(f"yahoo: no result for {text}")
        result = results[0]
        if not isinstance(result, dict):
            raise DataValidationError("yahoo: malformed chart result")

        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quotes = indicators.get("quote") or []
        quote = quotes[0] if quotes else {}
        if not isinstance(quote, dict):
            raise DataValidationError("yahoo: malformed quote block")

        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []

        bars: list[Bar] = []
        for i, epoch in enumerate(timestamps):
            try:
                open_ = opens[i]
                high = highs[i]
                low = lows[i]
                close = closes[i]
            except IndexError:
                break
            if open_ is None or high is None or low is None or close is None:
                continue  # Yahoo pads holidays / halts with nulls
            volume = volumes[i] if i < len(volumes) and volumes[i] is not None else 0.0
            try:
                ts = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
                bars.append(
                    Bar(
                        ts=ts,
                        open=float(open_),
                        high=float(high),
                        low=float(low),
                        close=float(close),
                        volume=float(volume),
                    )
                )
            except (TypeError, ValueError, OverflowError, OSError):
                continue

        usable = normalise_bars(
            bars, start=start_utc, end=end_utc, timeframe=timeframe
        )
        if not usable:
            raise DataUnavailableError(
                f"yahoo: no usable candles for {text} "
                f"({start_utc.isoformat()}..{end_utc.isoformat()})"
            )
        return usable
