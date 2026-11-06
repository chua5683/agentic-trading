"""High-level market-data service used by the agents, backtest and CLI.

It glues together :class:`~agentic_trading.data.base.ProviderChain` (fallback
across vendors), :class:`~agentic_trading.data.cache.OHLCVCache` (disk cache)
and :func:`agentic_trading.data.indicators.compute_features` (frozen, causal
features) so that a caller only ever sees ``history()`` and ``context()``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

from ..config import AppConfig, DataConfig
from ..contracts import Bar, DataProvider, MarketContext
from ..errors import DataUnavailableError, DataValidationError
from .base import ProviderChain, ensure_utc, timeframe_seconds
from .cache import OHLCVCache
from .indicators import compute_features

__all__ = ["MarketDataService", "make_service"]


class MarketDataService:
    """Cached, multi-provider OHLCV access + ``MarketContext`` construction.

    ``providers`` may be supplied explicitly (tests / custom routing); otherwise
    the chain is built from ``cfg.provider_order`` (and ``cfg.allow_network``).
    """

    def __init__(
        self,
        cfg: DataConfig,
        providers: Sequence[DataProvider] | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else DataConfig()
        self.chain = ProviderChain(self.cfg, providers=providers)
        self.cache = OHLCVCache(self.cfg.cache_dir, enabled=bool(self.cfg.cache_enabled))
        self.last_provider: str = "none"

    # ------------------------------------------------------------------ #
    def history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str | None = None,
    ) -> list[Bar]:
        """Closed bars for ``symbol`` in ``[start, end]`` (ascending, UTC).

        The return value is always clipped to the requested window, whether it
        came from the cache or the network: the cache may hold a wider span
        (widest span wins on disk), but a caller -- notably a backtest -- must
        get the same bars for the same ``(start, end)`` regardless of what
        previous runs cached.  Served from the disk cache when it is fresh,
        otherwise fetched through the provider chain and merged back into the
        cache.  Synthetic bars are never persisted (that would contaminate a
        later real-data run).
        """
        timeframe = str(timeframe or self.cfg.timeframe or "1d")
        step = timeframe_seconds(timeframe)
        step_delta = timedelta(seconds=step)
        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if end_utc < start_utc:
            raise DataValidationError(
                f"end ({end_utc.isoformat()}) precedes start ({start_utc.isoformat()})"
            )

        cached = self.cache.read(symbol, timeframe)
        if self._cache_usable(cached, start_utc, end_utc, step_delta):
            usable = self._slice(cached, start_utc, end_utc, step_delta)
            if usable:
                self.last_provider = "cache"
                return usable

        try:
            fetched = self.chain.fetch_ohlcv(symbol, start_utc, end_utc, timeframe)
        except DataUnavailableError:
            # Total vendor failure: degraded mode still serves whatever the
            # cache can honestly provide for this window.
            usable = self._slice(cached, start_utc, end_utc, step_delta)
            if usable:
                self.last_provider = "cache"
                return usable
            raise

        provider = self.chain.last_provider
        self.last_provider = provider
        if provider == "synthetic" or not self.cache.enabled:
            return self._slice(fetched, start_utc, end_utc, step_delta)

        merged = self.cache.merge(symbol, timeframe, fetched)
        usable = self._slice(merged, start_utc, end_utc, step_delta)
        return usable or self._slice(fetched, start_utc, end_utc, step_delta)

    # ------------------------------------------------------------------ #
    def context(
        self,
        symbol: str,
        asof: datetime,
        timeframe: str | None = None,
        lookback: int | None = None,
    ) -> MarketContext:
        """Point-in-time :class:`MarketContext` at ``asof`` (never looks ahead).

        The returned bars are the last ``lookback`` *closed* bars whose open is
        ``<= asof``; a bar still forming at ``asof`` is dropped.  Features come
        from the frozen causal implementation in ``data.indicators``.
        """
        timeframe = str(timeframe or self.cfg.timeframe or "1d")
        step = timeframe_seconds(timeframe)
        step_delta = timedelta(seconds=step)
        asof_utc = ensure_utc(asof)
        lookback_bars = int(self.cfg.lookback_bars if lookback is None else lookback)
        if lookback_bars <= 0:
            raise DataValidationError("lookback must be a positive number of bars")

        # Fetch a generous margin so we still have ``lookback_bars`` closed bars
        # after dropping the forming bar and after provider gaps (weekends,
        # exchange holidays).  Over-fetching is cheap; lookahead is not: every
        # bar kept here is strictly closed at ``asof``.
        margin = lookback_bars * 3 + 10
        start = asof_utc - timedelta(seconds=step * margin)
        bars = self.history(symbol, start, asof_utc, timeframe)

        closed = [
            bar for bar in bars
            if ensure_utc(bar.ts) + step_delta <= asof_utc
        ]
        closed = closed[-lookback_bars:]
        if not closed:
            raise DataUnavailableError(
                f"no closed bars for {symbol} {timeframe} asof {asof_utc.isoformat()}"
            )

        features = compute_features(closed, timeframe=timeframe)
        return MarketContext(
            symbol=str(symbol),
            asof=asof_utc,
            bars=tuple(closed),
            features=features,
            timeframe=timeframe,
            provider=self.last_provider,
            meta={"lookback_bars": len(closed), "last_bar_ts": closed[-1].ts.isoformat()},
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _slice(
        bars: Sequence[Bar],
        start: datetime,
        end: datetime,
        step_delta: timedelta,
    ) -> list[Bar]:
        if not bars:
            return []
        return [
            bar for bar in bars
            if ensure_utc(bar.ts) >= start and ensure_utc(bar.ts) + step_delta <= end
        ]

    @staticmethod
    def _cache_usable(
        cached: Sequence[Bar],
        start: datetime,
        end: datetime,
        step_delta: timedelta,
    ) -> bool:
        """True when the cache already covers the whole requested window.

        * the cache must reach past ``end``: the next bar after its last one
          would close after ``end`` (i.e. no new closed bar can exist yet);
        * it must reach back to (roughly) ``start``.
        """
        if not cached:
            return False
        last_ts = ensure_utc(cached[-1].ts)
        if last_ts + 2 * step_delta <= end:
            return False  # at least one newer closed bar should exist
        first_ts = ensure_utc(cached[0].ts)
        return first_ts <= start + step_delta


def make_service(cfg: AppConfig | DataConfig | None = None) -> MarketDataService:
    """Build a :class:`MarketDataService` from an ``AppConfig`` or ``DataConfig``."""
    if cfg is None:
        data_cfg = DataConfig()
    elif isinstance(cfg, DataConfig):
        data_cfg = cfg
    elif isinstance(cfg, AppConfig):
        data_cfg = cfg.data
    else:
        raise TypeError(
            "make_service expects an AppConfig or DataConfig, "
            f"got {type(cfg).__name__}"
        )
    return MarketDataService(data_cfg)
