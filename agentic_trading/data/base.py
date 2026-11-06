"""Shared plumbing for the market-data layer.

This module owns everything that is common to every vendor adapter:

* timeframe arithmetic and UTC normalisation;
* the closed-bar rule (a bar is only usable once ``ts + step <= end``);
* defensive bar validation/sorting, so one bad vendor row cannot poison the
  feature frame;
* bounded, timeout-enforced JSON HTTP with retries;
* :class:`ProviderChain` -- the fallback chain the rest of the system calls;
* :func:`build_providers` -- the keyless provider factory.

Importing this module never touches the network and needs no API key.
"""

from __future__ import annotations

import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

import requests

from ..config import DataConfig
from ..contracts import SUPPORTED_TIMEFRAMES, Bar, DataProvider
from ..errors import DataUnavailableError, DataValidationError

__all__ = [
    "TIMEFRAME_SECONDS",
    "timeframe_seconds",
    "ensure_utc",
    "floor_to_step",
    "last_closed_bar_start",
    "normalise_bars",
    "window_bars",
    "request_json",
    "ProviderChain",
    "build_providers",
    "make_provider",
    "make_session",
]

# Canonical timeframe -> seconds.  Mirrors contracts.SUPPORTED_TIMEFRAMES.
TIMEFRAME_SECONDS: dict[str, int] = {"15m": 900, "1h": 3600, "1d": 86400}

_PROVIDER_NAMES = ("coinbase", "twelvedata", "yahoo", "synthetic")
_NETWORK_PROVIDER_NAMES = frozenset({"coinbase", "twelvedata", "yahoo"})


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def timeframe_seconds(timeframe: str) -> int:
    """Seconds per bar for ``timeframe``; raises ``DataValidationError`` if bad."""
    key = str(timeframe or "1d").strip().lower()
    try:
        return TIMEFRAME_SECONDS[key]
    except KeyError:
        raise DataValidationError(
            f"unsupported timeframe {timeframe!r}; expected one of "
            f"{', '.join(SUPPORTED_TIMEFRAMES)}"
        ) from None


def ensure_utc(value: Any) -> datetime:
    """Coerce ``value`` to a timezone-aware UTC datetime.

    Accepts ``datetime`` (naive values are assumed UTC), ``date``, ISO-8601
    strings and unix epoch seconds.  Anything else raises
    ``DataValidationError`` rather than a bare ``TypeError``/``ValueError``.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise DataValidationError(f"invalid ISO datetime {value!r}: {exc}") from exc
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise DataValidationError(f"invalid epoch timestamp {value!r}: {exc}") from exc
    else:
        raise DataValidationError(f"cannot interpret {value!r} as a datetime")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def floor_to_step(ts: datetime, step: int) -> datetime:
    """Floor a UTC datetime to a multiple of ``step`` seconds since the epoch."""
    ts = ensure_utc(ts)
    seconds = math.floor(ts.timestamp() / step) * step
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc)


def last_closed_bar_start(end: datetime, step: int) -> datetime:
    """Start time of the newest bar whose *close* (``ts + step``) is ``<= end``."""
    end = ensure_utc(end)
    seconds = math.floor((end.timestamp() - step) / step) * step
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc)


# --------------------------------------------------------------------------- #
# Bar validation
# --------------------------------------------------------------------------- #
def normalise_bars(
    bars: Sequence[Bar] | Sequence[Any] | None,
    start: datetime | None = None,
    end: datetime | None = None,
    timeframe: str = "1d",
) -> list[Bar]:
    """Return valid bars sorted ascending, deduplicated by ``ts``.

    * timestamps coerced to tz-aware UTC;
    * only closed bars are kept when ``end`` is given
      (``ts + step <= end``);
    * bars before ``start`` are dropped when ``start`` is given;
    * non-finite / non-positive prices and negative volume are discarded;
    * ``high``/``low`` are clamped so ``low <= open, close <= high``
      (some vendors publish slightly inconsistent candles);
    * duplicate timestamps keep the *last* value seen.
    """
    if not bars:
        return []
    step = timeframe_seconds(timeframe)
    step_delta = timedelta(seconds=step)
    start_utc = ensure_utc(start) if start is not None else None
    end_utc = ensure_utc(end) if end is not None else None

    by_ts: dict[datetime, Bar] = {}
    for raw in bars:
        try:
            ts = ensure_utc(raw.ts)
            open_ = float(raw.open)
            high = float(raw.high)
            low = float(raw.low)
            close = float(raw.close)
            volume = float(getattr(raw, "volume", 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            continue

        if not all(math.isfinite(x) for x in (open_, high, low, close, volume)):
            continue
        if min(open_, high, low, close) <= 0.0 or volume < 0.0:
            continue
        if end_utc is not None and ts + step_delta > end_utc:
            continue
        if start_utc is not None and ts < start_utc:
            continue

        if high < low:
            continue
        high = max(high, open_, close)
        low = min(low, open_, close)
        by_ts[ts] = Bar(ts=ts, open=open_, high=high, low=low, close=close, volume=volume)

    return [by_ts[key] for key in sorted(by_ts)]


def window_bars(
    bars: Sequence[Bar],
    start: datetime | None,
    end: datetime | None,
    timeframe: str = "1d",
) -> list[Bar]:
    """Slice already-valid bars to ``start <= ts`` and ``ts + step <= end``."""
    step = timeframe_seconds(timeframe)
    step_delta = timedelta(seconds=step)
    start_utc = ensure_utc(start) if start is not None else None
    end_utc = ensure_utc(end) if end is not None else None
    out: list[Bar] = []
    for bar in bars:
        ts = ensure_utc(bar.ts)
        if start_utc is not None and ts < start_utc:
            continue
        if end_utc is not None and ts + step_delta > end_utc:
            continue
        out.append(bar)
    return out


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def make_session(cfg: DataConfig | None = None) -> requests.Session:
    """A ``requests`` session with the configured user agent (never used offline)."""
    session = requests.Session()
    ua = getattr(cfg, "request_user_agent", None) if cfg is not None else None
    if ua:
        session.headers["User-Agent"] = str(ua)
    return session


_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def request_json(
    cfg: DataConfig,
    url: str,
    params: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    provider: str = "provider",
) -> Any:
    """GET ``url`` and decode JSON, with timeout + bounded exponential retries.

    Always raises exactly ``DataUnavailableError`` on transport/HTTP failure or
    ``DataValidationError`` on undecodable bodies -- never a raw ``requests``
    exception.  Honours ``cfg.allow_network``, ``cfg.timeout_s``,
    ``cfg.max_retries`` and ``cfg.retry_backoff_s``.
    """
    if not bool(getattr(cfg, "allow_network", True)):
        raise DataUnavailableError(f"{provider}: network access disabled (allow_network=False)")

    timeout = float(getattr(cfg, "timeout_s", 20.0) or 20.0)
    attempts = max(1, int(getattr(cfg, "max_retries", 0) or 0) + 1)
    backoff = max(0.0, float(getattr(cfg, "retry_backoff_s", 0.0) or 0.0))

    hdrs: dict[str, str] = {}
    default_ua = getattr(cfg, "request_user_agent", None)
    if default_ua:
        hdrs["User-Agent"] = str(default_ua)
    if headers:
        hdrs.update(headers)

    last_error = "unknown error"
    for attempt in range(attempts):
        response: requests.Response | None = None
        try:
            response = requests.get(url, params=params, headers=hdrs, timeout=timeout)
        except requests.RequestException as exc:  # timeout, DNS, connection reset, ...
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise DataValidationError(
                        f"{provider}: invalid JSON from {url}: {exc}"
                    ) from exc
            last_error = f"HTTP {response.status_code}"
            if response.status_code not in _RETRYABLE_STATUS:
                snippet = (response.text or "")[:200].replace("\n", " ")
                raise DataUnavailableError(
                    f"{provider}: {last_error} from {url}: {snippet}"
                )

        if attempt + 1 < attempts and backoff > 0.0:
            time.sleep(backoff * (2 ** attempt))

    raise DataUnavailableError(
        f"{provider}: request failed after {attempts} attempt(s): {last_error}"
    )


# --------------------------------------------------------------------------- #
# Provider factory
# --------------------------------------------------------------------------- #
def make_provider(name: str, cfg: DataConfig | None = None) -> DataProvider:
    """Instantiate one provider by config name.

    Providers are cheap to construct and do *not* touch the network here; they
    raise ``DataUnavailableError`` from ``fetch_ohlcv`` when unusable (missing
    key, network disabled, bad symbol).  Unknown names raise ``KeyError``.
    """
    key = str(name).strip().lower()
    config = cfg if cfg is not None else DataConfig()
    if key == "coinbase":
        from .coinbase import CoinbaseProvider

        return CoinbaseProvider(config)
    if key == "twelvedata":
        from .twelvedata import TwelveDataProvider

        return TwelveDataProvider(config)
    if key == "yahoo":
        from .yahoo import YahooProvider

        return YahooProvider(config)
    if key == "synthetic":
        from .synthetic import SyntheticProvider

        return SyntheticProvider(config)
    raise KeyError(f"unknown data provider {name!r}; known: {', '.join(_PROVIDER_NAMES)}")


def build_providers(cfg: DataConfig) -> list[DataProvider]:
    """Build the ordered provider list from ``cfg.provider_order``.

    Unknown names are ignored.  When ``cfg.allow_network`` is false the network
    vendors are omitted entirely (so an offline run cannot even attempt a
    socket), and if that leaves nothing the synthetic provider is used.
    """
    config = cfg if cfg is not None else DataConfig()
    order = [str(name).strip().lower() for name in (config.provider_order or [])]
    providers: list[DataProvider] = []
    seen: set[str] = set()
    for name in order:
        if not name or name in seen:
            continue
        seen.add(name)
        if not bool(getattr(config, "allow_network", True)) and name in _NETWORK_PROVIDER_NAMES:
            continue
        try:
            providers.append(make_provider(name, config))
        except KeyError:
            continue

    if not providers:
        from .synthetic import SyntheticProvider

        providers.append(SyntheticProvider(config))
    return providers


# --------------------------------------------------------------------------- #
# Fallback chain
# --------------------------------------------------------------------------- #
class ProviderChain:
    """Try providers in order; the first usable response wins.

    ``last_provider`` is the name of the provider that served the most recent
    successful request (``"none"`` before the first success or after total
    failure).  ``last_errors`` keeps per-provider failure notes for diagnostics.
    """

    def __init__(
        self,
        cfg: DataConfig,
        providers: Sequence[DataProvider] | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else DataConfig()
        self.providers: list[DataProvider] = (
            list(providers) if providers is not None else build_providers(self.cfg)
        )
        self.last_provider: str = "none"
        self.last_errors: list[str] = []

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Bar]:
        """Return ascending, UTC, closed bars for ``symbol`` from the first
        provider that can serve them; raise ``DataUnavailableError`` otherwise."""
        timeframe = str(timeframe or getattr(self.cfg, "timeframe", "1d"))
        timeframe_seconds(timeframe)  # validate early -> DataValidationError
        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if end_utc < start_utc:
            raise DataValidationError(
                f"end ({end_utc.isoformat()}) precedes start ({start_utc.isoformat()})"
            )

        errors: list[str] = []

        def _note(provider_name: str, message: str) -> str:
            return message if message.startswith(f"{provider_name}:") else f"{provider_name}: {message}"

        for provider in self.providers:
            name = str(getattr(provider, "name", type(provider).__name__))
            try:
                raw_bars = provider.fetch_ohlcv(symbol, start_utc, end_utc, timeframe)
            except (DataUnavailableError, DataValidationError) as exc:
                errors.append(_note(name, str(exc)))
                continue
            except Exception as exc:  # defensive: a provider bug must not kill the run
                errors.append(_note(name, f"unexpected {type(exc).__name__}: {exc}"))
                continue

            bars = normalise_bars(raw_bars, end=end_utc, timeframe=timeframe)
            if not bars:
                errors.append(f"{name}: returned no usable bars")
                continue

            self.last_provider = name
            self.last_errors = errors
            return bars

        self.last_provider = "none"
        self.last_errors = errors
        detail = "; ".join(errors) if errors else "no providers configured"
        raise DataUnavailableError(
            f"no provider could serve {symbol} {timeframe} "
            f"[{start_utc.isoformat()} .. {end_utc.isoformat()}]: {detail}"
        )
