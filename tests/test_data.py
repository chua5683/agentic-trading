"""Offline tests for the market-data layer.

Everything here runs with **zero network and zero API keys**: network is either
disabled by ``allow_network=False`` or ``requests`` is monkeypatched, and the
synthetic provider is asserted to be fully reproducible.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from agentic_trading.config import AppConfig, DataConfig
from agentic_trading.contracts import REQUIRED_FEATURES, Bar
from agentic_trading.data.base import (
    ProviderChain,
    build_providers,
    ensure_utc,
    last_closed_bar_start,
    normalise_bars,
    timeframe_seconds,
)
from agentic_trading.data.cache import OHLCVCache
from agentic_trading.data.coinbase import CoinbaseProvider
from agentic_trading.data.service import MarketDataService, make_service
from agentic_trading.data.synthetic import SyntheticProvider
from agentic_trading.data.twelvedata import TwelveDataProvider
from agentic_trading.data.yahoo import YahooProvider
from agentic_trading.errors import DataUnavailableError, DataValidationError

UTC = timezone.utc
DAY = 86400
END = datetime(2026, 11, 6, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def offline_cfg(tmp_path, **overrides) -> DataConfig:
    values = dict(
        cache_dir=str(tmp_path / "cache"),
        allow_network=False,
        max_retries=0,
        retry_backoff_s=0.0,
        timeout_s=5.0,
    )
    values.update(overrides)
    return DataConfig(**values)


def make_bars(
    n: int,
    start: datetime,
    step: int = DAY,
    base: float = 100.0,
) -> list[Bar]:
    bars: list[Bar] = []
    for i in range(n):
        close = base + i
        bars.append(
            Bar(
                ts=start + timedelta(seconds=step * i),
                open=close - 0.5,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1_000.0 + i,
            )
        )
    return bars


def forbid_network(monkeypatch) -> None:
    """Fail the test loudly if any code path opens a socket via requests."""

    def boom(*args, **kwargs):  # noqa: ANN001 - test double
        raise AssertionError("network access attempted in an offline test")

    monkeypatch.setattr("requests.get", boom)
    monkeypatch.setattr("requests.request", boom)
    monkeypatch.setattr("requests.Session.request", boom)


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)
        self.headers: dict[str, str] = {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class BoomProvider:
    name = "boom"

    def fetch_ohlcv(self, symbol, start, end, timeframe="1d"):
        raise DataUnavailableError("vendor exploded")


class GoodProvider:
    name = "good"

    def __init__(self, bars: list[Bar] | None = None) -> None:
        self.bars = bars
        self.calls = 0

    def fetch_ohlcv(self, symbol, start, end, timeframe="1d"):
        self.calls += 1
        if self.bars is not None:
            return list(self.bars)
        start = ensure_utc(start)
        end = ensure_utc(end)
        step = timeframe_seconds(timeframe)
        count = int((end - start).total_seconds() // step) + 1
        return make_bars(max(1, count), start, step=step)


# --------------------------------------------------------------------------- #
# provider factory / chain
# --------------------------------------------------------------------------- #
def test_build_providers_offline_only_synthetic(tmp_path):
    providers = build_providers(offline_cfg(tmp_path))
    assert [p.name for p in providers] == ["synthetic"]

    chain = ProviderChain(offline_cfg(tmp_path))
    bars = chain.fetch_ohlcv("BTC-USD", END - timedelta(days=60), END)
    assert chain.last_provider == "synthetic"
    assert len(bars) >= 400


def test_build_providers_respects_order_and_ignores_unknown(tmp_path):
    cfg = offline_cfg(tmp_path, allow_network=True, provider_order=["yahoo", "nope", "synthetic"])
    providers = build_providers(cfg)
    assert [p.name for p in providers] == ["yahoo", "synthetic"]


def test_chain_falls_through_and_reports_last_provider(tmp_path):
    start = END - timedelta(days=10)
    chain = ProviderChain(offline_cfg(tmp_path), providers=[BoomProvider(), GoodProvider()])
    bars = chain.fetch_ohlcv("BTC-USD", start, END)
    assert chain.last_provider == "good"
    assert len(bars) == 10
    assert bars == sorted(bars, key=lambda b: b.ts)
    assert any("boom" in note for note in chain.last_errors)


def test_chain_falls_through_on_empty_and_unexpected_errors(tmp_path):
    class EmptyProvider:
        name = "empty"

        def fetch_ohlcv(self, *args, **kwargs):
            return []

    class BuggyProvider:
        name = "buggy"

        def fetch_ohlcv(self, *args, **kwargs):
            raise RuntimeError("vendor SDK bug")

    chain = ProviderChain(
        offline_cfg(tmp_path),
        providers=[EmptyProvider(), BuggyProvider(), GoodProvider()],
    )
    bars = chain.fetch_ohlcv("BTC-USD", END - timedelta(days=5), END)
    assert chain.last_provider == "good"
    assert len(bars) == 5  # the bar at ts=END is still forming and dropped


def test_chain_raises_data_unavailable_when_all_fail(tmp_path):
    chain = ProviderChain(offline_cfg(tmp_path), providers=[BoomProvider()])
    with pytest.raises(DataUnavailableError):
        chain.fetch_ohlcv("BTC-USD", END - timedelta(days=5), END)
    assert chain.last_provider == "none"


def test_chain_validates_timeframe_and_range(tmp_path):
    chain = ProviderChain(offline_cfg(tmp_path))
    with pytest.raises(DataValidationError):
        chain.fetch_ohlcv("BTC-USD", END - timedelta(days=5), END, timeframe="7m")
    with pytest.raises(DataValidationError):
        chain.fetch_ohlcv("BTC-USD", END, END - timedelta(days=5))


# --------------------------------------------------------------------------- #
# bar normalisation
# --------------------------------------------------------------------------- #
def test_normalise_bars_sorts_dedupes_and_drops_forming():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = make_bars(4, start)  # ts Jan 1..4, each closes next day
    shuffled = [bars[2], bars[0], bars[3], bars[1]]
    end = datetime(2024, 1, 4, tzinfo=UTC)  # bar Jan 4 is still forming
    out = normalise_bars(shuffled, start=start, end=end, timeframe="1d")
    assert [b.ts for b in out] == [bars[0].ts, bars[1].ts, bars[2].ts]

    # duplicate timestamp: newer value wins
    dupe = Bar(ts=bars[1].ts, open=1.0, high=2.0, low=0.5, close=1.5, volume=9.0)
    merged = normalise_bars([bars[1], dupe], timeframe="1d")
    assert len(merged) == 1 and merged[0].close == 1.5

    # clamping keeps OHLC coherent and drops impossible bars
    sloppy = Bar(ts=bars[0].ts, open=10.0, high=9.0, low=6.0, close=7.0, volume=1.0)
    fixed = normalise_bars([sloppy], timeframe="1d")[0]
    assert fixed.low <= min(fixed.open, fixed.close) <= max(fixed.open, fixed.close) <= fixed.high
    bad = Bar(ts=bars[0].ts, open=-1.0, high=1.0, low=0.0, close=1.0, volume=1.0)
    assert normalise_bars([bad], timeframe="1d") == []


def test_timeframe_and_closed_bar_helpers():
    assert timeframe_seconds("1d") == 86400
    assert timeframe_seconds("1h") == 3600
    assert timeframe_seconds("15m") == 900
    with pytest.raises(DataValidationError):
        timeframe_seconds("2d")
    # A bar is only closed when ts + step <= end.
    assert last_closed_bar_start(datetime(2024, 1, 10, 12, tzinfo=UTC), DAY) == datetime(
        2024, 1, 9, tzinfo=UTC
    )
    assert last_closed_bar_start(datetime(2024, 1, 10, 0, tzinfo=UTC), DAY) == datetime(
        2024, 1, 9, tzinfo=UTC
    )


# --------------------------------------------------------------------------- #
# synthetic provider
# --------------------------------------------------------------------------- #
def test_synthetic_determinism_and_contract(tmp_path):
    provider = SyntheticProvider(offline_cfg(tmp_path))
    start = END - timedelta(days=90)

    first = provider.fetch_ohlcv("BTC-USD", start, END)
    second = provider.fetch_ohlcv("BTC-USD", start, END)
    assert first == second
    assert len(first) >= 400  # padded to the minimum useful window

    # strictly monotonic, timezone-aware, closed bars only
    assert all(a.ts < b.ts for a, b in zip(first, first[1:]))
    assert all(bar.ts.tzinfo is not None and bar.ts.utcoffset() == timedelta(0) for bar in first)
    assert all(bar.ts + timedelta(days=1) <= END for bar in first)
    assert all(bar.ts <= END for bar in first)

    # coherent OHLCV
    for bar in first:
        assert bar.low <= min(bar.open, bar.close)
        assert bar.high >= max(bar.open, bar.close)
        assert bar.low > 0 and bar.high > 0
        assert bar.volume > 0

    # different symbols are different series
    other = provider.fetch_ohlcv("ETH-USD", start, END)
    assert [b.close for b in other] != [b.close for b in first]


def test_synthetic_history_is_stable_as_end_moves(tmp_path):
    provider = SyntheticProvider(offline_cfg(tmp_path))
    start = END - timedelta(days=120)
    early = provider.fetch_ohlcv("BTC-USD", start, END)
    later = provider.fetch_ohlcv("BTC-USD", start, END + timedelta(days=1))

    early_by_ts = {bar.ts: bar for bar in early}
    overlap = [early_by_ts[bar.ts] for bar in later if bar.ts in early_by_ts]
    assert len(overlap) == len(early) - 1
    assert overlap == later[: len(overlap)]
    assert later[-1].ts == early[-1].ts + timedelta(days=1)


def test_synthetic_equities_are_weekdays_only(tmp_path):
    provider = SyntheticProvider(offline_cfg(tmp_path))
    bars = provider.fetch_ohlcv("AAPL", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC))
    assert len(bars) >= 400
    assert all(bar.ts.weekday() < 5 for bar in bars)
    assert all(bar.ts + timedelta(days=1) <= datetime(2026, 3, 1, tzinfo=UTC) for bar in bars)

    crypto = provider.fetch_ohlcv("BTC-USD", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC))
    assert any(bar.ts.weekday() >= 5 for bar in crypto)


def test_synthetic_works_with_zero_network(tmp_path, monkeypatch):
    forbid_network(monkeypatch)
    provider = SyntheticProvider(offline_cfg(tmp_path))
    for timeframe in ("1d", "1h", "15m"):
        bars = provider.fetch_ohlcv("BTC-USD", END - timedelta(days=10), END, timeframe)
        assert bars
        assert all(a.ts < b.ts for a, b in zip(bars, bars[1:]))


# --------------------------------------------------------------------------- #
# disk cache
# --------------------------------------------------------------------------- #
def test_cache_roundtrip_and_merge(tmp_path):
    cache = OHLCVCache(str(tmp_path / "cache"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = make_bars(5, start)

    cache.write("BTC-USD", "1d", bars)
    assert cache.read("BTC-USD", "1d") == bars
    file_path = cache.path_for("BTC-USD", "1d")
    assert file_path.exists()
    assert "BTC-USD" in file_path.name and "/" not in file_path.name

    # same timestamp updated + two new bars appended
    replacement = Bar(ts=bars[1].ts, open=1.0, high=200.0, low=0.5, close=150.0, volume=42.0)
    fresh = make_bars(7, start)[5:7]
    merged = cache.merge("BTC-USD", "1d", [replacement, *fresh])
    assert len(merged) == 7
    assert [b.ts for b in merged] == sorted(b.ts for b in merged)
    assert merged[1].close == 150.0
    assert merged[-1].ts == start + timedelta(days=6)
    assert cache.read("BTC-USD", "1d") == merged

    # filenames are sanitised for exotic symbols / timeframes
    weird = cache.path_for("BRK/B.US", "1d")
    assert "/" not in weird.name and weird.parent == cache.cache_dir.parent / "cache"


def test_cache_disabled_and_corrupt_files(tmp_path):
    disabled = OHLCVCache(str(tmp_path / "cache"), enabled=False)
    bars = make_bars(3, datetime(2024, 1, 1, tzinfo=UTC))
    disabled.write("BTC-USD", "1d", bars)
    assert disabled.read("BTC-USD", "1d") == []
    assert disabled.merge("BTC-USD", "1d", bars) == bars
    assert not (tmp_path / "cache").exists()

    cache = OHLCVCache(str(tmp_path / "cache2"))
    cache.path_for("BTC-USD", "1d").parent.mkdir(parents=True, exist_ok=True)
    cache.path_for("BTC-USD", "1d").write_text("{not valid json", encoding="utf-8")
    assert cache.read("BTC-USD", "1d") == []


# --------------------------------------------------------------------------- #
# coinbase (mocked HTTP -- never hits the real API)
# --------------------------------------------------------------------------- #
def test_coinbase_maps_row_order_and_filters_forming(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None, **kwargs):
        calls.append(dict(params))
        assert timeout is not None
        assert "/products/BTC-USD/candles" in url
        assert params["granularity"] == 86400
        start = datetime.strptime(params["start"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        end = datetime.strptime(params["end"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        n = int((end - start).total_seconds() // DAY) + 1
        rows = []
        for i in range(n):  # newest first: [time, low, high, open, close, volume]
            t = int(end.timestamp()) - i * DAY
            rows.append([t, 100.0 + i, 110.0 + i, 105.0 + i, 101.0 + i, 1_000.0 + i])
        return FakeResponse(200, rows)

    monkeypatch.setattr("requests.get", fake_get)
    cfg = offline_cfg(tmp_path, allow_network=True)
    provider = CoinbaseProvider(cfg)
    end = datetime(2024, 1, 5, tzinfo=UTC)
    start = end - timedelta(days=3)
    bars = provider.fetch_ohlcv("btc-usd", start, end)

    assert [bar.ts for bar in bars] == sorted(bar.ts for bar in bars)
    # row j (j = days before end) maps low/high/open/close from positions 1..4
    by_ts = {bar.ts: bar for bar in bars}
    jan4 = by_ts[datetime(2024, 1, 4, tzinfo=UTC)]
    assert (jan4.low, jan4.high, jan4.open, jan4.close) == (101.0, 111.0, 106.0, 102.0)
    jan2 = by_ts[datetime(2024, 1, 2, tzinfo=UTC)]
    assert (jan2.low, jan2.high, jan2.open, jan2.close) == (103.0, 113.0, 108.0, 104.0)
    # bar at Jan 5 (still forming at Jan 5 00:00) is dropped
    assert datetime(2024, 1, 5, tzinfo=UTC) not in by_ts
    assert len(bars) == 3


def test_coinbase_paginates_backwards(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None, **kwargs):
        calls.append(dict(params))
        start = datetime.strptime(params["start"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        end = datetime.strptime(params["end"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        n = int((end - start).total_seconds() // DAY) + 1
        if n > 300:
            return FakeResponse(400, {"message": "too many candles"})
        rows = []
        for i in range(n):
            t = int(end.timestamp()) - i * DAY
            rows.append([t, 100.0 + i, 110.0 + i, 105.0 + i, 101.0 + i, 1_000.0 + i])
        return FakeResponse(200, rows)

    monkeypatch.setattr("requests.get", fake_get)
    provider = CoinbaseProvider(offline_cfg(tmp_path, allow_network=True))
    end = datetime(2026, 10, 1, tzinfo=UTC)
    start = end - timedelta(days=449)
    bars = provider.fetch_ohlcv("BTC-USD", start, end)

    assert len(bars) == 449  # 450 candles inclusive, minus the still-forming one
    assert len(calls) == 2  # 300 + 150, backwards pagination
    assert bars[0].ts == start and bars[-1].ts == end - timedelta(days=1)
    assert all(a.ts < b.ts for a, b in zip(bars, bars[1:]))


def test_coinbase_rejects_non_usd_and_non_200(tmp_path, monkeypatch):
    provider = CoinbaseProvider(offline_cfg(tmp_path, allow_network=True))
    with pytest.raises(DataUnavailableError):
        provider.fetch_ohlcv("AAPL", END - timedelta(days=5), END)

    def fake_get(*args, **kwargs):
        return FakeResponse(429, {"message": "rate limited"})

    monkeypatch.setattr("requests.get", fake_get)
    with pytest.raises(DataUnavailableError):
        provider.fetch_ohlcv("BTC-USD", END - timedelta(days=5), END)


# --------------------------------------------------------------------------- #
# yahoo / twelvedata (mocked HTTP)
# --------------------------------------------------------------------------- #
def test_yahoo_non_200_raises_data_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResponse(429, text="Too Many Requests"))
    provider = YahooProvider(offline_cfg(tmp_path, allow_network=True))
    with pytest.raises(DataUnavailableError):
        provider.fetch_ohlcv("AAPL", END - timedelta(days=5), END)


def test_yahoo_parses_chart_payload(tmp_path, monkeypatch):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [int((start + timedelta(days=i)).timestamp()) for i in range(5)]
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                                "high": [110.0] * 5,
                                "low": [90.0] * 5,
                                "close": [105.0, 106.0, 107.0, 108.0, 109.0],
                                "volume": [1000] * 5,
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResponse(200, payload))
    provider = YahooProvider(offline_cfg(tmp_path, allow_network=True))
    bars = provider.fetch_ohlcv("AAPL", start, start + timedelta(days=5))
    assert len(bars) == 5
    assert bars[0].close == 105.0
    assert bars[-1].ts == start + timedelta(days=4)
    assert all(bar.volume == 1000.0 for bar in bars)


def test_twelvedata_missing_key_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)
    provider = TwelveDataProvider(offline_cfg(tmp_path, allow_network=True))
    with pytest.raises(DataUnavailableError):
        provider.fetch_ohlcv("AAPL", END - timedelta(days=5), END)


def test_twelvedata_parses_values_with_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TWELVEDATA_API_KEY", "demo")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    values = [
        {"datetime": "2024-01-03", "open": "103", "high": "110", "low": "95", "close": "105", "volume": "1000"},
        {"datetime": "2024-01-02", "open": "102", "high": "109", "low": "94", "close": "104", "volume": "900"},
        {"datetime": "2024-01-01", "open": "101", "high": "108", "low": "93", "close": "103", "volume": "800"},
    ]
    payload = {"meta": {"symbol": "AAPL"}, "status": "ok", "values": values}
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResponse(200, payload))
    provider = TwelveDataProvider(offline_cfg(tmp_path, allow_network=True))
    bars = provider.fetch_ohlcv("AAPL", start, start + timedelta(days=3))
    assert [bar.close for bar in bars] == [103.0, 104.0, 105.0]
    assert bars[-1].ts == start + timedelta(days=2)


def test_chain_falls_from_yahoo_to_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResponse(429, text="Too Many Requests"))
    cfg = offline_cfg(
        tmp_path, allow_network=True, provider_order=["yahoo", "synthetic"]
    )
    chain = ProviderChain(cfg)
    bars = chain.fetch_ohlcv("BTC-USD", END - timedelta(days=30), END)
    assert chain.last_provider == "synthetic"
    assert len(bars) >= 400
    assert any("yahoo" in note for note in chain.last_errors)


# --------------------------------------------------------------------------- #
# market data service
# --------------------------------------------------------------------------- #
def test_service_context_offline_has_all_features_and_no_lookahead(tmp_path, monkeypatch):
    forbid_network(monkeypatch)
    service = MarketDataService(offline_cfg(tmp_path))
    ctx = service.context("BTC-USD", END)

    assert service.last_provider == "synthetic"
    assert ctx.provider == "synthetic"
    assert set(ctx.features) == set(REQUIRED_FEATURES)
    assert all(isinstance(value, float) for value in ctx.features.values())
    assert len(ctx.bars) == DataConfig().lookback_bars == 320
    assert ctx.bars == tuple(sorted(ctx.bars, key=lambda b: b.ts))
    assert all(bar.ts + timedelta(days=1) <= ctx.asof for bar in ctx.bars)
    assert ctx.features["close"] == pytest.approx(ctx.bars[-1].close)
    assert ctx.last_close == ctx.bars[-1].close
    assert not math.isnan(ctx.features["sma_50"])


def test_service_context_lookback_override_and_equities(tmp_path):
    service = MarketDataService(offline_cfg(tmp_path))
    asof = datetime(2026, 6, 1, 15, tzinfo=UTC)
    ctx = service.context("AAPL", asof, lookback=120)
    assert len(ctx.bars) == 120
    assert all(bar.ts.weekday() < 5 for bar in ctx.bars)
    assert all(bar.ts + timedelta(days=1) <= asof for bar in ctx.bars)
    assert set(ctx.features) == set(REQUIRED_FEATURES)


def test_service_history_uses_cache_then_skips_synthetic_cache(tmp_path):
    cfg = offline_cfg(tmp_path)
    cfg.cache_enabled = True
    start = END - timedelta(days=60)
    provider = GoodProvider()
    service = MarketDataService(cfg, providers=[provider])
    first = service.history("BTC-USD", start, END)
    assert service.last_provider == "good"
    assert provider.calls == 1
    second = service.history("BTC-USD", start, END)
    assert service.last_provider == "cache"
    assert provider.calls == 1  # no refetch
    assert second == first
    assert service.cache.path_for("BTC-USD", "1d").exists()

    synthetic_dir = tmp_path / "synthetic_cache"
    synth_cfg = offline_cfg(tmp_path, cache_dir=str(synthetic_dir))
    synth_service = MarketDataService(synth_cfg)
    synth_service.history("BTC-USD", start, END)
    synth_service.history("BTC-USD", start, END)
    assert synth_service.last_provider == "synthetic"
    assert not synthetic_dir.exists()  # synthetic data must never be persisted


def test_service_history_slices_requested_window(tmp_path):
    service = MarketDataService(offline_cfg(tmp_path))
    start = END - timedelta(days=60)
    bars = service.history("BTC-USD", start, END)
    assert bars[0].ts >= start
    assert bars[-1].ts + timedelta(days=1) <= END
    assert all(bar.ts + timedelta(days=1) <= END for bar in bars)
    assert len(bars) == 60


def test_history_clips_wide_cache_to_requested_window(tmp_path):
    """Regression: a wide cached span must never widen a narrow history() call.

    Reproducibility rule: ``history(start, end)`` returns exactly the same
    in-range bars on every call, no matter what earlier runs put in the cache.
    """
    cfg = offline_cfg(tmp_path)
    cfg.cache_enabled = True
    provider = GoodProvider()
    service = MarketDataService(cfg, providers=[provider])

    # Populate a wide cache first (e.g. an earlier, longer backtest).
    wide = service.history("BTC-USD", END - timedelta(days=200), END)
    assert wide[0].ts == END - timedelta(days=200)
    assert service.last_provider == "good"

    narrow_start = END - timedelta(days=30)
    narrow_end = END
    first = service.history("BTC-USD", narrow_start, narrow_end)
    second = service.history("BTC-USD", narrow_start, narrow_end)

    assert first == second
    assert service.last_provider == "cache"
    assert len(first) == 30
    assert first[0].ts == narrow_start
    assert first[-1].ts == narrow_end - timedelta(days=1)
    assert all(narrow_start <= bar.ts and bar.ts + timedelta(days=1) <= narrow_end for bar in first)
    assert provider.calls == 1  # both narrow calls were served from the cache

    # The cache itself keeps the widest span -- only the return value is clipped.
    cached = service.cache.read("BTC-USD", "1d")
    assert cached[0].ts == END - timedelta(days=200)
    assert cached[-1].ts == END - timedelta(days=1)


def test_service_context_rejects_empty_lookback(tmp_path):
    service = MarketDataService(offline_cfg(tmp_path))
    with pytest.raises(DataValidationError):
        service.context("BTC-USD", END, lookback=0)


def test_make_service_accepts_app_and_data_config(tmp_path):
    cfg = offline_cfg(tmp_path)
    assert isinstance(make_service(cfg), MarketDataService)
    app = AppConfig()
    app.data = cfg
    service = make_service(app)
    assert isinstance(service, MarketDataService)
    assert service.cfg is cfg
    with pytest.raises(TypeError):
        make_service(object())  # type: ignore[arg-type]


def test_offline_history_never_touches_network(tmp_path, monkeypatch):
    forbid_network(monkeypatch)
    service = MarketDataService(offline_cfg(tmp_path))
    bars = service.history("ETH-USD", END - timedelta(days=30), END)
    assert len(bars) == 30
    assert service.last_provider == "synthetic"
