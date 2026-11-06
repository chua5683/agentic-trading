"""Causality tests for the frozen indicator implementation.

``agentic_trading/data/indicators.py`` is the one place features are computed
(both live decisions and the backtest engine use it).  If it ever looked ahead,
every backtest in the project would be a lie.  These tests pin the contract:

* ``compute_features(bars[:k])`` equals row ``k-1`` of
  ``feature_frame(bars)`` for many ``k`` -- no future data can leak in;
* the full frame is prefix-stable (truncating the bars only deletes rows);
* NaN warm-up behaves as expected (rolling windows start after their window);
* a single/empty bar never crashes and still yields every REQUIRED_FEATURES key.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from agentic_trading.config import DataConfig
from agentic_trading.contracts import REQUIRED_FEATURES
from agentic_trading.data.indicators import (
    bars_to_frame,
    compute_features,
    feature_frame,
    latest_features,
)
from agentic_trading.data.synthetic import SyntheticProvider

UTC = timezone.utc
END = datetime(2026, 11, 6, tzinfo=UTC)
N_BARS = 420


@pytest.fixture(scope="module")
def bars():
    provider = SyntheticProvider(DataConfig(allow_network=False))
    # end is exclusive for closed bars: N_BARS days back yields N_BARS bars.
    out = provider.fetch_ohlcv("BTC-USD", END - timedelta(days=N_BARS), END)
    assert len(out) >= N_BARS
    return out[:N_BARS]


@pytest.fixture(scope="module")
def frame(bars):
    return feature_frame(bars, timeframe="1d")


def assert_same_feature(got: float, expected) -> None:
    if expected is None or (isinstance(expected, float) and math.isnan(expected)):
        assert isinstance(got, float) and math.isnan(got), f"expected NaN, got {got!r}"
    else:
        assert got == pytest.approx(float(expected), rel=1e-12, abs=1e-12)


# --------------------------------------------------------------------------- #
# schema / shape
# --------------------------------------------------------------------------- #
def test_feature_frame_schema(bars, frame):
    assert list(frame.columns) == list(REQUIRED_FEATURES)
    assert len(frame) == len(bars)
    assert list(frame.index) == [bar.ts for bar in bars]
    assert all(dtype == "float64" for dtype in frame.dtypes)


def test_bars_to_frame_is_ascending_and_deduplicated(bars):
    df = bars_to_frame([bars[2], bars[0], bars[1], bars[1]])
    assert list(df.index) == [bars[0].ts, bars[1].ts, bars[2].ts]
    assert df.index.tz is not None


# --------------------------------------------------------------------------- #
# causality: truncated recomputation == precomputed row
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 5, 14, 20, 21, 60, 61, 150, 200, 240, N_BARS])
def test_compute_features_matches_frame_row(bars, frame, k):
    got = compute_features(bars[:k], timeframe="1d")
    expected_row = frame.iloc[k - 1]
    assert set(got) == set(REQUIRED_FEATURES)
    for name in REQUIRED_FEATURES:
        assert_same_feature(got[name], expected_row[name])


def test_feature_frame_is_prefix_stable(bars, frame):
    """Computing on a prefix must not change any of that prefix's rows."""
    for k in (20, 60, 200, N_BARS):
        prefix = feature_frame(bars[:k], timeframe="1d")
        pd.testing.assert_frame_equal(prefix, frame.iloc[:k])


@pytest.mark.parametrize("i", [0, 1, 19, 60, 199, 200, N_BARS - 1])
def test_latest_features_matches_compute_features(bars, frame, i):
    fast = latest_features(frame, i)
    slow = compute_features(bars[: i + 1], timeframe="1d")
    for name in REQUIRED_FEATURES:
        assert_same_feature(fast[name], slow[name])


def test_latest_features_out_of_range_returns_nans(frame):
    for index in (-1, len(frame)):
        result = latest_features(frame, index)
        assert set(result) == set(REQUIRED_FEATURES)
        assert all(math.isnan(value) for value in result.values())


# --------------------------------------------------------------------------- #
# NaN warm-up behaviour
# --------------------------------------------------------------------------- #
def test_warmup_nans(frame):
    # close is always present
    assert frame["close"].notna().all()
    # a 1-bar return needs one prior bar
    assert math.isnan(frame["ret_1"].iloc[0])
    assert not math.isnan(frame["ret_1"].iloc[1])
    # pct_change(5) first valid on the 6th row
    assert frame["ret_5"].iloc[:5].isna().all()
    assert not math.isnan(frame["ret_5"].iloc[5])
    # rolling windows start after their warm-up
    assert frame["sma_20"].iloc[:19].isna().all()
    assert not math.isnan(frame["sma_20"].iloc[19])
    assert frame["sma_200"].iloc[:199].isna().all()
    assert not math.isnan(frame["sma_200"].iloc[199])
    # EMA-26 / RSI-14 / ATR-14 all have a non-trivial warm-up
    for column in ("ema_26", "rsi_14", "atr_14"):
        assert frame[column].iloc[:10].isna().all()
        assert frame[column].iloc[-1] == pytest.approx(frame[column].iloc[-1])
        assert not math.isnan(frame[column].iloc[-1])
    # regime flags are 0.0/1.0 (float) once defined, never "warm up backwards"
    for column in ("trend_up", "above_sma50", "above_sma200"):
        values = frame[column].dropna()
        assert set(values.unique()) <= {0.0, 1.0}


def test_single_bar_and_empty_inputs_do_not_crash(bars):
    features = compute_features(bars[:1], timeframe="1d")
    assert set(features) == set(REQUIRED_FEATURES)
    assert features["close"] == bars[0].close
    assert all(math.isnan(features[name]) for name in REQUIRED_FEATURES if name != "close")

    empty_frame = feature_frame([], timeframe="1d")
    assert empty_frame.empty
    assert list(empty_frame.columns) == list(REQUIRED_FEATURES)

    empty_features = compute_features([], timeframe="1d")
    assert set(empty_features) == set(REQUIRED_FEATURES)
    assert all(math.isnan(value) for value in empty_features.values())
