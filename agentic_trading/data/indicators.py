"""Vectorised, strictly-causal technical indicators.

Owned by the Lead because both the data layer and the backtest engine depend on
it: if features were computed differently in the two places, a backtest would
silently disagree with live decisions.

Causality rule: every value at row ``i`` is computed from rows ``<= i`` only.
No ``shift(-n)``, no centred windows.  ``tests/test_indicators.py`` enforces this
by recomputing on a truncated prefix and comparing.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from ..contracts import REQUIRED_FEATURES, Bar

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "bars_to_frame",
    "feature_frame",
    "compute_features",
    "latest_features",
    "annualisation_factor",
]

FEATURE_SCHEMA_VERSION = "1.0.0"


def annualisation_factor(timeframe: str, periods_per_year: float | None = None) -> float:
    """sqrt-of-periods scalar for annualising volatility of ``timeframe`` bars."""
    if periods_per_year is not None:
        return float(math.sqrt(periods_per_year))
    per_year = {"1d": 365.0, "1h": 24 * 365.0, "15m": 96 * 365.0}
    return float(math.sqrt(per_year.get(timeframe, 365.0)))


def bars_to_frame(bars: Sequence[Bar]) -> pd.DataFrame:
    """Convert bars to a strictly ascending OHLCV frame indexed by UTC timestamp."""
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], name="ts"))
    rows = [b.as_row() for b in bars]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    # Deduplicate defensively: vendors occasionally repeat a candle.
    df = df[~df.index.duplicated(keep="last")]
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # All-gain window -> RSI 100, all-loss -> 0.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0.0)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def feature_frame(
    bars: Sequence[Bar] | pd.DataFrame,
    timeframe: str = "1d",
    periods_per_year: float | None = None,
) -> pd.DataFrame:
    """Full feature matrix, one row per bar, every column of REQUIRED_FEATURES."""
    df = bars_to_frame(bars) if not isinstance(bars, pd.DataFrame) else bars
    if df.empty:
        return pd.DataFrame(columns=list(REQUIRED_FEATURES))

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    out = pd.DataFrame(index=df.index)
    ann = annualisation_factor(timeframe, periods_per_year)

    out["close"] = close

    # --- returns ----------------------------------------------------------- #
    for n in (1, 5, 20, 60):
        out[f"ret_{n}"] = close.pct_change(n)

    # --- trend ------------------------------------------------------------- #
    out["sma_20"] = close.rolling(20, min_periods=20).mean()
    out["sma_50"] = close.rolling(50, min_periods=50).mean()
    out["sma_200"] = close.rolling(200, min_periods=200).mean()
    out["ema_12"] = _ema(close, 12)
    out["ema_26"] = _ema(close, 26)

    # --- RSI / MACD -------------------------------------------------------- #
    out["rsi_14"] = _rsi(close, 14)
    macd_line = out["ema_12"] - out["ema_26"]
    out["macd"] = macd_line
    out["macd_signal"] = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # --- volatility -------------------------------------------------------- #
    atr = _atr(high, low, close, 14)
    out["atr_14"] = atr
    out["atr_pct"] = atr / close.replace(0.0, np.nan)
    ret1 = out["ret_1"]
    out["vol_20"] = ret1.rolling(20, min_periods=20).std(ddof=0) * ann
    out["vol_60"] = ret1.rolling(60, min_periods=60).std(ddof=0) * ann

    # --- Bollinger --------------------------------------------------------- #
    mid = out["sma_20"]
    sd = close.rolling(20, min_periods=20).std(ddof=0)
    out["bb_upper"] = mid + 2.0 * sd
    out["bb_lower"] = mid - 2.0 * sd
    width = (out["bb_upper"] - out["bb_lower"])
    out["bb_pctb"] = (close - out["bb_lower"]) / width.replace(0.0, np.nan)
    out["bb_width"] = width / mid.replace(0.0, np.nan)

    # --- volume ------------------------------------------------------------ #
    v_sma = volume.rolling(20, min_periods=20).mean()
    v_sd = volume.rolling(20, min_periods=20).std(ddof=0)
    out["volume_z"] = (volume - v_sma) / v_sd.replace(0.0, np.nan)

    # --- position / structure ---------------------------------------------- #
    out["mom_20"] = close.pct_change(20)
    out["price_vs_sma50"] = close / out["sma_50"].replace(0.0, np.nan) - 1.0
    out["price_vs_sma200"] = close / out["sma_200"].replace(0.0, np.nan) - 1.0

    up = (
        (out["sma_20"] > out["sma_50"])
        & (out["sma_50"] > out["sma_200"])
    )
    out["trend_up"] = up.astype(float).where(out["sma_200"].notna())
    out["above_sma50"] = (close > out["sma_50"]).astype(float).where(out["sma_50"].notna())
    out["above_sma200"] = (close > out["sma_200"]).astype(float).where(out["sma_200"].notna())

    hi60 = close.rolling(60, min_periods=20).max()
    lo60 = close.rolling(60, min_periods=20).min()
    out["dist_from_high_60"] = close / hi60.replace(0.0, np.nan) - 1.0
    out["dist_from_low_60"] = close / lo60.replace(0.0, np.nan) - 1.0

    # Enforce the schema: exact columns, in order, floats only.
    for col in REQUIRED_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    out = out[list(REQUIRED_FEATURES)].astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_features(
    bars: Sequence[Bar],
    timeframe: str = "1d",
    periods_per_year: float | None = None,
) -> dict[str, float]:
    """Feature dict for the *latest* bar.  Missing values are ``float('nan')``."""
    frame = feature_frame(bars, timeframe=timeframe, periods_per_year=periods_per_year)
    if frame.empty:
        return {name: float("nan") for name in REQUIRED_FEATURES}
    row = frame.iloc[-1]
    return {name: float(row[name]) if pd.notna(row[name]) else float("nan")
            for name in REQUIRED_FEATURES}


def latest_features(frame: pd.DataFrame, i: int) -> dict[str, float]:
    """Row ``i`` of a precomputed feature frame as a plain dict (backtest path).

    This is the fast path: the engine computes the frame once and slices it,
    which is exactly equivalent to ``compute_features(bars[: i + 1])``.
    """
    if frame.empty or i < 0 or i >= len(frame):
        return {name: float("nan") for name in REQUIRED_FEATURES}
    row = frame.iloc[i]
    return {name: float(row[name]) if pd.notna(row[name]) else float("nan")
            for name in REQUIRED_FEATURES}
