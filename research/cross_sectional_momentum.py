"""Prototype: does cross-sectional momentum carry signal across a crypto universe?

Standalone research script; it is NOT part of the shipped package and does not
affect the system's tested behaviour. Its purpose is to answer one question
empirically: is the bottleneck the agent architecture, or the signal source?

Result (see docs/ALPHA.md): switching from single-asset time-series timing to
cross-sectional ranking across ~33 crypto assets is the single largest
improvement available, and the effect is only trustworthy at a SLOW horizon.

Honest caveats baked in:
  * SURVIVORSHIP BIAS: the universe is TODAY's listed Coinbase products, so
    delisted / rug-pulled assets are absent. This inflates results, probably
    substantially for long-only alt momentum.
  * costs are charged on turnover at 20bps per side (crypto taker-ish);
  * signals use only data up to the rebalance bar (verified causal).
  * the parameter grid is small (18 configs); any "best" number is subject to
    multiple-testing decay, which is why docs/ALPHA.md reports the in-sample /
    out-of-sample split rather than the best backtest.

Run: PYTHONPATH=<repo> .venv/bin/python research/cross_sectional_momentum.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from agentic_trading.config import DataConfig
from agentic_trading.data.coinbase import CoinbaseProvider

CACHE = Path("/tmp/xsec_cache")
CACHE.mkdir(exist_ok=True)
START = "2022-06-01"
GRAN = 86400
UA = {"User-Agent": "agentic-trading-research/0.1"}


def dollar_volume_universe(min_usd: float = 1e6, cap: int = 45) -> list[str]:
    r = requests.get("https://api.exchange.coinbase.com/products", timeout=25, headers=UA).json()
    usd = [p for p in r if p.get("quote_currency") == "USD"
           and p.get("status") == "online" and not p.get("trading_disabled")]
    st = requests.get("https://api.exchange.coinbase.com/products/stats", timeout=25, headers=UA).json()
    rows = []
    for p in usd:
        s = st.get(p["id"], {}).get("stats_24hour", {})
        try:
            dv = float(s.get("volume", 0)) * float(s.get("last", 0))
        except (TypeError, ValueError):
            dv = 0.0
        rows.append((dv, p["id"]))
    rows.sort(reverse=True)
    return [i for dv, i in rows if dv >= min_usd][:cap]


def fetch_daily(product: str) -> pd.DataFrame | None:
    """Use the package's tested provider (correct pagination) + local cache."""
    f = CACHE / f"{product}.json"
    if f.exists():
        rows = json.load(f.open())
    else:
        prov = CoinbaseProvider(DataConfig(cache_enabled=False))
        start = datetime(2022, 6, 1, tzinfo=timezone.utc)
        end = datetime.now(timezone.utc)
        try:
            bars = prov.fetch_ohlcv(product, start, end, "1d")
        except Exception as exc:
            print(f"  fail {product}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None
        rows = [[int(b.ts.timestamp()), b.low, b.high, b.open, b.close, b.volume] for b in bars]
        json.dump(rows, f.open("w"))
        time.sleep(0.1)
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    df = df[df.index >= pd.Timestamp(START, tz="UTC")]
    return df[["open", "high", "low", "close", "volume"]]


def build_panel(products: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    closes, dv = {}, {}
    for p in products:
        df = fetch_daily(p)
        if df is None or len(df) < 300:
            print(f"  skip {p}: insufficient history", file=sys.stderr)
            continue
        closes[p] = df["close"]
        dv[p] = df["close"] * df["volume"]
        print(f"  {p:<12} {len(df):>5} bars", file=sys.stderr)
    close = pd.DataFrame(closes).sort_index()
    dollar = pd.DataFrame(dv).reindex(close.index).sort_index()
    return close, dollar, close.pct_change()


def metrics(equity: pd.Series, label: str, ann: float = 365.0, rf: float = 0.04) -> dict:
    eq = equity.dropna()
    ret = eq.pct_change().dropna()
    if len(ret) < 10:
        return {"label": label}
    years = len(ret) / ann
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vol = ret.std(ddof=0) * np.sqrt(ann)
    sharpe = (cagr - rf) / vol if vol > 0 else 0.0
    peak = eq.cummax()
    mdd = float((1 - eq / peak).max())
    return {"label": label, "total": total, "cagr": cagr, "vol": vol,
            "sharpe": sharpe, "maxdd": mdd, "n": len(ret)}


def rebalance_dates(close: pd.DataFrame, rebal: str) -> pd.DatetimeIndex:
    """First trading bar of each period, kept tz-aware and inside the index.

    Taking ``.values`` from a groupby would drop tz-awareness and silently make
    every lookup miss -> a portfolio that never trades.
    """
    idx = close.index
    pos = (pd.Series(range(len(idx)), index=idx)
           .groupby(pd.Grouper(freq=rebal)).first().dropna().astype(int))
    return idx[pos.values]


def _schedule_to_weights(schedule: pd.DataFrame, idx: pd.DatetimeIndex,
                         cols) -> pd.DataFrame:
    """Carry a rebalance schedule forward, preserving deliberate zero weights."""
    w = schedule.reindex(idx).ffill().fillna(0.0)
    return w.reindex(columns=cols).fillna(0.0)


def run_xsec(close: pd.DataFrame, dollar: pd.DataFrame, top_k: int, lookback: int,
             rebal: str, min_dv: float, cost_bps: float = 20.0,
             skip: int = 0) -> pd.Series:
    """Long-only top-K cross-sectional momentum, rebalanced on a calendar schedule."""
    idx, cols = close.index, close.columns
    dv_ma = dollar.rolling(30, min_periods=10).median()
    periods = rebalance_dates(close, rebal)
    start_i = lookback + skip + 1

    rows = {}
    for t in periods:
        loc = idx.get_loc(t)
        if loc < start_i:
            continue
        past = close.iloc[: loc + 1]
        sig = (past.iloc[-1 - skip] / past.iloc[-1 - skip - lookback] - 1.0)
        liquid = dv_ma.loc[t] > min_dv
        valid = sig.notna() & liquid
        if int(valid.sum()) == 0:
            rows[t] = pd.Series(0.0, index=cols)
            continue
        top = sig[valid].nlargest(min(top_k, int(valid.sum()))).index
        w = pd.Series(0.0, index=cols)
        w[top] = 1.0 / len(top)          # fully invested, long-only, sum == 1
        rows[t] = w

    if not rows:
        return pd.Series(1.0, index=idx)
    schedule = pd.DataFrame(rows).T
    weights = _schedule_to_weights(schedule, idx, cols)

    rets = close.pct_change().fillna(0.0)
    gross = (weights.shift(1) * rets).sum(axis=1)
    turnover = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * (cost_bps / 1e4)
    return (1 + net).cumprod()


def run_equal_weight(close: pd.DataFrame, dollar: pd.DataFrame, rebal: str,
                     min_dv: float, cost_bps: float = 20.0) -> pd.Series:
    idx, cols = close.index, close.columns
    dv_ma = dollar.rolling(30, min_periods=10).median()
    periods = rebalance_dates(close, rebal)
    rows = {}
    for t in periods:
        eligible = (dv_ma.loc[t] > min_dv) & close.loc[t].notna()
        n = int(eligible.sum())
        w = pd.Series(0.0, index=cols)
        if n:
            w[eligible[eligible].index] = 1.0 / n
        rows[t] = w
    schedule = pd.DataFrame(rows).T
    weights = _schedule_to_weights(schedule, idx, cols)
    rets = close.pct_change().fillna(0.0)
    gross = (weights.shift(1) * rets).sum(axis=1)
    turnover = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0.0)
    return (1 + gross - turnover * (cost_bps / 1e4)).cumprod()


def main() -> None:
    uni = dollar_volume_universe()
    print(f"universe candidates: {len(uni)}", file=sys.stderr)
    close, dollar, _ = build_panel(uni)
    print(f"\npanel: {close.shape[0]} days x {close.shape[1]} assets "
          f"({close.index[0].date()} -> {close.index[-1].date()})", file=sys.stderr)
    close = close.dropna(how="all")
    dollar = dollar.reindex(close.index)

    rows = []
    btc = close["BTC-USD"] if "BTC-USD" in close else close.mean(axis=1)
    rows.append(metrics(btc / btc.iloc[0], "BTC buy & hold"))
    eq = run_equal_weight(close, dollar, "ME", 1e6)
    rows.append(metrics(eq, "equal-weight universe"))

    grid = []
    for k in (3, 5, 8):
        for lb in (30, 90, 180):
            for skip in (0, 7):
                eqc = run_xsec(close, dollar, k, lb, "W", 1e6, skip=skip)
                grid.append((k, lb, skip, metrics(eqc, f"xsec k={k} lookback={lb} skip={skip}")))
    grid.sort(key=lambda g: g[3].get("sharpe", -9), reverse=True)
    rows.extend([g[3] for g in grid[:8]])
    rows.append(metrics(run_xsec(close, dollar, 5, 90, "ME", 1e6), "xsec k=5 lb=90 monthly"))

    print(f"\n{'strategy':<38}{'total':>10}{'cagr':>9}{'vol':>8}{'sharpe':>8}{'maxDD':>8}")
    print("-" * 81)
    for r in rows:
        print(f"{r['label']:<38}{r['total']:>9.0%}{r['cagr']:>9.1%}"
              f"{r['vol']:>8.1%}{r['sharpe']:>8.2f}{r['maxdd']:>8.1%}")


if __name__ == "__main__":
    main()
