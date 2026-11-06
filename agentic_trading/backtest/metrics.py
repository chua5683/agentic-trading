"""Risk/return metrics for the backtest engine.

All public functions are pure, stdlib-only, and total: they never divide by
zero, never raise on empty input, and always return finite floats.  A backtest
with one bar, zero trades, flat prices or zero volatility must still produce a
usable metrics dictionary -- a crash in the reporting layer is not an acceptable
way to discover a degenerate period.

``compute_metrics`` returns *exactly* ``contracts.METRIC_KEYS``: no extra keys,
no missing keys.  Definitions used here (and reproduced in the HTML report):

    total_return          equity[-1] / equity[0] - 1
    cagr                  geometric annualisation over (num_points - 1) periods
    volatility            sample stdev of per-bar returns * sqrt(periods_per_year)
    sharpe                mean excess return / stdev excess * sqrt(periods_per_year)
    sortino               mean excess return / downside deviation * sqrt(periods)
    max_drawdown          max peak-to-trough decline as a positive fraction
    calmar                cagr / max_drawdown (0 when there is no drawdown)
    win_rate              share of *closed* round trips with positive net PnL
    num_trades            number of fills (one ``Trade`` per fill)
    turnover              sum(|trade notional|) / mean equity (equity turns)
    exposure              share of curve bars with a non-zero position
    best_day / worst_day  best and worst single-bar return
    benchmark_*           the same statistics for the buy-and-hold curve
    excess_return         total_return - benchmark_total_return
    alpha_cagr            cagr - benchmark_cagr
    num_bars              len(equity_curve)
    num_decisions         len(decisions)
    risk_vetoes           decisions whose risk layer reported at least one veto

Win rate uses average-cost accounting over the chronological fill list: buy
fees enter the cost basis, sell fees are charged against proceeds, and a round
trip is closed on each sell.  A strategy that never sells therefore has a
``win_rate`` of 0.0 even while its equity rises -- it has no closed trades yet.
That is deliberate and is reported alongside ``total_return`` so the two are
never confused.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Sequence

from ..contracts import METRIC_KEYS, PortfolioDecision, Trade

__all__ = [
    "compute_metrics",
    "max_drawdown",
    "sharpe",
    "sortino",
    "cagr",
    "period_returns",
]

#: Amounts below this are treated as zero (float dust from fills/rounding).
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Small numeric helpers (stdlib only: these metrics must be auditable by hand)
# --------------------------------------------------------------------------- #
def _finite(value: float, default: float = 0.0) -> float:
    """Coerce to a finite float; anything NaN/inf/non-numeric becomes ``default``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: Sequence[float], ddof: int = 1) -> float:
    """Sample standard deviation; 0.0 when there are too few observations."""
    n = len(values)
    if n - ddof <= 0:
        return 0.0
    mu = _mean(values)
    variance = sum((v - mu) ** 2 for v in values) / (n - ddof)
    return math.sqrt(max(variance, 0.0))


def _total_return(values: Sequence[float]) -> float:
    if len(values) < 2 or values[0] <= 0:
        return 0.0
    if values[-1] <= 0:
        return -1.0
    return _finite(values[-1] / values[0] - 1.0)


def _annualise_vol(returns: Sequence[float], periods_per_year: float) -> float:
    return _finite(_stdev(returns) * math.sqrt(periods_per_year))


def period_returns(equity: Sequence[float]) -> list[float]:
    """Per-bar simple returns of an equity series (length ``len(equity) - 1``)."""
    out: list[float] = []
    prev = None
    for raw in equity:
        cur = _finite(raw)
        if prev is not None:
            out.append(_finite(cur / prev - 1.0) if prev > 0 else 0.0)
        prev = cur
    return out


def cagr(equity: Sequence[float], periods_per_year: float = 365.0) -> float:
    """Geometric annual growth rate; 0.0 for degenerate or overflowing series."""
    values = [_finite(v) for v in equity]
    periods = len(values) - 1
    if periods < 1 or values[0] <= 0:
        return 0.0
    growth = values[-1] / values[0]
    if growth <= 0:
        return -1.0
    try:
        return _finite(math.expm1(math.log(growth) * periods_per_year / periods))
    except (OverflowError, ValueError):
        # An absurd annualisation (e.g. one period of 10x at 365/yr) would blow
        # up; a non-finite CAGR is worse than an honest 0.
        return 0.0


def max_drawdown(equity: Sequence[float]) -> float:
    """Maximum peak-to-trough decline as a *positive* fraction (0.23 == -23%)."""
    peak: float | None = None
    worst = 0.0
    for raw in equity:
        value = _finite(raw, float("nan"))
        if not math.isfinite(value):
            continue
        if peak is None or value > peak:
            peak = value
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return _finite(worst)


def sharpe(
    returns: Sequence[float],
    rf: float = 0.0,
    periods: float = 365.0,
) -> float:
    """Annualised Sharpe ratio; 0.0 when volatility is zero or history too short."""
    rets = [_finite(r) for r in returns]
    periods = _finite(periods, 365.0)
    if periods <= 0:
        periods = 365.0
    if len(rets) < 2:
        return 0.0
    rf_per_period = _finite(rf) / periods
    excess = [r - rf_per_period for r in rets]
    vol = _stdev(excess)
    if vol <= _EPS:
        return 0.0
    return _finite(_mean(excess) / vol * math.sqrt(periods))


def sortino(
    returns: Sequence[float],
    rf: float = 0.0,
    periods: float = 365.0,
) -> float:
    """Annualised Sortino ratio; 0.0 when there is no downside deviation."""
    rets = [_finite(r) for r in returns]
    periods = _finite(periods, 365.0)
    if periods <= 0:
        periods = 365.0
    if not rets:
        return 0.0
    rf_per_period = _finite(rf) / periods
    excess = [r - rf_per_period for r in rets]
    downside = math.sqrt(sum(min(e, 0.0) ** 2 for e in excess) / len(excess))
    if downside <= _EPS:
        return 0.0
    return _finite(_mean(excess) / downside * math.sqrt(periods))


# --------------------------------------------------------------------------- #
# Trade-book helpers
# --------------------------------------------------------------------------- #
def _sorted_trades(trades: Sequence[Trade]) -> list[Trade]:
    return sorted(trades, key=lambda t: (t.asof, 0 if t.side == "buy" else 1))


def _realized_pnls(trades: Sequence[Trade]) -> list[float]:
    """Closed round-trip PnLs using average-cost accounting, fees included."""
    pnls: list[float] = []
    basis_qty = 0.0
    basis_cost = 0.0
    for trade in _sorted_trades(trades):
        qty = _finite(trade.qty)
        price = _finite(trade.price)
        fee = _finite(trade.fee)
        if qty <= 0:
            continue
        if trade.side == "buy":
            basis_qty += qty
            basis_cost += qty * price + fee
            continue
        if trade.side != "sell" or basis_qty <= _EPS:
            continue
        sold = min(qty, basis_qty)
        avg_cost = basis_cost / basis_qty
        fee_share = fee * (sold / qty)
        proceeds = sold * price - fee_share
        pnls.append(_finite(proceeds - avg_cost * sold))
        basis_qty -= sold
        basis_cost = max(basis_cost - avg_cost * sold, 0.0)
        if basis_qty <= _EPS:
            basis_qty, basis_cost = 0.0, 0.0
    return pnls


def _exposure(curve: Sequence[tuple[datetime, float]], trades: Sequence[Trade]) -> float:
    """Fraction of marked bars during which the book held a position."""
    if not curve:
        return 0.0
    deltas: dict[datetime, float] = {}
    for trade in _sorted_trades(trades):
        signed = _finite(trade.qty) * (1.0 if trade.side == "buy" else -1.0)
        deltas[trade.asof] = deltas.get(trade.asof, 0.0) + signed
    qty = 0.0
    held = 0
    for ts, _ in curve:
        qty += deltas.get(ts, 0.0)
        if abs(qty) > 1e-9:
            held += 1
    return held / len(curve)


def _turnover(curve: Sequence[tuple[datetime, float]], trades: Sequence[Trade]) -> float:
    """Traded notional divided by mean equity (how many times equity turned over)."""
    if not curve:
        return 0.0
    mean_equity = _mean([_finite(v) for _, v in curve])
    if mean_equity <= _EPS:
        return 0.0
    traded = sum(abs(_finite(t.notional)) for t in trades)
    return _finite(traded / mean_equity)


def _risk_vetoes(decisions: Sequence[PortfolioDecision]) -> float:
    return float(
        sum(1 for d in decisions if d.risk is not None and bool(d.risk.vetoes))
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def compute_metrics(
    equity_curve: list[tuple[datetime, float]],
    benchmark_curve: list[tuple[datetime, float]],
    trades: list[Trade],
    decisions: list[PortfolioDecision],
    periods_per_year: float = 365.0,
    risk_free_rate: float = 0.04,
) -> dict[str, float]:
    """Compute every key in :data:`contracts.METRIC_KEYS` from one backtest run.

    Parameters mirror the engine's output so this function is also usable on
    hand-built curves (which is how the tests pin the arithmetic).
    """
    periods = _finite(periods_per_year, 365.0)
    if periods <= 0:
        periods = 365.0
    rf = _finite(risk_free_rate)

    equity_values = [_finite(v) for _, v in equity_curve]
    benchmark_values = [_finite(v) for _, v in benchmark_curve]

    returns = period_returns(equity_values)
    benchmark_returns = period_returns(benchmark_values)

    total_return = _total_return(equity_values)
    benchmark_total_return = _total_return(benchmark_values)
    strategy_cagr = cagr(equity_values, periods)
    benchmark_cagr = cagr(benchmark_values, periods)
    drawdown = max_drawdown(equity_values)
    benchmark_drawdown = max_drawdown(benchmark_values)

    pnls = _realized_pnls(trades)
    win_rate = (
        sum(1 for pnl in pnls if pnl > 0) / len(pnls) if pnls else 0.0
    )

    raw: dict[str, float] = {
        "total_return": total_return,
        "cagr": strategy_cagr,
        "volatility": _annualise_vol(returns, periods),
        "sharpe": sharpe(returns, rf, periods),
        "sortino": sortino(returns, rf, periods),
        "max_drawdown": drawdown,
        "calmar": _finite(strategy_cagr / drawdown) if drawdown > _EPS else 0.0,
        "win_rate": _finite(win_rate),
        "num_trades": float(len(trades)),
        "turnover": _turnover(equity_curve, trades),
        "exposure": _finite(_exposure(equity_curve, trades)),
        "best_day": _finite(max(returns)) if returns else 0.0,
        "worst_day": _finite(min(returns)) if returns else 0.0,
        "benchmark_total_return": benchmark_total_return,
        "benchmark_cagr": benchmark_cagr,
        "benchmark_sharpe": sharpe(benchmark_returns, rf, periods),
        "benchmark_max_drawdown": benchmark_drawdown,
        "excess_return": _finite(total_return - benchmark_total_return),
        "alpha_cagr": _finite(strategy_cagr - benchmark_cagr),
        "num_bars": float(len(equity_values)),
        "num_decisions": float(len(decisions)),
        "risk_vetoes": _risk_vetoes(decisions),
    }

    # Enforce the frozen contract: exact keys, insertion order, finite floats.
    return {key: _finite(raw.get(key, 0.0)) for key in METRIC_KEYS}
