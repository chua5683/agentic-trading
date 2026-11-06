"""Multi-regime evaluation harness.

A single backtest over a single window proves nothing: any parameter set can be
made to look good on a bull run, and defensive strategies look terrible there.
This module runs the same strategy over several pre-declared regime windows and
several symbols, and reports strategy vs buy-and-hold side by side.

Windows are fixed calendar periods chosen *before* looking at results, so the
table cannot be assembled by cherry-picking. Anything the strategy loses is
printed just as prominently as anything it wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

from .config import AppConfig

__all__ = ["RegimeWindow", "DEFAULT_WINDOWS", "EvalRow", "evaluate_grid", "render_table"]


@dataclass(frozen=True)
class RegimeWindow:
    name: str
    start: datetime
    end: datetime
    note: str = ""


def _utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _default_windows() -> list[RegimeWindow]:
    now = datetime.now(timezone.utc)
    return [
        RegimeWindow("bear-2022", _utc(2022, 1, 1), _utc(2023, 1, 1),
                     "BTC/ETH drawdown year"),
        RegimeWindow("recovery-2023", _utc(2023, 1, 1), _utc(2024, 1, 1),
                     "choppy recovery"),
        RegimeWindow("bull-2024-25", _utc(2024, 1, 1), _utc(2025, 11, 1),
                     "strong sustained uptrend"),
        RegimeWindow("cycle-2024-now", _utc(2024, 1, 1), now.replace(hour=0, minute=0, second=0),
                     "full cycle including the later correction"),
        RegimeWindow("last-12m", now - timedelta(days=365), now.replace(hour=0, minute=0, second=0),
                     "most recent twelve months"),
    ]


DEFAULT_WINDOWS: list[RegimeWindow] = _default_windows()


@dataclass
class EvalRow:
    symbol: str
    window: str
    note: str
    bars: int
    provider: str
    total_return: float
    benchmark_total_return: float
    excess_return: float
    cagr: float
    sharpe: float
    benchmark_sharpe: float
    max_drawdown: float
    benchmark_max_drawdown: float
    exposure: float
    num_trades: float
    risk_vetoes: float

    @property
    def beats(self) -> bool:
        return self.excess_return > 0


def evaluate_grid(
    cfg: AppConfig,
    symbols: Sequence[str] | None = None,
    windows: Sequence[RegimeWindow] | None = None,
    service=None,
) -> list[EvalRow]:
    """Run the configured strategy across every (symbol, window) pair."""
    from .agents.pipeline import build_strategy
    from .backtest.engine import BacktestEngine
    from .data.service import make_service

    symbols = list(symbols or cfg.data.universe)
    windows = list(windows or DEFAULT_WINDOWS)
    svc = service or make_service(cfg)
    engine = BacktestEngine(cfg.backtest.to_contract())

    rows: list[EvalRow] = []
    for symbol in symbols:
        for w in windows:
            try:
                bars = svc.history(symbol, w.start, w.end, timeframe=cfg.data.timeframe)
            except Exception as exc:  # noqa: BLE001 - evaluation must not die on one vendor
                print(f"  skip {symbol} {w.name}: {type(exc).__name__}: {exc}")
                continue
            if len(bars) < cfg.backtest.warmup_bars + 10:
                print(f"  skip {symbol} {w.name}: only {len(bars)} bars")
                continue
            # A fresh strategy per row: no state carried between windows, so every
            # row is an independent backtest (and reproduces standalone runs).
            strategy = build_strategy(cfg)
            result = engine.run(
                bars, strategy, symbol=symbol, timeframe=cfg.data.timeframe,
                periods_per_year=cfg.backtest.periods_per_year,
                risk_free_rate=cfg.backtest.risk_free_rate,
            )
            m = result.metrics
            rows.append(EvalRow(
                symbol=symbol, window=w.name, note=w.note, bars=len(bars),
                provider=getattr(svc, "last_provider", "?"),
                total_return=m["total_return"],
                benchmark_total_return=m["benchmark_total_return"],
                excess_return=m["excess_return"],
                cagr=m["cagr"], sharpe=m["sharpe"],
                benchmark_sharpe=m["benchmark_sharpe"],
                max_drawdown=m["max_drawdown"],
                benchmark_max_drawdown=m["benchmark_max_drawdown"],
                exposure=m["exposure"], num_trades=m["num_trades"],
                risk_vetoes=m["risk_vetoes"],
            ))
    return rows


def render_table(rows: Sequence[EvalRow], title: str = "Multi-regime evaluation") -> str:
    """Markdown table plus a win/loss ledger that cannot hide the losses."""
    if not rows:
        return f"## {title}\n\n_no results_\n"

    out: list[str] = [f"## {title}", ""]
    out.append("| symbol | window | regime | return | buy&hold | excess | sharpe | b&h sharpe "
               "| maxDD | b&h maxDD | exposure | trades |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        sign = "+" if r.excess_return >= 0 else ""
        out.append(
            f"| {r.symbol} | {r.window} | {r.note} "
            f"| {r.total_return:+.1%} | {r.benchmark_total_return:+.1%} "
            f"| {sign}{r.excess_return:.1%} "
            f"| {r.sharpe:+.2f} | {r.benchmark_sharpe:+.2f} "
            f"| {r.max_drawdown:.1%} | {r.benchmark_max_drawdown:.1%} "
            f"| {r.exposure:.0%} | {r.num_trades:.0f} |"
        )

    wins = [r for r in rows if r.beats]
    losses = [r for r in rows if not r.beats]
    mean_excess = sum(r.excess_return for r in rows) / len(rows)

    out += ["", "### Ledger", ""]
    out.append(f"- windows evaluated: **{len(rows)}**")
    out.append(f"- beat buy & hold: **{len(wins)}** · lost to buy & hold: **{len(losses)}**")
    out.append(f"- mean excess return: **{mean_excess:+.2%}**")
    out.append(f"- mean strategy max drawdown: "
               f"**{sum(r.max_drawdown for r in rows)/len(rows):.1%}** vs buy & hold "
               f"**{sum(r.benchmark_max_drawdown for r in rows)/len(rows):.1%}**")

    # Aggregate character of the strategy across regimes.
    avg_exposure = sum(r.exposure for r in rows) / len(rows)
    out += ["", "### Reading", ""]
    out.append(
        f"Average exposure is **{avg_exposure:.0%}** of equity. The strategy is "
        "structurally defensive: it holds a fraction of the asset, so it captures "
        "less of a rally and loses less in a drawdown. That trade-off is the "
        "design, not an excuse — the drawdown columns are the evidence."
    )
    if losses:
        worst = min(losses, key=lambda r: r.excess_return)
        out.append(
            f"\nWorst relative window: **{worst.symbol} {worst.window}** "
            f"({worst.excess_return:+.1%} vs buy & hold) — reported, not hidden."
        )
    best = max(rows, key=lambda r: r.excess_return)
    out.append(
        f"Best relative window: **{best.symbol} {best.window}** "
        f"({best.excess_return:+.1%} vs buy & hold)."
    )
    out.append("")
    out.append("> Single-window results are not evidence. Read the spread across "
               "regimes, and note that none of this is investment advice.")
    return "\n".join(out) + "\n"
