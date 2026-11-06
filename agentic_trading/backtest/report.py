"""Human-readable reports for a :class:`~agentic_trading.contracts.BacktestResult`.

Three artefacts, zero dependencies:

* ``render_markdown`` -- a GitHub-flavoured summary table plus the trade list.
* ``render_html`` -- a *self-contained* page: inline CSS, an inline SVG
  ``<polyline>`` equity-vs-benchmark chart, no JavaScript, no CDN, no fonts or
  images loaded from the network.  It opens correctly from ``file://`` and can
  be attached to a PR or an email.
* ``save_report`` -- write all three (``.md`` / ``.html`` / ``.json``) and return
  the paths.

The chart is deliberately drawn from the engine's own curves rather than from
pre-computed decoration: if the equity path is wrong, the picture is wrong in
the same way, and the numbers in the same report prove it.

``print_summary`` returns a short terminal-friendly string and does **not**
print; the CLI decides where it goes (``print(print_summary(result))``).
"""

from __future__ import annotations

import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..contracts import BacktestResult

__all__ = ["render_markdown", "render_html", "save_report", "print_summary"]


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _finite(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pct(value: object, signed: bool = True) -> str:
    num = _finite(value)
    if num is None:
        return "n/a"
    return f"{num * 100:+.2f}%" if signed else f"{num * 100:.2f}%"


def _num(value: object, digits: int = 2, suffix: str = "") -> str:
    num = _finite(value)
    if num is None:
        return "n/a"
    return f"{num:,.{digits}f}{suffix}"


def _int(value: object) -> str:
    num = _finite(value)
    if num is None:
        return "n/a"
    return f"{int(round(num)):,}"


def _money(value: object) -> str:
    num = _finite(value)
    if num is None:
        return "n/a"
    return f"${num:,.2f}"


def _date(value: datetime | None) -> str:
    if value is None:
        return "n/a"
    try:
        return value.date().isoformat()
    except AttributeError:  # pragma: no cover - defensive for odd inputs
        return str(value)


def _period(result: BacktestResult) -> tuple[str, str]:
    curve = result.equity_curve
    if not curve:
        return "n/a", "n/a"
    return _date(curve[0][0]), _date(curve[-1][0])


def _fmt(value: object, kind: str) -> str:
    if kind == "pct":
        return _pct(value)
    if kind == "pct_plain":
        return _pct(value, signed=False)
    if kind == "num":
        return _num(value)
    if kind == "numx":
        return _num(value, suffix="x")
    if kind == "int":
        return _int(value)
    return str(value)


#: (label, metric key, benchmark key or None, formatter kind)
_METRIC_ROWS: tuple[tuple[str, str, str | None, str], ...] = (
    ("Total return", "total_return", "benchmark_total_return", "pct"),
    ("CAGR", "cagr", "benchmark_cagr", "pct"),
    ("Volatility (annualised)", "volatility", None, "pct_plain"),
    ("Sharpe", "sharpe", "benchmark_sharpe", "num"),
    ("Sortino", "sortino", None, "num"),
    ("Max drawdown", "max_drawdown", "benchmark_max_drawdown", "pct_plain"),
    ("Calmar", "calmar", None, "num"),
    ("Best bar", "best_day", None, "pct"),
    ("Worst bar", "worst_day", None, "pct"),
    ("Win rate (closed trades)", "win_rate", None, "pct_plain"),
    ("Trades", "num_trades", None, "int"),
    ("Turnover", "turnover", None, "numx"),
    ("Exposure", "exposure", None, "pct_plain"),
    ("Decisions", "num_decisions", None, "int"),
    ("Bars", "num_bars", None, "int"),
    ("Risk vetoes", "risk_vetoes", None, "int"),
    ("Excess return vs benchmark", "excess_return", None, "pct"),
    ("Alpha CAGR vs benchmark", "alpha_cagr", None, "pct"),
)


def _rows(result: BacktestResult) -> list[tuple[str, str, str]]:
    """Pre-formatted (label, strategy, benchmark) rows; '—' when unpaired."""
    metrics = result.metrics
    rows: list[tuple[str, str, str]] = []
    for label, key, bench_key, kind in _METRIC_ROWS:
        strategy_value = _fmt(metrics.get(key), kind)
        benchmark_value = _fmt(metrics.get(bench_key), kind) if bench_key else "—"
        rows.append((label, strategy_value, benchmark_value))
    return rows


def _config_lines(result: BacktestResult) -> list[str]:
    cfg = result.config
    if cfg is None:
        return []
    return [
        f"initial_cash={cfg.initial_cash:,.2f}",
        f"fee_bps={cfg.fee_bps:g}",
        f"slippage_bps={cfg.slippage_bps:g}",
        f"warmup_bars={cfg.warmup_bars}",
        f"allow_short={cfg.allow_short}",
        f"trade_at={cfg.trade_at}",
    ]


def _trade_rows(result: BacktestResult, limit: int = 100) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for trade in result.trades[:limit]:
        weight = _finite(trade.target_weight)
        out.append(
            (
                _date(trade.asof),
                trade.side,
                _num(trade.qty, 6),
                _money(trade.price),
                _money(trade.fee),
                _money(trade.notional),
                _money(trade.equity_after),
                f"{weight * 100:.1f}%" if weight is not None else "n/a",
                trade.reason,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def render_markdown(result: BacktestResult) -> str:
    """A compact markdown report: headline, metrics vs benchmark, trades."""
    start, end = _period(result)
    lines: list[str] = [
        f"# Backtest report — {result.symbol} ({result.timeframe})",
        "",
        f"**Period:** {start} → {end}  ",
        f"**Initial cash:** {_money(result.initial_cash)}  ",
        f"**Bars:** {_int(result.metrics.get('num_bars'))}  "
        f"**Decisions:** {_int(result.metrics.get('num_decisions'))}  "
        f"**Trades:** {_int(result.metrics.get('num_trades'))}",
        "",
        "## Performance",
        "",
        "| Metric | Strategy | Buy & hold |",
        "| --- | ---: | ---: |",
    ]
    for label, strategy_value, benchmark_value in _rows(result):
        lines.append(f"| {label} | {strategy_value} | {benchmark_value} |")
    lines.append("")

    config_lines = _config_lines(result)
    if config_lines:
        lines.extend(["## Execution assumptions", "", "```", *config_lines, "```", ""])

    trade_rows = _trade_rows(result, limit=100)
    if trade_rows:
        lines.extend(
            [
                f"## Trades ({len(result.trades)})",
                "",
                "| Date | Side | Qty | Price | Fee | Notional | Equity after | Target | Reason |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in trade_rows:
            noted = [str(cell).replace("|", "\\|") for cell in row]
            lines.append("| " + " | ".join(noted) + " |")
        if len(result.trades) > len(trade_rows):
            lines.extend(["", f"_Showing the first {len(trade_rows)} of {len(result.trades)} fills._"])
        lines.append("")
    else:
        lines.extend(["## Trades", "", "_No fills: the strategy never traded._", ""])

    lines.extend(
        [
            "---",
            "",
            "_Generated by AgenticTrading. Costs are modelled; past results are not a "
            "claim of future alpha. Decisions see only bars up to their own timestamp "
            "(no lookahead)._",
            "",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML (inline CSS + inline SVG; no JS, no network)
# --------------------------------------------------------------------------- #
_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 980px; padding: 32px 20px 64px;
       font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
             "Helvetica Neue", Arial, sans-serif;
       color: #111827; background: #ffffff; }
h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 16px; margin: 32px 0 10px; text-transform: uppercase;
     letter-spacing: 0.06em; color: #4b5563; }
.sub { color: #6b7280; margin: 0 0 22px; font-size: 13.5px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0 8px; }
.card { flex: 1 1 180px; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 12px 14px; background: #f9fafb; }
.card .label { font-size: 12px; color: #6b7280; text-transform: uppercase;
               letter-spacing: 0.05em; }
.card .value { font-size: 22px; font-weight: 650; margin-top: 4px;
               font-variant-numeric: tabular-nums; }
.pos { color: #047857; } .neg { color: #b91c1c; }
.chart { border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px;
         margin: 10px 0 4px; background: #ffffff; }
.legend { display: flex; gap: 18px; font-size: 12.5px; color: #4b5563;
          margin: 0 0 6px 4px; }
.swatch { display: inline-block; width: 18px; height: 3px; border-radius: 2px;
          vertical-align: middle; margin-right: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px;
        font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid #f0f1f3; }
th:first-child, td:first-child { text-align: left; }
th { color: #6b7280; font-weight: 600; font-size: 12px; text-transform: uppercase;
     letter-spacing: 0.04em; }
tbody tr:hover { background: #f9fafb; }
.muted { color: #9ca3af; font-size: 12.5px; }
footer { margin-top: 32px; color: #9ca3af; font-size: 12px; line-height: 1.5;
         border-top: 1px solid #f0f1f3; padding-top: 14px; }
"""

_SVG_WIDTH = 900
_SVG_HEIGHT = 340
_PAD_LEFT = 74
_PAD_RIGHT = 18
_PAD_TOP = 14
_PAD_BOTTOM = 30


def _equity_svg(result: BacktestResult) -> str:
    """Inline SVG with one ``<polyline>`` per series (strategy + benchmark)."""
    strategy = [(_finite(v) or 0.0) for _, v in result.equity_curve]
    benchmark = [(_finite(v) or 0.0) for _, v in result.benchmark_curve]
    series: list[tuple[str, list[float], str, str | None]] = []
    if strategy:
        series.append(("Strategy", strategy, "#2563eb", None))
    if benchmark:
        series.append(("Buy & hold", benchmark, "#6b7280", "7 5"))
    if not series:
        return '<p class="muted">No equity data for this run.</p>'

    values = [v for _, vals, _, _ in series for v in vals]
    lo, hi = min(values), max(values)
    if hi - lo <= max(abs(hi), 1.0) * 1e-9:   # completely flat curve
        span = max(abs(hi) * 0.01, 1.0)
        lo, hi = lo - span, hi + span
    else:
        padding = (hi - lo) * 0.06
        lo, hi = lo - padding, hi + padding

    plot_w = _SVG_WIDTH - _PAD_LEFT - _PAD_RIGHT
    plot_h = _SVG_HEIGHT - _PAD_TOP - _PAD_BOTTOM

    def x_at(index: int, length: int) -> float:
        if length <= 1:
            return _PAD_LEFT + plot_w / 2.0
        return _PAD_LEFT + plot_w * index / (length - 1)

    def y_at(value: float) -> float:
        return _PAD_TOP + (hi - value) / (hi - lo) * plot_h

    def points(vals: Sequence[float]) -> str:
        return " ".join(
            f"{x_at(i, len(vals)):.2f},{y_at(v):.2f}" for i, v in enumerate(vals)
        )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SVG_WIDTH} {_SVG_HEIGHT}" '
        f'width="{_SVG_WIDTH}" height="{_SVG_HEIGHT}" '
        f'style="max-width:100%;height:auto;display:block" '
        f'role="img" aria-label="Equity curve vs buy and hold">'
    ]
    # Horizontal grid + y-axis money labels.
    for tick in range(5):
        frac = tick / 4.0
        value = hi - (hi - lo) * frac
        y = _PAD_TOP + plot_h * frac
        parts.append(
            f'<line x1="{_PAD_LEFT}" y1="{y:.2f}" x2="{_SVG_WIDTH - _PAD_RIGHT}" '
            f'y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{_PAD_LEFT - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#6b7280" font-family="system-ui, sans-serif">'
            f"{html.escape(_money(value))}</text>"
        )
    # x-axis date labels (first / middle / last bar).
    if result.equity_curve:
        stamps = [ts for ts, _ in result.equity_curve]
        picks = sorted({0, len(stamps) // 2, len(stamps) - 1})
        for index in picks:
            x = x_at(index, len(stamps))
            anchor = "start" if index == 0 else ("end" if index == len(stamps) - 1 else "middle")
            parts.append(
                f'<text x="{x:.2f}" y="{_SVG_HEIGHT - 9}" text-anchor="{anchor}" '
                f'font-size="11" fill="#6b7280" font-family="system-ui, sans-serif">'
                f"{html.escape(_date(stamps[index]))}</text>"
            )
    # Series lines.
    for _name, vals, color, dash in series:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_attr} '
            f'points="{points(vals)}" />'
        )
    parts.append("</svg>")
    return "".join(parts)


def _html_table(result: BacktestResult) -> str:
    rows: list[str] = []
    for label, strategy_value, benchmark_value in _rows(result):
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(strategy_value)}</td>"
            f"<td>{html.escape(benchmark_value)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Metric</th><th>Strategy</th><th>Buy &amp; hold</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _html_trades(result: BacktestResult) -> str:
    trade_rows = _trade_rows(result, limit=100)
    if not trade_rows:
        return '<p class="muted">No fills: the strategy never traded.</p>'
    header = (
        "<tr><th>Date</th><th>Side</th><th>Qty</th><th>Price</th><th>Fee</th>"
        "<th>Notional</th><th>Equity after</th><th>Target</th><th>Reason</th></tr>"
    )
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in trade_rows
    )
    note = ""
    if len(result.trades) > len(trade_rows):
        note = (
            f'<p class="muted">Showing the first {len(trade_rows)} of '
            f"{len(result.trades)} fills.</p>"
        )
    return f"<table><thead>{header}</thead><tbody>{body}</tbody></table>{note}"


def _card(label: str, value: str, tone: str = "") -> str:
    cls = f"value {tone}".strip()
    return (
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="{cls}">{html.escape(value)}</div></div>'
    )


def _tone(value: object) -> str:
    num = _finite(value)
    if num is None:
        return ""
    return "pos" if num > 0 else ("neg" if num < 0 else "")


def render_html(result: BacktestResult) -> str:
    """A single self-contained HTML file (inline CSS, inline SVG, no JS/CDN)."""
    start, end = _period(result)
    metrics = result.metrics
    cards = "".join(
        [
            _card("Total return", _pct(metrics.get("total_return")),
                  _tone(metrics.get("total_return"))),
            _card("Sharpe", _num(metrics.get("sharpe"))),
            _card("Max drawdown", _pct(metrics.get("max_drawdown"), signed=False)),
            _card("Excess vs buy & hold", _pct(metrics.get("excess_return")),
                  _tone(metrics.get("excess_return"))),
        ]
    )
    config_text = " · ".join(html.escape(line) for line in _config_lines(result))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = f"Backtest report — {result.symbol} ({result.timeframe})"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="icon" href="data:," />
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{html.escape(result.symbol)} <span class="muted">({html.escape(result.timeframe)})</span></h1>
<p class="sub">{html.escape(start)} → {html.escape(end)} · initial cash {html.escape(_money(result.initial_cash))} · {html.escape(_int(metrics.get('num_bars')))} bars · {html.escape(_int(metrics.get('num_decisions')))} decisions · {html.escape(_int(metrics.get('num_trades')))} trades</p>
<div class="cards">{cards}</div>
<h2>Equity curve</h2>
<div class="chart">
<div class="legend"><span><span class="swatch" style="background:#2563eb"></span>Strategy</span><span><span class="swatch" style="background:#6b7280"></span>Buy &amp; hold</span></div>
{_equity_svg(result)}
</div>
<h2>Metrics vs buy &amp; hold</h2>
{_html_table(result)}
<h2>Trades</h2>
{_html_trades(result)}
<footer>
Generated {html.escape(generated)} by AgenticTrading · {config_text}<br />
Fills occur at the decision bar's close with slippage and fees; the benchmark buys once at the first post-warm-up close and holds. Every decision saw only bars up to its own timestamp — no lookahead. Past performance is not a forecast.
</footer>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Persistence and terminal summary
# --------------------------------------------------------------------------- #
def _safe_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "backtest"


def save_report(result: BacktestResult, outdir: str, stem: str | None = None) -> dict[str, str]:
    """Write ``.md``, ``.html`` and ``.json`` reports; return their paths.

    ``stem`` defaults to ``<symbol>_<timeframe>``.  The returned mapping is
    ``{"md": ..., "html": ..., "json": ...}``.
    """
    directory = Path(outdir)
    directory.mkdir(parents=True, exist_ok=True)
    name = _safe_stem(stem if stem is not None else f"{result.symbol}_{result.timeframe}")

    markdown_path = directory / f"{name}.md"
    html_path = directory / f"{name}.html"
    json_path = directory / f"{name}.json"

    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    html_path.write_text(render_html(result), encoding="utf-8")

    payload = result.to_dict()
    payload["report"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "agentic_trading.backtest.engine",
    }
    try:
        json_text = json.dumps(payload, indent=2, default=str, allow_nan=False)
    except (TypeError, ValueError):
        # Diagnostics produced by an agent may contain NaN or exotic objects;
        # reporting must never be the thing that crashes a finished backtest.
        json_text = json.dumps(payload, indent=2, default=str)
    json_path.write_text(json_text + "\n", encoding="utf-8")

    return {
        "md": str(markdown_path),
        "html": str(html_path),
        "json": str(json_path),
    }


def print_summary(result: BacktestResult) -> str:
    """Print a short terminal summary and return the same text.

    The function both prints (so ``print_summary(result)`` in a CLI does the
    obvious thing) and returns the string (so callers can log or embed it).
    """
    metrics = result.metrics
    start, end = _period(result)
    lines = [
        f"{result.symbol} {result.timeframe}  {start} → {end}  "
        f"({_int(metrics.get('num_bars'))} bars, {_int(metrics.get('num_trades'))} trades)",
        f"  Return        {_pct(metrics.get('total_return')):>10}   "
        f"buy & hold {_pct(metrics.get('benchmark_total_return')):>10}   "
        f"excess {_pct(metrics.get('excess_return')):>10}",
        f"  CAGR          {_pct(metrics.get('cagr')):>10}   "
        f"buy & hold {_pct(metrics.get('benchmark_cagr')):>10}   "
        f"alpha  {_pct(metrics.get('alpha_cagr')):>10}",
        f"  Sharpe        {_num(metrics.get('sharpe')):>10}   "
        f"buy & hold {_num(metrics.get('benchmark_sharpe')):>10}",
        f"  Max drawdown  {_pct(metrics.get('max_drawdown'), signed=False):>10}   "
        f"buy & hold {_pct(metrics.get('benchmark_max_drawdown'), signed=False):>10}",
        f"  Exposure {_pct(metrics.get('exposure'), signed=False)} · "
        f"decisions {_int(metrics.get('num_decisions'))} · "
        f"risk vetoes {_int(metrics.get('risk_vetoes'))}",
    ]
    text = "\n".join(lines)
    print(text)
    return text
