"""Command-line interface for AgenticTrading.

Subcommands
-----------
    doctor    check environment, config, data vendors and LLM availability
    backtest  run a leakage-free historical backtest and write reports
    analyze   produce one auditable decision for the latest bar
    broker    inspect the execution broker (paper or Robinhood MCP)
    live      run the decision -> risk -> execution loop (paper by default)

Safety: nothing here can place a real order unless ``execution.live_enabled()``
is true, which itself requires an explicit opt-in in both the config file and the
environment, plus the absence of a kill-switch file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import __version__
from .config import AppConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _parse_date(text: str | None) -> datetime | None:
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise SystemExit(f"error: cannot parse date {text!r} (expected YYYY-MM-DD)")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load(args) -> AppConfig:
    cfg = AppConfig.load(args.config)
    if getattr(args, "offline", False):
        cfg.data.allow_network = False
        # Force the synthetic provider first: deterministic and network-free.
        order = [p for p in cfg.data.provider_order if p == "synthetic"] + \
                [p for p in cfg.data.provider_order if p != "synthetic"]
        cfg.data.provider_order = order
    if getattr(args, "timeframe", None):
        cfg.data.timeframe = args.timeframe
    return cfg


def _section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def _kv(key: str, value) -> None:
    print(f"  {key:<26} {value}")


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def cmd_doctor(args) -> int:
    cfg = _load(args)
    _section("AgenticTrading doctor")
    _kv("version", __version__)
    _kv("python", sys.version.split()[0])
    _kv("config", args.config or "(defaults)")
    _kv("repo", REPO_ROOT)

    ok = True

    _section("Data")
    _kv("allow_network", cfg.data.allow_network)
    _kv("provider_order", " -> ".join(cfg.data.provider_order))
    _kv("timeframe", cfg.data.timeframe)
    _kv("cache_dir", cfg.data.cache_dir)
    try:
        from .data.service import make_service
        svc = make_service(cfg)
        bars = svc.history(cfg.data.universe[0],
                           _utcnow() - timedelta(days=400), _utcnow())
        _kv("probe", f"{cfg.data.universe[0]}: {len(bars)} bars via {svc.last_provider}")
        if not bars:
            ok = False
    except Exception as exc:  # noqa: BLE001 - doctor must not crash
        _kv("probe", f"FAILED: {type(exc).__name__}: {exc}")
        ok = False

    _section("LLM")
    _kv("mode", cfg.llm.enabled)
    _kv("resolved", "enabled" if cfg.llm.resolved_enabled() else "disabled (deterministic mode)")
    _kv("model", cfg.llm.model)
    _kv("api_key_env", f"{cfg.llm.api_key_env}: {'set' if cfg.llm.api_key else 'not set'}")

    _section("Execution")
    _kv("enabled", cfg.execution.enabled)
    _kv("broker", cfg.execution.broker)
    _kv("mode", cfg.execution.mode)
    _kv("dry_run", cfg.execution.dry_run)
    _kv("live", "YES" if cfg.execution.live_enabled() else "no")
    for reason in cfg.execution.why_not_live():
        _kv("  gate", reason)
    _kv("max_order_notional", cfg.execution.max_order_notional)
    _kv("kill_switch", "ENGAGED" if cfg.execution.kill_switch_engaged() else "clear")

    _section("Robinhood MCP")
    _kv("url", cfg.execution.mcp_url)
    token = cfg.execution.mcp_access_token
    _kv("token_env", f"{cfg.execution.mcp_access_token_env}: {'set' if token else 'not set'}")
    if token:
        try:
            from .brokers import RobinhoodMCPBroker
            broker = RobinhoodMCPBroker(cfg.execution)
            _kv("available", broker.is_available())
            if getattr(broker, "tool_map", None):
                _kv("tool_map", json.dumps(broker.tool_map, indent=None))
        except Exception as exc:  # noqa: BLE001
            _kv("available", f"FAILED: {type(exc).__name__}: {exc}")

    _section("Verdict")
    print("  ready" if ok else "  degraded - see failures above")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
def cmd_backtest(args) -> int:
    from .backtest.engine import BacktestEngine
    from .backtest.report import print_summary, save_report
    from .agents.pipeline import build_strategy

    cfg = _load(args)
    start = _parse_date(args.start) or (_utcnow() - timedelta(days=730))
    end = _parse_date(args.end) or _utcnow()

    _section(f"Data: {args.symbol} {start.date()} -> {end.date()}")
    from .data.service import make_service
    svc = make_service(cfg)
    bars = svc.history(args.symbol, start, end, timeframe=cfg.data.timeframe)
    print(f"  {len(bars)} bars via {svc.last_provider}")
    if len(bars) < cfg.backtest.warmup_bars + 5:
        print(f"  error: need > {cfg.backtest.warmup_bars + 5} bars for warmup, got {len(bars)}")
        return 2

    strategy = build_strategy(cfg)
    engine = BacktestEngine(cfg.backtest.to_contract())
    _section(f"Backtest: strategy={strategy.name} mode={getattr(strategy, 'mode', '?')}")
    result = engine.run(
        bars, strategy, symbol=args.symbol, timeframe=cfg.data.timeframe,
        periods_per_year=cfg.backtest.periods_per_year,
        risk_free_rate=cfg.backtest.risk_free_rate,
    )
    print_summary(result)

    if args.out:
        paths = save_report(result, args.out)
        _section("Reports")
        for kind, path in paths.items():
            _kv(kind, path)
    return 0


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def cmd_analyze(args) -> int:
    from .agents.pipeline import build_strategy
    from .data.service import make_service
    from .contracts import PortfolioState

    cfg = _load(args)
    asof = _parse_date(args.asof) or _utcnow()
    svc = make_service(cfg)
    ctx = svc.context(args.symbol, asof, timeframe=cfg.data.timeframe)

    # Flat book: this is a research question, not a portfolio action.
    portfolio = PortfolioState(
        cash=cfg.backtest.initial_cash,
        positions={},
        equity=cfg.backtest.initial_cash,
        peak_equity=cfg.backtest.initial_cash,
        last_prices={args.symbol: ctx.last_close},
    )
    strategy = build_strategy(cfg)
    decision = strategy.decide(ctx, portfolio)

    if args.json:
        print(json.dumps(decision.to_dict(), indent=2))
        return 0

    _section(f"Decision: {decision.symbol} as of {decision.asof.date()}")
    _kv("mode", decision.mode)
    _kv("action", decision.action.value)
    _kv("target_weight", f"{decision.target_weight:.2%}")
    _kv("confidence", f"{decision.confidence:.2f}")
    _kv("data provider", ctx.provider)
    _kv("last close", ctx.last_close)

    if decision.analysis:
        _section("Analysts")
        for stance in decision.analysis:
            _kv(stance.agent, f"{stance.label:<8} {stance.confidence:.2f}  [{stance.source}]")
            if args.verbose:
                print(f"      {stance.rationale}")

    if decision.debate:
        _section("Research debate")
        _kv("consensus", decision.debate.consensus)
        _kv("net_score", f"{decision.debate.net_score:+.3f}")

    if decision.proposal:
        _section("Trader proposal")
        _kv("action", decision.proposal.action.value)
        _kv("target_weight", f"{decision.proposal.target_weight:.2%}")
        _kv("thesis", decision.proposal.thesis[:300])

    if decision.risk:
        _section("Deterministic risk gate")
        _kv("approved", decision.risk.approved)
        _kv("final_weight", f"{decision.risk.target_weight:.2%}")
        for v in decision.risk.vetoes:
            _kv("VETO", v)
        for a in decision.risk.adjustments:
            _kv("adjusted", a)
        if decision.risk.stop_loss:
            _kv("stop_loss", decision.risk.stop_loss)
        if decision.risk.take_profit:
            _kv("take_profit", decision.risk.take_profit)
    return 0


# --------------------------------------------------------------------------- #
# broker
# --------------------------------------------------------------------------- #
def cmd_broker(args) -> int:
    cfg = _load(args)
    from .brokers import build_broker
    broker = build_broker(cfg.execution, cfg.data)
    _section(f"Broker: {broker.name} (live={broker.is_live})")
    _kv("available", broker.is_available())
    if not broker.is_available():
        print("  hint: paper is always available; robinhood needs "
              f"{cfg.execution.mcp_access_token_env} and a connected MCP client")
        return 1
    try:
        account = broker.get_account()
        _section("Account")
        _kv("account_id", account.account_id)
        _kv("cash", f"{account.cash:,.2f}")
        _kv("equity", f"{account.equity:,.2f}")
        _kv("buying_power", f"{account.buying_power:,.2f}")
        if account.positions:
            _section("Positions")
            for sym, pos in account.positions.items():
                _kv(sym, f"qty={pos.qty:g} avg={pos.avg_price:,.2f}")
        quotes = broker.get_quotes(list(cfg.data.universe))
        _section("Quotes")
        for sym, q in quotes.items():
            _kv(sym, f"{q.price:,.2f} ({q.source})")
    except Exception as exc:  # noqa: BLE001
        print(f"  error: {type(exc).__name__}: {exc}")
        return 1
    finally:
        close = getattr(broker, "close", None)
        if callable(close):
            close()
    return 0


# --------------------------------------------------------------------------- #
# live (paper by default)
# --------------------------------------------------------------------------- #
def cmd_live(args) -> int:
    from .agents.pipeline import build_strategy
    from .brokers import ExecutionEngine, build_broker
    from .data.service import make_service

    cfg = _load(args)
    if args.paper:
        cfg.execution.broker = "paper"
        cfg.execution.mode = "paper"
        cfg.execution.dry_run = True
        cfg.execution.enabled = True

    if cfg.execution.live_enabled():
        print("\033[31mLIVE TRADING ENABLED\033[0m - real orders can be placed.")
    else:
        _section("Live trading is NOT enabled; running in preview")
        for reason in cfg.execution.why_not_live():
            _kv("gate", reason)
        _kv("effect", "decisions and orders are journaled but never sent")

    strategy = build_strategy(cfg)
    broker = build_broker(cfg.execution, cfg.data)
    if not broker.is_available():
        print(f"error: broker {broker.name!r} unavailable")
        return 2
    engine = ExecutionEngine(broker, cfg.execution, cfg.data)
    svc = make_service(cfg)

    iterations = 1 if args.once else args.max_iterations
    symbols = args.symbols or cfg.data.universe
    _section(f"Loop: {iterations} iteration(s), symbols={symbols}, broker={broker.name}")

    account = engine.sync_from_broker()
    for i in range(iterations):
        asof = _utcnow()
        for symbol in symbols:
            try:
                ctx = svc.context(symbol, asof, timeframe=cfg.data.timeframe)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{symbol}] data error: {type(exc).__name__}: {exc}")
                continue
            equity = account.equity if account else cfg.backtest.initial_cash
            from .contracts import PortfolioState
            portfolio = PortfolioState(
                cash=account.cash if account else equity,
                positions=dict(account.positions) if account else {},
                equity=equity,
                peak_equity=equity,
                last_prices={symbol: ctx.last_close},
            )
            decision = strategy.decide(ctx, portfolio)
            result = engine.execute(decision)
            status = result.status.value if result else "no-op"
            print(f"  [{symbol}] {decision.action.value:<4} "
                  f"w={decision.target_weight:6.2%} "
                  f"conf={decision.confidence:.2f} -> {status}")
        if i + 1 < iterations:
            time.sleep(max(1, args.interval))
    return 0


# --------------------------------------------------------------------------- #
# evaluate (multi-regime)
# --------------------------------------------------------------------------- #
def cmd_evaluate(args) -> int:
    from .evaluation import DEFAULT_WINDOWS, RegimeWindow, evaluate_grid, render_table

    cfg = _load(args)
    symbols = args.symbols or cfg.data.universe
    _section(f"Multi-regime evaluation: {len(symbols)} symbol(s) x "
             f"{len(DEFAULT_WINDOWS)} windows")
    rows = evaluate_grid(cfg, symbols=symbols)
    table = render_table(rows, title=f"{cfg.data.timeframe} bars")
    print()
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
        _section("Written")
        _kv("markdown", str(out))
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agentic-trading",
        description="AgenticTrading - multi-agent trading with deterministic risk gates.",
    )
    p.add_argument("--version", action="version", version=f"agentic-trading {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"),
                        help="path to YAML config")
        sp.add_argument("--offline", action="store_true",
                        help="never touch the network; use the synthetic provider")
        sp.add_argument("--timeframe", choices=["15m", "1h", "1d"], default=None)

    sp = sub.add_parser("doctor", help="check environment and connectivity")
    common(sp)
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("backtest", help="run a leakage-free backtest")
    common(sp)
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--start", default=None)
    sp.add_argument("--end", default=None)
    sp.add_argument("--out", default=None, help="report output directory")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("analyze", help="one auditable decision for the latest bar")
    common(sp)
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--asof", default=None)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("broker", help="inspect the execution broker")
    common(sp)
    sp.set_defaults(func=cmd_broker)

    sp = sub.add_parser("evaluate", help="multi-regime evaluation vs buy & hold")
    common(sp)
    sp.add_argument("--symbols", nargs="*", default=None)
    sp.add_argument("--out", default=None, help="write the markdown table here")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("live", help="decision -> risk -> execution loop (paper by default)")
    common(sp)
    sp.add_argument("--symbols", nargs="*", default=None)
    sp.add_argument("--paper", action="store_true", help="force paper broker + preview")
    sp.add_argument("--once", action="store_true", help="single iteration")
    sp.add_argument("--max-iterations", type=int, default=1)
    sp.add_argument("--interval", type=int, default=60, help="seconds between iterations")
    sp.set_defaults(func=cmd_live)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
