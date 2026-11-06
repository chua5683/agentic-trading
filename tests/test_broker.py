"""Tests for the brokerage + guarded execution layer.

Every test is offline: the Robinhood MCP adapter is exercised through a fake
transport that mimics the ``initialize`` / ``tools/list`` / ``tools/call``
handshake, and the paper broker needs no sockets at all.

The safety tests are the point of this file: defaults must be inert, guardrails
must fire in deterministic code *before* ``broker.submit_order``, and every
attempt must land in the audit journal.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentic_trading.brokers import (
    AuditJournal,
    ExecutionEngine,
    MCPClient,
    MCPError,
    PaperBroker,
    RobinhoodMCPBroker,
    build_broker,
)
from agentic_trading.brokers.execution import MIN_TRADE_NOTIONAL
from agentic_trading.config import DataConfig, ExecutionConfig
from agentic_trading.contracts import (
    AccountSnapshot,
    Action,
    BacktestConfig,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioDecision,
    Position,
    Quote,
    RiskDecision,
)
from agentic_trading.errors import AgenticTradingError

NOW = datetime(2026, 11, 5, 21, 0, 0, tzinfo=timezone.utc)
TOKEN_ENV = "ROBINHOOD_MCP_ACCESS_TOKEN"
LIVE_ENV = "AGENTIC_TRADING_LIVE"
LIVE_PHRASE = "yes-i-understand-the-risk"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def decision(
    symbol: str = "AAA",
    action: Action = Action.BUY,
    target_weight: float = 0.1,
    *,
    asof: datetime = NOW,
    mode: str = "deterministic",
    thesis: str = "unit-test thesis",
    risk: RiskDecision | None = None,
) -> PortfolioDecision:
    return PortfolioDecision(
        asof=asof,
        symbol=symbol,
        action=action,
        target_weight=target_weight,
        confidence=0.72,
        thesis=thesis,
        mode=mode,
        risk=risk,
    )


def exec_cfg(tmp_path: Path, **overrides) -> ExecutionConfig:
    """Inert-by-default test config with explicit tmp journal/kill-switch paths."""
    values = dict(
        enabled=True,
        broker="paper",
        mode="paper",
        dry_run=True,
        symbol_allowlist=["AAA", "BBB"],
        audit_log_path=str(tmp_path / "orders.jsonl"),
        kill_switch_file=str(tmp_path / "STOP_TRADING"),
        max_order_notional=1_000.0,
        max_daily_orders=10,
        max_daily_notional=10_000.0,
    )
    values.update(overrides)
    return ExecutionConfig(**values)


def paper(price: float = 100.0, cash: float = 10_000.0, **kwargs) -> PaperBroker:
    return PaperBroker(start_cash=cash, price_source={"AAA": price, "BBB": price}, **kwargs)


class SpyBroker:
    """Records submissions; never talks to the network."""

    name = "spy"

    def __init__(
        self,
        *,
        is_live: bool = False,
        price: float = 100.0,
        equity: float = 10_000.0,
        positions: dict[str, Position] | None = None,
        status: OrderStatus = OrderStatus.FILLED,
    ) -> None:
        self.is_live = is_live
        self.price = price
        self.equity = equity
        self.positions = dict(positions or {})
        self.result_status = status
        self.submitted: list[OrderRequest] = []
        self.account_calls = 0
        self.quote_calls = 0

    def is_available(self) -> bool:
        return True

    def get_account(self) -> AccountSnapshot:
        self.account_calls += 1
        return AccountSnapshot(
            account_id="spy-account",
            cash=self.equity,
            equity=self.equity,
            buying_power=self.equity,
            positions=dict(self.positions),
            asof=NOW,
        )

    def get_quotes(self, symbols):
        self.quote_calls += 1
        return {symbol: Quote(symbol, self.price, asof=NOW, source="spy")
                for symbol in symbols}

    def submit_order(self, request: OrderRequest) -> OrderResult:
        self.submitted.append(request)
        qty = float(request.qty or 0.0)
        return OrderResult(
            client_order_id=request.client_order_id,
            status=self.result_status,
            broker_order_id="spy-order-1",
            symbol=request.symbol,
            side=request.side,
            qty=request.qty,
            filled_qty=qty if self.result_status is OrderStatus.FILLED else 0.0,
            avg_fill_price=self.price if self.result_status is OrderStatus.FILLED else None,
            notional=qty * self.price,
            submitted_at=NOW,
            message="spy fill",
        )

    def list_orders(self, limit: int = 50) -> list[OrderResult]:
        return []

    def cancel_order(self, broker_order_id: str) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Fake Robinhood MCP server (transport-level)
# --------------------------------------------------------------------------- #
FAKE_TOOLS = [
    {
        "name": "get_accounts",
        "description": "Get account balances, buying power and positions",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_equity_quotes",
        "description": "Get the latest equity quotes for one or more symbols",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "symbols": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "place_equity_order",
        "description": "Place a new equity order in the Agentic trading account",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "quantity": {"type": "number"},
                "type": {"type": "string",
                         "enum": ["market", "limit", "stop_market", "stop_limit"]},
                "limit_price": {"type": "number"},
                "stop_price": {"type": "number"},
                "time_in_force": {"type": "string"},
            },
            "required": ["symbol", "side", "quantity"],
        },
    },
    {
        "name": "get_equity_orders",
        "description": "List historical equity orders",
        "inputSchema": {"type": "object",
                        "properties": {"limit": {"type": "integer"}}},
    },
    {
        "name": "cancel_equity_order",
        "description": "Cancel a resting equity order by order id",
        "inputSchema": {"type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"]},
    },
    {
        "name": "execute_trade",
        "description": "Execute a trade",
        "inputSchema": {"type": "object"},  # deliberately schema-less
    },
]


class FakeMCPTransport:
    """A stateful fake of the Robinhood MCP server transport."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.session_id = "sess-test-1"
        self.order_status = "filled"
        self.last_order_args: dict | None = None

    def __call__(self, url, headers, payload):
        self.calls.append({"url": url, "headers": dict(headers), "payload": dict(payload)})
        method = payload.get("method")
        request_id = payload.get("id")

        def json_response(result: dict):
            return 200, {"Content-Type": "application/json"}, json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "result": result})

        if method == "initialize":
            return 200, {"Content-Type": "application/json",
                         "Mcp-Session-Id": self.session_id}, json.dumps({
                "jsonrpc": "2.0", "id": request_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-robinhood", "version": "0.0.1"},
                },
            })
        if method == "notifications/initialized":
            return 202, {}, ""
        if method == "tools/list":
            return json_response({"tools": FAKE_TOOLS})
        if method == "tools/call":
            params = payload.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "get_accounts":
                result = {"data": {
                    "account_number": "MCP-ACCOUNT-1",
                    "cash": 2500.0,
                    "equity": 12500.0,
                    "buying_power": 5000.0,
                    "positions": [
                        {"symbol": "AAPL", "quantity": 10,
                         "average_buy_price": 150.0},
                    ],
                }}
            elif name == "get_equity_quotes":
                symbol = args.get("symbol") or (args.get("symbols") or ["AAPL"])[0]
                result = {"quotes": [
                    {"symbol": symbol, "last_trade_price": 189.5,
                     "bid": 189.4, "ask": 189.6},
                ]}
            elif name == "place_equity_order":
                self.last_order_args = dict(args)
                result = {"order": {
                    "id": "RH-ORDER-42",
                    "symbol": args.get("symbol", "AAPL"),
                    "side": args.get("side", "buy"),
                    "quantity": args.get("quantity", 0),
                    "status": self.order_status,
                    "filled_quantity": args.get("quantity", 0),
                    "average_fill_price": 189.5,
                }}
            elif name == "get_equity_orders":
                result = {"orders": [
                    {"id": "RH-ORDER-1", "symbol": "AAPL", "side": "buy",
                     "quantity": 5, "filled_quantity": 5,
                     "average_fill_price": 180.0, "status": "filled"},
                ]}
            elif name == "cancel_equity_order":
                result = {"order": {"id": args.get("order_id"),
                                    "status": "cancelled"}}
            else:
                return 200, {"Content-Type": "application/json"}, json.dumps({
                    "jsonrpc": "2.0", "id": request_id,
                    "error": {"code": -32601, "message": f"unknown tool {name}"},
                })
            return json_response({
                "content": [{"type": "text", "text": json.dumps(result)}],
                "isError": False,
            })
        return 404, {}, json.dumps({"error": "not found"})


def rh_broker(monkeypatch, *, live: bool = True, tool_map=None, token: str | None = "test-token"):
    monkeypatch.setenv(TOKEN_ENV, token or "")
    cfg = ExecutionConfig(
        enabled=live,
        mcp_access_token_env=TOKEN_ENV,
        mcp_url="https://agent.robinhood.com/mcp/trading",
        audit_log_path="/tmp/never-used-robinhood.jsonl",
    )
    client = MCPClient(cfg.mcp_url, "test-token" if token else None,
                       transport=FakeMCPTransport())
    transport = client._transport
    broker = RobinhoodMCPBroker(cfg, client=client, live=live, tool_map=tool_map)
    return broker, transport


# --------------------------------------------------------------------------- #
# PaperBroker
# --------------------------------------------------------------------------- #
def test_paper_broker_fill_position_and_cash_math():
    broker = PaperBroker(start_cash=10_000.0, price_source={"AAA": 100.0})
    buy = broker.submit_order(OrderRequest(
        symbol="AAA", side=OrderSide.BUY.value, order_type=OrderType.MARKET.value,
        qty=10, client_order_id="buy-1"))
    assert buy.status is OrderStatus.FILLED
    assert buy.filled_qty == pytest.approx(10.0)
    assert buy.avg_fill_price == pytest.approx(100.0)
    assert broker.cash == pytest.approx(9_000.0)

    position = broker.position("AAA")
    assert position is not None
    assert position.qty == pytest.approx(10.0)
    assert position.avg_price == pytest.approx(100.0)

    account = broker.get_account()
    assert account.cash == pytest.approx(9_000.0)
    assert account.equity == pytest.approx(10_000.0)
    assert account.positions["AAA"].qty == pytest.approx(10.0)

    sell = broker.submit_order(OrderRequest(
        symbol="AAA", side=OrderSide.SELL.value, order_type=OrderType.MARKET.value,
        qty=4, client_order_id="sell-1"))
    assert sell.status is OrderStatus.FILLED
    assert broker.cash == pytest.approx(9_400.0)
    assert broker.position("AAA").qty == pytest.approx(6.0)

    # You cannot short in the paper account.
    reject = broker.submit_order(OrderRequest(
        symbol="AAA", side=OrderSide.SELL.value, order_type=OrderType.MARKET.value,
        qty=999, client_order_id="sell-2"))
    assert reject.status is OrderStatus.REJECTED
    assert "short" in reject.message.lower()
    assert broker.cash == pytest.approx(9_400.0)
    # Rejected orders are journalled by the execution engine, not by the
    # paper venue; only accepted orders appear in the venue order list.
    assert [o.status for o in broker.list_orders()] == [
        OrderStatus.FILLED, OrderStatus.FILLED]


def test_paper_broker_applies_backtest_costs():
    cfg = BacktestConfig(initial_cash=10_000.0, fee_bps=10.0, slippage_bps=5.0)
    broker = PaperBroker(cfg=cfg, price_source={"AAA": 100.0})
    result = broker.submit_order(OrderRequest(
        symbol="AAA", side=OrderSide.BUY.value, order_type=OrderType.MARKET.value,
        qty=10, client_order_id="cost-1"))
    expected_fill = 100.0 * (1.0 + 5.0 / 10_000.0)
    expected_fee = 10 * expected_fill * 10.0 / 10_000.0
    assert result.avg_fill_price == pytest.approx(expected_fill)
    assert broker.cash == pytest.approx(10_000.0 - 10 * expected_fill - expected_fee)


def test_paper_broker_persists_and_resumes(tmp_path: Path):
    state = tmp_path / "paper_state.json"
    first = PaperBroker(state_path=state, price_source={"AAA": 100.0})
    first.submit_order(OrderRequest(symbol="AAA", side=OrderSide.BUY.value,
                                    order_type=OrderType.MARKET.value, qty=3,
                                    client_order_id="persist-1"))
    assert state.exists()

    resumed = PaperBroker(state_path=state)  # no price source: falls back to avg price
    assert resumed.cash == pytest.approx(first.cash)
    assert resumed.position("AAA").qty == pytest.approx(3.0)
    assert [o.client_order_id for o in resumed.list_orders()] == ["persist-1"]
    assert resumed.get_quotes(["AAA"])["AAA"].price == pytest.approx(100.0)

    restored = PaperBroker(cfg=BacktestConfig(initial_cash=5_000.0), state_path=state)
    assert restored.cash == pytest.approx(first.cash)


# --------------------------------------------------------------------------- #
# Broker factory / Robinhood live gate
# --------------------------------------------------------------------------- #
def test_build_broker_defaults_to_paper_and_never_goes_live_without_gates(monkeypatch):
    monkeypatch.delenv(LIVE_ENV, raising=False)
    default = ExecutionConfig()
    assert default.live_enabled() is False
    broker = build_broker(default)
    assert isinstance(broker, PaperBroker)
    assert broker.is_live is False

    rh = build_broker(ExecutionConfig(broker="robinhood"))
    assert isinstance(rh, RobinhoodMCPBroker)
    assert rh.is_live is True          # the venue is live ...
    assert rh.live is False            # ... but this instance may not place orders

    unknown = build_broker(ExecutionConfig(broker="definitely-not-a-broker"))
    assert isinstance(unknown, PaperBroker)


def test_build_broker_only_marks_robinhood_live_when_every_gate_passes(
        monkeypatch, tmp_path: Path):
    monkeypatch.setenv(LIVE_ENV, LIVE_PHRASE)
    gated = ExecutionConfig(
        enabled=True, broker="robinhood", mode="live", dry_run=False,
        kill_switch_file=str(tmp_path / "STOP"),
    )
    assert gated.live_enabled() is True
    broker = build_broker(gated)
    assert isinstance(broker, RobinhoodMCPBroker)
    assert broker.live is True

    # A kill-switch file disarms the same config again.
    (tmp_path / "STOP").write_text("stop", encoding="utf-8")
    assert gated.live_enabled() is False
    assert build_broker(gated).live is False


def test_robinhood_submit_without_live_flag_previews_and_calls_nothing(monkeypatch):
    def should_not_be_called(*args, **kwargs):  # pragma: no cover
        raise AssertionError("transport must not be used for a preview")

    monkeypatch.setenv(TOKEN_ENV, "test-token")
    cfg = ExecutionConfig(mcp_access_token_env=TOKEN_ENV)
    client = MCPClient(cfg.mcp_url, "test-token", transport=should_not_be_called)
    broker = RobinhoodMCPBroker(cfg, client=client, live=False)
    result = broker.submit_order(OrderRequest(
        symbol="AAPL", side=OrderSide.BUY.value, order_type=OrderType.MARKET.value,
        qty=1, client_order_id="cid-1"))
    assert result.status is OrderStatus.PREVIEW
    assert result.client_order_id == "cid-1"
    assert "live=False" in result.message
    assert broker.tool_map == {}  # never even discovered tools


def test_robinhood_tool_map_override_is_visible_before_discovery(monkeypatch):
    broker, _ = rh_broker(monkeypatch, live=False,
                          tool_map={"submit_order": "my_place_order"})
    assert broker.tool_map["submit_order"] == "my_place_order"


# --------------------------------------------------------------------------- #
# ExecutionEngine: the safety properties
# --------------------------------------------------------------------------- #
def test_default_config_can_never_reach_a_live_or_paper_submit(tmp_path: Path):
    """Defaults: enabled=false, paper broker, dry_run=true -> preview only."""
    spy = SpyBroker(is_live=False)
    cfg = ExecutionConfig(  # literal defaults, plus tmp journal paths
        symbol_allowlist=["AAA"],
        audit_log_path=str(tmp_path / "orders.jsonl"),
        kill_switch_file=str(tmp_path / "STOP_TRADING"),
    )
    assert cfg.enabled is False and cfg.dry_run is True
    assert cfg.live_enabled() is False

    engine = ExecutionEngine(spy, cfg)
    result = engine.execute(decision(target_weight=0.2))

    assert result is not None and result.status is OrderStatus.PREVIEW
    assert spy.submitted == []
    assert engine.orders_today == 1
    events = engine.journal.replay()
    assert len(events) == 1 and events[0]["submitted"] is False

    # ... and the same holds against the real paper broker.
    real_paper = PaperBroker(price_source={"AAA": 100.0})
    paper_engine = ExecutionEngine(real_paper, cfg)
    preview = paper_engine.execute(decision(target_weight=0.2))
    assert preview.status is OrderStatus.PREVIEW
    assert real_paper.orders == []
    assert real_paper.cash == pytest.approx(10_000.0)


def test_dry_run_previews_and_never_calls_submit_order(tmp_path: Path):
    spy = SpyBroker(is_live=False)
    engine = ExecutionEngine(spy, exec_cfg(tmp_path))
    result = engine.execute(decision(target_weight=0.1))
    assert result.status is OrderStatus.PREVIEW
    assert result.submitted_at is None
    assert spy.submitted == []
    assert spy.account_calls == 1 and spy.quote_calls == 1


def test_engine_clamps_notional_to_max_order_notional(tmp_path: Path):
    spy = SpyBroker(is_live=False, price=100.0, equity=10_000.0)
    cfg = exec_cfg(tmp_path, max_order_notional=100.0)
    engine = ExecutionEngine(spy, cfg)
    result = engine.execute(decision(target_weight=1.0))  # wants $10,000
    assert result.status is OrderStatus.PREVIEW
    assert result.notional == pytest.approx(100.0)
    assert result.qty == pytest.approx(1.0)
    event = engine.journal.replay()[-1]
    assert event["guardrails"]["checks"]["max_order_notional"]["clamped"] is True
    assert any("max_order_notional" in note
               for note in event["guardrails"]["adjustments"])
    assert spy.submitted == []


def test_daily_order_cap_stops_third_order(tmp_path: Path):
    spy = SpyBroker(is_live=False, price=100.0, equity=10_000.0)
    cfg = exec_cfg(tmp_path, max_daily_orders=2, max_daily_notional=10_000.0)
    engine = ExecutionEngine(spy, cfg)

    first = engine.execute(decision(target_weight=0.05))
    second = engine.execute(decision(target_weight=0.10, asof=NOW.replace(hour=22)))
    third = engine.execute(decision(target_weight=0.15, asof=NOW.replace(hour=23)))

    assert first.status is OrderStatus.PREVIEW
    assert second.status is OrderStatus.PREVIEW
    assert third.status is OrderStatus.REJECTED
    assert "max_daily_orders" in third.message
    assert engine.orders_today == 2
    assert spy.submitted == []
    assert engine.journal.replay()[-1]["outcome"] == "blocked"


def test_daily_notional_cap_blocks(tmp_path: Path):
    spy = SpyBroker(is_live=False, price=100.0, equity=10_000.0)
    cfg = exec_cfg(tmp_path, max_daily_notional=600.0, max_order_notional=1_000.0)
    engine = ExecutionEngine(spy, cfg)

    first = engine.execute(decision(target_weight=0.04))   # $400 preview
    second = engine.execute(decision(target_weight=0.08))  # $400 would total $800
    assert first.status is OrderStatus.PREVIEW
    assert second.status is OrderStatus.REJECTED
    assert "max_daily_notional" in second.message
    assert engine.notional_today == pytest.approx(400.0)


def test_kill_switch_file_blocks_everything(tmp_path: Path):
    spy = SpyBroker()
    cfg = exec_cfg(tmp_path)
    kill = Path(cfg.kill_switch_file)
    kill.write_text("stop trading now", encoding="utf-8")
    assert cfg.kill_switch_engaged() is True

    engine = ExecutionEngine(spy, cfg)
    result = engine.execute(decision())
    assert result.status is OrderStatus.REJECTED
    assert "kill switch" in result.message
    assert spy.submitted == []
    assert spy.account_calls == 0  # blocked before touching the broker at all
    assert engine.journal.replay()[-1]["guardrails"]["blocks"]


def test_symbol_not_in_allowlist_blocks(tmp_path: Path):
    spy = SpyBroker()
    cfg = exec_cfg(tmp_path, symbol_allowlist=["BBB"])
    engine = ExecutionEngine(spy, cfg)
    result = engine.execute(decision(symbol="AAA"))
    assert result.status is OrderStatus.REJECTED
    assert "not in allowlist" in result.message
    assert spy.submitted == []
    assert spy.account_calls == 0


def test_allowlist_falls_back_to_data_universe(tmp_path: Path):
    spy = SpyBroker()
    cfg = exec_cfg(tmp_path, symbol_allowlist=[])
    engine = ExecutionEngine(spy, cfg, data_cfg=DataConfig(universe=["ZZZ"]))
    result = engine.execute(decision(symbol="AAA"))
    assert result.status is OrderStatus.REJECTED
    assert "not in allowlist" in result.message


def test_no_shorting_enforced(tmp_path: Path):
    held = {"AAA": Position("AAA", 5.0, 100.0)}
    spy = SpyBroker(positions=held)
    engine = ExecutionEngine(spy, exec_cfg(tmp_path))

    close = engine.execute(decision(target_weight=0.0))
    assert close.status is OrderStatus.PREVIEW
    assert close.side is OrderSide.SELL
    assert close.qty == pytest.approx(5.0)  # exactly the held quantity

    flipped = engine.execute(decision(action=Action.SELL, target_weight=-0.2,
                                      asof=NOW.replace(hour=22)))
    assert flipped.status is OrderStatus.REJECTED
    assert "shorting" in flipped.message.lower()
    assert len(spy.submitted) == 0


def test_hold_is_a_noop_that_is_still_journalled(tmp_path: Path):
    spy = SpyBroker()
    engine = ExecutionEngine(spy, exec_cfg(tmp_path))
    result = engine.execute(decision(action=Action.HOLD, target_weight=0.0))
    assert result is None
    assert spy.submitted == []
    events = engine.journal.replay()
    assert len(events) == 1
    assert events[0]["outcome"] == "noop"
    assert events[0]["result"] is None


def test_already_at_target_is_a_noop(tmp_path: Path):
    spy = SpyBroker(positions={"AAA": Position("AAA", 10.0, 100.0)})  # $1,000 = 10%
    engine = ExecutionEngine(spy, exec_cfg(tmp_path))
    result = engine.execute(decision(target_weight=0.10))
    assert result is None
    assert engine.journal.replay()[-1]["outcome"] == "noop"
    assert "minimum trade size" in " ".join(
        engine.journal.replay()[-1]["guardrails"]["adjustments"])


def test_live_venue_is_blocked_without_all_gates(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(LIVE_ENV, raising=False)
    spy = SpyBroker(is_live=True)
    cfg = exec_cfg(tmp_path, enabled=True, mode="live", dry_run=False)
    assert cfg.live_enabled() is False
    engine = ExecutionEngine(spy, cfg)
    result = engine.execute(decision())
    assert result.status is OrderStatus.REJECTED
    assert "live venue blocked" in result.message
    assert spy.submitted == []


def test_live_path_works_only_when_every_gate_passes(tmp_path: Path, monkeypatch):
    spy = SpyBroker(is_live=True)
    cfg = exec_cfg(tmp_path, enabled=True, mode="live", dry_run=False,
                   live_confirm_env=LIVE_ENV, live_confirm_phrase=LIVE_PHRASE)
    monkeypatch.setenv(LIVE_ENV, LIVE_PHRASE)
    assert cfg.live_enabled() is True

    engine = ExecutionEngine(spy, cfg)
    result = engine.execute(decision(target_weight=0.1))
    assert result.status is OrderStatus.FILLED
    assert len(spy.submitted) == 1
    sent = spy.submitted[0]
    assert sent.side is OrderSide.BUY
    assert sent.qty == pytest.approx(10.0)
    assert "mode=deterministic" in sent.reason
    assert "thesis=unit-test thesis" in sent.reason
    event = engine.journal.replay()[-1]
    assert event["outcome"] == "submitted" and event["submitted"] is True


def test_paper_live_like_loop_sends_when_explicitly_enabled(tmp_path: Path):
    broker = paper()
    cfg = exec_cfg(tmp_path, enabled=True, dry_run=False)
    engine = ExecutionEngine(broker, cfg)
    result = engine.execute(decision(target_weight=0.1))
    assert result.status is OrderStatus.FILLED
    assert broker.cash == pytest.approx(9_000.0)
    assert broker.position("AAA").qty == pytest.approx(10.0)
    assert engine.journal.replay()[-1]["outcome"] == "submitted"


def test_risk_provenance_is_carried_into_the_order_reason(tmp_path: Path):
    # Deliberately pass bare strings for tuple-typed fields: contract consumers
    # must normalise them instead of char-exploding the string in the join.
    risk = RiskDecision(approved=True, target_weight=0.1,
                        vetoes="drawdown close to limit",
                        adjustments="size halved by risk")
    spy = SpyBroker(is_live=True)
    cfg = exec_cfg(tmp_path, enabled=True, mode="live", dry_run=False,
                   live_confirm_env=LIVE_ENV, live_confirm_phrase=LIVE_PHRASE)
    engine = ExecutionEngine(spy, cfg)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv(LIVE_ENV, LIVE_PHRASE)
    try:
        engine.execute(decision(target_weight=0.1, risk=risk))
    finally:
        monkeypatch.undo()
    sent = spy.submitted[0]
    assert "risk_vetoes=drawdown close to limit" in sent.reason
    assert "risk_adjustments=size halved by risk" in sent.reason
    # The audit record also carries the risk verdict.
    event = engine.journal.replay()[-1]
    assert event["decision"]["risk"]["vetoes"] == ["drawdown close to limit"]


def test_sync_from_broker_returns_snapshot(tmp_path: Path):
    spy = SpyBroker(equity=12_345.0)
    engine = ExecutionEngine(spy, exec_cfg(tmp_path))
    snapshot = engine.sync_from_broker()
    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.equity == pytest.approx(12_345.0)
    assert engine.last_snapshot is snapshot


def test_client_order_id_is_stable_for_the_same_decision(tmp_path: Path):
    broker = paper()
    engine = ExecutionEngine(broker, exec_cfg(tmp_path))
    one = engine.execute(decision(target_weight=0.1))
    two = engine.execute(decision(target_weight=0.1))
    assert one.client_order_id == two.client_order_id

    other = engine.execute(decision(target_weight=0.2))
    assert other.client_order_id != one.client_order_id


# --------------------------------------------------------------------------- #
# Audit journal
# --------------------------------------------------------------------------- #
def test_audit_journal_appends_jsonl_and_is_replayable(tmp_path: Path):
    journal = AuditJournal(tmp_path / "nested" / "orders.jsonl")
    journal.record({"event": "one", "value": 1})
    journal.record({"event": "two", "when": NOW})
    last = journal.tail(1)[0]
    assert last["event"] == "two"
    assert last["when"] == NOW.isoformat()
    assert "ts" in last
    replay = journal.replay()
    assert [event["event"] for event in replay] == ["one", "two"]
    assert all("ts" in event for event in replay)
    assert len(journal) == 2

    lines = journal.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["event"] for line in lines] == ["one", "two"]


def test_engine_journals_every_attempt(tmp_path: Path):
    spy = SpyBroker(is_live=False)
    cfg = exec_cfg(tmp_path, max_daily_orders=2)
    engine = ExecutionEngine(spy, cfg)

    preview = engine.execute(decision(target_weight=0.05))
    blocked = engine.execute(decision(target_weight=0.10, asof=NOW.replace(hour=22)))
    noop = engine.execute(decision(action=Action.HOLD, target_weight=0.10,
                                   asof=NOW.replace(hour=23)))
    third = engine.execute(decision(target_weight=0.15, asof=NOW.replace(hour=23)))

    assert preview.status is OrderStatus.PREVIEW
    assert blocked.status is OrderStatus.PREVIEW
    assert noop is None
    assert third.status is OrderStatus.REJECTED
    events = engine.journal.replay()
    assert len(events) == 4
    outcomes = [event["outcome"] for event in events]
    assert outcomes == ["preview", "preview", "noop", "blocked"]

    for event in events:
        assert event["event"] == "order_attempt"
        assert "decision" in event and "guardrails" in event
        assert "order" in event and "result" in event
        assert event["broker"] == "spy"
        assert event["broker_is_live"] is False

    # JSONL: one object per line, no partial records.
    raw_lines = Path(cfg.audit_log_path).read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 4
    for line in raw_lines:
        assert isinstance(json.loads(line), dict)


# --------------------------------------------------------------------------- #
# MCP client: JSON-RPC framing, sessions, SSE
# --------------------------------------------------------------------------- #
def test_mcp_client_parses_sse_responses_and_reuses_session_header():
    class SSETransport:
        def __init__(self):
            self.calls = []

        def __call__(self, url, headers, payload):
            self.calls.append({"headers": dict(headers), "payload": dict(payload)})
            method = payload["method"]
            if method == "initialize":
                body = (
                    ": keep-alive\n\n"
                    "event: message\n"
                    'data: {"jsonrpc":"2.0","id":%d,"result":{"protocolVersion":"2025-06-18",'
                    '"capabilities":{},"serverInfo":{"name":"sse-server"}}}\n\n'
                    % payload["id"]
                )
                return 200, {"Content-Type": "text/event-stream",
                             "Mcp-Session-Id": "sse-session"}, body
            if method == "notifications/initialized":
                return 202, {}, ""
            if method == "tools/list":
                body = (
                    "event: message\n"
                    'data: {"jsonrpc":"2.0","id":%d,"result":{"tools":[{"name":"get_accounts",'
                    '"description":"account balances","inputSchema":{"type":"object",'
                    '"properties":{}}}]}}\n\n' % payload["id"]
                )
                return 200, {"Content-Type": "text/event-stream"}, body
            if method == "tools/call":
                body = (
                    "data: {\"jsonrpc\":\"2.0\",\"id\":%d,\"result\":{\"content\":"
                    "[{\"type\":\"text\",\"text\":\"{\\\"ok\\\":true}\"}]}}\n\n"
                    % payload["id"]
                )
                return 200, {"Content-Type": "text/event-stream"}, body
            return 404, {}, ""

    transport = SSETransport()
    client = MCPClient("https://example.invalid/mcp/trading", "token-abc",
                       transport=transport)
    info = client.initialize()
    assert info["serverInfo"]["name"] == "sse-server"
    assert client.session_id == "sse-session"

    tools = client.list_tools()
    assert [tool["name"] for tool in tools] == ["get_accounts"]
    result = client.call_tool("get_accounts", {})
    assert result["content"][0]["text"] == '{"ok":true}'

    first_headers = transport.calls[0]["headers"]
    assert first_headers["Accept"] == "application/json, text/event-stream"
    assert first_headers["Authorization"] == "Bearer token-abc"
    for call in transport.calls[1:]:
        assert call["headers"]["Mcp-Session-Id"] == "sse-session"
    assert [call["payload"]["method"] for call in transport.calls] == [
        "initialize", "notifications/initialized", "tools/list", "tools/call"]
    assert transport.calls[1]["payload"].get("id") is None  # notification


def test_mcp_client_raises_mcp_error_on_http_and_jsonrpc_errors():
    assert issubclass(MCPError, AgenticTradingError)

    def http_500(url, headers, payload):
        return 500, {}, "server exploded"

    with pytest.raises(MCPError, match="HTTP 500"):
        MCPClient("https://example.invalid/mcp", "token",
                  transport=http_500).initialize()

    def jsonrpc_error(url, headers, payload):
        return 200, {"Content-Type": "application/json"}, json.dumps({
            "jsonrpc": "2.0", "id": payload["id"],
            "error": {"code": -32602, "message": "bad params"},
        })

    with pytest.raises(MCPError, match="bad params"):
        MCPClient("https://example.invalid/mcp", "token",
                  transport=jsonrpc_error).initialize()


def test_mcp_client_rejects_non_json_body():
    def bad_body(url, headers, payload):
        return 200, {"Content-Type": "application/json"}, "<html>not json</html>"

    client = MCPClient("https://example.invalid/mcp", "token", transport=bad_body)
    with pytest.raises(MCPError):
        client.initialize()


# --------------------------------------------------------------------------- #
# RobinhoodMCPBroker against the fake server
# --------------------------------------------------------------------------- #
def test_robinhood_broker_discovers_tools_and_reads_account(monkeypatch):
    broker, transport = rh_broker(monkeypatch, live=True)
    assert broker.is_available() is True

    assert broker.tool_map == {
        "cancel_order": "cancel_equity_order",
        "list_orders": "get_equity_orders",
        "submit_order": "place_equity_order",
        "get_account": "get_accounts",
        "get_quotes": "get_equity_quotes",
    }

    snapshot = broker.get_account()
    assert snapshot.account_id == "MCP-ACCOUNT-1"
    assert snapshot.cash == pytest.approx(2_500.0)
    assert snapshot.equity == pytest.approx(12_500.0)
    assert snapshot.buying_power == pytest.approx(5_000.0)
    assert snapshot.positions["AAPL"].qty == pytest.approx(10.0)
    assert snapshot.positions["AAPL"].avg_price == pytest.approx(150.0)

    quotes = broker.get_quotes(["AAPL"])
    assert quotes["AAPL"].price == pytest.approx(189.5)
    assert quotes["AAPL"].bid == pytest.approx(189.4)
    assert quotes["AAPL"].ask == pytest.approx(189.6)
    assert quotes["AAPL"].source == "robinhood_mcp"

    # JSON-RPC session protocol was followed exactly once.
    methods = [call["payload"]["method"] for call in transport.calls]
    assert methods[:3] == ["initialize", "notifications/initialized", "tools/list"]


def test_robinhood_broker_submits_and_parses_order_result(monkeypatch):
    broker, transport = rh_broker(monkeypatch, live=True)
    result = broker.submit_order(OrderRequest(
        symbol="AAPL", side=OrderSide.BUY.value, order_type=OrderType.MARKET.value,
        qty=2, client_order_id="cid-42"))
    assert result.status is OrderStatus.FILLED
    assert result.broker_order_id == "RH-ORDER-42"
    assert result.client_order_id == "cid-42"
    assert result.filled_qty == pytest.approx(2.0)
    assert result.avg_fill_price == pytest.approx(189.5)

    # The order was translated into the venue's schema, not sent raw.
    assert transport.last_order_args == {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 2.0,
        "type": "market",
        "time_in_force": "gfd",
    }

    orders = broker.list_orders()
    assert len(orders) == 1
    assert orders[0].broker_order_id == "RH-ORDER-1"
    assert orders[0].status is OrderStatus.FILLED
    assert broker.cancel_order("RH-ORDER-1") is True


def test_robinhood_broker_unknown_schema_fails_closed(monkeypatch):
    broker, transport = rh_broker(monkeypatch, live=True,
                                  tool_map={"submit_order": "execute_trade"})
    result = broker.submit_order(OrderRequest(
        symbol="AAPL", side=OrderSide.BUY.value, order_type=OrderType.MARKET.value,
        qty=1, client_order_id="cid-x"))
    assert result.status is OrderStatus.UNKNOWN
    assert "schema" in result.message.lower()
    # No order tool was invoked: the adapter refuses to guess the shape.
    called_tools = [call["payload"].get("params", {}).get("name")
                    for call in transport.calls
                    if call["payload"]["method"] == "tools/call"]
    assert called_tools == []


def test_robinhood_broker_without_token_is_unavailable_and_offline(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    def should_not_be_called(*args, **kwargs):  # pragma: no cover
        raise AssertionError("no network without a token")

    cfg = ExecutionConfig(mcp_access_token_env=TOKEN_ENV)
    broker = RobinhoodMCPBroker(
        cfg, client=MCPClient(cfg.mcp_url, None, transport=should_not_be_called))
    assert broker.is_available() is False
    assert TOKEN_ENV in (broker.last_error or "")
