"""Brokerage adapters and the guarded execution engine.

``agentic_trading.brokers`` is the only package in the project allowed to turn a
:class:`~agentic_trading.contracts.PortfolioDecision` into a real order.  The
public surface is intentionally small:

* :class:`PaperBroker` -- deterministic local simulator (default, no network);
* :class:`RobinhoodMCPBroker` -- adapter for Robinhood's official Agentic
  Trading MCP server; preview-only unless explicitly constructed ``live=True``;
* :class:`ExecutionEngine` -- sizes orders from target weights and enforces
  every blast-radius limit in deterministic code before calling a broker;
* :func:`build_broker` -- config -> broker factory that fails closed;
* :class:`MCPClient` -- minimal JSON-RPC/SSE MCP client with an injectable
  transport (no MCP SDK, no network in tests);
* :class:`AuditJournal` -- append-only JSONL journal of every order attempt.

Safety model: with ``ExecutionConfig()`` defaults
(``enabled=false, broker=paper, mode=paper, dry_run=true``) no code path can
reach ``Broker.submit_order`` on a live venue, and every attempt -- blocked,
previewed, sent or failed -- is journalled.
"""

from __future__ import annotations

from ..contracts import Broker
from .audit import AuditJournal
from .execution import MIN_TRADE_NOTIONAL, ExecutionEngine
from .mcp_client import MCPClient, MCPError, PROTOCOL_VERSION
from .paper import PaperBroker
from .robinhood_mcp import TOOL_MAP_ENV, RobinhoodMCPBroker, build_broker

__all__ = [
    "AuditJournal",
    "Broker",
    "ExecutionEngine",
    "MCPClient",
    "MCPError",
    "MIN_TRADE_NOTIONAL",
    "PROTOCOL_VERSION",
    "PaperBroker",
    "RobinhoodMCPBroker",
    "TOOL_MAP_ENV",
    "build_broker",
]
