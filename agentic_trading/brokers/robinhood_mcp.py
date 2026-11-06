"""Robinhood broker adapter for the official Agentic Trading MCP server.

Robinhood exposes agent trading as an **official MCP server**
(``https://agent.robinhood.com/mcp/trading``, streamable HTTP + OAuth), not as a
REST trading API.  There is no supported Python trading SDK, so this adapter:

1. discovers the server's tools at runtime with ``tools/list``;
2. maps tool names onto our small ``Broker`` vocabulary using case-insensitive
   name/description pattern matching (with an explicit user override);
3. translates :class:`~agentic_trading.contracts.OrderRequest` into the venue's
   argument schema by inspecting that tool's JSON schema;
4. parses the venue's response back into ``AccountSnapshot`` / ``OrderResult``.

Safety posture
--------------
* The constructor takes an explicit ``live: bool = False``.  Unless it is
  ``True``, :meth:`submit_order` returns ``OrderStatus.PREVIEW`` and **never
  calls any order tool**.  :func:`build_broker` only sets ``live=True`` when
  :meth:`ExecutionConfig.live_enabled` passes every gate.
* User's "trade approvals" mode in Robinhood is mirrored by our preview/dry-run
  default: the safest thing an agent can do is propose an order.
* Anything the adapter cannot understand becomes an ``OrderStatus.UNKNOWN``
  result with a diagnostic message -- it never guesses an order shape and it
  never crashes the caller.

Tests inject a fake ``transport`` into :class:`MCPClient`, so nothing here ever
touches the network under test.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from ..config import ExecutionConfig
from ..contracts import (
    AccountSnapshot,
    Broker,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
)
from ..errors import AgenticTradingError
from .mcp_client import MCPClient, MCPError, Transport
from .paper import PaperBroker

__all__ = ["RobinhoodMCPBroker", "build_broker", "TOOL_MAP_ENV"]

log = logging.getLogger(__name__)

#: JSON object mapping a capability to a tool name, e.g.
#: ``{"submit_order": "place_equity_order"}``.  Lets users fix a bad discovery
#: match without changing code.
TOOL_MAP_ENV = "ROBINHOOD_MCP_TOOL_MAP"

CAPABILITIES = ("get_account", "get_quotes", "submit_order", "list_orders", "cancel_order")

# Substring scores over ``name + " " + description``.  Higher wins; a tool is
# assigned to at most one capability and a capability to at most one tool.
_TOOL_PATTERNS: dict[str, tuple[tuple[str, float], ...]] = {
    "cancel_order": (
        ("cancel_order", 8.0), ("cancel", 4.0),
    ),
    "list_orders": (
        ("list_orders", 8.0), ("order_history", 8.0), ("get_orders", 7.0),
        ("list order", 5.0), ("orders", 2.0), ("history", 2.0),
    ),
    "submit_order": (
        ("place_equity_order", 9.0), ("place_option_order", 9.0),
        ("place_crypto_order", 9.0), ("place_market_order", 9.0),
        ("place_limit_order", 9.0), ("place_order", 8.0),
        ("submit_order", 8.0), ("create_order", 7.0), ("execute_order", 6.0),
        ("place", 1.5), ("buy", 1.5), ("sell", 1.5), ("trade", 1.0),
    ),
    "get_account": (
        ("get_account", 8.0), ("account", 3.0), ("buying_power", 4.0),
        ("portfolio", 3.0), ("balance", 3.0), ("positions", 2.0),
    ),
    "get_quotes": (
        ("get_quotes", 8.0), ("quotes", 5.0), ("quote", 4.0),
        ("market_data", 4.0), ("last_trade_price", 4.0), ("price", 2.0),
    ),
}

_CAP_PRIORITY = {"cancel_order": 0, "list_orders": 1, "submit_order": 2,
                 "get_account": 3, "get_quotes": 4}

_READ_PREFIXES = (
    "get", "list", "search", "fetch", "read", "show", "view", "describe",
    "preview", "review", "approve", "validate", "simulate", "check", "cancel",
)

# Field aliases used to translate our OrderRequest into a venue JSON schema.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol": ("symbol", "ticker", "instrument_symbol", "instrument", "asset",
               "asset_symbol", "product_id"),
    "side": ("side", "order_side", "direction", "action"),
    "qty": ("quantity", "qty", "shares", "num_shares", "amount_shares", "size",
            "share_quantity"),
    "notional": ("notional", "amount", "dollar_amount", "total_amount",
                 "dollar_based_amount", "quantity_dollars", "order_value"),
    "type": ("type", "order_type", "order_kind", "order_type_id"),
    "limit_price": ("limit_price", "limit", "limitprice"),
    "stop_price": ("stop_price", "stop", "trigger_price", "stop_trigger_price"),
    "tif": ("time_in_force", "tif", "timeinforce"),
    "client_order_id": ("client_order_id", "idempotency_key", "reference_id",
                        "ref_id", "client_ref"),
}

_ORDER_ID_KEYS = ("order_id", "order_uuid", "broker_order_id", "order_reference_id",
                  "id", "reference_id")
_STATUS_KEYS = ("status", "order_status", "state", "order_state", "status_code")
_SYMBOL_KEYS = ("symbol", "ticker", "instrument_symbol", "asset_symbol", "product_id")
_QTY_KEYS = ("quantity", "qty", "shares", "position_quantity", "num_shares", "units")
_AVG_KEYS = ("average_buy_price", "avg_price", "average_price", "avg_entry_price",
             "cost_basis_price", "average_cost")
_COST_KEYS = ("cost_basis", "total_cost", "book_value")
_PRICE_KEYS = ("price", "last_trade_price", "last_price", "last", "mark_price",
               "mark", "current_price", "close")
_BID_KEYS = ("bid", "bid_price", "best_bid")
_ASK_KEYS = ("ask", "ask_price", "best_ask")
_CASH_KEYS = ("cash", "cash_balance", "cash_available", "available_cash",
              "settled_cash")
_EQUITY_KEYS = ("equity", "total_equity", "portfolio_value", "account_value",
                "net_liquidation", "total_account_value")
_BUYING_POWER_KEYS = ("buying_power", "buyingpower", "available_buying_power")
_ACCOUNT_ID_KEYS = ("account_id", "account_number", "accountid", "account")
_FILLED_QTY_KEYS = ("filled_quantity", "filled_qty", "cumulative_quantity",
                    "executed_quantity", "filled_shares")
_AVG_FILL_KEYS = ("average_fill_price", "avg_fill_price", "average_price",
                  "avg_price", "fill_price")
_NOTIONAL_KEYS = ("notional", "total_notional", "order_value", "amount")
_MESSAGE_KEYS = ("message", "reject_reason", "rejection_reason", "status_detail",
                 "detail", "reason")

_STATUS_MAP: dict[str, OrderStatus] = {
    "queued": OrderStatus.PENDING,
    "pending": OrderStatus.PENDING,
    "new": OrderStatus.PENDING,
    "submitted": OrderStatus.SUBMITTED,
    "open": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "working": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "partial_fill": OrderStatus.PARTIALLY_FILLED,
    "partial": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "executed": OrderStatus.FILLED,
    "completed": OrderStatus.FILLED,
    "complete": OrderStatus.FILLED,
    "cancelled": OrderStatus.CANCELLED,
    "canceled": OrderStatus.CANCELLED,
    "voided": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "failed": OrderStatus.REJECTED,
    "denied": OrderStatus.REJECTED,
    "error": OrderStatus.REJECTED,
    "preview": OrderStatus.PREVIEW,
    "previewed": OrderStatus.PREVIEW,
}

_ORDER_TYPE_TO_VENUE = {
    OrderType.MARKET: "market",
    OrderType.MARKET_DOLLAR: "market",
    OrderType.LIMIT: "limit",
    OrderType.STOP_MARKET: "stop_market",
    OrderType.STOP_LIMIT: "stop_limit",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _norm_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _scalar(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple, set)):
        return None
    return value


def _walk(obj: Any):
    """Yield every mapping nested anywhere inside ``obj`` (pre-order)."""
    if isinstance(obj, Mapping):
        yield obj
        for value in obj.values():
            yield from _walk(value)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            yield from _walk(item)


def _find_first(obj: Any, keys: Sequence[str]) -> Any:
    """First scalar value whose key matches ``keys``, trying keys in order."""
    for key in keys:
        wanted = _norm_key(key)
        for mapping in _walk(obj):
            for raw_key, value in mapping.items():
                if _norm_key(raw_key) != wanted:
                    continue
                scalar = _scalar(value)
                if scalar is not None:
                    return scalar
    return None


def _find_number(obj: Any, keys: Sequence[str]) -> float | None:
    return _to_float(_find_first(obj, keys))


def _find_str(obj: Any, keys: Sequence[str]) -> str | None:
    value = _find_first(obj, keys)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# --------------------------------------------------------------------------- #
# Tool discovery / classification
# --------------------------------------------------------------------------- #
def _tool_matches_cap(capability: str, name: str, text: str) -> bool:
    norm = _norm_key(name)
    if not norm:
        return False
    if capability == "cancel_order":
        return "cancel" in norm and ("order" in norm or "order" in text)
    if capability == "list_orders":
        if any(verb in norm for verb in ("place", "submit", "create", "execute")):
            return False
        has_read = any(verb in norm for verb in ("list", "get", "search", "history", "show"))
        return has_read and "order" in norm
    if capability == "submit_order":
        if norm.startswith(_READ_PREFIXES):
            return False
        if "preview" in norm or "review" in norm or "approve" in norm:
            return False
        has_verb = any(verb in norm for verb in
                       ("place", "submit", "create", "buy", "sell", "execute", "trade"))
        has_object = any(word in norm for word in
                         ("order", "trade", "asset", "equity", "option", "crypto"))
        return has_verb and has_object
    if capability == "get_account":
        if any(verb in norm for verb in ("place", "submit", "create", "cancel", "order")):
            return False
        return any(word in text for word in
                   ("account", "portfolio", "balance", "buying_power", "buying power",
                    "position"))
    if capability == "get_quotes":
        if any(verb in norm for verb in ("place", "submit", "create", "cancel",
                                         "order", "account", "portfolio")):
            return False
        return any(word in text for word in
                   ("quote", "price", "market_data", "market data", "last trade",
                    "last_trade"))
    return False


def _classify_tools(tools: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map ``capability -> venue tool name`` using name/description patterns."""
    candidates: list[tuple[float, int, int, str, str]] = []
    for tool in tools:
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        text = f"{name} {tool.get('description') or ''}".lower()
        for capability, patterns in _TOOL_PATTERNS.items():
            if not _tool_matches_cap(capability, name, text):
                continue
            score = sum(weight for pattern, weight in patterns if pattern in text)
            if score > 0:
                candidates.append((score, -_CAP_PRIORITY[capability], -len(name),
                                   name, capability))
    candidates.sort(reverse=True)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for _score, _priority, _length, name, capability in candidates:
        if capability in mapping or name in used:
            continue
        mapping[capability] = name
        used.add(name)
    return mapping


def _env_tool_map() -> dict[str, str]:
    raw = (os.environ.get(TOOL_MAP_ENV) or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("%s is not valid JSON (%s); ignoring tool map override", TOOL_MAP_ENV, exc)
        return {}
    if not isinstance(parsed, Mapping):
        log.warning("%s must be a JSON object; ignoring override", TOOL_MAP_ENV)
        return {}
    return {str(key): str(value) for key, value in parsed.items() if value}


# --------------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------------- #
def _schema_properties(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        return {}
    props = schema.get("properties")
    if isinstance(props, Mapping):
        return {str(key): (value if isinstance(value, Mapping) else {})
                for key, value in props.items()}
    return {}


def _schema_required(schema: Any) -> list[str]:
    if not isinstance(schema, Mapping):
        return []
    required = schema.get("required")
    if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
        return [str(item) for item in required]
    return []


def _match_property(properties: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    wanted = {_norm_key(alias) for alias in aliases}
    for key in properties:
        if _norm_key(key) in wanted:
            return str(key)
    return None


def _coerce_enum(value: Any, enum: Any) -> Any:
    if not isinstance(enum, Sequence) or isinstance(enum, (str, bytes)) or not enum:
        return value
    text = str(value)
    wanted = _norm_key(text)
    for option in enum:
        if _norm_key(option) == wanted:
            return option
    return value


def _venue_order_type(order_type: OrderType) -> str:
    return _ORDER_TYPE_TO_VENUE.get(order_type, "market")


def _venue_order_arguments(request: OrderRequest, schema: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Translate an ``OrderRequest`` into ``(arguments, problems)``.

    ``arguments is None`` means "do not send": the schema could not be mapped
    safely.  We never guess field names for required fields.
    """
    properties = _schema_properties(schema)
    if not properties:
        return None, ["tool input schema has no properties; refusing to guess an order shape"]

    arguments: dict[str, Any] = {}
    problems: list[str] = []

    def put(field: str, value: Any) -> bool:
        key = _match_property(properties, _FIELD_ALIASES[field])
        if key is None:
            return False
        if value is None:
            return True
        spec = properties.get(key) or {}
        enum = spec.get("enum") if isinstance(spec, Mapping) else None
        arguments[key] = _coerce_enum(value, enum) if enum else value
        return True

    if not put("symbol", request.symbol):
        problems.append("no symbol field in tool schema")
    if not put("side", request.side.value):
        problems.append("no side field in tool schema")
    put("type", _venue_order_type(request.order_type))  # often implicit in the tool name

    if request.order_type is OrderType.MARKET_DOLLAR:
        if request.notional is None:
            problems.append("dollar-based market order requires notional")
        elif not put("notional", round(float(request.notional), 2)):
            problems.append("no dollar amount field in tool schema")
    else:
        if request.qty is None:
            problems.append("share-based order requires qty")
        elif not put("qty", abs(float(request.qty))):
            problems.append("no quantity field in tool schema")

    if request.limit_price is not None and not put("limit_price", float(request.limit_price)):
        problems.append("no limit-price field in tool schema")
    if request.stop_price is not None and not put("stop_price", float(request.stop_price)):
        problems.append("no stop-price field in tool schema")
    put("tif", request.time_in_force)
    put("client_order_id", request.client_order_id or None)

    present = {_norm_key(key) for key in arguments}
    for required in _schema_required(schema):
        if _norm_key(required) not in present:
            problems.append(f"required tool field {required!r} cannot be mapped")
    if problems:
        return None, problems
    return arguments, []


def _quote_arguments(symbol: str, schema: Any) -> dict[str, Any]:
    properties = _schema_properties(schema)
    plural = _match_property(properties, ("symbols", "tickers", "symbol_list"))
    if plural:
        return {plural: [symbol]}
    single = _match_property(properties, ("symbol", "ticker", "instrument_symbol"))
    if single:
        return {single: symbol}
    return {"symbol": symbol}


def _account_arguments(schema: Any, account_id: str | None) -> dict[str, Any]:
    properties = _schema_properties(schema)
    key = _match_property(properties, ("account_id", "account_number", "account"))
    if key and account_id:
        return {key: account_id}
    return {}


def _list_arguments(limit: int, schema: Any) -> dict[str, Any]:
    properties = _schema_properties(schema)
    key = _match_property(properties, ("limit", "per_page", "page_size", "count",
                                       "max_results", "first"))
    return {key: int(limit)} if key else {}


def _cancel_arguments(broker_order_id: str, schema: Any) -> dict[str, Any] | None:
    properties = _schema_properties(schema)
    key = _match_property(properties, ("order_id", "id", "order_uuid",
                                       "broker_order_id", "order_reference_id"))
    if key is None:
        return None
    return {key: broker_order_id}


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
def _tool_payload(result: Any) -> Any:
    """Unwrap an MCP ``tools/call`` result into the venue's JSON payload."""
    if not isinstance(result, Mapping):
        return result
    structured = result.get("structuredContent")
    if structured is not None:
        return structured
    content = result.get("content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        texts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
        if texts:
            joined = "\n".join(texts).strip()
            try:
                return json.loads(joined)
            except json.JSONDecodeError:
                return {"text": joined}
    return result


def _parse_status(payload: Any) -> OrderStatus | None:
    value = _find_first(payload, _STATUS_KEYS)
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _STATUS_MAP.get(text)


def _looks_like_order(mapping: Mapping[str, Any]) -> bool:
    keys = {_norm_key(key) for key in mapping}
    has_status = bool(keys & {_norm_key(k) for k in _STATUS_KEYS})
    has_id = bool(keys & {_norm_key(k) for k in _ORDER_ID_KEYS})
    has_body = bool(keys & {_norm_key(k) for k in
                            ("symbol", "ticker", "side", "quantity", "qty", "shares")})
    return has_status and (has_id or has_body)


def _first_order_object(payload: Any) -> Mapping[str, Any]:
    for mapping in _walk(payload):
        if _looks_like_order(mapping):
            return mapping
    return payload if isinstance(payload, Mapping) else {}


def _find_broker_order_id(payload: Any, exclude: Sequence[str] = ()) -> str | None:
    excluded = {str(value) for value in exclude if value}
    for key in _ORDER_ID_KEYS:
        wanted = _norm_key(key)
        for mapping in _walk(payload):
            for raw_key, value in mapping.items():
                if _norm_key(raw_key) != wanted:
                    continue
                scalar = _scalar(value)
                if scalar is None:
                    continue
                text = str(scalar).strip()
                if text and text not in excluded:
                    return text
    return None


def _parse_order_result(
    payload: Any,
    *,
    client_order_id: str,
    symbol: str = "",
    side: OrderSide = OrderSide.BUY,
    qty: float | None = None,
    arguments: Mapping[str, Any] | None = None,
) -> OrderResult:
    order = _first_order_object(payload)
    status = _parse_status(order)
    if status is None:
        status = _parse_status(payload)
    raw: dict[str, Any] = {"arguments": dict(arguments or {})}
    raw["payload"] = payload if isinstance(payload, (Mapping, list)) else {"text": str(payload)}
    if status is None:
        return OrderResult(
            client_order_id=client_order_id,
            status=OrderStatus.UNKNOWN,
            symbol=symbol,
            side=side,
            qty=qty,
            message="unrecognised order response from Robinhood MCP (no status field); "
                    "inspect raw payload before trusting this order",
            raw=raw,
        )

    broker_order_id = _find_broker_order_id(order, exclude=(client_order_id,)
                                             ) or _find_broker_order_id(payload,
                                                                        exclude=(client_order_id,))
    response_symbol = _find_str(order, _SYMBOL_KEYS) or symbol
    filled_qty = _find_number(order, _FILLED_QTY_KEYS) or 0.0
    avg_fill = _find_number(order, _AVG_FILL_KEYS)
    notional = _find_number(order, _NOTIONAL_KEYS)
    message = _find_str(order, _MESSAGE_KEYS) or ""
    return OrderResult(
        client_order_id=client_order_id,
        status=status,
        broker_order_id=broker_order_id,
        symbol=response_symbol,
        side=side,
        qty=qty,
        filled_qty=float(filled_qty),
        avg_fill_price=avg_fill,
        notional=notional,
        submitted_at=_utcnow(),
        message=message,
        raw=raw,
    )


def _extract_orders(payload: Any) -> list[Mapping[str, Any]]:
    """Pull the list of order-shaped mappings out of a tool response."""
    if isinstance(payload, list):
        orders = [item for item in payload if isinstance(item, Mapping)]
        if orders:
            return orders
    found: list[Mapping[str, Any]] = []
    for mapping in _walk(payload):
        for value in mapping.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping) and _looks_like_order(item):
                        found.append(item)
    if found:
        return found
    if isinstance(payload, Mapping) and _looks_like_order(payload):
        return [payload]
    return []


def _parse_account(payload: Any) -> AccountSnapshot:
    positions: dict[str, Position] = {}
    for mapping in _walk(payload):
        symbol = _find_str(mapping, _SYMBOL_KEYS)
        if not symbol:
            continue
        qty = _find_number(mapping, _QTY_KEYS)
        if qty is None:
            continue
        avg = _find_number(mapping, _AVG_KEYS)
        if avg is None:
            cost = _find_number(mapping, _COST_KEYS)
            if cost is not None and abs(qty) > 1e-12:
                avg = cost / qty
        if symbol not in positions:
            positions[symbol] = Position(symbol=symbol, qty=float(qty),
                                         avg_price=float(avg or 0.0))

    cash = _find_number(payload, _CASH_KEYS)
    equity = _find_number(payload, _EQUITY_KEYS)
    buying_power = _find_number(payload, _BUYING_POWER_KEYS)
    account_id = _find_str(payload, _ACCOUNT_ID_KEYS) or "robinhood-mcp"

    if equity is None and cash is not None:
        equity = cash + sum(position.qty * position.avg_price
                            for position in positions.values())
    if cash is None and equity is not None:
        cash = equity - sum(position.qty * position.avg_price
                            for position in positions.values())
    if cash is None and equity is None:
        raise MCPError(
            "could not parse the Robinhood MCP account payload: no cash or equity "
            "field was present; refusing to invent balances"
        )
    return AccountSnapshot(
        account_id=account_id,
        cash=float(cash if cash is not None else 0.0),
        equity=float(equity if equity is not None else 0.0),
        buying_power=float(buying_power if buying_power is not None
                           else (cash if cash is not None else 0.0)),
        positions=positions,
        asof=_utcnow(),
        raw=payload if isinstance(payload, Mapping) else {"payload": payload},
    )


def _parse_quote(payload: Any, symbol: str) -> Quote | None:
    price = _find_number(payload, _PRICE_KEYS)
    bid = _find_number(payload, _BID_KEYS)
    ask = _find_number(payload, _ASK_KEYS)
    if price is None and bid is not None and ask is not None:
        price = (bid + ask) / 2.0
    if price is None or price <= 0:
        return None
    return Quote(symbol=symbol, price=float(price), asof=_utcnow(), bid=bid, ask=ask,
                 source="robinhood_mcp")


# --------------------------------------------------------------------------- #
# Broker
# --------------------------------------------------------------------------- #
class RobinhoodMCPBroker:
    """``Broker`` adapter over Robinhood's official Agentic Trading MCP server.

    Parameters
    ----------
    cfg:
        ``ExecutionConfig``; supplies the MCP URL, token env var name, timeout.
    client:
        Optional pre-built :class:`MCPClient` (used by tests to inject a fake
        transport).  Built from ``cfg`` when omitted.
    live:
        **Explicit live-order permission.**  ``False`` (the default) means
        :meth:`submit_order` always returns ``OrderStatus.PREVIEW`` and never
        calls an order tool.  :func:`build_broker` derives this from
        :meth:`ExecutionConfig.live_enabled`.
    tool_map:
        Optional ``capability -> tool name`` override.  A ``{"submit_order":
        "place_equity_order"}``-style mapping read from the
        ``ROBINHOOD_MCP_TOOL_MAP`` JSON env var is merged in too (explicit
        argument wins).
    account_id:
        Optional account selector forwarded to the account tool when the server
        exposes one (Robinhood orders only ever go to the dedicated Agentic
        account, but read calls sometimes need an identifier).
    """

    name = "robinhood"
    is_live = True  # the *venue* is real money; `self.live` gates placing orders

    def __init__(
        self,
        cfg: ExecutionConfig,
        *,
        client: MCPClient | None = None,
        live: bool = False,
        tool_map: Mapping[str, str] | None = None,
        transport: Transport | None = None,
        account_id: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.live = bool(live)
        self.account_id = account_id
        self._token = cfg.mcp_access_token
        self.client = client or MCPClient(
            cfg.mcp_url,
            self._token,
            timeout_s=cfg.mcp_timeout_s,
            transport=transport,
            client_name="agentic-trading-robinhood",
        )
        overrides = _env_tool_map()
        if tool_map:
            overrides.update({str(key): str(value) for key, value in tool_map.items()})
        self._overrides = overrides
        # Until discovery runs this exposes only explicit overrides.
        self.tool_map: dict[str, str] = dict(overrides)
        self._schemas: dict[str, dict[str, Any]] = {}
        self._discovered = False
        self.last_error: str | None = None

    # -- discovery ----------------------------------------------------------
    def _ensure_discovered(self) -> dict[str, str]:
        if self._discovered:
            return self.tool_map
        if not self._token:
            raise MCPError(
                f"missing Robinhood MCP access token in environment variable "
                f"{self.cfg.mcp_access_token_env!r}; run the OAuth flow in docs/ROBINHOOD.md"
            )
        if not self.client.initialized:
            self.client.initialize()
        tools = self.client.list_tools()
        discovered = _classify_tools(tools)
        discovered.update(self._overrides)
        self.tool_map = discovered
        self._schemas = {
            str(tool.get("name")): (tool.get("inputSchema")
                                    or tool.get("input_schema") or {})
            for tool in tools if tool.get("name")
        }
        self._discovered = True
        if not self.tool_map:
            self.last_error = "no broker tools matched the expected patterns; set " + TOOL_MAP_ENV
        return self.tool_map

    def is_available(self) -> bool:
        if not self._token:
            self.last_error = (f"missing Robinhood MCP access token in "
                               f"{self.cfg.mcp_access_token_env!r}")
            return False
        try:
            self._ensure_discovered()
        except AgenticTradingError as exc:
            self.last_error = str(exc)
            return False
        return True

    def _schema_for(self, capability: str) -> Any:
        tool = self.tool_map.get(capability)
        return self._schemas.get(tool or "", {})

    def _client_order_id(self, request: OrderRequest) -> str:
        return request.client_order_id or f"rh-{_utcnow().strftime('%Y%m%dT%H%M%S%f')}"

    # -- protocol -----------------------------------------------------------
    def get_account(self) -> AccountSnapshot:
        self._ensure_discovered()
        tool = self.tool_map.get("get_account")
        if not tool:
            raise MCPError(
                "Robinhood MCP server exposes no account/portfolio tool; "
                f"tool_map={self.tool_map!r}. Set {TOOL_MAP_ENV} to override."
            )
        schema = self._schemas.get(tool, {})
        raw = self.client.call_tool(tool, _account_arguments(schema, self.account_id))
        return _parse_account(_tool_payload(raw))

    def get_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        if not symbols:
            return {}
        if not self.tool_map.get("get_quotes"):
            try:
                self._ensure_discovered()
            except AgenticTradingError as exc:
                self.last_error = str(exc)
                return {}
        tool = self.tool_map.get("get_quotes")
        if not tool:
            self.last_error = "no quote tool discovered on the Robinhood MCP server"
            return {}
        schema = self._schemas.get(tool, {})
        quotes: dict[str, Quote] = {}
        for raw_symbol in symbols:
            symbol = str(raw_symbol)
            try:
                result = self.client.call_tool(tool, _quote_arguments(symbol, schema))
            except AgenticTradingError as exc:
                self.last_error = str(exc)
                continue
            quote = _parse_quote(_tool_payload(result), symbol)
            if quote is not None:
                quotes[symbol] = quote
        return quotes

    def submit_order(self, request: OrderRequest) -> OrderResult:
        client_order_id = self._client_order_id(request)

        # ---- HARD SAFETY GATE: no live flag, no live order tool call. ----
        if not self.live:
            return OrderResult(
                client_order_id=client_order_id,
                status=OrderStatus.PREVIEW,
                symbol=str(request.symbol).upper(),
                side=request.side,
                qty=request.qty,
                notional=request.notional,
                message=(
                    "RobinhoodMCPBroker constructed with live=False: order previewed "
                    "only. Live orders require cfg.live_enabled() (enabled=true, "
                    "mode=live, dry_run=false, no kill switch, confirmation env var) "
                    "and an explicit live=True broker."
                ),
                raw={"live": False, "reason": "broker_live_flag_disabled"},
            )

        try:
            self._ensure_discovered()
        except AgenticTradingError as exc:
            return self._unknown(client_order_id, request,
                                 f"MCP discovery failed: {exc}")
        tool = self.tool_map.get("submit_order")
        if not tool:
            return self._unknown(
                client_order_id, request,
                "no order-placement tool discovered; refusing to send. "
                f"tool_map={self.tool_map!r}. Set {TOOL_MAP_ENV} to override.")
        schema = self._schemas.get(tool, {})
        arguments, problems = _venue_order_arguments(request, schema)
        if arguments is None:
            return self._unknown(
                client_order_id, request,
                "cannot translate order into the venue's tool schema (" +
                "; ".join(problems) + "); order NOT sent")
        try:
            raw = self.client.call_tool(tool, arguments)
        except AgenticTradingError as exc:
            return self._unknown(
                client_order_id, request,
                f"MCP order call failed ({exc}); order state is unknown -- reconcile "
                "with the venue before retrying")
        try:
            return _parse_order_result(
                _tool_payload(raw),
                client_order_id=client_order_id,
                symbol=str(request.symbol).upper(),
                side=request.side,
                qty=request.qty,
                arguments=arguments,
            )
        except Exception as exc:  # noqa: BLE001 - adapter must never crash the caller
            return self._unknown(client_order_id, request,
                                 f"could not parse the venue's order response: {exc}")

    def list_orders(self, limit: int = 50) -> list[OrderResult]:
        try:
            self._ensure_discovered()
        except AgenticTradingError as exc:
            self.last_error = str(exc)
            return []
        tool = self.tool_map.get("list_orders")
        if not tool:
            self.last_error = "no order-history tool discovered"
            return []
        schema = self._schemas.get(tool, {})
        try:
            raw = self.client.call_tool(tool, _list_arguments(int(limit), schema))
        except AgenticTradingError as exc:
            self.last_error = str(exc)
            return []
        payload = _tool_payload(raw)
        results: list[OrderResult] = []
        for item in _extract_orders(payload)[:max(0, int(limit))]:
            results.append(_parse_order_result(
                item,
                client_order_id=_find_str(item, ("client_order_id",)) or "",
            ))
        return results

    def cancel_order(self, broker_order_id: str) -> bool:
        try:
            self._ensure_discovered()
        except AgenticTradingError as exc:
            self.last_error = str(exc)
            return False
        tool = self.tool_map.get("cancel_order")
        if not tool:
            self.last_error = "no cancel-order tool discovered"
            return False
        arguments = _cancel_arguments(str(broker_order_id), self._schemas.get(tool, {}))
        if arguments is None:
            self.last_error = "cancel tool schema has no order-id field"
            return False
        try:
            raw = self.client.call_tool(tool, arguments)
        except AgenticTradingError as exc:
            self.last_error = str(exc)
            return False
        payload = _tool_payload(raw)
        status = _parse_status(payload)
        if status is OrderStatus.REJECTED:
            self.last_error = _find_str(payload, _MESSAGE_KEYS) or "venue rejected the cancel"
            return False
        return True

    # -- helpers ------------------------------------------------------------
    def _unknown(self, client_order_id: str, request: OrderRequest, message: str) -> OrderResult:
        self.last_error = message
        return OrderResult(
            client_order_id=client_order_id,
            status=OrderStatus.UNKNOWN,
            symbol=str(request.symbol).upper(),
            side=request.side,
            qty=request.qty,
            notional=request.notional,
            message=message,
            raw={"live": self.live, "tool_map": dict(self.tool_map)},
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_broker(cfg: ExecutionConfig, data_cfg: Any = None) -> Broker:
    """Build the broker selected by ``cfg``, failing *closed*.

    * ``broker: robinhood`` -> :class:`RobinhoodMCPBroker`.  Its live-order flag
      is ``cfg.live_enabled()``, so an agent can never place a live order unless
      every gate in :class:`ExecutionConfig` passes.
    * ``broker: paper`` -> :class:`PaperBroker` (deterministic, no network).
    * anything unknown -> :class:`PaperBroker`, never a live venue.
    """
    del data_cfg  # accepted for symmetry with the rest of the config plumbing
    name = (cfg.broker or "").strip().lower()
    if name in ("robinhood", "rh", "mcp"):
        return RobinhoodMCPBroker(cfg, live=cfg.live_enabled(), tool_map=_env_tool_map())
    if name in ("paper", "sim", "simulated", ""):
        return PaperBroker()
    log.warning("unknown execution.broker %r; falling back to PaperBroker for safety",
                cfg.broker)
    return PaperBroker()
