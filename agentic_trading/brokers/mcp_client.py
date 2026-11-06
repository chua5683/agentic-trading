"""Minimal MCP (Model Context Protocol) client over streamable HTTP.

Robinhood's "Agentic Trading" integration is an official **MCP server**, not a
REST brokerage API, so the only way for an agent to read an account or place an
order is a sequence of JSON-RPC 2.0 tool calls:

    initialize  ->  notifications/initialized  ->  tools/list  ->  tools/call

This module implements just enough of the client side of that specification to
be useful, and nothing more:

  * ``POST`` JSON-RPC 2.0 request bodies with
    ``Accept: application/json, text/event-stream``;
  * capture and reuse of the ``Mcp-Session-Id`` response header;
  * plain JSON responses *and* SSE (``text/event-stream``) responses whose
    ``data:`` lines carry the JSON-RPC result;
  * ``MCPError`` (a subclass of :class:`AgenticTradingError`) for every
    protocol / HTTP / JSON-RPC failure.

The HTTP transport is a plain callable, so tests and offline use never touch a
socket::

    transport(url, headers, payload) -> (status, headers, body)

The default transport is :mod:`requests`.

No MCP SDK dependency is used or wanted; this is stdlib + requests only.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Mapping, Sequence

import requests

from ..errors import AgenticTradingError

__all__ = ["MCPError", "MCPClient", "Transport", "PROTOCOL_VERSION"]

log = logging.getLogger(__name__)

#: MCP protocol revision we speak.  Servers may negotiate a different one during
#: ``initialize``; we then echo the negotiated value back in
#: ``MCP-Protocol-Version`` on subsequent requests.
PROTOCOL_VERSION = "2025-06-18"

#: ``transport(url, headers, payload) -> (status, headers, body)``.
Transport = Callable[..., "tuple[int, Mapping[str, Any], Any]"]

_MAX_TOOL_PAGES = 25
_MAX_ERROR_PREVIEW = 500

_MISSING = object()


class MCPError(AgenticTradingError):
    """Any MCP transport, HTTP, or JSON-RPC protocol failure.

    ``code`` carries the JSON-RPC or HTTP status code when one is available;
    ``data`` carries the JSON-RPC ``error.data`` (or the offending body).
    """

    def __init__(self, message: str, *, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


# --------------------------------------------------------------------------- #
# Default transport
# --------------------------------------------------------------------------- #
def _requests_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    *,
    timeout_s: float = 30.0,
) -> tuple[int, Mapping[str, Any], str]:
    """Default transport: one ``requests.post`` per JSON-RPC message."""
    response = requests.post(
        url,
        headers=dict(headers),
        data=json.dumps(payload),
        timeout=float(timeout_s),
    )
    return response.status_code, dict(response.headers), response.text


# --------------------------------------------------------------------------- #
# Response decoding helpers
# --------------------------------------------------------------------------- #
def _preview(value: Any, limit: int = _MAX_ERROR_PREVIEW) -> str:
    if isinstance(value, (bytes, bytearray)):
        text = bytes(value).decode("utf-8", "replace")
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def _as_text(body: Any) -> str:
    if body is None:
        return ""
    if isinstance(body, (bytes, bytearray)):
        return bytes(body).decode("utf-8", "replace")
    if isinstance(body, str):
        return body
    return json.dumps(body, default=str)


def _looks_like_sse(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("data:") or stripped.startswith("event:"):
            return True
    return False


def _parse_sse(text: str) -> list[Any]:
    """Decode an ``text/event-stream`` body into the JSON documents it carries.

    Per the SSE spec an event is a block of lines terminated by a blank line;
    multiple ``data:`` lines in one event are joined with ``\\n``.  Comments
    (lines starting with ``:``), ``event:``, ``id:`` and ``retry:`` fields are
    ignored because MCP servers only ever put JSON in ``data:`` for responses.
    """
    messages: list[Any] = []
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal data_lines
        if not data_lines:
            return
        raw = "\n".join(data_lines).strip()
        data_lines = []
        if not raw or raw == "[DONE]":
            return
        try:
            messages.append(json.loads(raw))
        except json.JSONDecodeError:
            # Some servers stream partial JSON across multiple events; treat an
            # unparseable event as noise rather than aborting the whole call.
            log.debug("ignoring non-JSON SSE event: %s", _preview(raw, 200))

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            flush()
            continue
        stripped = line.lstrip()
        if stripped.startswith(":"):
            continue
        if ":" in stripped:
            field, _, value = stripped.partition(":")
        else:
            field, value = stripped, ""
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
        # event:/id:/retry: are ignored.
    flush()
    return messages


def _decode_messages(body: Any, content_type: str = "") -> list[Any]:
    """Return a list of JSON-RPC messages decoded from ``body``."""
    if body is None:
        return []
    if isinstance(body, (dict, list)):
        return [body] if isinstance(body, dict) else list(body)
    if isinstance(body, (bytes, bytearray)):
        text = bytes(body).decode("utf-8", "replace")
    else:
        text = str(body)
    if not text.strip():
        return []
    ctype = (content_type or "").lower()
    if "text/event-stream" in ctype or _looks_like_sse(text):
        return _parse_sse(text)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MCPError(
            f"MCP response was not valid JSON ({exc}): {_preview(text)}"
        ) from exc
    return [decoded] if isinstance(decoded, dict) else list(decoded)


def _normalise_headers(headers: Any) -> dict[str, str]:
    if not headers:
        return {}
    items = headers.items() if hasattr(headers, "items") else headers
    out: dict[str, str] = {}
    try:
        for key, value in items:
            out[str(key).lower()] = str(value)
    except (TypeError, ValueError):
        return {}
    return out


def _message_error(message: Mapping[str, Any]) -> MCPError | None:
    error = message.get("error")
    if error is None:
        return None
    if isinstance(error, Mapping):
        text = str(error.get("message") or error)
        code = error.get("code")
        try:
            code_int = int(code) if code is not None else None
        except (TypeError, ValueError):
            code_int = None
        return MCPError(f"MCP JSON-RPC error: {text}", code=code_int, data=error.get("data"))
    return MCPError(f"MCP JSON-RPC error: {error}")


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class MCPClient:
    """A tiny, dependency-free MCP client for streamable HTTP servers.

    Parameters
    ----------
    base_url:
        Full MCP endpoint, e.g. ``https://agent.robinhood.com/mcp/trading``.
    access_token:
        OAuth bearer token.  ``None`` means unauthenticated (useful for tests).
    timeout_s:
        Timeout handed to the default ``requests`` transport.
    transport:
        Optional callable ``transport(url, headers, payload) ->
        (status, headers, body)``.  Inject a fake in tests; the default posts
        with :mod:`requests`.
    """

    def __init__(
        self,
        base_url: str,
        access_token: str | None,
        timeout_s: float = 30.0,
        transport: Transport | None = None,
        *,
        client_name: str = "agentic-trading",
        client_version: str = "0.1.0",
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.access_token = access_token or None
        self.timeout_s = float(timeout_s)
        self.client_name = client_name
        self.client_version = client_version
        self.session_id: str | None = None
        self.server_info: dict[str, Any] = {}
        self.protocol_version = PROTOCOL_VERSION
        self.initialized = False
        self.last_error: str | None = None

        if transport is None:
            def transport(url, headers, payload, _timeout=self.timeout_s):
                return _requests_transport(url, headers, payload, timeout_s=_timeout)

        self._transport: Transport = transport
        self._next_id = 0

    # -- transport / framing ------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": f"{self.client_name}/{self.client_version}",
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
            headers["MCP-Protocol-Version"] = self.protocol_version
        return headers

    def _send(self, method: str, params: Mapping[str, Any] | None, *, notification: bool) -> Any:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notification:
            self._next_id += 1
            payload["id"] = self._next_id
        if params is not None:
            payload["params"] = dict(params)

        headers = self._headers()
        try:
            status, response_headers, body = self._transport(self.base_url, headers, payload)
        except AgenticTradingError:
            raise
        except Exception as exc:  # noqa: BLE001 - translate every transport failure
            self.last_error = str(exc)
            raise MCPError(f"MCP transport failed during {method!r}: {exc}") from exc

        try:
            status_int = int(status)
        except (TypeError, ValueError):
            raise MCPError(f"MCP transport returned a non-integer status for {method!r}: {status!r}")

        response_headers = _normalise_headers(response_headers)
        session_id = response_headers.get("mcp-session-id")
        if session_id:
            self.session_id = session_id

        if status_int >= 400:
            self.last_error = f"HTTP {status_int}"
            raise MCPError(
                f"MCP HTTP {status_int} for {method!r}: {_preview(body)}",
                code=status_int,
                data=body,
            )

        if notification:
            return None

        messages = _decode_messages(body, response_headers.get("content-type", ""))
        return self._select_response(method, payload["id"], messages)

    def _select_response(self, method: str, request_id: int, messages: Sequence[Any]) -> Any:
        # A server may interleave notifications; keep only response-shaped ones.
        responses = [
            m for m in messages
            if isinstance(m, Mapping) and ("result" in m or "error" in m)
        ]
        for message in responses:
            if message.get("id") == request_id:
                error = _message_error(message)
                if error is not None:
                    raise error
                return message.get("result")
        if len(responses) == 1:
            # Tolerate servers that omit/alter the id on a single response.
            error = _message_error(responses[0])
            if error is not None:
                raise error
            return responses[0].get("result")
        if not responses:
            raise MCPError(
                f"MCP response for {method!r} contained no JSON-RPC result "
                f"(got {_preview(messages)})"
            )
        raise MCPError(
            f"MCP response for {method!r} had no JSON-RPC message with id {request_id} "
            f"(got {_preview(messages)})"
        )

    # -- protocol -----------------------------------------------------------
    def initialize(self) -> dict[str, Any]:
        """Perform the MCP handshake and send ``notifications/initialized``."""
        result = self._send(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": self.client_name, "version": self.client_version},
            },
            notification=False,
        )
        self.server_info = dict(result) if isinstance(result, Mapping) else {}
        negotiated = self.server_info.get("protocolVersion")
        if isinstance(negotiated, str) and negotiated:
            self.protocol_version = negotiated
        self._send("notifications/initialized", {}, notification=True)
        self.initialized = True
        return dict(self.server_info)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return every tool the server advertises (following pagination)."""
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(_MAX_TOOL_PAGES):
            params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
            result = self._send("tools/list", params, notification=False)
            if not isinstance(result, Mapping):
                raise MCPError(f"MCP tools/list returned {type(result).__name__}, expected object")
            page = result.get("tools") or []
            if isinstance(page, Sequence) and not isinstance(page, (str, bytes)):
                tools.extend(dict(t) for t in page if isinstance(t, Mapping))
            next_cursor = result.get("nextCursor")
            if not next_cursor:
                return tools
            cursor = str(next_cursor)
        raise MCPError(f"MCP tools/list did not terminate after {_MAX_TOOL_PAGES} pages")

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Invoke one MCP tool.  Returns the raw ``tools/call`` result object."""
        result = self._send(
            "tools/call",
            {"name": str(name), "arguments": dict(arguments or {})},
            notification=False,
        )
        if result is None:
            return {}
        if not isinstance(result, Mapping):
            raise MCPError(f"MCP tools/call returned {type(result).__name__}, expected object")
        payload = dict(result)
        if payload.get("isError"):
            raise MCPError(
                f"MCP tool {name!r} returned an error: {_preview(_tool_text(payload))}",
                data=payload,
            )
        return payload


def _tool_text(result: Mapping[str, Any]) -> Any:
    """Best-effort human-readable text out of an MCP tool result."""
    content = result.get("content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and item.get("text"):
                parts.append(str(item["text"]))
        if parts:
            return "\n".join(parts)
    return result
