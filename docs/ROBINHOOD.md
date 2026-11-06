# Robinhood Agentic Trading over MCP — setup & safety guide

> **Read this first.** This integration can place real orders with real money.
> The code in this repository defaults to **paper + dry-run**, and every live
> order has to pass several independent, deterministic gates before it is even
> allowed to reach the venue. Do not disable those gates until you have watched
> a full paper/dry-run cycle, read the audit log, and understand that a bug, a
> stale quote, or an LLM hallucination can lose money. You are responsible for
> every order this code sends.

---

## 1. What the integration actually is

Robinhood launched official **"Agentic Trading"** in May 2026. The agent-facing
interface is an **MCP server**, not a REST trading API:

| Item | Value |
| --- | --- |
| Endpoint | `https://agent.robinhood.com/mcp/trading` |
| Transport | MCP **streamable HTTP** (JSON-RPC 2.0 over HTTP `POST`; SSE responses supported) |
| Auth | **OAuth bearer token** (Robinhood-issued; not a username/password) |
| Protocol | `initialize` → `notifications/initialized` → `tools/list` → `tools/call` |
| Account scope | Orders are only permitted inside the dedicated **"Agentic"/MCP account** |
| Approval mode | Robinhood supports **trade approvals**, where an agent may only *propose* orders |

There is **no official Python trading SDK**. This repository implements the
MCP client itself (`agentic_trading/brokers/mcp_client.py`, stdlib +
`requests`) and refuses to use unofficial scrapers such as `robin_stocks`.
Tool names are **discovered at runtime**, so the adapter maps them by pattern
and lets you override the mapping without code changes.

Docs this guide is based on (verified 2026-11-05):

- <https://robinhood.com/us/en/support/articles/onboarding-an-external-agent/>
- <https://robinhood.com/us/en/support/articles/trading-with-your-agent/>

---

## 2. Paper mode (the default) — no Robinhood account needed

Execution defaults are inert:

```yaml
execution:
  enabled: false       # master switch
  broker: "paper"      # local simulator
  mode: "paper"
  dry_run: true        # validate + journal, never send
```

```python
from agentic_trading.brokers import PaperBroker, ExecutionEngine
from agentic_trading.config import ExecutionConfig

broker = PaperBroker(start_cash=10_000.0, price_source={"AAPL": 190.0})
engine = ExecutionEngine(broker, ExecutionConfig())
result = engine.execute(decision)  # -> OrderStatus.PREVIEW, broker.orders == []
```

Paper mode is deterministic: fills happen at the quote price you supply (dict,
callable, or the last observed price / position average). Set a `state_path` to
persist cash/positions/orders to JSON so a CLI session resumes. Daily counters
are **per process** — restarting the process resets `max_daily_orders` /
`max_daily_notional`, so keep the audit journal as the durable record.

---

## 3. Gating model (how a live order is allowed)

Order of checks, all in deterministic Python **before** `broker.submit_order`:

1. **`execution.enabled`** must be `true`.
2. **`execution.mode`** must be `"live"`.
3. **`execution.dry_run`** must be `false`.
4. **Kill switch file** (`execution.kill_switch_file`, default `STOP_TRADING`)
   must not exist.
5. **Confirmation env var** (`execution.live_confirm_env`) must equal
   `execution.live_confirm_phrase` exactly.
6. **`ExecutionConfig.live_enabled()`** must return `True`; `build_broker()`
   passes that as `live=` to `RobinhoodMCPBroker`. A broker constructed with
   `live=False` returns `OrderStatus.PREVIEW` and **never calls an order tool**.
7. **Blast-radius caps** enforced by `ExecutionEngine` regardless of what any
   model says: symbol allowlist, no shorting, minimum trade size,
   `max_order_notional` (clamped), `max_daily_orders` (blocked),
   `max_daily_notional` (blocked).

If the broker is the live Robinhood venue but any gate 1–6 fails, the engine
returns a `REJECTED` result and logs the block. This is defence in depth: the
engine gate and the broker's own `live=True` requirement must **both** pass.

Once every gate passes the order is still a normal market/limit order inside
the dedicated Agentic account, subject to Robinhood's own limits/approvals.

---

## 4. Connecting the official MCP server

### 4.1 Prerequisites

- A Robinhood account eligible for Agentic Trading, with the dedicated
  **Agentic/MCP account** funded (orders outside it are rejected by Robinhood).
- An OAuth access token issued through Robinhood's agent onboarding flow
  (see the onboarding article above). Prefer the most narrowly scoped token
  available.
- Start with **trade approvals / propose-only** enabled on Robinhood's side if
  the product offers it: it mirrors this project's preview-first design.

### 4.2 Export the token

The adapter reads the token from the environment variable named by
`execution.mcp_access_token_env` (default `ROBINHOOD_MCP_ACCESS_TOKEN`):

```bash
export ROBINHOOD_MCP_ACCESS_TOKEN="<your OAuth bearer token>"
```

Never commit the token, never put it in `configs/*.yaml`, never log it. The
audit journal (`reports/orders.jsonl`) deliberately contains no credentials;
it is written `0600` on creation, but check your directory permissions anyway.

### 4.3 Verify discovery without trading

```python
from agentic_trading.brokers import MCPClient, RobinhoodMCPBroker
from agentic_trading.config import ExecutionConfig

cfg = ExecutionConfig(broker="robinhood")   # live defaults remain inert
broker = RobinhoodMCPBroker(cfg)            # live=False by default
print("available:", broker.is_available())  # token present + tools listable
print("tool map :", broker.tool_map)
print("account  :", broker.get_account())   # read-only
```

Expected capability mapping (discovered names can differ):

| Capability | Pattern examples |
| --- | --- |
| `get_account` | `get_account`, `get_accounts`, `portfolio`, `balance`, `positions` |
| `get_quotes` | `get_equity_quotes`, `quotes`, `market_data`, `price` |
| `submit_order` | `place_equity_order`, `place_order`, `submit_order`, `create_order` |
| `list_orders` | `get_equity_orders`, `list_orders`, `order_history` |
| `cancel_order` | `cancel_equity_order`, `cancel_order` |

Read-only tools (`get_/list_/search_...`) are never mapped to
`submit_order`, and preview/review/approve tools are ignored for placement.

### 4.4 Fix a bad mapping without code changes

```bash
export ROBINHOOD_MCP_TOOL_MAP='{"submit_order": "place_equity_order", "get_account": "get_accounts"}'
```

The JSON object wins over discovered/pattern-mapped names and is also accepted
as the `tool_map=` constructor argument. Use it if Robinhood renames a tool or
returns an unhelpful schema.

### 4.5 Order translation and failure behaviour

`submit_order` translates `contracts.OrderRequest` into the venue's tool
`inputSchema` by matching field aliases (`quantity`/`qty`/`shares`,
`amount`/`notional`, `limit_price`, `stop_price`, `time_in_force`, …), and
checks required fields. If the schema cannot be mapped safely, the adapter
returns `OrderStatus.UNKNOWN` with a message and **does not call the order
tool** — it never guesses an order shape. Protocol/HTTP failures also return
`UNKNOWN` (not `REJECTED`) because the venue may have received the call:
reconcile with Robinhood before retrying. Every MCP problem raises/returns
`MCPError`/an explanatory result instead of crashing the caller.

`build_broker(ExecutionConfig(broker="robinhood"))` fails closed: an unknown
`execution.broker` value falls back to `PaperBroker`.

---

## 5. Going live safely — exact steps

Do these in order and keep each one recorded in your change log.

1. **Run paper + dry-run for several days.** Use `broker: paper`,
   `dry_run: true`. Read `reports/orders.jsonl`; every attempt is one JSON line
   with the decision, guardrail verdict, order and result.
2. **Make the limits absurdly small** in `configs/local.yaml` while testing:
   `max_order_notional: 25`, `max_daily_orders: 2`, `max_daily_notional: 50`,
   and a non-empty `symbol_allowlist` (e.g. `["AAPL"]`). Never rely on an empty
   allowlist — empty means "fall back to `data.universe`", and if both are empty
   the engine treats the allowlist as unconfigured.
3. **Connect the MCP with `live=False`** and verify `is_available()`,
   `tool_map`, `get_account()` and `get_quotes()` return sane data for the
   Agentic account. Do not place orders yet.
4. **Flip the config deliberately:**
   ```yaml
   execution:
     enabled: true
     broker: "robinhood"
     mode: "live"
     dry_run: false
     symbol_allowlist: ["AAPL"]
     max_order_notional: 25
     max_daily_orders: 2
     max_daily_notional: 50
   ```
5. **Set the confirmation phrase for this process only:**
   ```bash
   export AGENTIC_TRADING_LIVE="yes-i-understand-the-risk"
   ```
   Run `python -c "from agentic_trading.config import ExecutionConfig as E; print(E().why_not_live())"`
   in the same environment; it must print `[]` before you start the loop.
   (`ExecutionConfig()` there is illustrative — check the config object you
   actually pass to `ExecutionEngine`.)
6. **Make sure the kill switch is easy to hit.** `touch STOP_TRADING` in the
   repo root (or the configured path) blocks all new orders immediately, even
   mid-session; the file is checked before every order. Deleting it re-arms the
   engine, so protect it with normal filesystem permissions.
7. **Watch the first live session.** Confirm the order appears in the Agentic
   account, that `reports/orders.jsonl` contains the matching entry, and that
   Robinhood's trade-approval notification (if enabled) arrives.
8. **Widen limits only after a clean week** of paper + tiny live orders. Keep
   the kill switch file one command away at all times.

To go back to safe mode instantly: `touch STOP_TRADING`, stop the process, and
set `enabled: false` or `dry_run: true`.

---

## 6. Safety properties this code guarantees (and tests prove)

`tests/test_broker.py` — run with:

```bash
.venv/bin/python -m pytest tests/test_broker.py -q
```

| # | Property |
| --- | --- |
| 1 | With `ExecutionConfig` defaults (`enabled=false, paper, dry_run=true`) no order can ever reach `broker.submit_order`. |
| 2 | `ExecutionEngine` enforces `max_order_notional`, `max_daily_orders`, `max_daily_notional`, symbol allowlist, kill-switch file, no-short and minimum trade size in deterministic code before calling the broker. |
| 3 | `dry_run=True` returns `OrderStatus.PREVIEW` and never calls `broker.submit_order` (spy broker asserts zero submissions). |
| 4 | Every attempt — blocked, previewed, sent or failed — is appended to the JSONL audit journal and replayable. |
| 5 | `RobinhoodMCPBroker` cannot place a live order unless constructed with `live=True`; otherwise `submit_order` returns `PREVIEW` without touching the transport. |
| 6 | Tests never touch the network: a fake `transport(url, headers, payload) -> (status, headers, body)` is injected into `MCPClient` / `RobinhoodMCPBroker`. |

Daily counters count **every attempt that passes the pre-trade guardrails**,
including dry-run previews, so a runaway loop hits the cap in preview exactly
as it would live. They reset at UTC date rollover and when a new engine is
constructed; the journal is the durable record.

---

## 7. Known limitations

- Tool schemas vary; the adapter handles the common field aliases but returns
  `UNKNOWN` rather than guessing. Use `ROBINHOOD_MCP_TOOL_MAP` plus a
  schema-aware fork if Robinhood changes the shapes.
- Options and crypto tool schemas are not specifically modelled; the adapter
  passes only symbol/side/qty/type/limit/stop/tif. Extend deliberately.
- Market orders sized from a quote are not price-protected; `max_order_notional`
  is a notional cap, not a guaranteed fill price.
- The MCP client is synchronous and non-streaming (one JSON-RPC response per
  call). Long-running sessions should be supervised by the CLI/loop.
- This repository's live path is **not** investment advice and has not been
  audited by a third party. Treat it as a research tool that can trade.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `is_available()` false, "missing … access token" | env var not exported in this shell/process |
| `MCPError: HTTP 401/403` | expired/insufficient OAuth token; re-run onboarding |
| `MCPError: HTTP 406` | proxy stripped `Accept: application/json, text/event-stream` |
| `tool_map` has `submit_order` missing | set `ROBINHOOD_MCP_TOOL_MAP`; check `tools/list` output |
| submit returns `UNKNOWN … no properties` | tool schema is empty; override the tool map or report it upstream |
| engine returns `blocked: live venue blocked …` | `why_not_live()` lists exactly which gate failed |
| orders never send but previews work | `dry_run` still true, or kill-switch file exists |
| `max_daily_orders reached` unexpectedly | counters include previews; wait for UTC date rollover or start a fresh engine |
