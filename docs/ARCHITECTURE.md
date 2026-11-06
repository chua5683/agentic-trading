# Architecture

AgenticTrading is a multi-agent trading system built around one uncomfortable
fact: **a language model should not be allowed to move money on its own.** The
architecture is an answer to that, not a workaround for it.

The system decomposes into four layers with exactly two seams. Everything below
is designed so that a decision can be replayed, audited, and reproduced.

```
                        ┌──────────────────────────────────────────┐
                        │            MarketContext                 │
                        │  bars[:i+1] + causal features at bar i   │
                        └───────────────────┬──────────────────────┘
                                            │  (read-only; no future data)
   ┌────────────────────────────────────────▼──────────────────────────────────┐
   │ AGENT LAYER                                    agentic_trading/agents/    │
   │                                                                           │
   │  analysts ──► debate ──► trader ──────────────► RISK GATE (deterministic) │
   │  technical    bull/bear   TradeProposal          approve/veto/resize      │
   │  momentum     net score   (LLM may refine)       stop-loss / take-profit  │
   │  mean_rev                                        turnover / drawdown cap  │
   │  risk                                                                │    │
   │                                                                      ▼    │
   │                                                          PortfolioDecision │
   └───────────────────────────────────┬───────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  BACKTEST ENGINE              EXECUTION ENGINE                DECISION JOURNAL
  bar-by-bar replay            guardrails + broker              JSONL audit trail
  metrics vs buy-and-hold      paper | Robinhood MCP            replayable
```

## 1. The two seams

### Seam A — `DataProvider` (`contracts.DataProvider`)
A vendor adapter yields `list[Bar]`: ascending, tz-aware UTC, never a bar whose
open time is after `end`, never a still-forming bar. Providers raise
`DataUnavailableError` when they cannot serve a request; the chain falls through
to the next vendor. This is what makes the system vendor-agnostic and what makes
"it runs offline" possible.

### Seam B — `DecisionStrategy` (`contracts.DecisionStrategy`)
```python
class DecisionStrategy(Protocol):
    name: str
    def decide(self, ctx: MarketContext, portfolio: PortfolioState) -> PortfolioDecision: ...
```
The backtest engine knows *nothing* about analysts, LLMs, or debates. It knows
only "give me a context, get a decision". That is why the same strategy object
runs in the backtester and in the live loop — if those two paths diverged, every
backtest number would be a lie.

## 2. Data flow and the leakage guarantee

Lookahead bias is the most common way LLM trading backtests produce imaginary
alpha. The fix here is structural rather than careful.

1. `feature_frame(bars)` is computed **once** for the whole series. Every
   indicator at row `i` is a function of rows `<= i` only (`min_periods` guards,
   no negative shifts, no centred windows).
2. The engine slices that single frame: at bar `i` the strategy receives
   `features = latest_features(frame, i)` and `bars = bars[:i+1]`.
3. A recording strategy in the test suite asserts that at decision step `k` it
   sees exactly `k` bars, the newest of which is the decision bar. If the engine
   ever leaks, that test fails.

Because the frame is computed once and sliced, the fast backtest path is
*provably identical* to recomputing indicators on each prefix — and
`tests/test_indicators.py` asserts exactly that equality.

Execution assumptions are explicit in `BacktestConfig`: fills at the decision
bar's close, `fee_bps` per side, `slippage_bps` per side, and `warmup_bars`
during which no decision is taken (indicators need history).

## 3. "The model proposes, the code disposes"

This is the core safety property, implemented as a type-level separation:

| Type | Who may create it | Authority |
|---|---|---|
| `Stance` | analyst code (rule-based and/or LLM) | opinion only |
| `DebateResult` | debate module | aggregated opinion |
| `TradeProposal` | trader module, **possibly LLM** | a request, no power |
| `RiskDecision` | `agents/risk.py`, **deterministic only** | approve/veto/resize |
| `PortfolioDecision` | pipeline | the auditable outcome |

The risk gate is pure arithmetic over `ctx.features`, `RiskConfig`, and the
current `PortfolioState`. When an LLM is in the loop it can change *what it wants*
but it cannot change *what is allowed*. Concretely, `agents/risk.py` enforces:

- `min_confidence` — a low-conviction proposal degrades to HOLD;
- `max_position_weight` — clamp, always, as the last step;
- `max_drawdown_stop` — no new exposure once the book is deep in drawdown;
- `stop_loss_pct` / `take_profit_pct` — an existing position is closed;
- `max_daily_turnover` — caps how much of equity may rotate per bar;
- `cooldown_bars_after_stop` — prevents immediate re-entry after a stop;
- `allow_short: false` — negative weights are rejected, not reinterpreted.

`tests/test_risk.py` asserts each of these against a deliberately hostile
proposal, including a weight of 90% that must come back at the configured cap.

## 4. LLM-optional by construction

There are no API keys in the default environment, and the system is expected to
work anyway. Every analyst therefore computes a **complete deterministic
stance** first. If `LLMConfig.resolved_enabled()` is true, one compact JSON call
may refine the label/confidence/rationale and the stance is marked
`source="llm"`; on any error, timeout, or malformed JSON the deterministic stance
is used unchanged.

Consequences:

- the same code path is testable in CI with no network and no secret;
- the LLM is an *enhancement* over a real baseline, so its contribution is
  measurable (run with `AGENTIC_TRADING_LLM=off` and diff the results);
- no prompt-injection in market data can escalate privileges, because LLM output
  is only ever a `Stance` or a bounded `TradeProposal`.

Any OpenAI-compatible endpoint works (`base_url` + `api_key_env`), so the same
project runs against OpenAI, a local server, or a proxy.

## 5. Execution and the Robinhood path

Robinhood did not ship a REST trading API; in May 2026 it shipped **Agentic
Trading** as a Model Context Protocol server
(`https://agent.robinhood.com/mcp/trading`). So the broker adapter speaks MCP:
`initialize` → `notifications/initialized` → `tools/list` → `tools/call`, over
streamable HTTP with an `Mcp-Session-Id`, OAuth bearer token from the
environment.

Because tool names are discovered at runtime rather than fixed by a spec, the
adapter maps them by pattern and exposes `tool_map` for a user override. It also
inspects each tool's input schema to build arguments, so a schema change degrades
to `OrderStatus.UNKNOWN` instead of a malformed order.

`ExecutionEngine` is the only component that may call `broker.submit_order`, and
it enforces blast-radius limits in deterministic code, in this order:

1. kill-switch file present → block everything;
2. symbol not in allowlist → block;
3. `dry_run` → journal and return `OrderStatus.PREVIEW` (broker never called);
4. `max_order_notional` → clamp or block;
5. `max_daily_orders` / `max_daily_notional` → block;
6. no-short → block sells that would go negative;
7. minimum trade size → suppress dust.

Going live requires **all** of: `execution.enabled=true`, `mode="live"`,
`dry_run=false`, no kill-switch file, and `AGENTIC_TRADING_LIVE` set to an exact
confirmation phrase. `ExecutionConfig.live_enabled()` fails closed and
`why_not_live()` explains every failed gate, which is what `doctor` prints.

Robinhood additionally offers a "trade approvals" mode where an agent can only
propose orders for a human to confirm in the app. Our `dry_run`/`paper` default
mirrors that posture.

## 6. Module map

```
agentic_trading/
  contracts.py      frozen shared types + the two Protocol seams   (Lead)
  config.py         typed YAML config, env overrides, safety gates (Lead)
  errors.py         exception hierarchy                            (Lead)
  cli.py            doctor / backtest / evaluate / analyze / broker / live (Lead)
  evaluation.py     multi-regime grid vs buy & hold                (Lead)
  data/
    indicators.py   causal, vectorised features (single source)    (Lead)
    base.py         provider chain + registry
    cache.py        on-disk OHLCV cache, merge-by-timestamp
    coinbase.py     keyless crypto OHLCV
    yahoo.py        equities (graceful when throttled -> 429)
    twelvedata.py   equities (needs key)
    synthetic.py    deterministic offline GBM, always available
    service.py      MarketDataService -> MarketContext
  agents/
    llm.py          OpenAI-compatible client + offline Null client
    analysts.py     technical / momentum / mean_reversion / risk
    debate.py       bull-vs-bear aggregation into DebateResult
    trader.py       TradeProposal sizing (vol-targeted)
    risk.py         deterministic gate -- the authority
    pipeline.py     AgentStrategy = the DecisionStrategy implementation
  backtest/
    engine.py       bar-by-bar replay, explicit costs, no leakage
    metrics.py      returns/sharpe/sortino/drawdown/calmar vs buy-and-hold
    report.py       markdown + self-contained HTML (inline SVG chart)
  brokers/
    mcp_client.py       minimal MCP-over-streamable-HTTP client
    robinhood_mcp.py    Robinhood Trading MCP adapter
    paper.py            deterministic simulated broker, resumable state
    execution.py        guardrails + OrderRequest construction
    audit.py            append-only JSONL journal, replayable
```

## 7. Extension points

- **New venue**: implement `Broker` (5 methods) and register it in
  `brokers.build_broker`.
- **New data vendor**: implement `DataProvider.fetch_ohlcv` and add its name to
  `data.provider_order`.
- **New analyst**: add a class in `agents/analysts.py` with `name` and
  `analyze(ctx, llm)` returning a `Stance`, then add its name to
  `agents.analyst_names`.
- **New feature**: add it to `REQUIRED_FEATURES` and `feature_frame`; the schema
  assertion in the indicator tests keeps both paths honest.
- **Different model**: change `llm.base_url` / `llm.model`; nothing else.

## 8. Evaluation, and what it forced us to change

A single window can flatter any parameter set, so `evaluation.py` runs the
strategy over five pre-declared regime windows across two assets and prints the
win/loss ledger including every loss.

That harness immediately contradicted the first implementation. In the
2024–25 bull window the agent returned **-6.6%** while a *dumb always-30%-long*
allocation returned **+23.0%** at lower drawdown. Diagnosis, not tuning,
followed: ablating the analysts showed the aggregation was *harmful* (technical
alone `+0.7%`, all four `-6.6%`), and the decision journal showed the book
flipping between 0 and ~10% every few days — i.e. whipsaw around the consensus
threshold, paying 15bps per leg.

The fix is in the deterministic policy layer and is deliberately boring:

- **causal EMA smoothing** of `net_score` (α=0.35) so one noisy bar cannot flip
  the book;
- **hysteresis** — enter on moderate conviction, exit only on a primary-trend
  break (close below SMA200) after `min_hold_bars`, or on a strong opposite
  signal; a risk-mandated stop bypasses this and forces flat;
- **regime-aware debate weights** — mean-reversion is halved when the SMA stack
  is aligned, because fading a confirmed trend is the expensive mistake;
- **conviction floor** 0.35 so a modest persistent edge produces a real
  position;
- **rebalance dead-band** so a target change below 30% relative is a HOLD.

Measured across the 10 fixed windows: trades **587 → 248 (-58%)**, mean excess
return **-4.26% → -1.3%**, mean max drawdown **5.1% → 4.3%**. One window
regressed (ETH recovery-2023, +3.1% → +0.1%) and is reported rather than hidden.

## 9. Known limitations

- **Absolute edge is weak.** The deterministic analysts are classical technical
  signals on daily bars. Across 10 fixed windows the strategy beats buy & hold
  5 times and loses 5 times; mean excess return is about -1.3%. It is a
  *credible, auditable baseline* and a defensible defensive allocator — not
  alpha. The honest next step is a better signal, not more agents.
- **Stop-losses are the largest residual whipsaw source.** Many stop-outs are
  sell-at-dip/rebuy-higher cycles. Widening `stop_loss_pct` from 8% improves the
  bull window and degrades the bear window, so the frozen config was left alone
  instead of being overfit to one regime.
- Regime-aware weights were chosen from a 10-window sample. That is small;
  treat parameter choices as provisional and re-run `evaluate` after any change.
- Single-asset focus. The `PortfolioState`/`BacktestConfig` layers support
  multiple symbols, but the orchestration and benchmark are per-symbol.
- Daily bars mostly. Intraday timeframes are plumbed through but data vendors
  were probed for daily coverage.
- The Robinhood adapter is written against the documented MCP contract and
  verified against a fake transport; it has not been run against a live,
  authenticated Robinhood account from this environment (no token present).
- Costs are modelled as bps of notional; real fills, borrow costs, and taxes are
  out of scope.
