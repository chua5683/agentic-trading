# AgenticTrading

A multi-agent trading system where **an LLM may propose, but deterministic code
disposes** — and where every decision is backtested without lookahead and
journalled so it can be replayed.

It runs with no API keys and no network (deterministic analysts + synthetic
market data), upgrades to LLM reasoning when you give it a key, and can place
paper or real orders through **Robinhood's official Agentic Trading MCP server**.

```
$ python -m agentic_trading backtest --symbol BTC-USD --start 2024-01-01 --out reports
```

> **Name note:** this is a local project directory name. There is an unrelated
> academic effort called *AgenticTrading* on GitHub
> (`Open-Finance-Lab/AgenticTrading`, ~884★). Nothing here is derived from it;
> the name is qualified here so the two are not confused.

---

## Why this one

There are many LLM-trading repos. Most share three weaknesses worth naming:
backtests that leak future data, risk limits that live inside a prompt, and
non-reproducible results. This project is built around fixing those three.

| | Typical LLM trading repo | AgenticTrading |
|---|---|---|
| Risk limits | written in a prompt, hoped for | enforced in `agents/risk.py`, unit-tested |
| Backtest integrity | indicators recomputed per step, easy to leak | one causal feature frame, sliced; test proves no leakage |
| Needs an API key | yes, to do anything | no — deterministic baseline + offline synthetic data |
| Reproducibility | depends on model temperature | deterministic mode is bit-reproducible; LLM is an optional layer |
| Execution | none, or scraped unofficial APIs | paper broker + official Robinhood MCP adapter |
| Audit trail | logs | append-only JSONL with decision provenance |

The point is not that it will beat the market. The point is that when it
doesn't, you can find out *why*.

---

## Quickstart (no API key required)

```bash
# 1. environment (Python 3.10+, uses uv)
make setup

# 2. is everything wired up?
make doctor

# 3. a real backtest on real Coinbase data
make backtest SYMBOL=BTC-USD START=2024-01-01 END=2025-11-01
# -> reports/BTC-USD.html, .md, .json

# 4. one auditable decision for the latest bar
make analyze SYMBOL=BTC-USD
```

Reports are self-contained HTML (inline SVG chart, no CDN/JS), Markdown, and raw
JSON with the full decision journal.

---

## How a decision is made

```
MarketContext  (bars up to bar i, causal features at bar i)
      │
      ├─► analysts ── technical / momentum / mean_reversion / risk   -> list[Stance]
      │        (each computes a deterministic stance first; an LLM may refine it
      │         only when a key is configured, and only after the rules ran)
      │
      ├─► debate ──── bull vs bear aggregation                        -> DebateResult
      │
      ├─► trader ──── vol-targeted sizing                             -> TradeProposal
      │
      └─► RISK GATE  deterministic, non-bypassable                    -> RiskDecision
                       clamp to max_position_weight
                       min_confidence -> HOLD
                       drawdown stop / stop-loss / take-profit
                       turnover cap / cooldown / no shorting
                                                                          │
                                                              PortfolioDecision (audited)
```

**The asymmetry is the design.** `TradeProposal` is the most an LLM can produce;
it has no authority. `RiskDecision` can only be constructed by `agents/risk.py`,
which is pure arithmetic. No prompt, no injection in a news headline, and no
model hallucination can raise a position above `risk.max_position_weight`.

Try it: `tests/test_risk.py` feeds the gate a hostile 90%-weight, low-confidence,
short-selling proposal and asserts what comes back.

---

## LLM-optional, not LLM-dependent

| `llm.enabled` | behaviour |
|---|---|
| `off` | deterministic analysts only; zero network, zero cost, bit-reproducible |
| `auto` (default) | deterministic baseline, upgraded by an LLM **iff** `OPENAI_API_KEY` is set |
| `on` | same as auto, but warns instead of silently falling back |

Any OpenAI-compatible endpoint works — set `llm.base_url`, `llm.model`,
`llm.api_key_env` in the config. Use a local server (Ollama/vLLM) and it never
leaves your machine.

Because the deterministic path is complete, the LLM's contribution is
*measurable*: run a backtest with `AGENTIC_TRADING_LLM=off` and again with it on,
and diff the metrics.

---

## Robinhood

Robinhood does not expose a REST trading API. In May 2026 it shipped **Agentic
Trading** as an official **MCP server** — so this is what we integrate with:

```
https://agent.robinhood.com/mcp/trading      (streamable HTTP + OAuth)
```

The adapter speaks JSON-RPC 2.0 MCP (`initialize` → `tools/list` → `tools/call`),
discovers the venue's tools at runtime, maps them onto our `Broker` protocol, and
inspects each tool's input schema to build orders. It supports the order types
Robinhood documents: market (share), market (dollar), limit, stop-limit,
stop-market.

```bash
export ROBINHOOD_MCP_ACCESS_TOKEN=...   # from your Robinhood MCP connection
python -m agentic_trading doctor        # shows which MCP tools were discovered
python -m agentic_trading broker        # account, buying power, positions, quotes
python -m agentic_trading live --paper  # decision -> risk -> execution, paper
```

See [docs/ROBINHOOD.md](docs/ROBINHOOD.md) for the full setup walkthrough.

### Safety gates for real money

Default posture is **inert**: `enabled=false`, `broker=paper`, `dry_run=true`.
A live order requires *all* of the following:

1. `execution.enabled: true` in config
2. `execution.mode: "live"`
3. `execution.dry_run: false`
4. no `STOP_TRADING` kill-switch file on disk
5. `AGENTIC_TRADING_LIVE=yes-i-understand-the-risk` in the environment

`ExecutionConfig.live_enabled()` fails closed on any doubt, and
`why_not_live()` lists every failed gate (surfaced by `doctor`). Independently of
signals, `ExecutionEngine` enforces `max_order_notional`, `max_daily_orders`,
`max_daily_notional`, a symbol allowlist, and no-short — in code, before the
broker is ever called. Every attempt, blocked or sent, lands in
`reports/orders.jsonl`.

```bash
touch STOP_TRADING     # instantly halt all order submission
```

---

## CLI

| command | what it does |
|---|---|
| `doctor` | config, data-vendor probe, LLM status, broker + live-gate report |
| `backtest --symbol X --start Y --end Z --out reports` | leakage-free replay + metrics vs buy-and-hold |
| `analyze --symbol X [--asof DATE] [--json]` | one decision, every agent's reasoning shown |
| `broker` | broker account, positions and quotes |
| `live [--paper] [--once] [--max-iterations N]` | the decision → risk → execution loop |

Common flags: `--config`, `--offline` (synthetic data, no network), `--timeframe`.

---

## Backtest results

### Method

One backtest over one window proves nothing, so results are reported over **five
pre-declared regime windows × two assets**, with the same strategy, costs and
warmup. Each row is an independent run (fresh strategy, no carried state).
Windows were fixed before results were inspected. Bars come from Coinbase
(keyless), fees 10bps and slippage 5bps per side, fills at the decision bar's
close. `make evaluate` reproduces the table.

### Strategy vs buy & hold

| symbol | window | regime | return | buy&hold | excess | maxDD | b&h maxDD | trades |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BTC-USD | bear-2022 | drawdown year | −1.8% | −62.4% | **+60.6%** | 1.8% | 66.8% | 3 |
| BTC-USD | recovery-2023 | choppy recovery | +3.9% | +79.9% | −76.0% | 5.9% | 20.1% | 26 |
| BTC-USD | bull-2024-25 | sustained uptrend | −1.8% | +75.2% | −77.0% | 6.2% | 28.2% | 58 |
| BTC-USD | cycle-2024-now | full cycle | +0.1% | +21.5% | −21.4% | 6.2% | 53.1% | 63 |
| BTC-USD | last-12m | recent year | +0.6% | −19.1% | **+19.6%** | 3.0% | 39.6% | 13 |
| ETH-USD | bear-2022 | drawdown year | −0.9% | −59.5% | **+58.6%** | 1.7% | 71.8% | 3 |
| ETH-USD | recovery-2023 | choppy recovery | +0.1% | +38.2% | −38.1% | 6.1% | 27.4% | 21 |
| ETH-USD | bull-2024-25 | sustained uptrend | +5.2% | +11.8% | −6.5% | 3.0% | 63.8% | 25 |
| ETH-USD | cycle-2024-now | full cycle | +5.2% | −35.1% | **+40.3%** | 3.0% | 67.6% | 29 |
| ETH-USD | last-12m | recent year | −5.8% | −32.3% | **+26.5%** | 6.4% | 53.3% | 13 |

**Beat buy & hold: 5 of 10 windows.** Mean excess return **−1.3%**.
Mean max drawdown **4.3% vs 49.2%**.

**This does not beat buy & hold on return, and the table says so.** What it does
is hold a fraction of the asset (average exposure ~22%): it loses almost nothing
in a −60% year, and therefore gives up most of a +75% one. That is a defensive
mandate, not an alpha claim. Anyone reading a single row — especially
`BTC-USD bull-2024-25` — will correctly conclude the agent underperforms.

### What the evaluation actually taught us

The first version of this strategy lost **6.6%** in the bull window while a
*dumb always-30%-long* allocation made **+23.0%** with lower drawdown. That is
not conservatism, that is adverse timing, so we found the cause instead of
tuning around it:

| BTC, bull-2024-25 | return | trades |
|---|---:|---:|
| always 10% long | +7.5% | 68 |
| always 30% long | +23.0% | 180 |
| agent, before fix | −6.6% | 133 |
| **agent, after fix** | **−1.8%** | **58** |

Ablating the analysts showed the aggregation itself was the problem — adding
mean-reversion and risk views *diluted* the one marginally useful trend signal:

| BTC, bull-2024-25 | technical | momentum | mean-reversion | tech+momentum | all four |
|---|---:|---:|---:|---:|---:|
| return (before fix) | +0.7% | −6.6% | −6.5% | −5.2% | −6.6% |

The diagnosis was whipsaw: `net_score` oscillated across the consensus
threshold, flipping the book between 0 and ~10% every few days — selling after
pullbacks and rebuying after recoveries, 15bps each way. The fix (all
deterministic, all unit-tested) was **causal smoothing** of the consensus,
**hysteresis** (`min_hold_bars`, exit on trend break rather than a lukewarm
score), **regime-aware debate weights** (mean-reversion halved when the SMA stack
is aligned), a **conviction floor**, and a **rebalance dead-band**. Across the
10 windows: trades **587 → 248 (−58%)**, mean excess **−4.26% → −1.3%**, mean
drawdown **5.1% → 4.3%**, bull-window return **−6.6% → −1.8%**.

Two honest caveats from the same exercise: the fix **regressed ETH
recovery-2023** (+3.1% → +0.1%), and stop-losses remain the largest residual
whipsaw source (25 stop-outs in the bull window before, 15 after) — widening the
8% stop helps that window and hurts the bear window, so we left the frozen risk
config alone rather than overfit it.

### What is actually established

- The pipeline runs, is auditable, and produces genuinely legible reasoning.
- The risk gate does what it claims: drawdown is 4–6% against 20–72% for holding.
- The deterministic signal set has **weak absolute edge**; the honest next step
  is a better signal, not more agents. The LLM layer is measurable against this
  baseline precisely because the baseline never disappears.


---

## Repository layout

```
agentic_trading/
  contracts.py      frozen shared types + the two Protocol seams
  config.py         typed YAML config, env overrides, live-trading gates
  cli.py            doctor / backtest / analyze / broker / live
  data/             providers (Coinbase, Yahoo, TwelveData, synthetic), cache,
                    causal indicators, MarketDataService
  agents/           llm client, analysts, debate, trader, risk gate, pipeline
  backtest/         engine, metrics, markdown+HTML reports
  brokers/          MCP client, Robinhood adapter, paper broker, execution, audit
docs/
  RESEARCH.md       landscape of agentic trading projects on GitHub
  ARCHITECTURE.md   design, seams, leakage guarantee, safety model
  ROBINHOOD.md      how to connect the Robinhood MCP
tests/              222 tests; every safety claim above is asserted
```

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first, then
[docs/RESEARCH.md](docs/RESEARCH.md) for the landscape this sits in.

---

## Data sources

Chosen for what actually works keyless, verified in this environment:

| provider | asset class | auth | note |
|---|---|---|---|
| Coinbase Exchange | crypto | none | primary; paginated OHLCV |
| Twelve Data | equities | API key | free tier |
| Yahoo Finance | equities | none | often rate-limited (HTTP 429) — degrades gracefully |
| synthetic | any | none | deterministic GBM; offline tests and demos |

The provider chain falls through on failure, so a throttled vendor degrades the
system instead of breaking it.

---

## Testing

```bash
make test          # full suite
make offline       # end-to-end with zero network dependency
```

The suite asserts the claims in this README rather than describing them:
no-lookahead in the engine, causality in the indicator frame, the risk gate's
domain boundaries, `dry_run` never reaching `broker.submit_order`, and MCP
parsing against a fake transport.

---

## Disclaimer

Research software, not investment advice. Trading involves substantial risk of
loss. The Robinhood adapter is written against the documented MCP contract and
is exercised against a fake transport in tests; it has not been run against a
live funded account from this environment. Read the disclaimers in
[docs/ROBINHOOD.md](docs/ROBINHOOD.md) before pointing this at real money.
