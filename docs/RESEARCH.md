# Research: the agentic-trading landscape on GitHub, and why AgenticTrading is built this way

**Retrieval date: 2026-11-05** (Pacific). Every star/fork/issue count below was read from
`api.github.com` on that date and is stored verbatim in
[`research/github_snapshot_2026-11-05.csv`](research/github_snapshot_2026-11-05.csv);
the raw top-40 search ranking is in
[`research/search_top40_2026-11-05.csv`](research/search_top40_2026-11-05.csv).
Exact queries and commands are in [§7](#7-reproducibility-appendix).

Scope note: this is a **landscape** report, not a benchmark. Nothing here reproduces anyone's
returns. Where a repo *claims* alpha, the claim is reported as a claim. Claims marked
**[unverified]** could not be checked from source or docs — they are called out rather than
smoothed over.

---

## 1. Executive summary

The field has converged on one shape and left four gaps around it.

**The converged shape.** Take a language model, give it tools, decompose the trading decision into
specialised roles, make the roles argue, and let a manager decide. TradingAgents codified it
(analysts → bull/bear debate → research manager → trader → risk debate → portfolio manager) and
120,871 stars later it is the reference implementation. 28 of the 57 agent projects in the
community [awesome-trading-agents list](https://github.com/LLMQuant/awesome-trading-agents) are
multi-agent systems of exactly this form.

**The four gaps.** (1) The converge-shaped systems are **LLM-mandatory** — every one of the top six
by stars needs at least one paid API key before it can emit a number. (2) Their **risk management is
roleplay** — the "risk manager" is another prompt, and the position size is prose. (3) Their
**backtests leak**, and the mature projects say so in their own changelogs. (4) Their **results are
not reproducible** by their own admission.

**The two genuinely new things since late 2025** are not architectural: they are
**agent-native interfaces** (MCP servers, `SKILL.md` bootstraps) and **execution venues that now
accept agents**. Robinhood shipped Agentic Trading as an MCP server on **2026-05-27** — verified
from Robinhood's own newsroom and support docs, endpoint
`https://agent.robinhood.com/mcp/trading` — and that single fact reprices the whole design space.
The broker is now a tool call. What is scarce is not connectivity; it is **containment,
reproducibility, and honesty about baseline**.

### What AgenticTrading takes

| Take | From whom | Why it is worth taking |
|---|---|---|
| Role decomposition (analysts → debate → trader → risk → portfolio) | TradingAgents | It is legible, it is the field's shared vocabulary, and it maps cleanly onto a typed pipeline |
| A provider **chain** with graceful fall-through instead of one hard-coded vendor | Vibe-Trading (28 sources), AI-Trader (Alpha Vantage → yfinance), TradingAgents (`dataflows/router.py`) | Every mature project in this space eventually grows one; the projects that did not are stale or dead |
| "Fail closed on the order path, fail open on the advice path" | QuantDinger (`JEV` gate) + LangAlpha (`order_governance.py`) | The only two projects that separate *what the model wants* from *what the code allows*, in code |
| Deterministic whole-series feature computation with point-in-time discipline | TradingAgents' five releases of PIT fixes (v0.3.1 → v0.7.1); Vibe-Trading's `22.5% → 18.4%` benchmark fix | The landscape proves leakage is not a theoretical worry: it is the single most-repeated bug class |
| Append-only decision journal with replay | nof0 (`go/cmd/journalreplay`), LangAlpha (order attempts ledger), TradeMemory (fills + refusals) | "Show me why" is the only defensible answer to an agent that lost money |
| Benchmark-relative reporting as a first-class output | TradingAgents (`alpha` vs regional benchmark), Vibe-Trading ("compare with buy-and-hold") | Absolute return from a single long-only window is noise |

### What AgenticTrading deliberately does differently

1. **It runs with zero keys and zero network.** Every comparable repo hard-fails without a key
   (see §3). Here the deterministic analysts produce complete stances first; a synthetic GBM
   provider is always available; the LLM is an *upgrade path*, not a dependency. Consequence: CI
   needs no secret, and the LLM's contribution is measurable instead of assumed.
2. **The risk gate is arithmetic, not a persona.** `RiskDecision` can only be produced by
   `agents/risk.py`. In the entire TradingAgents tree, the only code touching deterministic risk
   vocabulary is **3 lines in `agents/schemas.py`**, all of them the model-filled `stop_loss` field
   of `TraderProposal`; sizing is `position_sizing: str | None` ("Optional sizing guidance, e.g.
   '5% of portfolio'") and nothing consumes that string. Here a 90% proposal comes back at the
   configured cap, and a test asserts it.
3. **Leakage is prevented structurally, not by care.** `feature_frame()` is computed **once** and
   sliced (`bars[:i+1]`); a recording strategy in the test suite asserts it sees exactly `k` bars at
   step `k`. TradingAgents needed five releases over four months (v0.3.1 → v0.7.1) to close
   point-in-time holes; Vibe-Trading was still fixing "Carhart momentum leaking today's price" in
   November 2026.
4. **It says "this is a baseline" out loud.** No repo in this survey publishes a return it can
   reproduce. AgenticTrading emits `total_return`, `cagr`, `sharpe`, `sortino`, `max_drawdown`,
   `turnover`, `exposure` and `risk_vetoes` **beside** `benchmark_*` (same statistics for
   buy-and-hold), plus `excess_return` and `alpha_cagr` — and claims nothing more.

**Name collision, flagged early:** `Open-Finance-Lab/AgenticTrading` already exists — 884 stars,
OpenMDW-1.0, an academic "Agentic Trading Lab" platform (FastAPI + leaderboard + Alpaca/Robinhood
adapters). This project is **not** that one, and shares no code with it. Anyone searching the name
will find it first. See §3.9.

---

## 2. Landscape table (skim in 60 seconds)

All counts observed 2026-11-05. "BT" = ships a backtester. "LLM req." = cannot produce a decision
without a model API key.

| Repo | Stars | Lang | Primary approach | Data sources | LLM req. | BT | Licence notes |
|---|---:|---|---|---|---|---|---|
| [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | **120,871** | Python | 11-role LangGraph firm: 4 analysts → bull/bear debate → research mgr → trader → 3-way risk debate → PM | Yahoo, Alpha Vantage, FRED, SEC EDGAR, Reddit, StockTwits, Polymarket (+optional Jev screening) | **yes** (18 providers) | decision grid only, *explicitly not a portfolio sim* | Apache-2.0 |
| [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) | **37,746** | Python | Agent + CLI/web/MCP; prompt → research → backtest → paper/live within a mandate | **28** sources, per-market fallback chains (46 loader modules) | yes | **yes** (17 engine modules) | MIT; 6,114 forks |
| [hsliuping/TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN) | **32,815** | Python | Chinese/A-share enhanced fork of TradingAgents | CN vendors (A-share, HK) | yes | yes | NOASSERTION (no SPDX id) |
| [HKUDS/AI-Trader](https://github.com/HKUDS/AI-Trader) | **23,116** | Python + TS | **Agent-native platform**: agent registers by reading a `SKILL.md` URL, then trades signals | Alpha Vantage → yfinance fallback; Polymarket paper | yes (agent-side) | no (live paper scoring) | README badge says MIT; **no LICENSE file in tree at `main`** |
| [OpenByteInc/QuantDinger](https://github.com/OpenByteInc/QuantDinger) | **13,556** | Python | Self-hosted "AI Trading OS": Flask API + Kafka + Celery + Postgres, MCP + Agent Gateway | Multi-broker; Factor Lab point-in-time panel | yes (optional JEV gate) | **yes** + PBO/Deflated Sharpe/BH-FDR | Apache-2.0 |
| [The-Swarm-Corporation/AutoHedge](https://github.com/The-Swarm-Corporation/AutoHedge) | **6,634** | Python | 4 prompt personas (Director/Quant/Risk/Execution) on Solana via Jupiter | Jupiter, Polygon, Yahoo, Exa | yes | **no** (a 416-line experimental script, not a backtester) | MIT; 2,364 LOC total |
| [Polymarket/agents](https://github.com/Polymarket/agents) | 3,810 | Python | RAG + news + Polymarket CLOB trading | Polymarket Gamma/CLOB, news, web search, Chroma | yes | no | MIT; last push **2024-11-05** |
| [wquguru/nof0](https://github.com/wquguru/nof0) | 2,756 | Go + TS | NOF1 Alpha Arena reproduction: N models, $10k each, real fills on Hyperliquid | Hyperliquid, crypto prices | yes | no (forward paper arena) | no licence file detected |
| [ginlix-ai/LangAlpha](https://github.com/ginlix-ai/LangAlpha) | 2,244 | Python + TS | "Claude Code for markets": sandbox + programmatic tool calling + **governed orders** | Yahoo/FMP, MCP servers, SEC | yes | via Python in sandbox | Apache-2.0 |
| [Lumiwealth/lumibot](https://github.com/Lumiwealth/lumibot) | 2,212 | Python | Mature non-LLM backtest/live framework; AI as strategy examples | Polygon, Yahoo, Databento, IBKR, Alpaca, ThetaData… | **no** (optional helpers) | **yes**, 12 data sources, 13 brokers | **GPL-3.0**; 494 MB repo |
| [mnemox-ai/tradememory-protocol](https://github.com/mnemox-ai/tradememory-protocol) | 1,440 | Python | MCP memory + **brake**: refuses orders that break your own rules | Broker fills (Hyperliquid, Alpaca) | optional | no | MIT; 20 MCP tools |
| [kyky2347/ALTA](https://github.com/kyky2347/ALTA) | 1,075 | Python + TS | 4 research "Scouts" + locked independent assessments + **code-based ranking** + audit gate | Date-aware search, issuer crawls, feeds | yes (Codex harness) | no (shadow + broker) | Apache-2.0; vendors `openai-codex` |
| [LLMQuant/awesome-trading-agents](https://github.com/LLMQuant/awesome-trading-agents) | 981 | — | Curated map: 57 agents, 35 MCP servers, 17 skills | — | — | — | CC0-1.0 |
| [pipiku915/FinMem-LLM-StockTrading](https://github.com/pipiku915/FinMem-LLM-StockTrading) | 970 | Python | Single agent, **layered memory** with decay/recency/importance, reflection | Prebuilt news/SEC pipeline | yes | **no** (offline study pipeline) | MIT; last push **2024-08-18** |
| [Open-Finance-Lab/AgenticTrading](https://github.com/Open-Finance-Lab/AgenticTrading) | 884 | Python | "Agentic Trading Lab" platform: dashboard, Agent API v2, nightly leaderboard | Alpaca, Robinhood, iFinD | yes | yes (per-run + dashboard backtests) | OpenMDW-1.0; **same project name as ours** |
| [FinStep-AI/ContestTrade](https://github.com/FinStep-AI/ContestTrade) | 680 | Python | **Internal contest**: data agents compete → factor portfolio; research agents compete → allocation | AKShare, US tools | yes | **no** (contest evaluator, no backtester) | Apache-2.0 |
| [TauricResearch/Trading-R1](https://github.com/TauricResearch/Trading-R1) | 549 | Python | Terminal research harness + 17 skills; **needs `TAURIC_API_KEY`** | Tauric API, web | yes | no | Apache-2.0; **no weights in repo** |

Two structural facts fall out of the table:

- **Stars track age and name recognition, not engineering.** AutoHedge (6,634★) has 2,364 lines of
  Python, no backtester in its package, and risk management implemented as prompt strings; lumibot
  (2,212★) has 2,387 files, 12 backtest data sources and 20 broker adapters. If you pick a model to
  copy by stars alone you copy the wrong thing.
- **The fork layer is bigger than the originals it forked.** TradingAgents-CN alone (32,815★) plus
  A-share forks (TradingAgents-astock 3,914★, TradingAgents-AShare 890★) is ~37.6k stars of demand
  for "the same architecture, but with data sources that actually answer in my market". That is a
  data-plumbing signal, not an architecture signal (§5.6).

---

## 3. Deep dives

### 3.1 TradingAgents — 120,871★, Apache-2.0, 28,791 LOC, 90 test files

The reference implementation, and the only one in this survey whose design decisions are documented
as a sequence of *corrections*. Read its changelog before reading anything else in this field.

**As implemented** (verified from `main` tree + source):

- `tradingagents/agents/` (2,453 LOC) — `analysts/{market_analyst.py 84, sentiment_analyst.py 199,
  fundamentals_analyst.py 64, news_analyst.py 63, turn.py 42}`,
  `researchers/{bull_researcher, bear_researcher}`, `managers/{research_manager.py 74,
  portfolio_manager.py 106}`, `trader/trader.py 98`,
  `risk_mgmt/{aggressive,conservative,neutral}_debator.py ~68 each`, plus `state.py`, `schemas.py`,
  `structured.py`, `rating.py`, `post_screen.py`.
- **Orchestration** — LangGraph `StateGraph` (`graph/setup.py`). Each analyst is *its own subgraph*
  with a private message history so parallel analysts cannot collide on state keys and cannot see
  each other's tool calls; a `ToolNode` runs its tools; `_tools_or_done` routes
  `"tools" if state["messages"][-1].tool_calls else END`. `_analyst_graph` caps tool rounds and
  force-wraps-up, "so a model that keeps calling tools cannot run the graph into its recursion
  limit (#1420)". `graph/conditional_logic.py` is impure Python, not an LLM: debate ends at
  `count >= 2 * max_debate_rounds`, risk debate ends at `count >= 3 * max_risk_discuss_rounds`, and
  the next speaker is chosen by `latest_speaker.startswith(...)`. `DEBATE_PATH_MAP` /
  `RISK_ANALYSIS_PATH_MAP` enumerate every returnable node name so a speaker-label drift can never
  produce a missing-edge crash (#1088).
- **State** — `AgentState(MessagesState)` with `trade_date`, `instrument_context` ("Deterministic
  ticker identity resolved at run start"), four report fields, two nested debate states, and
  `final_rating` (5 tiers, or `REVIEW`).
- **Data** — `dataflows/router.py` maps tool categories (`core_stock_apis`, `technical_indicators`,
  `fundamental_data`, `news_data`, …) onto vendor implementations: `vendors/{yahoo,alpha_vantage,
  fred,sec_edgar,reddit,stocktwits,polymarket}`. Point-in-time is a first-class concern: SEC EDGAR
  figures are served **as filed** ("Apple's 2008 total assets were filed as $39.6B and restated to
  $36.2B in 2010, so a run dated in between reads $39.6B"), Yahoo-dated statements are *withheld*
  for past dates because Yahoo cannot say when they were public, and insider trades are withheld
  because they are dated by event rather than filing.
- **LLM layer** — `llm_clients/` (1,824 LOC): OpenAI, Anthropic, Google, Azure, Bedrock, xAI,
  DeepSeek, Qwen (dual-region), GLM, MiniMax, OpenRouter, Mistral, Moonshot, Groq, NVIDIA, Ollama,
  OpenAI-compatible, and ChatGPT-subscription sign-in. Two model tiers: quick (analysts/researchers/
  trader) and deep (managers), each with its own provider.
- **Evaluation** — `backtest.py` runs the graph over a (ticker × date) grid and scores each cell on
  realised **alpha vs the instrument's regional benchmark**, entered at the first close after the
  decision. Its own docstring: *"It is not a portfolio simulator, and must not grow one. Turning a
  rating into a filled order needs a quantity, a fill price and a cash ledger, none of which the
  system has; inventing them here would put an execution model behind an evaluation tool."*

**Strengths.** (a) Point-in-time integrity is treated as a correctness property with tests
(`tests/test_memory_pointintime.py`, `test_tool_date_enforcement.py`, `test_undated_tools_as_of.py`,
`test_date_boundaries.py`). (b) Determinism where it is cheap: the ticker→company identity is
resolved *before* any agent runs, the market analyst grounds price claims in a verified snapshot,
and the docstring names the exact failure modes those two mechanisms fix ("different companies" and
"fabricated price levels"). (c) A memory log that **settles** past decisions against realised return
and alpha and writes a reflection the PM reads — a real feedback loop, not a decorative one.
(d) Explicit refusal to oversell: the README states backtest results "are not guaranteed to match
any published figure" and the framework is "a research scaffold … not a strategy with a fixed,
replicable return."

**Limitations.**
- **No execution model at all.** `TraderProposal` carries `entry_price`, `stop_loss` and
  `position_sizing: str | None` — "Optional sizing guidance, e.g. '5% of portfolio'". Nothing parses
  or enforces it; there is no cash ledger, no fill model, no venue. "Risk management" is three LLM
  debators plus a PM prompt. Grep for `max_position|position_size|kelly|stop_loss|drawdown` across
  `tradingagents/` returns **3 hits, all in `agents/schemas.py`**.
- **Cost and latency per decision.** The topology alone implies **12 model calls per decision** at
  the class defaults (`max_debate_rounds=1`, `max_risk_discuss_rounds=1`): 4 analysts + 2 debate
  turns + 1 research manager + 1 trader + 3 risk turns + 1 PM — before any analyst's tool-call
  rounds. At the README's `max_debate_rounds=2` it is 14. That is a per-*bar* cost for what a
  moving-average rule computes in microseconds.
- **Non-reproducible by construction**, and the README says so: "two runs of the same ticker and
  date can differ … the current curated models are reasoning-first and largely ignore temperature."
- **LLM-mandatory.** No key → no decision. There is no keyless path, no offline mode, and no
  deterministic baseline to compare the agents against.

> **So what.** *Copy:* the role vocabulary, the analyst-as-subgraph pattern (it is the cleanest way
> to keep parallel agents from polluting each other's state), the deterministic-where-cheap posture
> (resolve identity before the LLM runs; ground every number the model may quote), the settled
> memory log, and the refusal to fake a portfolio simulator inside an evaluation harness.
> *Avoid:* treating a risk *persona* as risk management; making the LLM the only path to a number;
> shipping a five-figure-star framework whose sizing output is a free-text string nobody consumes.
> Their `backtest.py` docstring is the best sentence in this field and we should keep its spirit:
> **an evaluation tool must not smuggle in an execution model, and an execution layer must not be a
> prompt.**

### 3.2 Vibe-Trading — 37,746★, MIT, 3,219 files, v0.1.19 shipped the same day as this survey

The most *operationally* serious open project: it is built like a broker-facing product, and its
changelog is a public log of data-plumbing bugs, which makes it uniquely useful evidence.

**As implemented.** `agent/` contains `backtest/{engines/, loaders/, metrics.py, run_card.py,
risk_xray.py, validation.py, constraints.py, loader_health.py, benchmark.py, regime.py}`,
`backtest/engines/{china_a, china_futures, crypto, forex, global_equity, global_futures, india_equity,
korea_equity, vietnam_equity, argentina_equity, brazil_equity, options_portfolio, composite}`,
`agent/mcp_server.py` + `vibe-trading-mcp` stdio entry point, `agent/cli/`, and **90 `SKILL.md`
files** (mostly under `agent/src/skills/`, a few under `agent/skills/`). Data: **28 sources** with
per-market fallback ("A-shares/HK starts with Tencent → Eastmoney → rest; US starts with Yahoo →
Stooq → Sina; crypto: OKX → Binance → CCXT"). Execution: **19 broker connectors** with per-broker
permissions, paper fills, a mandate, and a kill switch.

**Strengths.**
- **Fallback chains are a first-class architectural object**, not an afterthought: `source: "auto"`
  infers the market from the symbol and walks the chain; `loaders/loader_health.py` reports *why* a
  source returned nothing, and credentials are redacted out of health output (#1643/#1698).
- **Reports that fail closed.** "report audits that fail closed when nothing was verified (#1362)";
  "grounding that fails closed on unresolved derived refs" (#1835/#1837). A run card names "the data
  source, the model, the execution record and the metric references behind a backtest" so a
  reported figure can be re-opened.
- **The best leakage evidence in the field.** From the release notes: *"backtest benchmarks are
  bought at the first evaluated open, as the strategy is, instead of from the last warm-up close;
  excess return no longer includes that move (AAPL 20/60 SMA example: **22.5% → 18.4%**).*" A pure
  benchmark-alignment bug was overstating excess return by **4.1 percentage points**.
- **Self-aware about contamination**: "run cards that flag backtests possibly inside the model's
  training data (#1618)".

**Limitations.**
- **Surface area is its own risk.** 3,219 files, 90 skills, 17 engines, 46 loaders, 19 brokers,
  116 merged PRs *in one release*. The November 2026 changelog is largely a list of correctness
  regressions across that surface — split detection (3:2), TTM sums across a missing quarter, NaN
  propagation through 16 alphas, "adjusted and raw prices never mixed in one series". Reproducing a
  number across two versions of this repo is not guaranteed.
- **Still requires a model provider** (`agent/.env` provider + key) for the agent loop; the
  backtest engine is deterministic but the thing that writes strategies is not.
- **Breadth over auditability of the core claim.** It reports Sharpe/drawdown/IC and validates
  walk-forward, but the headline artefact is a research report, not a reproducible strategy object
  with a frozen dataset hash.

> **So what.** *Copy:* the fallback-chain-with-health-reporting idea, run cards that bind every
> number to its source and metric definition, next-bar fills, benchmark bought at the same bar as
> the strategy, and "report fails closed if nothing was verified". *Avoid:* growing surface area
> faster than the correctness story. Our equivalent of "90 skills" is one feature frame with a
> schema assertion.

### 3.3 AI-Trader — 23,116★, no licence file, agent-native via `SKILL.md`

**As implemented.** Not a trading engine — a **platform** for agents to trade against. 149 files:
`skills/{ai4trade,copytrade,tradesync,heartbeat,market-intel,polymarket}/SKILL.md`,
`service/server/` (FastAPI: `routes_{agent,trading,signals,challenges,experiments,research,market}.py`,
`challenge_scoring.py`, `experiment_metrics.py`, `fees.py`, `price_fetcher.py`),
`service/frontend/` (React/Vite), `research/schemas/` (**25 JSON schemas**: trades, positions,
signals, rewards, quality_scores, team_missions…) and `research/scripts/` (feature building, network
edges, metrics, figure generation).

**The onboarding mechanic is the interesting part.** An agent joins by reading
`https://ai4trade.ai/skill/ai4trade` — a `SKILL.md` with a front-matter `name`/`description`, a table
of skill-file URLs, and execution rules ("read this file first, then fetch specialised skills"). No
MCP, no SDK, no install: **the integration surface is a markdown file over HTTPS**. That is cheaper
than an MCP server and works with any agent that can fetch.

**Strengths.** (a) The `SKILL.md`-as-bootstrap pattern is the lowest-friction agent interface in this
survey. (b) Paper-first with explicit fee modelling (`fees.py`) and **30 typed schemas** so research
exports are analysable rather than scraped. (c) Honest operational evolution visible in the changelog:
"yfinance fallback for US stock prices" (2026-06-08) when Alpha Vantage was missing/rate-limited;
"FastAPI service now runs separately from background workers" (2026-04-10).

**Limitations.**
- **Governance hygiene:** README badge says MIT, but no `LICENSE` file exists at `main`, and the
  GitHub API reports no detected licence. For a platform that asks agents to register tokens, that
  ambiguity matters.
- No backtester: it scores *live paper* outcomes. That is a legitimate design, but it means nothing
  here can be reproduced offline, and challenge scores are exposed to survivorship in the
  participant pool.
- It is a venue, not a strategy. It does not answer "is this agent's edge real".

> **So what.** *Copy:* the "one markdown file is the integration contract" idea — our `doctor`
> command and JSONL journal should be readable by an agent as easily as by a human, and our MCP
> adapter should not be the only way in. *Avoid:* shipping a platform with no licence file and no
> reproducible evaluation path.

### 3.4 QuantDinger — 13,556★, Apache-2.0, 1,347 files

The most **production-shaped** system in the survey: 11 distinct container roles (`backend`,
`trading-worker`, `strategy-dispatcher-worker`, `strategy-evaluator-worker`, `kafka-audit-worker`,
`scheduler-worker`, `celery-{worker,backtest-worker,beat}`, `migration`, `kafka-init`), Kafka as the
event backbone, Postgres + two Redis instances with different eviction policies, leases/fencing
tokens for runtime ownership.

**Strengths — the statistics are unusually honest.**
- **Factor Lab** loads one point-in-time universe panel once, then benchmarks ≤40 factors with
  "time-ordered holdout, out-of-sample IC retention, **Benjamini-Hochberg false-discovery control**,
  cost-aware scoring, factor-life-cycle labels, cross-factor IC correlation".
- **Strategy evolution** with "walk-forward validation and a final blind holdout", plus "**PBO**,
  **Deflated Sharpe**, block-bootstrap Monte Carlo, and transaction-cost stress tests" — and the
  disclaimer that "results compare parameter robustness; they do not forecast future returns."
- **The JEV pre-trade gate is the best model of "LLM proposes, code disposes" in the field.** Entry
  signals pass *deterministic* risk/budget checks **first**; only then does an AI gate see a typed
  Decision Context V2; a typed result must clear a confidence threshold to reach the order queue.
  Crucially: "rejected entries never reach the exchange; **exits, stop-loss, take-profit, and
  emergency actions bypass the filter**", and if the provider fails the order is **allowed** and the
  fail-open is logged — "so an AI outage cannot trap an existing position."
- Agent privileges: trading token scope + `paper_only=false` + `AGENT_LIVE_TRADING_ENABLED=true` +
  operator limits/allowlists — four independent gates, and paper-only by default.

**Limitations.** Complexity is the cost: Kafka + Celery + 11 roles is not a 500-line project, and the
`README` is 49 KB. The AI gate is optional and off by default, so the "agentic" part is a bolt-on to
a conventional strategy engine rather than the spine. Solana/equities coverage is uneven.

> **So what.** This is the project to imitate for *risk containment ergonomics*: deterministic checks
> first, typed AI decision second, exits never blocked by AI, provider failure → logged fail-open on
> the entry path. We adopt the same ordering and the same asymmetry, minus the Kafka.

### 3.5 AutoHedge — 6,634★, MIT, 2,364 LOC

The cleanest illustration of the star/quality gap, so it earns a deep dive for the wrong reason.

**As implemented.** `autohedge/` = 15 Python files, 1,645 lines: `main.py 63`, `cli.py 230`,
`workers.py 93`, `prompts.py 202`, `tools/{exa_search_tool 104, jupiter_price 74, jupiter_search 67,
polygon_api 189, ultra_tools 282, yahoo_api 259, tools_registry 14}`. Of the 51 files in the repo,
**24 are `.github/` workflow and template YAML**. Four "agents" (Director, Quant, Risk, Execution)
are assembled from prompt strings in `prompts.py` + `workers.py`.

**Strengths.** It is a working end-to-end Solana loop (Jupiter quotes → thesis → size → swap) in
under 2k lines, and it is honestly scoped in code if not in README ("enterprise-grade autonomous
agent hedge fund", "institutional reliability").

**Limitations — all verifiable in the tree.**
- **No backtester.** `grep -ri backtest` hits only `experimental/market_making.py` (a 416-line
  example). Nothing in the shipped package tests a strategy against history.
- **Risk management is prompt text.** `workers.py`: *"Provide risk assessment including:
  1. Recommended position size 2. Maximum drawdown risk 3. Market risk exposure 4. Overall risk
  score"* — the model writes the size; no code clamps it.
- **No status/health story**: two committed CSV logs from March 2025 are the entire audit trail.
- Last push 2026-05-11: stale for six months while claiming autonomous live trading.

> **So what.** *Copy:* almost nothing. *Avoid:* the pattern of shipping prompts as architecture. The
> lesson is calibration: **stars are a popularity signal, not an engineering one**, and any
> "autonomous hedge fund" with no backtester and no deterministic sizing should be treated as a demo.

### 3.6 lumibot — 2,212★, GPL-3.0, 2,387 files

The control group: what a decade of iteration looks like when the hard part is treated as
engineering. No LLM in the core loop.

**As implemented.** `lumibot/backtesting/` with **12 data-source backtesters** (Yahoo, Polygon,
Alpaca, Alpha Vantage, Databento ×3, CCXT, ThetaData ×2, IBKR REST, Kalshi, Polymarket, ProjectX,
routed), `lumibot/brokers/` with 13 broker adapters, `entities/{order, position, bar, cash_event,
trading_fee, trading_slippage}.py`, `strategies/{strategy, strtategy_executor → strategy_executor,
session_manager, scheduled_timing}.py`, and — the detail worth stealing —
`backtesting/data_provenance.py` and `backtesting/market_input_evidence.py`: the backtest records
where its inputs came from. `example_strategies/` has ~20 "AI trading team" strategies, i.e. **LLM
decisions are a strategy plugged into a real execution model**, not the other way round.

**Strengths.** Same code path for backtest and live; fees and slippage are first-class entities;
data provenance is modelled; 423 forks and 6 years of issue history.

**Limitations.** GPL-3.0 (a real constraint for reuse), 494 MB repo, and the AI surface is examples
rather than an agent framework — no role decomposition, no debate, no memory, no research loop.

> **So what.** *Copy:* the inversion — **decide inside a real execution/portfolio model, rather than
> bolting execution onto a research pipeline** — plus `data_provenance` as an explicit artefact. This
> is precisely the inversion our `DecisionStrategy` seam makes possible: the backtest engine knows
> nothing about agents, so one strategy object runs in both paths.

### 3.7 LangAlpha — 2,244★, Apache-2.0, 5,927 files

The most sophisticated **harness** in the field, and the closest thing to a reference for our
execution layer.

**As implemented.** Agent runs in a sandbox (Daytona/E2B/Docker/SSH) with a workspace filesystem,
`agent.md`, memory, and MCP servers **turned into importable Python modules** ("programmatic tool
calling") so tables stay in code and only results enter context. `mcp_servers/` with a shared
envelope (`symbol, currency, timezone, count, data, source`, oldest-first, typed error codes).
`src/ptc_agent/agent/middleware/order_governance.py` + `src/server/app/orders.py` +
`src/server/database/{order_attempts,order_reconciliation}.py` + migrations
`042_order_approval_modes`, `043_order_attempts`, `086_order_attempts_handed_off`,
`094_order_attempts_expired`. Four brokerages (Robinhood, IBKR, moomoo, Webull).

**Strengths — the order path is the best-specified in the survey.**
- **Order tools are pinned to direct JSON calls and never exposed inside the sandbox** — "a sandbox
  script could place any number of orders in a single execution".
- **Consent per connection** (capability groups the relay refuses outside of, whatever the model
  asks), **approval per order** (live/staged by default, paper not), **exactly one execution**
  (short-lived token bound to attempt + tool + argument hash; a changed argument or a replay fails at
  the relay), **credentials never enter the sandbox**, and **a ledger** ("every attempt, approval,
  rejection and fill lands on the Orders page, and a reconciler settles orders against the broker's
  own records").
- **Data provenance layer** outside the model's context: every source touched gets provider,
  redacted arguments, timestamp, result fingerprint, up to 64 KB of result, tied to turn and agent.
- Leak redaction of secrets in every tool result before it reaches the model.

**Limitations.** Requires an LLM and a sandbox stack (Docker minimum); research-side backtesting is
ad-hoc Python written by the agent, so a number is as reproducible as the code the model happened to
write; hosted-vs-self-host feature gaps (options chains, live quotes) are significant.

> **So what.** *Copy:* the four ideas that matter — order tools never reachable from model-written
> code, an approval token bound to a hash of the arguments (we get this for free by making
> `ExecutionEngine` the only caller of `broker.submit_order`), reconcile-by-identity instead of
> blind retry, and a provenance record per data source. *Avoid:* letting the agent author the
> backtest it then reports on.

### 3.8 ALTA — 1,075★, Apache-2.0, 6,338 files (short)

The strongest **epistemics** in the field: "**Research is open-ended; authority is explicit.** LLMs
choose what to investigate and propose how to use it. Code enforces freshness, risk, account
isolation and durable transitions. A persuasive thesis cannot approve its own execution."
Assessment and challenge are written to **private, locked views before either can see the other**;
ranking is "software, not an extra Agent voting for its favorite"; `Wait` is a first-class outcome that
must record *what evidence was missing*; "source pacing and recorded partial results keep research
inspectable"; "a stale proposal is not permission to chase"; "reconcile the persisted identity with
the broker instead of submitting another order blindly". One review records **916 passing tests**.
Separate `alta-runtime/capital-python` and `alta-runtime/broker-python` processes, i.e. dollars are
isolated from research code — and it vendors a **pinned** `openai-codex` harness rather than
reimplementing an agent loop.

**Limitations.** Heavy (Node gateway + Python services + Postgres + Redis + Codex App Server);
requires an LLM; no classic backtester — evaluation is shadow-mode forward.

> **So what.** *Copy:* the vocabulary of `Wait`, "code-based ranking", "locked independent views",
> and separate authority for execution. In our terms: HOLD is a real decision with a reason, the risk
> gate is not a voter, and execution authority lives in one place.

### 3.9 `Open-Finance-Lab/AgenticTrading` — 884★, OpenMDW-1.0 — **name collision**

Same name as this project, different project. It is the "Agentic Trading Lab": `dashboard/backend`
(FastAPI, `api/v2`, `domain/`, `execution/` with "backtest live; paper stub",
`infrastructure/` with "LLM validator, market data, Alpaca + Robinhood brokers", a Discord bot),
`packaging/agentictrading/` Python SDK, `orchestration/` = the FinAgent framework (published at
NeurIPS 2025 Workshop: *Orchestration Framework for Financial Agents: From Algorithmic Trading to
Agentic Trading*). It hosts a curated leaderboard that advances nightly on simulated fills, and it
ships a **TradingAgents adapter** so that project's decisions replay through its SDK. Licence is
OpenMDW-1.0 (hence `NOASSERTION` from the GitHub API), created 2025-05-21, 131 open issues.

> **So what.** *Avoid:* naming collisions in a crowded namespace — do not name-drop "AgenticTrading"
> in a PR, package or README without qualifiers, and check PyPI before publishing
> (`agentic-trading` may be contested **[unverified — PyPI not queried]**). *Copy:* its leaderboard
> discipline — a nightly forward run on simulated fills, with the roster curated so nobody can
> submit-selection-game the board.

### 3.10 Secondary evidence (checked, no deep dive)

- **FinMem (970★, MIT, last push 2024-08-18).** The memory design the field cites: `puppy/` =
  `agent.py`, `memorydb.py`, `reflection.py`, `portfolio.py`, `prompts.py`, and
  `memory_functions/{decay, recency, importance_score, compound_score, access_counter}.py` — layered
  memory with a decayed compound retrieval score. Stale codebase, live idea: our journal should be
  equally *retrievable* (importance × recency), not just append-only.
- **Trading-R1 (549★, Apache-2.0).** Ships a terminal **harness** — `trading_r1/runtime/{agent,
  memory, skill_registry, tool_registry, tool_events}.py`, `trading_r1/tools/{compute,core,data,web}`,
  **17 `SKILL.md`** files — but **no model weights in the tree** and it wants `TAURIC_API_KEY`. The
  arXiv paper it cites (2509.11420, *Trading-R1: Financial Trading with LLM Reasoning via
  Reinforcement Learning*) is about RL training; the repo is not that training run. Treat "R1" as a
  research terminal, not an open model.
- **ContestTrade (680★, Apache-2.0).** The most interesting *aggregation* idea after debate:
  two internal **contests**. Data agents compete to produce "text factors"; a factor portfolio is
  built from winners; research agents with distinct "trading beliefs" compete on proposals; a second
  contest synthesises an allocation. Competition as the selection mechanism instead of a manager LLM.
  Its weakness is the same as its strength — the judges are also LLMs.
- **nof0 (2,756★, Go + Next.js, no licence file).** Open reproduction of NOF1's Alpha Arena, and the
  field's honest alternative to backtesting: run N models forward on real fills and let the
  leaderboard decide. `mcp/data/analytics-{gpt-5,claude-sonnet-4-5,deepseek-chat-v3.1,gemini-2.5-pro,
  grok-4,qwen3-max}.json` are the per-model artefacts; `go/cmd/journalreplay/main.go` is a rejected-
  vs-executed replay tool. Its own README scores progress honestly (frontend 100%, backend 70%,
  engine 80%).
- **Polymarket/agents (3,810★, MIT).** 37 files, `agents/{application/{creator,executor,trade,
  prompts,cron},connectors/{chroma,news,search},polymarket/{gamma,polymarket}}`. A clean, small RAG
  loop — and abandoned since **2024-11-05**, i.e. it has been dead for the entire period in which
  this field grew. Stars are a lagging indicator.
- **TradeMemory Protocol (1,440★, MIT, 20 MCP tools).** A memory/brake MCP server: sync fills, find
  where your own history loses money, **refuse an order that breaks rules you set, before it reaches
  the broker**. Also the source of the survey's best second-hand claim on broker guardrails: Webull
  and tastytrade set size/buying-power limits, IBKR only lets an agent draft an order, **Robinhood and
  Public set no cap you can impose on an external agent** (citing StockBrokers, 2026-09-30)
  **[second-hand — not verified against the primary source]**.
- **lumibot's GPL-3.0** and **QuantDinger/ALTA/LangAlpha's Apache-2.0** are the licence reality of
  this space: Apache/MIT dominates, GPL is the exception, and "no licence file" is common enough
  (AI-Trader, nof0) to be a genuine hazard when copying code.

> **So what.** From this group we take exactly three things: FinMem's *retrievable* memory (a journal
> you can rank, not just append), TradeMemory's *refusal* framing (the memory's job includes saying
> "you have lost money this way before"), and nof0's forward-run honesty (if a number cannot be
> produced reproducibly, run forward and publish the losing streak too). We take nothing from them
> architecturally — none of them resolves the four gaps in §1.

---

## 4. Cross-cutting patterns

### 4.1 Role decomposition is the field's shared grammar

Analysts → debate → manager → risk → portfolio. TradingAgents uses 11 agents; AutoHedge 4; ALTA 4
scouts + assessor + challenger + moderator + expression + auditor. The interesting variation is not
*how many roles* but **who holds authority**: TradingAgents and AutoHedge put it in a prompt;
QuantDinger, LangAlpha and ALTA put it in code. The second group is the only one where a persuasive
argument cannot become an order.

### 4.2 Debate as signal aggregation — and its unmeasured cost

Bull/bear rounds (TradingAgents), 3-way risk debate (aggressive/conservative/neutral), ALTA's locked
independent assessments, ContestTrade's two contests, Vibe-Trading's Swarm "investment committee".
**No project in this survey publishes an ablation showing the debate beats a simple aggregation of
the same analyst signals.** TradingAgents' own README warns results vary by model and temperature and
"are not guaranteed to match any published figure", which makes such an ablation nearly impossible to
run from outside. Debate is architecture that *feels* like reasoning; until someone shows
debate > weighted-sum of stances at equal token budget, it is an unproven cost. Our design keeps the
debate as a pure function over stances (`debate.py`) precisely so it *can* be ablated against a
zero-debate baseline.

### 4.3 Memory layers

Four distinct kinds, often confused:
- **Episodic with settlement** (TradingAgents' memory log): each decision is later scored against
  realised return/alpha and reflected on; the PM reads it next time. Best-in-class, and the only one
  with a real feedback loop.
- **Layered decay** (FinMem): importance/recency/access-count/compound score over a memory DB.
  Elegant, unreproducible today (stale repo, offline dataset).
- **Durable workspace** (LangAlpha, ALTA): `agent.md`, memos, transcripts, per-workspace memory as
  files on a FUSE mount. This is memory as *working state*, not as learning.
- **Brake** (TradeMemory): memory whose purpose is refusal — put your own losing trades in front of
  the agent before it orders again.

### 4.4 Reflection / self-improvement

Present as (a) settled-decision reflection (TradingAgents), (b) per-agent "Trader Mind" lessons
(ALTA), (c) "skills" the agent writes and reloads (Vibe-Trading's 90 `SKILL.md`, LangAlpha's
user skills, Trading-R1's skill registry), (d) parameter search over strategies (QuantDinger's
TPE/grid/random + PBO). Only (a) and (d) have an outcome signal attached; (c) is self-improvement
without a scoreboard and should be treated as prompt engineering with a filesystem.

### 4.5 **MCP / agent-native interfaces — the cross-cutting shift** (deserves its own subsection)

This is the one genuinely new axis since the TradingAgents paper, and it cuts across every project.

**The venue moved first.** Robinhood's newsroom post *"Robinhood is Now Open to Agents"* is dated
**2026-05-27** and states customers can "connect them to Robinhood's AI-native Model Context Protocol
(MCP) servers". Robinhood's own support docs give the literal command:
`claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading`, and the
same URL for Claude Desktop and ChatGPT. Terms of the beta, verified from those pages: **equities
only**; trades only inside a **separate Agentic Trading account**; the agent gets **read access to all
your accounts, positions, balances and order history**; you may opt into **manual approvals** and
agents "will preview orders with customers whenever they ask"; disconnect is one tap.
([newsroom](https://robinhood.com/newsroom/robinhood-is-now-open-to-agents/),
[support: onboarding an external agent](https://robinhood.com/us/en/support/articles/onboarding-an-external-agent/))

**The projects followed, unevenly:**

| Project | Interface | Evidence |
|---|---|---|
| Vibe-Trading | **Real MCP server over stdio** (`vibe-trading-mcp`), plus HTTP API | `agent/mcp_server.py`; `README §MCP Plugin`; **`agent/tests/robinhood_mcp_helpers.py`** — it explicitly targets Robinhood's MCP |
| QuantDinger | MCP server + Agent Gateway `/api/agent/v1` | `mcp_server/`; four independent gates for live agent trading |
| LangAlpha | MCP servers as importable Python modules; order tools excluded from that path | `mcp_servers/`, `order_governance.py` |
| Trading-R1 | Terminal + 17 skills; `tauric` as a model provider | `trading_r1/skills/*/SKILL.md` |
| AI-Trader | **Not MCP**: `SKILL.md` bootstrap over HTTPS | `skills/*/SKILL.md` |
| ALTA | MCP through `vendor/openai-codex` harness | `vendor/openai-codex/` |
| TradingAgents | LangGraph only; MCP appears as a **data vendor** (Polymarket) not an interface | `dataflows/vendors/polymarket.py` |
| awesome list | **35 MCP servers** catalogued in a list created 2026-04-22 (17 market-data, 9 brokerage/exchange, 5 research, 2 prediction-market, 1 TradingView, 1 backtesting) | `README §MCPs` |

Two consequences for us. (1) **MCP is not a differentiator any more** — a 981-star curated list
already catalogues 35 servers, and the broker side is now the venue's own product. The differentiator
is what you *do* with a tool call: our `ExecutionEngine` is the only caller of `broker.submit_order`
and it applies seven deterministic gates before the wire. (2) **Tool-name drift is the failure mode.**
Robinhood's tool names are discovered at runtime, not specified; our adapter maps by pattern, exposes
`tool_map` for override, and degrades to `OrderStatus.UNKNOWN` when a schema changes instead of
sending a malformed order. That design was chosen *because* the field's MCP integrations are all
pattern-matching against an evolving surface.

### 4.6 Execution venue integration

Spectrum, observed: **simulation only** (TradingAgents — explicitly *no* execution model),
**paper/arena** (nof0, Polymarket/agents, AI-Trader), **venue MCP** (Vibe-Trading 19 brokers, LangAlpha
4 brokerages incl. Robinhood, QuantDinger Binance/OKX/IBKR/Alpaca, lumibot 13 adapters),
**isolation** (ALTA's separate capital/broker processes; LangAlpha's egress relay that attaches
credentials on the host so sandbox code never sees them). The mature answer is that **the execution
path is a separate program with its own authority and its own ledger** — not a function the research
loop can call.

---

## 5. Honest weaknesses of the state of the art

### 5.1 LLM cost and latency per bar
TradingAgents' topology implies **12 model calls per decision** at default debate settings (4
analysts + 2 debate + 1 research manager + 1 trader + 3 risk + 1 PM), 14 at `max_debate_rounds=2`,
before counting each analyst's tool rounds. Vibe-Trading surfaces token usage in Run Detail; LangAlpha
exposes `max_timeout` per plan (Hobby 3600 s). Nobody in this survey publishes a **cost per decision
in dollars**, and nobody compares 12 calls against 1 call on the same signal. Treat per-bar agentic
inference as ~an order of magnitude more expensive than a rule, with no public measurement of the
benefit **[unverified: no project publishes $/decision]**.

### 5.2 Lookahead leakage is not hypothetical — it is the field's dominant bug class
Evidence, all from project-owned artefacts:
- TradingAgents shipped **point-in-time fixes in v0.3.1 (Alpha Vantage), v0.4.0 (FRED, social
  sentiment, decision-log memory), v0.5.0 (all dated paths, SEC as filed), v0.5.2 ("backtests that
  see only data published by each analysis date"), v0.7.1 ("a fix for model dates that could read
  filings made after the run date")** — five releases over four months.
- Vibe-Trading: *"excess return no longer includes that move (AAPL 20/60 SMA example: **22.5% →
  18.4%**)"* — 4.1 pp of phantom alpha from mis-timing the benchmark; plus "Carhart momentum no
  longer leaking today's price (#1578)", "walk-forward ML training without future-label leakage
  (#1392)", "adjusted and raw prices never mixed in one series (#1684)".
- TradingAgents' `backtest.py` enters "at the **first close after** its date, because a run reads
  news published **after** that day's close" — a correction that silently changes every number.
- The most sophisticated acknowledgement is a *product feature*: Vibe-Trading's "run cards that flag
  backtests possibly inside the model's training data".

### 5.3 "Risk management" without a risk manager
Three distinct anti-patterns, all observed:
1. **Risk as persona** — TradingAgents' risk debate is three LLM prompts; the only arithmetic in the
   entire tree touching sizing is 3 lines in `agents/schemas.py`, and `position_sizing` is a
   free-text string.
2. **Risk as prompt output** — AutoHedge asks the model for "recommended position size" and "maximum
   drawdown risk" in `workers.py`; nothing clamps the answer.
3. **Risk as a separate service you must opt into** — QuantDinger's JEV gate is well-designed but
   optional and off by default, so the default configuration is unprotected.
Nothing in the survey combines (a) deterministic sizing, (b) deterministic portfolio limits and
(c) an auditable record of *why* a size was refused. Our `RiskDecision` type exists to make (a)–(c)
the default.

### 5.4 Unvalidated alpha claims
- AutoHedge: no backtester, yet "autonomous hedge fund".
- Polymarket/agents: dead for two years, still 3,810 stars.
- TradingAgents: scores alpha vs a regional benchmark (good) but explicitly does not model fills,
  cash, or portfolio (so the number is decision quality, not a strategy return) and warns
  "backtest results are not guaranteed to match any published figure".
- Vibe-Trading and QuantDinger do the statistics properly — walk-forward, blind holdout, PBO,
  Deflated Sharpe, BH-FDR — and both then say results "do not forecast future returns" / compare
  robustness. That is the honest end state: **the field's mature projects have stopped claiming alpha
  and started reporting robustness.** Any project still claiming returns is behind on epistemics.

### 5.5 Non-reproducibility
TradingAgents publishes the admission: "two runs of the same ticker and date can differ … reasoning
models … largely ignore temperature." Vibe-Trading's 116-PR release includes nan-propagation and
formula corrections across 16 alphas, i.e. a number computed in October is not the number computed in
November. AI-Trader and nof0 are live-only. **No repo in this survey offers a frozen dataset + frozen
code + one-command reproduction of a reported figure.** A JSONL journal with a replay command is
cheap and nobody has it except nof0 (`go/cmd/journalreplay`) and LangAlpha's order-attempts ledger.

### 5.6 Vendor API fragility
The industry's real response to this is the fork layer (TradingAgents-CN 32,815★ + A-share forks
~4.8k★) and the loader layer (Vibe-Trading: 46 loader modules, per-market fallback chains,
`loader_health.py`, redacted credential reporting; AI-Trader adding a yfinance fallback in June 2026;
TradingAgents' `VendorNotConfiguredError` / `VendorUnavailableError` / `NoMarketDataError` hierarchy).
**Single-vendor projects die**; the survey's two dead repos (Polymarket/agents, FinMem) are exactly
the ones whose data path was bespoke and is now unmaintained.

### 5.7 The gap nobody fills: keyless, offline, reproducible **by default**

Every one of the top six by stars fails to produce a decision without a paid key. That means:
no clean-room reproduction, no CI without secrets, no measurable LLM contribution, and no way for a
reviewer to separate "the model is good" from "the vendor was up". This is the specific hole
AgenticTrading is built into.

---

## 6. Design implications → the decisions this repo makes

Cross-reference: [`ARCHITECTURE.md`](ARCHITECTURE.md) §1 (two seams), §2 (leakage guarantee),
§3 ("the model proposes, the code disposes"), §4 (LLM-optional), §5 (execution and the Robinhood
path), §6 (module map), §8 (known limitations).

Module paths in the "Where it lives" column are the ones specified in ARCHITECTURE §6. Where a path
is not yet present in the working tree, it is the intended location, not a claim that the file
already exists.

| Landscape failure (evidence in §5) | Decision here | Where it lives |
|---|---|---|
| **§5.7** every top repo needs a paid key; nothing runs offline | **Keyless multi-provider chain + always-available synthetic fallback.** `configs/default.yaml` ships `provider_order: ["coinbase", "twelvedata", "yahoo", "synthetic"]` — Coinbase keyless crypto, TwelveData (key), Yahoo (graceful on 429), then a deterministic offline GBM. A provider raises `DataUnavailableError`; the chain falls through; the run never dies for lack of a key. `--offline` reorders synthetic to the front (`cli.py`). | `agentic_trading/data/{base,service,synthetic}.py`, `configs/default.yaml`; ARCHITECTURE §1 Seam A, §6 |
| **§5.6** vendor fragility; every mature project grows its own loader fan-out | **One `DataProvider` protocol, not 46 loaders.** Normalised `list[Bar]`: ascending, tz-aware UTC, never a bar whose open is after `end`, never a forming bar. Adding a vendor is one class + one name in `provider_order`. | ARCHITECTURE §1, §7 |
| **§5.1** 12–14 model calls per decision; cost/latency unbounded | **Deterministic-first analysts, LLM optional and bounded.** Each analyst computes a *complete* stance; if `LLMConfig.resolved_enabled()` is true, **one compact JSON call** may refine label/confidence/rationale and the stance is marked `source="llm"`; on any error/timeout/malformed JSON the deterministic stance stands. `AGENTIC_TRADING_LLM=off` gives a byte-identical, network-free run to diff against. | `agents/analysts.py`, `agents/llm.py` (incl. offline Null client); ARCHITECTURE §4 |
| **§5.3** risk as persona / as prompt output | **"LLM proposes, deterministic code disposes" as a type-level separation.** `Stance` (opinion) → `DebateResult` (aggregated opinion) → `TradeProposal` (request; the only type an LLM may influence) → **`RiskDecision` — constructible only by `agents/risk.py`, pure arithmetic** → `PortfolioDecision` (outcome). The gate enforces `min_confidence`, `max_position_weight` (clamp, always last), `max_drawdown_stop`, `stop_loss_pct`/`take_profit_pct`, `max_daily_turnover`, `cooldown_bars_after_stop`, and rejects negative weights (`allow_short: false`) rather than reinterpreting them. Modelled on QuantDinger's ordering (deterministic checks first, typed AI decision second) and on QuantDinger's asymmetry: **exits and stops are never blocked by an LLM**. | `agents/risk.py`, `contracts.py`; ARCHITECTURE §3 |
| **§5.2** leakage is the dominant bug class; five TradingAgents releases and a 22.5%→18.4% correction | **Leakage-free single feature frame.** `feature_frame(bars)` is computed **once** for the whole series with causal indicator definitions only (`min_periods` guards, no negative shifts, no centred windows); the engine *slices* it — at bar `i` the strategy gets `features = latest_features(frame, i)` and `bars = bars[:i+1]`. A recording strategy test asserts it sees exactly `k` bars at step `k`, newest = decision bar; a separate test asserts the fast path equals recomputing indicators on each prefix. Execution assumptions are explicit (`fee_bps`, `slippage_bps`, `warmup_bars`, fills at the decision bar's close). | `data/indicators.py`, `backtest/engine.py`; ARCHITECTURE §2 |
| **§5.5** nothing reproduces, nothing replays | **Append-only JSONL decision journal, replayable.** Every proposal, veto, resize and order status lands as one line, so a decision can be re-read and replayed without the LLM present. Inspired by nof0's `journalreplay`, LangAlpha's order-attempt ledger, and TradeMemory's refusals-as-records. | `brokers/audit.py`; ARCHITECTURE §6 |
| **§5.4** alpha claims nobody can check | **Metrics vs buy-and-hold, side by side, no headline claim.** `backtest/metrics.py` emits `total_return`, `cagr`, `sharpe`, `sortino`, `max_drawdown`, `turnover`, `exposure`, `risk_vetoes` next to `benchmark_total_return`, `benchmark_cagr`, `benchmark_sharpe`, `benchmark_max_drawdown`, plus `excess_return` and `alpha_cagr` — and §8 of ARCHITECTURE states plainly that the deterministic analysts are "a credible baseline, not a claim of alpha". | `backtest/metrics.py`, `backtest/report.py`; ARCHITECTURE §8 |
| **§5.3 + §4.5** the order path is where agents lose money; MCP tool names drift | **Deterministic blast-radius limits in front of an MCP broker.** `ExecutionEngine` is the only caller of `broker.submit_order`, and it applies, in order: kill-switch file → symbol allowlist → `dry_run`/PREVIEW (broker never called) → `max_order_notional` → `max_daily_orders`/`max_daily_notional` → no-short → minimum size. Live requires **all** of `execution.enabled`, `mode="live"`, `dry_run=false`, no kill-switch, and `AGENTIC_TRADING_LIVE` set to an exact confirmation phrase; `why_not_live()` explains every failed gate (what `doctor` prints). The MCP adapter discovers tools at runtime, maps by pattern, exposes `tool_map`, and degrades to `OrderStatus.UNKNOWN` on schema change. Mirrors Robinhood's own posture (separate account, manual approvals, previews). | `brokers/{mcp_client,robinhood_mcp,execution,paper}.py`; ARCHITECTURE §5 |
| **§5.1 + §3.1** debate is an unmeasured cost | **Debate as a pure function over stances** (`debate.py`), so "no debate" is one config value away and an ablation is a test, not a rewrite. | `agents/debate.py`; ARCHITECTURE §4, §7 |

Deliberate non-goals, chosen from the landscape's failure modes rather than from ambition: no 19
brokers, no 28 data sources, no 90 skills, no Kafka, no sandbox, no 11th agent. Every one of those is
something a mature project earned by surviving its own bug reports; starting there would import the
maintenance load without the evidence base.

---

## 7. Reproducibility appendix

**Retrieval date: 2026-11-05** (machine clock: `Fri Nov 6 06:54 UTC 2026` = 2026-11-05 22:54 Pacific).
All HTTP calls unauthenticated (`curl -s -m 25 ...`), so core-API budget was 60/hr and search 10/min;
calls were spaced ~1 s apart and metadata was fetched once per repo. No `GITHUB_TOKEN` was used, so
every number here is reproducible by anyone, subject to the counts moving after the date.

### 7.1 Star/fork/issue counts (per repo, 1 call each)

```
GET https://api.github.com/repos/{owner}/{repo}
```
for `TauricResearch/TradingAgents`, `HKUDS/Vibe-Trading`, `HKUDS/AI-Trader`,
`OpenByteInc/QuantDinger`, `The-Swarm-Corporation/AutoHedge`, `Lumiwealth/lumibot`,
`ginlix-ai/LangAlpha`, `pipiku915/FinMem-LLM-StockTrading`, `kyky2347/ALTA`,
`TauricResearch/Trading-R1`, `LLMQuant/awesome-trading-agents`, `FinStep-AI/ContestTrade`,
`wquguru/nof0`, `Polymarket/agents`, `Open-Finance-Lab/AgenticTrading`,
`hsliuping/TradingAgents-CN`, `mnemox-ai/tradememory-protocol`.
Fields used: `stargazers_count`, `forks_count`, `open_issues_count`, `language`, `license.spdx_id`,
`created_at`, `pushed_at`, `size`, `default_branch`. **Output:
[`research/github_snapshot_2026-11-05.csv`](research/github_snapshot_2026-11-05.csv).**

### 7.2 Rankings and the wider field (2 search calls)

```
GET https://api.github.com/search/repositories?q=trading+agent&sort=stars&order=desc&per_page=100
    -> total_count 17,339
GET https://api.github.com/search/repositories?q=llm+trading+agent+in:name,description&sort=stars&order=desc&per_page=60
    -> total_count 1,287
```
**Output of the first: [`research/search_top40_2026-11-05.csv`](research/search_top40_2026-11-05.csv).**

### 7.3 Source layout (1 call per repo)

```
GET https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1
```
for TradingAgents (236 entries), Vibe-Trading (3,563), AI-Trader (173), QuantDinger (1,347),
AutoHedge (58), LangAlpha (6,509), FinMem (75), ALTA (7,122), ContestTrade (134), Trading-R1 (396),
nof0 (485), Polymarket/agents (51), and `.../git/trees/dev?recursive=1` for lumibot (2,387).
None were truncated. File paths quoted in §3 come from these trees.

### 7.4 READMEs and source files

```
GET https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md
```
(tried `main`, then `master`, then `dev`; 15 primary READMEs retrieved, plus
`mnemox-ai/tradememory-protocol`, whose default branch is `master`).
Individual source files quoted:
- `tradingagents/graph/{setup.py,conditional_logic.py}`, `tradingagents/agents/state.py`,
  `tradingagents/agents/schemas.py`, `tradingagents/backtest.py`,
  `tradingagents/dataflows/router.py`, `tradingagents/default_config.py`
- `HKUDS/AI-Trader` `skills/ai4trade/SKILL.md`
- `mnemox-ai/tradememory-protocol` `README.md`

### 7.5 Line counts (2 tarballs, no API cost)

```
curl -sL https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/main | tar xz -C src
find src/<repo>-main -name '*.py' | xargs wc -l
```
Results: TradingAgents **28,791** LOC Python total (`agents/` 2,453 · `dataflows/` 3,874 ·
`llm_clients/` 1,824 · `graph/` 950 · memory 629), 90 files in `tests/`; AutoHedge **2,364** LOC
total (1,645 in `autohedge/`). Keyword greps for deterministic risk code:
`grep -rniE 'max_position|position_size|kelly|stop_loss|drawdown|turnover|risk_limit|notional'` →
TradingAgents **3 hits, all in `agents/schemas.py`**; AutoHedge `backtest` hits only in
`experimental/market_making.py`.

### 7.6 Venue claim (Robinhood MCP)

- `web_search` → `web_fetch https://robinhood.com/newsroom/robinhood-is-now-open-to-agents/`
  — date **May 27, 2026**, "Robinhood's AI-native Model Context Protocol (MCP) servers",
  Agentic Trading launches **in beta with equities only**.
- `web_fetch https://robinhood.com/us/en/support/articles/onboarding-an-external-agent/`
  — `claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading`;
  read access to all accounts/positions/transactions; orders only in the Agentic account; manual
  approval and order preview described.

**Could not verify as claimed.** The Lead's brief said Robinhood shipped agentic trading "in May
2026" — confirmed, exact date **2026-05-27**. I attempted a live MCP `initialize` handshake against
`https://agent.robinhood.com/mcp/trading`; **the harness rejected that command before execution, so
the endpoint is verified from Robinhood's documentation only, not from a live handshake.**

### 7.7 Things this report could not verify (consolidated)

1. **Any claimed trading performance, anywhere.** No repo publishes a dataset+commit+command that
   reproduces a stated return. All performance references above are *claims by the project* or
   corrections *the project itself* reported.
2. **Robinhood MCP live behaviour** — documented only (see 7.6); no authenticated token in this
   environment.
3. **Broker guardrail comparison** (Webull/tastytrade caps, IBKR draft-only, "Robinhood and Public
   set no cap") — taken from TradeMemory's README citing StockBrokers 2026-09-30; **primary source
   not fetched**.
4. **`hsliuping/TradingAgents-CN`, `simonlin1212/TradingAgents-astock`, `wquguru/nof0`** — metadata
   and (for nof0) README read; their internal architecture was **not** inspected beyond file trees,
   so nothing is asserted about their implementation quality.
5. **Licence of `AI-Trader` and `nof0`** — no `LICENSE` file in the trees at `main` and no SPDX id
   from the API, while AI-Trader's README badge says MIT. Recorded as ambiguous, not as MIT.
6. **PyPI naming availability** for `agentic-trading` — not queried.
7. **Cost per LLM decision in dollars** — no project publishes it; the 12–14 calls figure is derived
   from the graph topology in `conditional_logic.py`, not measured.

### 7.8 Artefacts produced

| File | Contents |
|---|---|
| `docs/RESEARCH.md` | This report |
| `docs/research/github_snapshot_2026-11-05.csv` | 17 repos × 11 observed fields (stars, forks, issues, licence, dates, size, branch, role) |
| `docs/research/search_top40_2026-11-05.csv` | Raw top-40 of `q=trading+agent&sort=stars`, unedited |

**References.** Xiao, Sun, Luo, Wang — *"TradingAgents: Multi-Agents LLM Financial Trading
Framework"*, arXiv:2412.20138 (q-fin.TR), 2025. Xiao et al. — *"Trading-R1: Financial Trading with
LLM Reasoning via Reinforcement Learning"*, arXiv:2509.11420, 2025. Li, Grover, Alpuerto, Cao, Liu —
*"Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading"*,
NeurIPS 2025 Workshop on Generative AI in Finance. FinMem — Yu et al., arXiv:2311.13743.
