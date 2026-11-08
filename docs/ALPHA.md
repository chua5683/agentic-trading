# How to turn AgenticTrading into an alpha engine

The uncomfortable premise: **the agent architecture is not what is stopping this
from making money.** The system already has auditable decisions, a deterministic
risk authority, and a leakage-free backtester. What it does not have is
*information*. Better orchestration of a weak signal produces a well-documented
weak signal.

Evidence for that claim is in this repo: across 10 fixed windows the current
4-analyst deterministic policy beats buy & hold 5 times and loses 5, mean excess
return ≈ −1.3%, and ablating the analysts showed they *dilute* the one
marginally useful trend view instead of improving it.

So "alpha engine" is four sequential problems, in order of leverage.

---

## 1. Change where the numbers come from (biggest lever)

Single-asset, daily-bar, OHLCV-only direction prediction is the most crowded,
least informative corner of the market. A language model reading the same 200
closing prices as everyone else has no informational advantage.

Alpha comes from **relative** information and from **non-price** information:

| Signal family | Why it works | Difficulty here |
|---|---|---|
| Cross-sectional momentum | ranks assets against each other; a much easier problem than timing one | **low** — needs a multi-asset engine |
| Funding-rate / basis carry | structural premium paid by leveraged longs; documented and persistent | medium — needs perp data |
| Volatility risk premium | implied > realised on average | high — needs options venue |
| Event / news extraction | where an LLM genuinely has an edge: structured events at scale | medium — needs a news feed |
| On-chain flows, exchange netflow | crypto-specific, not in price yet | medium — needs new vendors |

### Measured: cross-sectional momentum

`research/cross_sectional_momentum.py` ranks ~33 Coinbase USD pairs by trailing
return, holds the top K equal-weighted, rebalances weekly, 20bps/side.
2022-06-01 → 2026-11-07, in-sample / out-of-sample split at 2024-11-01.

| strategy | IS CAGR | IS Sharpe | OOS CAGR | OOS Sharpe | OOS maxDD |
|---|---:|---:|---:|---:|---:|
| BTC buy & hold | 23% | 0.40 | 4% | 0.01 | 53% |
| equal-weight universe | 37% | 0.48 | 7% | 0.04 | 71% |
| best IS config (k=5, 30d lookback) | 67% | 0.83 | 39% | **0.43** | 74% |
| **k=3, 180d lookback** | 51% | 0.61 | 70% | **0.78** | 56% |
| **k=5, 180d lookback** | 50% | 0.66 | 59% | **0.71** | 59% |

Three conclusions that matter more than the returns:

1. **Signal source dwarfs architecture.** Moving from single-asset timing
   (OOS Sharpe 0.01) to cross-sectional ranking (OOS 0.71) is a far bigger jump
   than anything achievable by improving the agents.
2. **Only the slow signal generalises.** The best in-sample config (30-day
   lookback, Sharpe 0.83) decays to 0.43 out-of-sample — textbook multiple-testing
   decay. The 180-day lookback was *less* impressive in-sample and *better* OOS.
   If you tune to the best backtest you will ship the 30-day version and lose.
3. **It is still mostly beta.** Regressed on the equal-weight universe the
   strategy has beta 0.98, correlation 0.86, and annualised alpha +27%. It is
   "better alt selection", not market-neutral alpha, and it carries a 56–74%
   drawdown.

> **Survivorship bias warning.** The universe is *today's* listed Coinbase
> products. Delisted and rug-pulled assets are missing, which flatters a
> long-only alt strategy. Treat these numbers as an upper bound and the ranking
> as directional, not as a backtest you can trade.

## 2. Fix the portfolio construction

The current engine is single-symbol, long-only, vol-targeted. Cross-sectional
strategies need:

- a **multi-asset backtest engine** (`PortfolioState` already models many
  positions; `BacktestEngine` does not yet replay a universe);
- **cluster/correlation caps** — crypto assets are highly correlated, so "top 5"
  can be one bet; cap exposure per cluster;
- **risk-parity / inverse-vol weighting** instead of equal weight, so a 200%-vol
  microcap does not dominate;
- **turnover budget** — weekly rebalancing at 20bps is optimistic at size;
- **beta hedging** if you want the alpha to be market-neutral rather than
  a geared version of the market.

## 3. Give the LLM a job it can actually win

Do not ask a model to predict price from OHLCV. Use it where it has an
information advantage that classical code does not:

- **extract structured events** from news, filings, exchange announcements,
  governance posts, unlocks, listings/delistings — `{asset, event_type, direction,
  magnitude, confidence, source, timestamp}`;
- **cross-reference at scale** — read 200 documents per rebalance and surface the
  3 that change a thesis;
- **maintain a thesis ledger** with explicit invalidation conditions, and score
  itself later (the codebase already journals decisions, so this is a natural
  extension);
- **propose hypotheses** to the research loop, not trades.

The right metric for the LLM layer is **information coefficient** (rank
correlation between its signal and forward returns), measured against the
deterministic baseline — not "does the backtest look better". A model that cannot
beat IC≈0 adds cost and latency for narrative.

## 4. Make the evaluation able to reject you

This is what separates an alpha engine from a compelling story. Add to
`evaluation.py`:

- **IC / IR analysis** per signal, per horizon, with t-stats;
- **purged / embargoed walk-forward** splits (no overlapping labels);
- **deflated Sharpe** (Bailey & López de Prado) or at least an explicit
  multiple-testing haircut — we tested 18 configs above, so the honest bar is
  higher than 1.0 Sharpe;
- **cost and capacity sensitivity** — show the strategy at 0, 10, 30, 60 bps;
- **bootstrap confidence interval** on Sharpe;
- **regime attribution** — the 10-window table already exists; extend it with
  factor attribution so "alpha" is not disguised beta;
- **shadow live trading** on paper with the real execution path before capital.

A candidate is only "alpha" when it survives all of the above *and* the
survivorship-corrected universe.

---

## Concrete 90-day sequence

| # | Step | Why first |
|---|---|---|
| 1 | Multi-asset backtest engine + cross-sectional weights | unlocks the whole signal family; ~1 week |
| 2 | Point-in-time universe incl. delisted assets | otherwise every number below is fiction |
| 3 | Slow momentum (90–180d) + inverse-vol weighting + cluster caps | the robust baseline from the table above |
| 4 | IC reporting + purged walk-forward + deflated Sharpe | makes results falsifiable |
| 5 | Add a second signal (funding carry, then events) | diversification is the only free lunch |
| 6 | Paper-trade the combination with `ExecutionEngine` | validates costs and operational risk |
| 7 | Only then: LLM event extraction, measured by IC uplift | LLM last — it must beat a real baseline |

Steps 1–2 are the ones that change the answer. Steps 5–7 are where the LLM earns
its place or does not.

---

## What this means for the current code

Nothing here invalidates the existing system — the risk gate, the leakage
discipline, the audit journal and the execution layer are exactly what you want
underneath a real signal. The work is additive:

- `BacktestEngine` → needs a universe-level replay (per-symbol replay exists and
  is reusable for the sleeve construction);
- `evaluation.py` → gains IC/IR and purged splits;
- `agents/` → stays; the analysts become one sleeve among several, not the whole
  system;
- `brokers/` → already sufficient for paper trading a multi-sleeve book.

## Honest bottom line

A realistic target is a **modest, robust premium** (mid-single-digit to low-double-digit
annualised excess over an appropriate benchmark with a survivorship-corrected
universe and real costs), delivered with controlled drawdown — not the 700%+
backtests that appear in this document, which are artefacts of survivorship bias
and a favourable alt cycle. Most "alpha" published in this space is one of:
lookahead, survivorship, costs ignored, or a single lucky window. The existing
evaluation harness exists specifically so those four can be ruled out.

*Research notes, not investment advice.*
