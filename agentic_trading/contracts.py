"""Frozen shared contracts for AgenticTrading.

This module is the *interface seam* between the data layer, the agent pipeline,
and the backtest engine.  Every other module imports its types from here; no
other module may redefine them.

Design rule that the whole project depends on:
    an LLM may *propose*, but only deterministic code may *dispose*.
`TradeProposal` is what an LLM is allowed to produce.  `RiskDecision` is produced
only by deterministic code.  `PortfolioDecision` is the final, auditable record.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

__all__ = [
    "Action",
    "Bar",
    "MarketContext",
    "Stance",
    "DebateResult",
    "TradeProposal",
    "RiskDecision",
    "PortfolioDecision",
    "Position",
    "PortfolioState",
    "Trade",
    "BacktestConfig",
    "BacktestResult",
    "METRIC_KEYS",
    "REQUIRED_FEATURES",
    "SUPPORTED_TIMEFRAMES",
    "DataProvider",
    "DecisionStrategy",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Quote",
    "OrderRequest",
    "OrderResult",
    "AccountSnapshot",
    "Broker",
]

SUPPORTED_TIMEFRAMES = ("15m", "1h", "1d")


class Action(str, Enum):
    """Discrete trade intent.  ``HOLD`` is a real, first-class outcome."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

    @classmethod
    def parse(cls, value: Any) -> "Action":
        """Lenient parse used by LLM output handling; raises ValueError if impossible."""
        if isinstance(value, Action):
            return value
        text = str(value).strip().lower()
        # str() on a str-mixin Enum yields "action.buy" on Python 3.11+.
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        aliases = {"long": "buy", "increase": "buy", "enter": "buy",
                   "short": "sell", "exit": "sell", "reduce": "sell",
                   "hold": "hold", "wait": "hold", "none": "hold", "noop": "hold"}
        text = aliases.get(text, text)
        return cls(text)


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV candle.  ``ts`` is the bar's *open* time, timezone-aware UTC.

    Only closed bars may ever reach an agent: the data layer is responsible for
    dropping the still-forming last bar before it becomes a `MarketContext`.
    """

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def day(self) -> date:
        return self.ts.date()

    def as_row(self) -> dict[str, Any]:
        return {"ts": self.ts, "open": self.open, "high": self.high,
                "low": self.low, "close": self.close, "volume": self.volume}


# Canonical feature names every provider-indicator implementation must emit.
# Values are floats computed *only* from bars with index <= asof (no lookahead).
REQUIRED_FEATURES: tuple[str, ...] = (
    "close", "ret_1", "ret_5", "ret_20", "ret_60",
    "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "atr_14", "atr_pct",
    "bb_upper", "bb_lower", "bb_pctb", "bb_width",
    "vol_20", "vol_60", "volume_z", "mom_20",
    "price_vs_sma50", "price_vs_sma200",
    "trend_up", "above_sma50", "above_sma200",
    "dist_from_high_60", "dist_from_low_60",
)


@dataclass(frozen=True, slots=True)
class MarketContext:
    """Everything an agent is allowed to know at decision time.

    ``bars`` is oldest -> newest and its final bar is the most recent *closed*
    bar.  Agents must not need data newer than this object.
    """

    symbol: str
    asof: datetime
    bars: tuple[Bar, ...]
    features: Mapping[str, float]
    timeframe: str = "1d"
    provider: str = "unknown"
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def last_close(self) -> float:
        return self.bars[-1].close

    @property
    def last_bar(self) -> Bar:
        return self.bars[-1]

    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    def window(self, n: int) -> tuple[Bar, ...]:
        return self.bars[-n:] if n > 0 else self.bars


# --------------------------------------------------------------------------- #
# Agent outputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Stance:
    """A single analyst's (or researcher's) opinion."""

    agent: str
    label: str            # "bullish" | "bearish" | "neutral"
    confidence: float     # 0..1
    rationale: str
    evidence: tuple[str, ...] = ()
    source: str = "deterministic"   # "deterministic" | "llm"

    def __post_init__(self) -> None:
        if self.label not in ("bullish", "bearish", "neutral"):
            raise ValueError(f"bad stance label: {self.label!r}")
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))

    @property
    def score(self) -> float:
        """Signed conviction in [-1, 1]; neutral collapses to 0."""
        sign = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}[self.label]
        return sign * self.confidence


@dataclass(frozen=True, slots=True)
class DebateResult:
    """Output of the bull/bear research debate."""

    bull: Stance
    bear: Stance
    rounds: int = 1
    consensus: str = "neutral"     # "bullish" | "bearish" | "neutral"
    net_score: float = 0.0         # in [-1, 1]

    @property
    def stance(self) -> Stance:
        return Stance(
            agent="research_team",
            label=self.consensus,
            confidence=min(1.0, abs(self.net_score)),
            rationale=f"debate net score {self.net_score:+.2f} over {self.rounds} round(s)",
            evidence=(self.bull.rationale, self.bear.rationale),
            source="debate",
        )


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """What the trader agent (possibly an LLM) proposes.  Not yet validated."""

    symbol: str
    action: Action
    target_weight: float          # desired fraction of equity, 0..1
    confidence: float
    thesis: str = ""
    invalidation: str = ""
    horizon_days: int = 5
    proposed_by: str = "trader"

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", Action.parse(self.action))
        object.__setattr__(self, "target_weight", float(self.target_weight))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Deterministic verdict on a proposal.  Only risk code may create this."""

    approved: bool
    target_weight: float                      # post-adjustment, 0..1
    vetoes: tuple[str, ...] = ()              # hard blocks (proposal ignored)
    adjustments: tuple[str, ...] = ()         # soft changes (sizing/stop)
    notes: tuple[str, ...] = ()
    stop_loss: float | None = None            # absolute price, if any
    take_profit: float | None = None

    @property
    def was_modified(self) -> bool:
        return bool(self.adjustments)


@dataclass(frozen=True, slots=True)
class PortfolioDecision:
    """The auditable end-of-bar decision.  One of these is journalled per bar."""

    asof: datetime
    symbol: str
    action: Action
    target_weight: float
    confidence: float
    thesis: str = ""
    analysis: tuple[Stance, ...] = ()
    debate: DebateResult | None = None
    proposal: TradeProposal | None = None
    risk: RiskDecision | None = None
    mode: str = "deterministic"        # "deterministic" | "llm" | "hybrid"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe projection used by the decision journal."""
        return {
            "asof": self.asof.isoformat(),
            "symbol": self.symbol,
            "action": self.action.value,
            "target_weight": round(self.target_weight, 6),
            "confidence": round(self.confidence, 4),
            "thesis": self.thesis,
            "mode": self.mode,
            "analysis": [
                {"agent": s.agent, "label": s.label, "confidence": round(s.confidence, 4),
                 "rationale": s.rationale, "source": s.source}
                for s in self.analysis
            ],
            "debate": None if self.debate is None else {
                "consensus": self.debate.consensus,
                "net_score": round(self.debate.net_score, 4),
                "rounds": self.debate.rounds,
            },
            "proposal": None if self.proposal is None else {
                "action": self.proposal.action.value,
                "target_weight": round(self.proposal.target_weight, 6),
                "confidence": round(self.proposal.confidence, 4),
                "thesis": self.proposal.thesis,
                "invalidation": self.proposal.invalidation,
                "horizon_days": self.proposal.horizon_days,
                "proposed_by": self.proposal.proposed_by,
            },
            "risk": None if self.risk is None else {
                "approved": self.risk.approved,
                "target_weight": round(self.risk.target_weight, 6),
                "vetoes": list(self.risk.vetoes),
                "adjustments": list(self.risk.adjustments),
                "notes": list(self.risk.notes),
                "stop_loss": self.risk.stop_loss,
                "take_profit": self.risk.take_profit,
            },
            "diagnostics": dict(self.diagnostics),
        }


# --------------------------------------------------------------------------- #
# Portfolio / execution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    qty: float
    avg_price: float

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealised_pct(self, price: float) -> float:
        if self.avg_price <= 0:
            return 0.0
        return (price - self.avg_price) / self.avg_price


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Read-only view of the book handed to a strategy each bar."""

    cash: float
    positions: Mapping[str, Position]
    equity: float
    peak_equity: float
    bar_index: int = 0
    last_prices: Mapping[str, float] = field(default_factory=dict)

    def qty(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.qty if pos else 0.0

    def price(self, symbol: str) -> float:
        return float(self.last_prices.get(symbol, 0.0))

    def weight(self, symbol: str) -> float:
        if self.equity <= 0:
            return 0.0
        return self.qty(symbol) * self.price(symbol) / self.equity

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return 1.0 - self.equity / self.peak_equity

    @property
    def is_flat(self) -> bool:
        return all(abs(p.qty) < 1e-12 for p in self.positions.values())


@dataclass(frozen=True, slots=True)
class Trade:
    asof: datetime
    symbol: str
    side: str            # "buy" | "sell"
    qty: float
    price: float
    fee: float
    equity_after: float
    target_weight: float = 0.0
    reason: str = ""

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Execution assumptions.  Kept deliberately small and explicit."""

    initial_cash: float = 10_000.0
    fee_bps: float = 10.0          # per side, 10 bps = 0.10%
    slippage_bps: float = 5.0      # per side, applied against the fill
    allow_short: bool = False
    warmup_bars: int = 60
    trade_at: str = "close"        # "close" -> fill at decision bar's close

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be > 0")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs must be >= 0")


METRIC_KEYS: tuple[str, ...] = (
    "total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown",
    "calmar", "win_rate", "num_trades", "turnover", "exposure",
    "best_day", "worst_day",
    "benchmark_total_return", "benchmark_cagr", "benchmark_sharpe",
    "benchmark_max_drawdown", "excess_return", "alpha_cagr",
    "num_bars", "num_decisions", "risk_vetoes",
)


@dataclass
class BacktestResult:
    """Container returned by the engine; ``metrics`` uses exactly METRIC_KEYS."""

    symbol: str
    timeframe: str
    initial_cash: float
    equity_curve: list[tuple[datetime, float]]
    benchmark_curve: list[tuple[datetime, float]]
    trades: list[Trade]
    decisions: list[PortfolioDecision]
    metrics: dict[str, float]
    config: BacktestConfig | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "initial_cash": self.initial_cash,
            "metrics": dict(self.metrics),
            "num_trades": len(self.trades),
            "num_decisions": len(self.decisions),
            "equity_curve": [[t.isoformat(), round(v, 4)] for t, v in self.equity_curve],
            "benchmark_curve": [[t.isoformat(), round(v, 4)] for t, v in self.benchmark_curve],
            "trades": [
                {"asof": t.asof.isoformat(), "side": t.side, "qty": round(t.qty, 8),
                 "price": round(t.price, 6), "fee": round(t.fee, 6),
                 "notional": round(t.notional, 4), "reason": t.reason}
                for t in self.trades
            ],
            "decisions": [d.to_dict() for d in self.decisions],
        }


# --------------------------------------------------------------------------- #
# Protocols (the two seams)
# --------------------------------------------------------------------------- #
@runtime_checkable
class DataProvider(Protocol):
    """Adapter interface for a market-data vendor.

    Contracts:
      * return bars sorted ascending by ``ts``, ``ts`` timezone-aware UTC;
      * never include a bar whose close time is after ``end``;
      * raise ``DataUnavailableError`` when the vendor cannot serve the request
        (callers fall through to the next provider).
    """

    name: str

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Bar]: ...


@runtime_checkable
class DecisionStrategy(Protocol):
    """The only thing the backtest engine knows about the agent."""

    name: str

    def decide(self, ctx: MarketContext, portfolio: PortfolioState) -> PortfolioDecision: ...


def as_plain(obj: Any) -> Any:
    """Best-effort dataclass -> dict for logging."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj


# --------------------------------------------------------------------------- #
# Execution / brokerage
#
# Robinhood exposes agent trading through an official MCP server
# (https://agent.robinhood.com/mcp/trading) rather than a REST trading API, so a
# broker adapter is necessarily "list tools, then call tools".  Everything here
# is transport-agnostic: the paper broker and the MCP broker both implement
# ``Broker``, and ``ExecutionEngine`` is the only place that decides what a live
# order is allowed to be.
# --------------------------------------------------------------------------- #
class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"              # share-based market order
    MARKET_DOLLAR = "market_dollar"  # dollar-based market order (Robinhood supports this)
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


def _coerce_enum(enum_cls, value, field_name: str):
    """Accept an enum member, its value, or its ``Class.MEMBER`` string form.

    Python 3.11+ renders ``str(SomeStrEnum.MEMBER)`` as ``"SomeStrEnum.MEMBER"``
    rather than the value, so constructing an enum from ``str(member)`` breaks.
    Broker adapters pass these values around, so coercion must be tolerant.
    """
    if isinstance(value, enum_cls):
        return value
    text = str(value).strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    try:
        return enum_cls(text.strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of "
            f"{[m.value for m in enum_cls]}, got {value!r}"
        ) from exc

class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PREVIEW = "preview"            # dry run: never sent to the venue
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    price: float
    asof: datetime | None = None
    bid: float | None = None
    ask: float | None = None
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """An intent to trade.  ``client_order_id`` makes submission idempotent."""

    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    qty: float | None = None
    notional: float | None = None        # for MARKET_DOLLAR
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "gfd"
    client_order_id: str = ""
    reason: str = ""                     # decision provenance, carried into the audit log

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _coerce_enum(OrderSide, self.side, "side"))
        object.__setattr__(self, "order_type",
                           _coerce_enum(OrderType, self.order_type, "order_type"))
        if self.qty is None and self.notional is None:
            raise ValueError("order requires qty or notional")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"{self.order_type.value} order requires limit_price")
        if self.order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} order requires stop_price")

    @property
    def estimated_notional(self) -> float:
        if self.notional is not None:
            return float(self.notional)
        px = self.limit_price or self.stop_price or 0.0
        return float(self.qty or 0.0) * px


@dataclass(frozen=True, slots=True)
class OrderResult:
    client_order_id: str
    status: OrderStatus
    broker_order_id: str | None = None
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    qty: float | None = None
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    notional: float | None = None
    submitted_at: datetime | None = None
    message: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                               OrderStatus.REJECTED, OrderStatus.PREVIEW)


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    account_id: str
    cash: float
    equity: float
    buying_power: float
    positions: Mapping[str, Position] = field(default_factory=dict)
    asof: datetime | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Broker(Protocol):
    """Transport-agnostic brokerage adapter (paper, Robinhood MCP, ...)."""

    name: str
    is_live: bool

    def is_available(self) -> bool: ...
    def get_account(self) -> AccountSnapshot: ...
    def get_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]: ...
    def submit_order(self, request: OrderRequest) -> OrderResult: ...
    def list_orders(self, limit: int = 50) -> list[OrderResult]: ...
    def cancel_order(self, broker_order_id: str) -> bool: ...
