"""Deterministic in-process paper broker.

``PaperBroker`` implements the frozen :class:`agentic_trading.contracts.Broker`
protocol without touching a network.  It is used by tests, by the ``--paper``
live-ish loop, and as the safe default of :func:`build_broker`.

Behaviour is deliberately boring and reproducible:

* fills happen at a supplied quote price (a ``dict`` or a callable), with
  optional backtest-style fee/slippage bps from a
  :class:`agentic_trading.contracts.BacktestConfig`;
* market / market-dollar / limit / stop orders are simulated deterministically;
  resting orders are recorded as ``SUBMITTED`` and never auto-filled;
* shorting is rejected unless the config allows it;
* cash, positions, last prices and order history are persisted to JSON when
  ``state_path`` is set, so a CLI session can be resumed.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..contracts import (
    AccountSnapshot,
    BacktestConfig,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
)

__all__ = ["PaperBroker"]

log = logging.getLogger(__name__)

_EPS = 1e-9


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


def _now_iso(clock: Callable[[], datetime] | None) -> datetime:
    now = clock() if clock is not None else _utcnow()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now


class PaperBroker:
    """A local, deterministic brokerage simulator.

    Parameters
    ----------
    cfg:
        Optional :class:`BacktestConfig`.  When given, ``cfg.initial_cash``,
        ``cfg.fee_bps``, ``cfg.slippage_bps`` and ``cfg.allow_short`` are the
        defaults (explicit keyword arguments still win).  With ``cfg=None`` the
        simulator fills exactly at the quoted price with zero costs and
        ``start_cash`` as the opening balance.
    state_path:
        Optional JSON file used to persist/restore cash, positions, last prices
        and orders between processes.
    start_cash:
        Opening cash when no state file exists (ignored when ``cfg`` is given).
    price_source:
        ``dict[symbol, price]`` or callable ``symbol -> price | None``.  Prices
        fall back to the last observed price / position average before giving
        up and reporting no quote.
    fee_bps, slippage_bps:
        Override the corresponding config values.
    clock:
        Optional ``() -> datetime`` used for ``submitted_at`` timestamps.
    """

    name = "paper"
    is_live = False

    def __init__(
        self,
        cfg: BacktestConfig | None = None,
        state_path: str | Path | None = None,
        start_cash: float = 10_000.0,
        price_source: Mapping[str, float] | Callable[[str], float | None] | None = None,
        *,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.cfg = cfg
        self.state_path = Path(state_path) if state_path else None
        self._clock = clock

        opening_cash = float(cfg.initial_cash) if cfg is not None else float(start_cash)
        if opening_cash <= 0:
            raise ValueError("start_cash must be > 0")

        self._start_cash = opening_cash
        self.cash: float = opening_cash
        self._positions: dict[str, Position] = {}
        self._orders: list[OrderResult] = []
        self._last_prices: dict[str, float] = {}

        self.fee_bps = float(fee_bps if fee_bps is not None
                             else (cfg.fee_bps if cfg is not None else 0.0))
        self.slippage_bps = float(slippage_bps if slippage_bps is not None
                                  else (cfg.slippage_bps if cfg is not None else 0.0))
        self.allow_short = bool(cfg.allow_short) if cfg is not None else False
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fee_bps/slippage_bps must be >= 0")

        self._price_source = price_source
        self._load_state()

    # -- price plumbing -----------------------------------------------------
    def set_price(self, symbol: str, price: float) -> None:
        """Manually mark ``symbol`` at ``price`` (also updates mark-to-market)."""
        value = _to_float(price)
        if value is None or value <= 0:
            raise ValueError(f"invalid price for {symbol!r}: {price!r}")
        self._last_prices[symbol] = value

    def set_prices(self, prices: Mapping[str, float]) -> None:
        for symbol, price in prices.items():
            self.set_price(symbol, price)

    def set_price_source(
        self, price_source: Mapping[str, float] | Callable[[str], float | None] | None
    ) -> None:
        self._price_source = price_source

    def _resolve_price(self, symbol: str) -> float | None:
        source = self._price_source
        price: float | None = None
        if source is not None:
            raw = source(symbol) if callable(source) else source.get(symbol)
            price = _to_float(raw)
        if price is None or price <= 0:
            price = _to_float(self._last_prices.get(symbol))
        if price is None or price <= 0:
            position = self._positions.get(symbol)
            if position is not None and position.avg_price > 0:
                price = float(position.avg_price)
        if price is None or price <= 0:
            return None
        self._last_prices[symbol] = float(price)
        return float(price)

    # -- protocol -----------------------------------------------------------
    def is_available(self) -> bool:
        return True

    def get_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        now = _now_iso(self._clock)
        quotes: dict[str, Quote] = {}
        for raw_symbol in symbols:
            symbol = str(raw_symbol)
            price = self._resolve_price(symbol)
            if price is None:
                log.debug("PaperBroker has no price for %s", symbol)
                continue
            quotes[symbol] = Quote(symbol=symbol, price=price, asof=now,
                                   bid=price, ask=price, source="paper")
        return quotes

    def get_account(self) -> AccountSnapshot:
        now = _now_iso(self._clock)
        positions = dict(self._positions)
        market_value = 0.0
        for symbol, position in positions.items():
            price = self._resolve_price(symbol)
            mark = price if price is not None else position.avg_price
            market_value += position.qty * mark
        equity = float(self.cash + market_value)
        return AccountSnapshot(
            account_id="paper",
            cash=float(self.cash),
            equity=equity,
            buying_power=float(self.cash),
            positions=positions,
            asof=now,
            raw={"broker": self.name, "start_cash": self._start_cash},
        )

    def submit_order(self, request: OrderRequest) -> OrderResult:
        if not isinstance(request, OrderRequest):
            raise TypeError("PaperBroker.submit_order expects an OrderRequest")

        now = _now_iso(self._clock)
        client_order_id = request.client_order_id or self._next_client_id()
        symbol = str(request.symbol).upper()

        price = self._resolve_price(symbol)
        if price is None:
            return self._reject(request, client_order_id, now,
                                f"no price available for {symbol!r}")

        qty = _to_float(request.qty)
        if qty is None or qty <= 0:
            if request.notional is None:
                return self._reject(request, client_order_id, now, "order has no quantity or notional")
            qty = float(request.notional) / price
        qty = abs(float(qty))
        if qty <= _EPS:
            return self._reject(request, client_order_id, now, "order quantity rounds to zero")

        fill_price, resting_reason = self._simulate_fill(request, price)
        side = request.side

        if side is OrderSide.SELL and not self.allow_short:
            held = self._positions.get(symbol)
            held_qty = held.qty if held else 0.0
            if qty > held_qty + _EPS:
                return self._reject(
                    request, client_order_id, now,
                    f"sell of {qty:g} {symbol} would create a short position "
                    f"(held {held_qty:g}); shorting disabled in PaperBroker",
                )

        if fill_price is None:
            result = OrderResult(
                client_order_id=client_order_id,
                status=OrderStatus.SUBMITTED,
                broker_order_id=self._next_broker_id(),
                symbol=symbol,
                side=side,
                qty=qty,
                filled_qty=0.0,
                avg_fill_price=None,
                notional=None,
                submitted_at=now,
                message=resting_reason or "resting order recorded (paper broker never auto-fills it)",
                raw={"order_type": request.order_type.value,
                     "reference_price": price, "seq": self._next_seq()},
            )
            self._orders.append(result)
            self._persist()
            return result

        gross = qty * fill_price
        fee = gross * self.fee_bps / 10_000.0
        if side is OrderSide.BUY:
            total = gross + fee
            if total > self.cash + 1e-6:
                return self._reject(
                    request, client_order_id, now,
                    f"insufficient paper cash: need {total:.2f}, have {self.cash:.2f}",
                )
            self.cash -= total
        else:
            self.cash += gross - fee

        self._apply_fill(symbol, side, qty, fill_price)
        self._last_prices[symbol] = float(fill_price)

        result = OrderResult(
            client_order_id=client_order_id,
            status=OrderStatus.FILLED,
            broker_order_id=self._next_broker_id(),
            symbol=symbol,
            side=side,
            qty=qty,
            filled_qty=qty,
            avg_fill_price=float(fill_price),
            notional=float(gross),
            submitted_at=now,
            message=f"paper fill {qty:g} {symbol} @ {fill_price:.4f} (fee {fee:.4f})",
            raw={
                "order_type": request.order_type.value,
                "reference_price": price,
                "fee": fee,
                "slippage_bps": self.slippage_bps,
                "seq": self._next_seq(),
            },
        )
        self._orders.append(result)
        self._persist()
        return result

    def list_orders(self, limit: int = 50) -> list[OrderResult]:
        try:
            count = max(0, int(limit))
        except (TypeError, ValueError):
            count = 50
        if count == 0:
            return []
        return list(reversed(self._orders[-count:]))

    def cancel_order(self, broker_order_id: str) -> bool:
        target = str(broker_order_id)
        for index, order in enumerate(self._orders):
            if order.broker_order_id != target:
                continue
            if order.status not in (OrderStatus.PENDING, OrderStatus.SUBMITTED,
                                    OrderStatus.PARTIALLY_FILLED):
                return False
            self._orders[index] = replace(
                order, status=OrderStatus.CANCELLED,
                message=f"cancelled: {order.message}")
            self._persist()
            return True
        return False

    # -- convenience --------------------------------------------------------
    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def orders(self) -> list[OrderResult]:
        return list(self._orders)

    @property
    def last_prices(self) -> dict[str, float]:
        return dict(self._last_prices)

    def position(self, symbol: str) -> Position | None:
        return self._positions.get(str(symbol).upper())

    def qty(self, symbol: str) -> float:
        position = self.position(symbol)
        return float(position.qty) if position is not None else 0.0

    def equity(self) -> float:
        return float(self.get_account().equity)

    def reset(self) -> None:
        """Restore the opening balance and forget every position/order."""
        self.cash = float(self._start_cash)
        self._positions.clear()
        self._orders.clear()
        self._last_prices.clear()
        self._persist()

    # -- internals ----------------------------------------------------------
    def _simulate_fill(self, request: OrderRequest, price: float) -> tuple[float | None, str]:
        """Return ``(fill_price, resting_reason)``; ``fill_price=None`` rests."""
        slip = self.slippage_bps / 10_000.0
        buy = request.side is OrderSide.BUY
        order_type = request.order_type

        if order_type in (OrderType.MARKET, OrderType.MARKET_DOLLAR):
            return price * (1.0 + slip) if buy else price * (1.0 - slip), ""

        if order_type is OrderType.LIMIT:
            limit = _to_float(request.limit_price)
            if limit is None:
                return None, "limit order without limit_price"
            marketable = price <= limit + _EPS if buy else price >= limit - _EPS
            if not marketable:
                return None, f"limit {limit:g} not marketable at {price:g}; resting"
            fill = price * (1.0 + slip) if buy else price * (1.0 - slip)
            fill = min(fill, limit) if buy else max(fill, limit)
            return fill, ""

        stop = _to_float(request.stop_price)
        if stop is None:
            return None, "stop order without stop_price"
        triggered = price >= stop - _EPS if buy else price <= stop + _EPS
        if not triggered:
            return None, f"stop {stop:g} not triggered at {price:g}; resting"

        if order_type is OrderType.STOP_LIMIT:
            limit = _to_float(request.limit_price)
            if limit is None:
                return None, "stop-limit order without limit_price"
            marketable = price <= limit + _EPS if buy else price >= limit - _EPS
            if not marketable:
                return None, f"stop-limit {limit:g} not marketable at {price:g}; resting"
            fill = price * (1.0 + slip) if buy else price * (1.0 - slip)
            return (min(fill, limit) if buy else max(fill, limit)), ""

        return price * (1.0 + slip) if buy else price * (1.0 - slip), ""

    def _apply_fill(self, symbol: str, side: OrderSide, qty: float, fill_price: float) -> None:
        position = self._positions.get(symbol)
        old_qty = float(position.qty) if position is not None else 0.0
        old_avg = float(position.avg_price) if position is not None else 0.0
        if side is OrderSide.BUY:
            new_qty = old_qty + qty
            new_avg = ((old_qty * old_avg) + (qty * fill_price)) / new_qty
        else:
            new_qty = old_qty - qty
            if new_qty > _EPS:
                new_avg = old_avg
            elif abs(new_qty) <= 1e-6:
                new_qty, new_avg = 0.0, 0.0
            else:  # only reachable when allow_short is on
                new_avg = fill_price
        if abs(new_qty) <= 1e-9:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = Position(symbol=symbol, qty=float(new_qty),
                                               avg_price=float(new_avg))

    def _reject(self, request: OrderRequest, client_order_id: str, now: datetime,
                message: str) -> OrderResult:
        return OrderResult(
            client_order_id=client_order_id,
            status=OrderStatus.REJECTED,
            symbol=str(request.symbol).upper(),
            side=request.side,
            qty=_to_float(request.qty),
            submitted_at=None,
            message=message,
            raw={"order_type": request.order_type.value, "rejected_by": "paper"},
        )

    def _next_client_id(self) -> str:
        return f"paper-{len(self._orders) + 1:06d}"

    def _next_broker_id(self) -> str:
        return f"paper-{len(self._orders) + 1:06d}"

    def _next_seq(self) -> int:
        return len(self._orders) + 1

    # -- persistence --------------------------------------------------------
    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read paper state %s: %s", self.state_path, exc)
            return
        self.cash = float(raw.get("cash", self.cash))
        for symbol, position in (raw.get("positions") or {}).items():
            qty = _to_float(position.get("qty"))
            avg = _to_float(position.get("avg_price"))
            if qty is None or avg is None:
                continue
            self._positions[str(symbol)] = Position(str(symbol), qty, avg)
        for symbol, price in (raw.get("last_prices") or {}).items():
            value = _to_float(price)
            if value is not None and value > 0:
                self._last_prices[str(symbol)] = value
        self._orders = [_order_from_dict(item) for item in (raw.get("orders") or [])
                        if isinstance(item, Mapping)]

    def _state_dict(self) -> dict[str, Any]:
        return {
            "broker": self.name,
            "start_cash": self._start_cash,
            "cash": self.cash,
            "positions": {symbol: asdict(position)
                          for symbol, position in self._positions.items()},
            "last_prices": dict(self._last_prices),
            "orders": [_order_to_dict(order) for order in self._orders],
        }

    def _persist(self) -> None:
        path = self.state_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                            dir=str(path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(self._state_dict(), handle, indent=2, sort_keys=True)
                    handle.write("\n")
                os.replace(tmp_name, path)
            finally:
                if os.path.exists(tmp_name):
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        except OSError as exc:
            log.warning("could not persist paper state %s: %s", path, exc)


# --------------------------------------------------------------------------- #
# Order (de)serialisation for the state file
# --------------------------------------------------------------------------- #
def _order_to_dict(order: OrderResult) -> dict[str, Any]:
    return {
        "client_order_id": order.client_order_id,
        "status": order.status.value,
        "broker_order_id": order.broker_order_id,
        "symbol": order.symbol,
        "side": order.side.value,
        "qty": order.qty,
        "filled_qty": order.filled_qty,
        "avg_fill_price": order.avg_fill_price,
        "notional": order.notional,
        "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
        "message": order.message,
        "raw": dict(order.raw),
    }


def _order_from_dict(raw: Mapping[str, Any]) -> OrderResult:
    submitted_at = raw.get("submitted_at")
    submitted = None
    if submitted_at:
        try:
            submitted = datetime.fromisoformat(str(submitted_at))
        except ValueError:
            submitted = None
    try:
        status = OrderStatus(str(raw.get("status", OrderStatus.UNKNOWN.value)))
    except ValueError:
        status = OrderStatus.UNKNOWN
    try:
        side = OrderSide(str(raw.get("side", OrderSide.BUY.value)))
    except ValueError:
        side = OrderSide.BUY
    return OrderResult(
        client_order_id=str(raw.get("client_order_id") or ""),
        status=status,
        broker_order_id=raw.get("broker_order_id"),
        symbol=str(raw.get("symbol") or ""),
        side=side,
        qty=_to_float(raw.get("qty")),
        filled_qty=_to_float(raw.get("filled_qty")) or 0.0,
        avg_fill_price=_to_float(raw.get("avg_fill_price")),
        notional=_to_float(raw.get("notional")),
        submitted_at=submitted,
        message=str(raw.get("message") or ""),
        raw=dict(raw.get("raw") or {}),
    )
