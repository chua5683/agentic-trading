"""The deterministic execution gate between a decision and a broker.

Architecture rule for the whole project: **an LLM may propose, only
deterministic code may dispose.**  ``ExecutionEngine`` is the place where a
:class:`~agentic_trading.contracts.PortfolioDecision` becomes (or is refused as)
an order.  Every guardrail lives here, in plain Python, evaluated before
``broker.submit_order`` is ever called:

* kill-switch file on disk;
* symbol allowlist (``execution.symbol_allowlist``, else ``data.universe``);
* no shorting (negative target weights are refused, sells are clamped to the
  held quantity);
* minimum trade size;
* ``max_order_notional`` (oversized orders are clamped, not silently sent);
* ``max_daily_orders`` and ``max_daily_notional`` (hard blocks);
* the live-trading gate (``ExecutionConfig.live_enabled()``) plus
  ``execution.enabled``;
* ``dry_run`` -> ``OrderStatus.PREVIEW`` and no submission at all.

Every call to :meth:`execute` appends exactly one JSON object to the JSONL audit
journal, whatever happened, so a missing journal entry means a bug in this
module rather than an unrecorded order.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from ..config import DataConfig, ExecutionConfig
from ..contracts import (
    AccountSnapshot,
    Action,
    Broker,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioDecision,
    Position,
    Quote,
)
from ..errors import AgenticTradingError
from .audit import AuditJournal

__all__ = ["ExecutionEngine", "MIN_TRADE_NOTIONAL"]

log = logging.getLogger(__name__)

#: Smallest dollar order this engine will send or preview.
MIN_TRADE_NOTIONAL = 1.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _short(text: Any, limit: int = 280) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _as_text_list(value: Any) -> list[str]:
    """Coerce a ``tuple[str, ...]``-typed field without char-exploding strings.

    The frozen contracts declare ``RiskDecision.vetoes/adjustments/notes`` as
    tuples, but a caller (or a hand-built test) can pass a bare string.  Joining
    that string directly would produce ``"s; i; z; e; ..."``, which is both
    useless and dangerous in an audit trail, so normalise defensively.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _join_text(value: Any, separator: str = "; ") -> str:
    return separator.join(_as_text_list(value))


class ExecutionEngine:
    """Sizes, validates, journals and (only then) submits orders.

    Parameters
    ----------
    broker:
        Any :class:`~agentic_trading.contracts.Broker` implementation.
    cfg:
        ``ExecutionConfig``; all blast-radius limits come from here.
    data_cfg:
        Optional ``DataConfig``; its ``universe`` is the fallback symbol
        allowlist when ``cfg.symbol_allowlist`` is empty.
    clock:
        Optional ``() -> datetime`` used for daily-counter rollover and
        timestamps (inject a fake in tests).
    journal:
        Optional pre-built :class:`AuditJournal` (otherwise built from
        ``cfg.audit_log_path``).
    """

    def __init__(
        self,
        broker: Broker,
        cfg: ExecutionConfig,
        data_cfg: DataConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        journal: AuditJournal | None = None,
    ) -> None:
        self.broker = broker
        self.cfg = cfg
        self.data_cfg = data_cfg
        self._clock = clock
        self.journal = journal if journal is not None else AuditJournal(cfg.audit_log_path)
        self._day = None
        self.orders_today = 0
        self.notional_today = 0.0
        self._last_snapshot: AccountSnapshot | None = None

    # -- public API ---------------------------------------------------------
    def sync_from_broker(self) -> AccountSnapshot:
        """Fetch and cache the broker's account snapshot."""
        snapshot = self.broker.get_account()
        self._last_snapshot = snapshot
        return snapshot

    @property
    def last_snapshot(self) -> AccountSnapshot | None:
        return self._last_snapshot

    def execute(self, decision: PortfolioDecision) -> OrderResult | None:
        """Turn one decision into at most one order.

        Returns ``None`` when the decision is a no-op (HOLD, or already at the
        target weight), ``OrderStatus.PREVIEW`` in dry-run mode,
        ``OrderStatus.REJECTED`` when a guardrail blocked it, and the broker's
        own :class:`OrderResult` otherwise.
        """
        now = self._now()
        self._roll_day(now)

        symbol = str(_safe_attr(decision, "symbol", "") or "").strip().upper()
        action = Action.parse(_safe_attr(decision, "action", Action.HOLD))
        target_weight = _to_float(_safe_attr(decision, "target_weight", 0.0)) or 0.0
        client_order_id = self._client_order_id(decision, symbol)

        checks: dict[str, Any] = {}
        blocks: list[str] = []
        adjustments: list[str] = []
        order: OrderRequest | None = None
        result: OrderResult | None = None
        outcome = "noop"
        submitted = False
        side = OrderSide.BUY

        try:
            # ---- trivial no-ops --------------------------------------------
            if not symbol:
                blocks.append("decision has no symbol")
            if action is Action.HOLD:
                outcome = "noop"

            # ---- 1. kill switch --------------------------------------------
            engaged = self.cfg.kill_switch_engaged()
            checks["kill_switch"] = {"path": self.cfg.kill_switch_file,
                                     "engaged": bool(engaged)}
            if engaged and action is not Action.HOLD:
                blocks.append(f"kill switch engaged: {self.cfg.kill_switch_file}")

            # ---- 2. symbol allowlist ---------------------------------------
            allowlist = self._allowlist()
            checks["symbol_allowlist"] = {
                "configured": bool(allowlist),
                "symbols": list(allowlist),
                "symbol": symbol,
            }
            if allowlist and symbol and symbol not in allowlist:
                blocks.append(f"symbol {symbol!r} not in allowlist {list(allowlist)!r}")

            # ---- 3. target weight / no-short policy ------------------------
            if target_weight < 0:
                blocks.append("negative target_weight: shorting is not permitted")
            if target_weight > 1.0:
                target_weight = 1.0
                adjustments.append("target_weight clamped to 1.0")
            checks["no_short"] = {"target_weight": target_weight,
                                   "policy": "long-only (shorting disabled)"}

            # ---- 4. sanity-check the configured caps -----------------------
            max_order_notional = _to_float(self.cfg.max_order_notional)
            max_daily_orders = self.cfg.max_daily_orders
            max_daily_notional = _to_float(self.cfg.max_daily_notional)
            if max_order_notional is None or max_order_notional <= 0:
                blocks.append("execution.max_order_notional must be > 0; refusing to trade")
            if not isinstance(max_daily_orders, int) or max_daily_orders <= 0:
                blocks.append("execution.max_daily_orders must be > 0; refusing to trade")
            if max_daily_notional is None or max_daily_notional <= 0:
                blocks.append("execution.max_daily_notional must be > 0; refusing to trade")

            # ---- 5. live-trading / master-switch gate ----------------------
            if action is not Action.HOLD and not self.cfg.dry_run:
                send_blocks = self._send_block_reasons()
                if send_blocks:
                    blocks.extend(send_blocks)
                    checks["live_gate"] = {"ok": False, "reasons": send_blocks}
                else:
                    checks["live_gate"] = {"ok": True, "reasons": []}

            # ---- 6. size against live account + quote ----------------------
            if not blocks and action is not Action.HOLD:
                order, result, outcome = self._attempt_order(
                    now, decision, symbol, action, target_weight,
                    checks, blocks, adjustments,
                )
                if order is not None:
                    side = order.side
            elif action is Action.HOLD:
                outcome = "noop"
            else:
                if blocks:
                    outcome = "blocked"
                    result = self._blocked_result(client_order_id, symbol, side,
                                                  checks, blocks)

            # A sizing no-op ("already at target") leaves `result` as None.
            if action is not Action.HOLD and order is None and result is None and not blocks:
                outcome = "noop"
        except Exception as exc:  # noqa: BLE001 - never lose an audit record
            log.exception("execution attempt raised unexpectedly")
            outcome = "error"
            blocks.append(f"internal execution error: {type(exc).__name__}: {exc}")
            result = OrderResult(
                client_order_id=client_order_id,
                status=OrderStatus.UNKNOWN,
                symbol=symbol,
                side=side,
                message=f"internal execution error: {type(exc).__name__}: {exc}",
                raw={"guardrails": {"blocks": list(blocks)}},
            )
        else:
            if blocks and result is None:
                outcome = "blocked"
                result = self._blocked_result(client_order_id, symbol, side,
                                              checks, blocks)

        if result is not None and result.status is OrderStatus.PREVIEW and outcome == "submitted":
            outcome = "preview"
            submitted = result.status not in (OrderStatus.PREVIEW, OrderStatus.REJECTED,
                                              OrderStatus.UNKNOWN, OrderStatus.CANCELLED)
        elif result is not None:
            submitted = result.status in (OrderStatus.SUBMITTED, OrderStatus.PENDING,
                                          OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED)

        if result is not None and outcome == "noop" and result.status is OrderStatus.REJECTED:
            outcome = "blocked"

        self._write_journal(now, decision, outcome=outcome, submitted=submitted,
                            checks=checks, blocks=blocks, adjustments=adjustments,
                            order=order, result=result)
        return result

    # -- order construction -------------------------------------------------
    def _attempt_order(
        self,
        now: datetime,
        decision: PortfolioDecision,
        symbol: str,
        action: Action,
        target_weight: float,
        checks: dict[str, Any],
        blocks: list[str],
        adjustments: list[str],
    ) -> tuple[OrderRequest | None, OrderResult | None, str]:
        client_order_id = self._client_order_id(decision, symbol)

        try:
            snapshot = self.sync_from_broker()
        except AgenticTradingError as exc:
            blocks.append(f"broker account unavailable: {exc}")
            return None, self._blocked_result(client_order_id, symbol, OrderSide.BUY,
                                              checks, blocks), "blocked"

        equity = float(snapshot.equity)
        checks["account"] = {"account_id": snapshot.account_id, "cash": snapshot.cash,
                             "equity": equity, "buying_power": snapshot.buying_power}
        if equity <= 0:
            blocks.append(f"non-positive account equity ({equity:.2f}); refusing to size")
            return None, self._blocked_result(client_order_id, symbol, OrderSide.BUY,
                                              checks, blocks), "blocked"

        try:
            quotes = self.broker.get_quotes([symbol]) or {}
        except AgenticTradingError as exc:
            blocks.append(f"quote unavailable: {exc}")
            return None, self._blocked_result(client_order_id, symbol, OrderSide.BUY,
                                              checks, blocks), "blocked"
        quote: Quote | None = quotes.get(symbol)
        if quote is None or float(quote.price) <= 0:
            blocks.append(f"no usable quote for {symbol}")
            return None, self._blocked_result(client_order_id, symbol, OrderSide.BUY,
                                              checks, blocks), "blocked"
        price = float(quote.price)
        checks["quote"] = {"symbol": symbol, "price": price, "source": quote.source}

        position = snapshot.positions.get(symbol)
        held_qty = float(position.qty) if position is not None else 0.0
        if held_qty < -1e-9:
            blocks.append(
                f"account holds a short position in {symbol} ({held_qty:g}); "
                "no-short policy requires manual intervention")
            return None, self._blocked_result(client_order_id, symbol, OrderSide.BUY,
                                              checks, blocks), "blocked"
        current_value = held_qty * price
        desired_value = target_weight * equity
        delta = desired_value - current_value
        checks["sizing"] = {
            "current_qty": held_qty,
            "current_value": round(current_value, 4),
            "target_weight": round(target_weight, 6),
            "desired_value": round(desired_value, 4),
            "delta_value": round(delta, 4),
        }

        if abs(delta) < MIN_TRADE_NOTIONAL:
            adjustments.append(
                f"delta {delta:.2f} below minimum trade size {MIN_TRADE_NOTIONAL:.2f}; no order")
            return None, None, "noop"

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        qty = abs(delta) / price

        # ---- no-short: never sell more than we hold ------------------------
        if side is OrderSide.SELL:
            if held_qty <= 1e-9:
                adjustments.append(f"no {symbol} position to sell; no order")
                checks["no_short"].update({"clamped": True, "held_qty": held_qty})
                return None, None, "noop"
            if qty > held_qty + 1e-9:
                qty = held_qty
                adjustments.append("sell quantity clamped to held quantity (no shorting)")
                checks["no_short"].update({"clamped": True, "held_qty": held_qty})

        # ---- max_order_notional: clamp, never send oversized orders --------
        requested_notional = qty * price
        if requested_notional > float(self.cfg.max_order_notional) + 1e-9:
            qty = float(self.cfg.max_order_notional) / price
            adjustments.append(
                f"order clamped to max_order_notional="
                f"{float(self.cfg.max_order_notional):.2f} "
                f"(was {requested_notional:.2f})")
        notional = qty * price
        checks["max_order_notional"] = {
            "limit": float(self.cfg.max_order_notional),
            "requested": round(requested_notional, 4),
            "final": round(notional, 4),
            "clamped": notional < requested_notional - 1e-9,
        }
        if notional < MIN_TRADE_NOTIONAL or qty <= 0:
            adjustments.append(
                f"clamped order notional {notional:.2f} below minimum trade size; no order")
            return None, None, "noop"

        # ---- daily order cap ----------------------------------------------
        checks["max_daily_orders"] = {"limit": int(self.cfg.max_daily_orders),
                                      "used": int(self.orders_today)}
        if self.orders_today >= int(self.cfg.max_daily_orders):
            blocks.append(
                f"max_daily_orders reached ({self.orders_today}/"
                f"{int(self.cfg.max_daily_orders)}); no further orders today")
            return None, self._blocked_result(client_order_id, symbol, side, checks, blocks), "blocked"

        # ---- daily notional cap -------------------------------------------
        checks["max_daily_notional"] = {
            "limit": float(self.cfg.max_daily_notional),
            "used": round(self.notional_today, 4),
            "requested": round(notional, 4),
        }
        if self.notional_today + notional > float(self.cfg.max_daily_notional) + 1e-9:
            blocks.append(
                f"max_daily_notional would be exceeded: used {self.notional_today:.2f} + "
                f"{notional:.2f} > {float(self.cfg.max_daily_notional):.2f}")
            return None, self._blocked_result(client_order_id, symbol, side, checks, blocks), "blocked"

        # ---- build the order ----------------------------------------------
        qty = round(qty, 8)
        notional = qty * price
        order = OrderRequest(
            symbol=symbol,
            side=side.value,
            order_type=OrderType.MARKET.value,
            qty=qty,
            client_order_id=client_order_id,
            reason=self._reason(decision, adjustments),
        )

        # ---- dry run: preview only, never submit ---------------------------
        if self.cfg.dry_run:
            self._count_order(notional)
            result = OrderResult(
                client_order_id=client_order_id,
                status=OrderStatus.PREVIEW,
                symbol=symbol,
                side=side,
                qty=qty,
                notional=round(notional, 2),
                submitted_at=None,
                message=("dry_run=true: validated and journalled, NOT sent. " +
                         ("; ".join(adjustments) if adjustments else "all guardrails passed")),
                raw={"guardrails": checks, "broker": getattr(self.broker, "name", "unknown"),
                     "broker_is_live": bool(getattr(self.broker, "is_live", False))},
            )
            return order, result, "preview"

        # ---- live send (all gates already checked) -------------------------
        self._count_order(notional)
        try:
            result = self.broker.submit_order(order)
        except AgenticTradingError as exc:
            result = OrderResult(
                client_order_id=client_order_id,
                status=OrderStatus.UNKNOWN,
                symbol=symbol,
                side=side,
                qty=qty,
                notional=round(notional, 2),
                message=(f"broker submit raised {type(exc).__name__}: {exc}; order state "
                         "is UNKNOWN -- reconcile with the venue before retrying"),
                raw={"guardrails": checks, "error": type(exc).__name__},
            )
            return order, result, "error"
        except Exception as exc:  # noqa: BLE001
            log.exception("broker.submit_order raised unexpectedly")
            result = OrderResult(
                client_order_id=client_order_id,
                status=OrderStatus.UNKNOWN,
                symbol=symbol,
                side=side,
                qty=qty,
                notional=round(notional, 2),
                message=f"broker submit raised {type(exc).__name__}: {exc}; state UNKNOWN",
                raw={"guardrails": checks, "error": type(exc).__name__},
            )
            return order, result, "error"
        if result.status is OrderStatus.PREVIEW:
            # Defence in depth: a broker that refuses to go live (Robinhood
            # MCP broker without live=True) still cannot produce a live order.
            return order, result, "preview"
        return order, result, "submitted"

    # -- guardrail helpers --------------------------------------------------
    def _allowlist(self) -> list[str]:
        configured = [str(symbol).strip().upper()
                      for symbol in (self.cfg.symbol_allowlist or [])
                      if str(symbol).strip()]
        if configured:
            return configured
        if self.data_cfg is not None:
            universe = getattr(self.data_cfg, "universe", None) or []
            return [str(symbol).strip().upper() for symbol in universe
                    if str(symbol).strip()]
        return []

    def _send_block_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.cfg.enabled:
            reasons.append("execution.enabled is false")
        if bool(getattr(self.broker, "is_live", False)) and not self.cfg.live_enabled():
            detail = self.cfg.why_not_live()
            reasons.append(
                "live venue blocked: " + ("; ".join(detail) if detail else "live gates failed"))
        return reasons

    def _count_order(self, notional: float) -> None:
        self.orders_today += 1
        self.notional_today += float(notional)

    def _blocked_result(
        self,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        checks: Mapping[str, Any],
        blocks: Sequence[str],
    ) -> OrderResult:
        message = "blocked: " + "; ".join(blocks)
        return OrderResult(
            client_order_id=client_order_id,
            status=OrderStatus.REJECTED,
            symbol=symbol,
            side=side,
            message=message,
            raw={"guardrails": dict(checks), "blocks": list(blocks)},
        )

    # -- identity / provenance ---------------------------------------------
    def _client_order_id(self, decision: PortfolioDecision, symbol: str) -> str:
        """Stable id for a decision, so a retry is not a second order."""
        asof = _safe_attr(decision, "asof", None)
        stamp = ""
        if isinstance(asof, datetime):
            stamp = asof.strftime("%Y%m%dT%H%M%S")
        else:
            stamp = self._now().strftime("%Y%m%dT%H%M%S")
        mode = str(_safe_attr(decision, "mode", "") or "")
        action = Action.parse(_safe_attr(decision, "action", Action.HOLD)).value
        weight = _to_float(_safe_attr(decision, "target_weight", 0.0)) or 0.0
        fingerprint = "|".join((symbol, action, f"{weight:.6f}", mode,
                                asof.isoformat() if isinstance(asof, datetime) else ""))
        digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
        return f"at-{stamp}-{symbol or 'NA'}-{digest}"

    def _reason(self, decision: PortfolioDecision, adjustments: Sequence[str]) -> str:
        action = Action.parse(_safe_attr(decision, "action", Action.HOLD))
        weight = _to_float(_safe_attr(decision, "target_weight", 0.0)) or 0.0
        confidence = _to_float(_safe_attr(decision, "confidence", 0.0)) or 0.0
        parts = [
            f"mode={_safe_attr(decision, 'mode', '')}",
            f"action={action.value}",
            f"target_weight={weight:.4f}",
            f"confidence={confidence:.3f}",
        ]
        thesis = _safe_attr(decision, "thesis", "")
        if thesis:
            parts.append(f"thesis={_short(thesis)}")
        risk = _safe_attr(decision, "risk", None)
        if risk is not None:
            vetoes = _as_text_list(_safe_attr(risk, "vetoes", ()))
            risk_adjustments = _as_text_list(_safe_attr(risk, "adjustments", ()))
            parts.append(f"risk_approved={bool(_safe_attr(risk, 'approved', False))}")
            if vetoes:
                parts.append("risk_vetoes=" + _join_text(vetoes))
            if risk_adjustments:
                parts.append("risk_adjustments=" + _join_text(risk_adjustments))
        if adjustments:
            parts.append("exec_adjustments=" + _join_text(adjustments))
        return " | ".join(parts)[:4000]

    # -- journal ------------------------------------------------------------
    def _write_journal(
        self,
        now: datetime,
        decision: PortfolioDecision,
        *,
        outcome: str,
        submitted: bool,
        checks: Mapping[str, Any],
        blocks: Sequence[str],
        adjustments: Sequence[str],
        order: OrderRequest | None,
        result: OrderResult | None,
    ) -> None:
        event = {
            "event": "order_attempt",
            "engine": "execution",
            "ts": now.isoformat(),
            "outcome": outcome,
            "submitted": bool(submitted),
            "broker": str(getattr(self.broker, "name", "unknown")),
            "broker_is_live": bool(getattr(self.broker, "is_live", False)),
            "dry_run": bool(self.cfg.dry_run),
            "live_enabled": bool(self.cfg.live_enabled()),
            "decision": self._decision_summary(decision),
            "guardrails": {
                "passed": not blocks,
                "blocks": list(blocks),
                "adjustments": list(adjustments),
                "checks": dict(checks),
                "orders_today": int(self.orders_today),
                "notional_today": round(float(self.notional_today), 4),
            },
            "order": self._order_request_dict(order),
            "result": self._order_result_dict(result),
        }
        try:
            self.journal.record(event)
        except OSError:
            # The order may already have been sent; crashing here helps nobody.
            log.error("AUDIT JOURNAL WRITE FAILED for %s -- event=%r",
                      self.journal.path, event.get("outcome"))

    @staticmethod
    def _decision_summary(decision: PortfolioDecision) -> dict[str, Any]:
        asof = _safe_attr(decision, "asof", None)
        summary: dict[str, Any] = {
            "asof": asof.isoformat() if isinstance(asof, datetime) else str(asof or ""),
            "symbol": str(_safe_attr(decision, "symbol", "") or ""),
            "action": Action.parse(_safe_attr(decision, "action", Action.HOLD)).value,
            "target_weight": round(_to_float(_safe_attr(decision, "target_weight", 0.0)) or 0.0, 6),
            "confidence": round(_to_float(_safe_attr(decision, "confidence", 0.0)) or 0.0, 4),
            "thesis": _short(_safe_attr(decision, "thesis", ""), 400),
            "mode": str(_safe_attr(decision, "mode", "") or ""),
        }
        risk = _safe_attr(decision, "risk", None)
        if risk is not None:
            summary["risk"] = {
                "approved": bool(_safe_attr(risk, "approved", False)),
                "target_weight": _to_float(_safe_attr(risk, "target_weight", None)),
                "vetoes": _as_text_list(_safe_attr(risk, "vetoes", ())),
                "adjustments": _as_text_list(_safe_attr(risk, "adjustments", ())),
                "notes": _as_text_list(_safe_attr(risk, "notes", ())),
            }
        proposal = _safe_attr(decision, "proposal", None)
        if proposal is not None:
            summary["proposal"] = {
                "action": Action.parse(_safe_attr(proposal, "action", Action.HOLD)).value,
                "target_weight": _to_float(_safe_attr(proposal, "target_weight", None)),
                "proposed_by": str(_safe_attr(proposal, "proposed_by", "") or ""),
            }
        return summary

    @staticmethod
    def _order_request_dict(order: OrderRequest | None) -> dict[str, Any] | None:
        if order is None:
            return None
        return {
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "qty": order.qty,
            "notional": order.notional,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price,
            "time_in_force": order.time_in_force,
            "client_order_id": order.client_order_id,
            "reason": order.reason,
        }

    @staticmethod
    def _order_result_dict(result: OrderResult | None) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "client_order_id": result.client_order_id,
            "status": result.status.value,
            "broker_order_id": result.broker_order_id,
            "symbol": result.symbol,
            "side": result.side.value,
            "qty": result.qty,
            "filled_qty": result.filled_qty,
            "avg_fill_price": result.avg_fill_price,
            "notional": result.notional,
            "submitted_at": (result.submitted_at.isoformat()
                             if isinstance(result.submitted_at, datetime) else None),
            "message": result.message,
        }

    # -- clock --------------------------------------------------------------
    def _now(self) -> datetime:
        now = self._clock() if self._clock is not None else _utcnow()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now

    def _roll_day(self, now: datetime) -> None:
        today = now.date()
        if self._day != today:
            self._day = today
            self.orders_today = 0
            self.notional_today = 0.0
