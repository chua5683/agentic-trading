"""Append-only JSONL audit journal for order attempts.

Every execution attempt -- blocked, previewed, sent, or failed -- is appended as
one JSON object per line.  The file is append-only by convention (nothing here
ever rewrites it), line-buffered and flushed on every record so a crash cannot
silently lose an order attempt.

The journal deliberately contains only order/decision metadata: never an OAuth
token or any other credential.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

__all__ = ["AuditJournal"]

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_default(value: Any) -> Any:
    """Make journal events JSON-safe without importing project contracts."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


class AuditJournal:
    """Append-only JSONL journal.

    Parameters
    ----------
    path:
        File to append to.  Parent directories are created on first write.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._warned_parent = False

    # -- writing ------------------------------------------------------------
    def record(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append ``event`` as one JSON line; returns the event as written.

        A ``ts`` field is added if the caller did not provide one.  Any
        dataclass/enum/datetime values are serialised with sensible defaults.
        """
        if not isinstance(event, dict):
            raise TypeError("AuditJournal.record expects a dict event")
        written = dict(event)
        written.setdefault("ts", _utcnow().isoformat())
        line = json.dumps(
            written,
            default=_json_default,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                created = not self.path.exists()
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    try:
                        os.fsync(handle.fileno())
                    except OSError:
                        pass
                if created:
                    try:
                        os.chmod(self.path, 0o600)
                    except OSError:
                        pass
            except OSError as exc:
                # A broken audit trail must never be silently ignored.
                log.error("could not append audit event to %s: %s", self.path, exc)
                raise
        return written

    # -- reading ------------------------------------------------------------
    def replay(self) -> list[dict[str, Any]]:
        """Return every recorded event, oldest first.

        Unparseable lines (e.g. a torn write after a power loss) are logged and
        skipped rather than making the whole journal unreadable.
        """
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self._lock:
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                log.error("could not read audit journal %s: %s", self.path, exc)
                raise
        for number, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping corrupt audit line %d in %s", number, self.path)
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the last ``n`` events, oldest first."""
        try:
            count = int(n)
        except (TypeError, ValueError):
            raise ValueError("n must be an integer")
        if count <= 0:
            return []
        return self.replay()[-count:]

    def count(self) -> int:
        """Number of readable events in the journal."""
        return len(self.replay())

    def __len__(self) -> int:
        return self.count()

    def __iter__(self) -> Iterable[dict[str, Any]]:
        return iter(self.replay())

    def __repr__(self) -> str:  # pragma: no cover - debugging nicety
        return f"AuditJournal({str(self.path)!r})"
