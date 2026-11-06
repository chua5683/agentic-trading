"""Small, dependency-free on-disk OHLCV cache.

One JSON file per ``(symbol, timeframe)`` under ``cache_dir``.  Files are
written atomically (temp file + ``os.replace``) and read defensively: a
truncated or hand-edited cache file degrades to "empty cache", never a crash.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..contracts import Bar
from .base import ensure_utc, normalise_bars, timeframe_seconds

__all__ = ["OHLCVCache"]

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(value: Any) -> str:
    """Filesystem-safe token for a symbol/timeframe (``BTC-USD`` stays intact)."""
    cleaned = _SAFE_CHARS.sub("_", str(value)).strip("._-")
    return cleaned or "unnamed"


class OHLCVCache:
    """JSON-backed bar cache.  All operations are best-effort when disabled."""

    def __init__(self, cache_dir: str | os.PathLike[str], enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = bool(enabled)

    # -- paths ------------------------------------------------------------- #
    def path_for(self, symbol: str, timeframe: str) -> Path:
        """Path of the JSON file backing ``symbol`` / ``timeframe``."""
        safe_tf = _safe_name(str(timeframe or "1d").lower())
        return self.cache_dir / f"{_safe_name(symbol)}__{safe_tf}.json"

    # -- public API -------------------------------------------------------- #
    def read(self, symbol: str, timeframe: str) -> list[Bar]:
        """Bars stored for ``symbol``/``timeframe`` (oldest -> newest)."""
        if not self.enabled:
            return []
        path = self.path_for(symbol, timeframe)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []

        records: Any
        if isinstance(raw, dict):
            records = raw.get("bars")
            timeframe = str(raw.get("timeframe") or timeframe)
        elif isinstance(raw, list):
            records = raw
        else:
            return []
        if not isinstance(records, list):
            return []

        bars: list[Bar] = []
        for item in records:
            if not isinstance(item, dict):
                continue
            try:
                bars.append(
                    Bar(
                        ts=ensure_utc(item["ts"]),
                        open=float(item["open"]),
                        high=float(item["high"]),
                        low=float(item["low"]),
                        close=float(item["close"]),
                        volume=float(item.get("volume", 0.0) or 0.0),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        try:
            timeframe_seconds(timeframe)
        except Exception:
            timeframe = "1d"
        return normalise_bars(bars, timeframe=timeframe)

    def write(self, symbol: str, timeframe: str, bars: Sequence[Bar]) -> None:
        """Atomically replace the cache file with ``bars`` (no-op when disabled)."""
        if not self.enabled:
            return
        ordered = normalise_bars(bars, timeframe=timeframe)
        payload = {
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "bars": [
                {
                    "ts": bar.ts.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in ordered
            ],
        }
        path = self.path_for(symbol, timeframe)
        tmp_name: str | None = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.cache_dir,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_name = handle.name
                json.dump(payload, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
            tmp_name = None
        except OSError:
            # A read-only or full disk must never break trading.
            return
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def merge(self, symbol: str, timeframe: str, bars: Sequence[Bar]) -> list[Bar]:
        """Union cached + new bars, newest value per timestamp wins, ascending."""
        merged: dict[datetime, Bar] = {}
        for bar in self.read(symbol, timeframe):
            merged[ensure_utc(bar.ts)] = bar
        for bar in normalise_bars(bars, timeframe=timeframe):
            merged[ensure_utc(bar.ts)] = bar
        ordered = [merged[key] for key in sorted(merged)]
        if ordered:
            self.write(symbol, timeframe, ordered)
        return ordered
