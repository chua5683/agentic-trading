"""Minimal OpenAI-compatible LLM client with hard offline degradation.

Product rule: **deterministic first, LLM optional**.  Nothing in this module
touches the network unless ``LLMClient.available`` is ``True``, and every public
method swallows failures and returns an empty/``None`` result, so a missing key,
a flaky endpoint or a hostile response can never crash a decision, a backtest or
the live loop.

The rest of the agent package only depends on the duck-typed surface

    .available  -> bool
    .model      -> str
    .complete_text(system, user) -> str            (``""`` on failure)
    .complete_json(system, user, schema_hint) -> dict | None

which keeps tests and alternative backends trivial to inject.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import LLMConfig

__all__ = ["BudgetedLLM", "LLMClient", "NullLLM", "extract_json", "make_llm"]


def extract_json(text: str | None) -> dict[str, Any] | None:
    """Best-effort extraction of a single JSON object from model output.

    Models wrap JSON in ``` fences or add prose despite instructions; this
    helper tries the raw string, fenced blocks and the first ``{`` .. last ``}``
    slice.  Returns ``None`` (never raises) when no dict can be recovered.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    candidates: list[str] = [text.strip()]
    if "```" in text:
        for part in text.split("```")[1:]:
            inner = part.strip()
            if inner[:4].lower() == "json":
                inner = inner[4:].strip()
            if inner:
                candidates.append(inner)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


class LLMClient:
    """A tiny OpenAI-compatible chat-completions client.

    Parameters
    ----------
    cfg:
        The frozen :class:`agentic_trading.config.LLMConfig`.  ``None`` is
        allowed and yields an offline client (useful in tests).
    available:
        Override for the computed availability.  By default a client is
        available only when ``cfg.resolved_enabled()`` *and* an API key is
        present -- ``enabled: "on"`` without a key still stays offline.
    """

    def __init__(self, cfg: LLMConfig | None = None, *, available: bool | None = None) -> None:
        self.cfg = cfg or LLMConfig()
        self.model = self.cfg.model
        self._api_key = self.cfg.api_key
        if available is None:
            available = bool(self.cfg.resolved_enabled() and self._api_key)
        self.available = bool(available)
        # Cheap, deterministic telemetry used by the pipeline diagnostics.
        self.calls = 0
        self.failures = 0
        self.last_error: str | None = None
        self.reason = "" if self.available else self._offline_reason()

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def _offline_reason(self) -> str:
        if not self.cfg.resolved_enabled():
            return f"llm disabled (enabled={self.cfg.enabled!r})"
        return f"missing API key env var {self.cfg.api_key_env!r}"

    def _fail(self, message: Any) -> None:
        self.failures += 1
        self.last_error = str(message)[:400]

    @property
    def endpoint(self) -> str:
        return self.cfg.base_url.rstrip("/") + "/chat/completions"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "available" if self.available else f"offline: {self.reason}"
        return f"LLMClient(model={self.model!r}, {state})"

    # ------------------------------------------------------------------ #
    # Transport seam (overridable in tests)
    # ------------------------------------------------------------------ #
    def _chat(self, system: str, user: str, *, require_json: bool = False) -> str | None:
        """One completion call.  Returns the raw content or ``None`` on failure."""
        if not self.available:
            return None
        if not self._api_key:
            self._fail("no API key configured")
            return None
        try:  # imported lazily so the package imports fine without `requests`
            import requests
        except Exception as exc:  # pragma: no cover - requests is a hard dep
            self._fail(f"requests unavailable: {exc}")
            return None

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": str(system)},
                {"role": "user", "content": str(user)},
            ],
            "temperature": float(self.cfg.temperature),
            "max_tokens": int(self.cfg.max_tokens),
        }
        if require_json and self.cfg.require_json:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        attempts = max(1, int(self.cfg.max_retries) + 1)
        for _ in range(attempts):
            self.calls += 1
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=float(self.cfg.timeout_s),
                )
                status = int(getattr(response, "status_code", 0) or 0)
                if status != 200:
                    body = getattr(response, "text", "") or ""
                    self._fail(f"HTTP {status}: {str(body)[:200]}")
                    continue
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    self._fail("empty completion content")
                    continue
                return content
            except Exception as exc:  # network, JSON, schema: all non-fatal
                self._fail(f"{type(exc).__name__}: {exc}")
        return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def complete_text(self, system: str, user: str) -> str:
        """Free-form completion.  Returns ``""`` on any failure; never raises."""
        try:
            content = self._chat(system, user, require_json=False)
        except Exception as exc:  # defensively swallow even transport bugs
            self._fail(f"{type(exc).__name__}: {exc}")
            return ""
        return content if isinstance(content, str) else ""

    def complete_json(self, system: str, user: str, schema_hint: str = "{}") -> dict[str, Any] | None:
        """JSON completion.  Returns a dict or ``None``; never raises."""
        if not self.available:
            return None
        prompt = str(user)
        if schema_hint:
            prompt = f"{prompt}\n\nReturn ONLY a single JSON object matching this schema: {schema_hint}"
        try:
            content = self._chat(system, prompt, require_json=True)
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            return None
        if content is None:
            return None
        parsed = extract_json(content)
        if parsed is None:
            self._fail("response was not valid JSON")
        return parsed


class NullLLM(LLMClient):
    """Offline stand-in: reports unavailable and never touches the network."""

    def __init__(self, cfg: LLMConfig | None = None, reason: str = "offline") -> None:
        super().__init__(cfg, available=False)
        self.reason = str(reason)

    def _chat(self, system: str, user: str, *, require_json: bool = False) -> str | None:
        return None

    def complete_text(self, system: str, user: str) -> str:
        return ""

    def complete_json(self, system: str, user: str, schema_hint: str = "{}") -> dict[str, Any] | None:
        return None


class BudgetedLLM:
    """Per-decision call budget wrapper.

    Keeps one bar's decision from accidentally fanning out into dozens of API
    calls (analysts + debate rounds + trader all share one budget).  Once the
    budget is exhausted ``available`` becomes ``False``, so every downstream
    agent transparently falls back to its deterministic path.
    """

    def __init__(self, inner: Any, budget: int = 8) -> None:
        self.inner = inner
        self.budget = max(0, int(budget))
        self.used = 0

    @property
    def available(self) -> bool:
        return bool(getattr(self.inner, "available", False)) and self.used < self.budget

    @property
    def model(self) -> str:
        return str(getattr(self.inner, "model", ""))

    @property
    def calls(self) -> int:
        return int(getattr(self.inner, "calls", 0))

    @property
    def failures(self) -> int:
        return int(getattr(self.inner, "failures", 0))

    @property
    def last_error(self) -> str | None:
        return getattr(self.inner, "last_error", None)

    def complete_text(self, system: str, user: str) -> str:
        if not self.available:
            return ""
        self.used += 1
        try:
            return self.inner.complete_text(system, user)
        except Exception:
            return ""

    def complete_json(self, system: str, user: str, schema_hint: str = "{}") -> dict[str, Any] | None:
        if not self.available:
            return None
        self.used += 1
        try:
            result = self.inner.complete_json(system, user, schema_hint)
        except Exception:
            return None
        return result if isinstance(result, dict) else None


def make_llm(cfg: LLMConfig | None) -> LLMClient:
    """Build the best client available for ``cfg`` -- offline when no key/disabled.

    This is the *only* factory the pipeline should use, so that a machine with
    no credentials provably never creates a network-capable client.
    """
    if cfg is None:
        return NullLLM(None, "no LLMConfig supplied")
    if not cfg.resolved_enabled():
        return NullLLM(cfg, f"llm disabled (enabled={cfg.enabled!r})")
    if not cfg.api_key:
        return NullLLM(cfg, f"missing API key env var {cfg.api_key_env!r}")
    return LLMClient(cfg, available=True)
