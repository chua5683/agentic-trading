"""Typed configuration with YAML loading, env overrides and sane defaults.

Nothing in this module touches the network.  ``AppConfig.load(path)`` merges
the shipped defaults, an optional YAML file, and environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = ["DataConfig", "LLMConfig", "RiskConfig", "AgentConfig", "BacktestRunConfig",
           "ExecutionConfig", "AppConfig"]

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _f(d: dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


@dataclass
class DataConfig:
    provider_order: list[str] = field(
        default_factory=lambda: ["coinbase", "twelvedata", "yahoo", "synthetic"])
    cache_dir: str = str(_REPO_ROOT / "data_cache")
    cache_enabled: bool = True
    timeout_s: float = 20.0
    max_retries: int = 2
    retry_backoff_s: float = 1.5
    timeframe: str = "1d"
    lookback_bars: int = 320          # history handed to agents
    allow_network: bool = True        # False -> synthetic/offline only
    twelvedata_api_key_env: str = "TWELVEDATA_API_KEY"
    request_user_agent: str = (
        "agentic-trading/0.1 (+https://github.com/; research use)")
    # Coinbase product ids are e.g. BTC-USD; equities use Yahoo symbols.
    universe: list[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataConfig":
        base = cls()
        return cls(
            provider_order=list(_f(d, "provider_order", base.provider_order)),
            cache_dir=str(_f(d, "cache_dir", base.cache_dir)),
            cache_enabled=bool(_f(d, "cache_enabled", base.cache_enabled)),
            timeout_s=float(_f(d, "timeout_s", base.timeout_s)),
            max_retries=int(_f(d, "max_retries", base.max_retries)),
            retry_backoff_s=float(_f(d, "retry_backoff_s", base.retry_backoff_s)),
            timeframe=str(_f(d, "timeframe", base.timeframe)),
            lookback_bars=int(_f(d, "lookback_bars", base.lookback_bars)),
            allow_network=bool(_f(d, "allow_network", base.allow_network)),
            twelvedata_api_key_env=str(_f(d, "twelvedata_api_key_env", base.twelvedata_api_key_env)),
            request_user_agent=str(_f(d, "request_user_agent", base.request_user_agent)),
            universe=list(_f(d, "universe", base.universe)),
        )


@dataclass
class LLMConfig:
    """Any OpenAI-compatible chat-completions endpoint works here."""

    enabled: str = "auto"             # "auto" | "on" | "off"
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 800
    timeout_s: float = 60.0
    max_retries: int = 1
    # Guardrails for the LLM path.
    require_json: bool = True
    allow_weight_override: bool = False   # LLM can never exceed risk limits anyway

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LLMConfig":
        base = cls()
        return cls(**{f.name: _f(d, f.name, getattr(base, f.name)) for f in fields(cls)})

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None

    def resolved_enabled(self) -> bool:
        mode = (self.enabled or "auto").lower()
        if mode == "off":
            return False
        if mode == "on":
            return True
        return bool(self.api_key)


@dataclass
class RiskConfig:
    """Deterministic limits.  These are enforced in code, never by the LLM."""

    max_position_weight: float = 0.30       # of equity, per symbol
    min_confidence: float = 0.55            # below this -> HOLD
    max_daily_turnover: float = 0.50        # fraction of equity traded per bar
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.25
    max_drawdown_stop: float = 0.25         # halt new risk beyond this drawdown
    vol_target_annual: float = 0.40         # scale exposure toward this vol
    allow_short: bool = False
    max_bars_hold: int = 0                  # 0 disables the time stop
    cooldown_bars_after_stop: int = 2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RiskConfig":
        base = cls()
        return cls(**{f.name: _f(d, f.name, getattr(base, f.name)) for f in fields(cls)})

    def validate(self) -> None:
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")


@dataclass
class AgentConfig:
    analyst_names: list[str] = field(
        default_factory=lambda: ["technical", "momentum", "mean_reversion", "risk"])
    debate_rounds: int = 1
    use_llm: bool = True                  # whether the LLM path is attempted at all
    llm_temperature_note: str = ""
    min_bars_for_signals: int = 60
    # Bars a position survives before a soft (regime-break) exit may close it.
    # 0 disables the hold requirement. Hysteresis is the main anti-whipsaw lever.
    min_hold_bars: int = 5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentConfig":
        base = cls()
        return cls(
            analyst_names=list(_f(d, "analyst_names", base.analyst_names)),
            debate_rounds=int(_f(d, "debate_rounds", base.debate_rounds)),
            use_llm=bool(_f(d, "use_llm", base.use_llm)),
            llm_temperature_note=str(_f(d, "llm_temperature_note", base.llm_temperature_note)),
            min_bars_for_signals=int(_f(d, "min_bars_for_signals", base.min_bars_for_signals)),
            min_hold_bars=int(_f(d, "min_hold_bars", base.min_hold_bars)),
        )


@dataclass
class ExecutionConfig:
    """Live-trading safety configuration.

    Defaults are deliberately inert: execution is disabled, the broker is a local
    paper simulator, dry-run is on, and going live requires an exact confirmation
    phrase in the environment *and* no kill-switch file on disk.  Reasoning is
    that an agent-produced signal is a hypothesis until a human opts in.
    """

    enabled: bool = False                    # master switch for the execution layer
    broker: str = "paper"                    # "paper" | "robinhood"
    mode: str = "paper"                      # "paper" | "live"
    dry_run: bool = True                     # validate + journal, never send
    live_confirm_env: str = "AGENTIC_TRADING_LIVE"
    live_confirm_phrase: str = "yes-i-understand-the-risk"
    # Robinhood Agentic Trading is exposed as an MCP server (streamable HTTP).
    mcp_url: str = "https://agent.robinhood.com/mcp/trading"
    mcp_access_token_env: str = "ROBINHOOD_MCP_ACCESS_TOKEN"
    mcp_timeout_s: float = 30.0
    # Hard blast-radius caps, enforced by ExecutionEngine regardless of signals.
    max_order_notional: float = 100.0
    max_daily_orders: int = 10
    max_daily_notional: float = 500.0
    symbol_allowlist: list[str] = field(default_factory=list)   # empty -> data.universe
    audit_log_path: str = str(_REPO_ROOT / "reports" / "orders.jsonl")
    kill_switch_file: str = str(_REPO_ROOT / "STOP_TRADING")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExecutionConfig":
        base = cls()
        return cls(**{f.name: _f(d, f.name, getattr(base, f.name)) for f in fields(cls)})

    @property
    def mcp_access_token(self) -> str | None:
        return os.environ.get(self.mcp_access_token_env) or None

    def kill_switch_engaged(self) -> bool:
        return Path(self.kill_switch_file).exists()

    def live_enabled(self) -> bool:
        """True only when every live gate passes.  Fails closed on any doubt."""
        if not self.enabled or self.mode != "live" or self.dry_run:
            return False
        if self.kill_switch_engaged():
            return False
        return os.environ.get(self.live_confirm_env, "").strip() == self.live_confirm_phrase

    def why_not_live(self) -> list[str]:
        """Human-readable list of failed live gates (for CLI diagnostics)."""
        reasons: list[str] = []
        if not self.enabled:
            reasons.append("execution.enabled is false")
        if self.mode != "live":
            reasons.append("execution.mode is not 'live'")
        if self.dry_run:
            reasons.append("execution.dry_run is true")
        if self.kill_switch_engaged():
            reasons.append(f"kill switch present at {self.kill_switch_file}")
        if os.environ.get(self.live_confirm_env, "").strip() != self.live_confirm_phrase:
            reasons.append(
                f"env {self.live_confirm_env} must equal the confirmation phrase")
        return reasons


@dataclass
class BacktestRunConfig:
    initial_cash: float = 10_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    warmup_bars: int = 60
    risk_free_rate: float = 0.04
    periods_per_year: float = 365.0       # crypto trades every day
    allow_short: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BacktestRunConfig":
        base = cls()
        return cls(**{f.name: _f(d, f.name, getattr(base, f.name)) for f in fields(cls)})

    def to_contract(self):
        from .contracts import BacktestConfig
        return BacktestConfig(
            initial_cash=self.initial_cash, fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps, allow_short=self.allow_short,
            warmup_bars=self.warmup_bars,
        )


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    backtest: BacktestRunConfig = field(default_factory=BacktestRunConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    seed: int = 7

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "AppConfig":
        d = d or {}
        return cls(
            data=DataConfig.from_dict(d.get("data") or {}),
            llm=LLMConfig.from_dict(d.get("llm") or {}),
            risk=RiskConfig.from_dict(d.get("risk") or {}),
            agents=AgentConfig.from_dict(d.get("agents") or {}),
            backtest=BacktestRunConfig.from_dict(d.get("backtest") or {}),
            execution=ExecutionConfig.from_dict(d.get("execution") or {}),
            seed=int(d.get("seed", 7)),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        raw: dict[str, Any] = {}
        if path is not None:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"config not found: {p}")
            raw = yaml.safe_load(p.read_text()) or {}
        cfg = cls.from_dict(raw)
        # Env overrides that matter most for CI/containers.
        if os.environ.get("AGENTIC_TRADING_OFFLINE") == "1":
            cfg.data.allow_network = False
        if os.environ.get("AGENTIC_TRADING_LLM") in ("0", "off", "false"):
            cfg.llm.enabled = "off"
        elif os.environ.get("AGENTIC_TRADING_LLM") in ("1", "on", "true"):
            cfg.llm.enabled = "on"
        cfg.risk.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
