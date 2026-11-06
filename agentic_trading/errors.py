"""Shared exception types."""

class AgenticTradingError(Exception):
    """Base class for all project errors."""

class DataUnavailableError(AgenticTradingError):
    """A data vendor could not serve the request; try the next provider."""

class DataValidationError(AgenticTradingError):
    """Vendor returned malformed or insufficient data."""

class LLMError(AgenticTradingError):
    """LLM call failed or returned unusable output."""

class RiskViolation(AgenticTradingError):
    """Proposal violated a hard risk rule (normally handled, not raised)."""
