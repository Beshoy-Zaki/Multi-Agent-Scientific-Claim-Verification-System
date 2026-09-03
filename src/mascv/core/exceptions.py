"""Custom exception hierarchy for MASCV."""


class MASCVException(Exception):
    """Base exception for all MASCV errors."""
    pass


class DocumentParsingError(MASCVException):
    """Raised when parsing a research paper fails."""
    pass


class AgentExecutionError(MASCVException):
    """Raised when an agent encounters a failure during execution."""
    pass


class InsufficientEvidenceError(MASCVException):
    """Raised when evidence threshold cannot be satisfied after max cycles."""
    pass


class StorageError(MASCVException):
    """Raised on failure interacting with evidence store or vector DB."""
    pass
