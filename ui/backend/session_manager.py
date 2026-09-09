"""In-memory session state management for MASCV backend API."""

from typing import Dict, Optional
from mascv.core.state import InvestigationState

_SESSIONS: Dict[str, InvestigationState] = {}


def get_session(paper_id: str) -> Optional[InvestigationState]:
    """Retrieve the InvestigationState for a given paper ID."""
    return _SESSIONS.get(paper_id)


def set_session(paper_id: str, state: InvestigationState) -> None:
    """Store the InvestigationState for a given paper ID."""
    _SESSIONS[paper_id] = state


def delete_session(paper_id: str) -> bool:
    """Remove a session from the store."""
    if paper_id in _SESSIONS:
        del _SESSIONS[paper_id]
        return True
    return False


def list_sessions() -> Dict[str, str]:
    """Return summary of all active paper sessions."""
    summaries = {}
    for pid, state in _SESSIONS.items():
        title = state.paper.metadata.title if state.paper and state.paper.metadata else "Unknown Paper"
        summaries[pid] = title
    return summaries
