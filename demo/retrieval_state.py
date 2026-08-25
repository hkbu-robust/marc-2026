"""Re-export Stage 2 retrieval state from the production module."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from marc_agent.retrieval_state import (  # noqa: E402, F401
    DEFAULT_GOAL_TOLERANCE,
    RetrievalState,
    search_area_approach_goal,
)
