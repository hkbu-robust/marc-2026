"""Stage 2 retrieval state: owner, grounding fallback, and reveal search area.

Nav reads ``search_xy`` / ``search_tol`` each tick so a late msg 411 can retarget
the drive without treating the hint as the object itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DEFAULT_GOAL_TOLERANCE = 0.7


def _xy(seq) -> Optional[tuple]:
    if seq is None or len(seq) < 2:
        return None
    return (float(seq[0]), float(seq[1]))


@dataclass
class RetrievalState:
    owner_xy: Optional[tuple] = None
    grounding_xy: Optional[tuple] = None
    hint_center: Optional[tuple] = None
    hint_radius: float = 0.0
    target_type: str = ""

    def adopt_owner(self, owner_position) -> None:
        xy = _xy(owner_position)
        if xy is not None:
            self.owner_xy = xy

    def adopt_grounding(self, target_coord) -> None:
        xy = _xy(target_coord)
        if xy is not None:
            self.grounding_xy = xy

    def adopt_reveal(self, reveal) -> None:
        if reveal is None:
            return
        xy = _xy(getattr(reveal, "hint_center", None))
        if xy is not None:
            self.hint_center = xy
        self.hint_radius = float(getattr(reveal, "hint_radius", 0.0) or 0.0)
        self.target_type = str(getattr(reveal, "target_type", "") or "")

    def search_xy(self) -> Optional[tuple]:
        """Reveal hint if present, else Stage 2 grounding target_coord."""
        if self.hint_center is not None:
            return self.hint_center
        return self.grounding_xy

    def search_tol(self, default: float = DEFAULT_GOAL_TOLERANCE) -> float:
        """Reached when inside the hint disc (or ``default`` if no radius)."""
        r = float(self.hint_radius or 0.0)
        if r > 0.0:
            return max(float(default), r)
        return float(default)
