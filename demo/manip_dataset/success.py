"""Sensor-only episode success filter."""

from __future__ import annotations


def evaluate_success(*, saw_grasp_after_close: bool, basket_occupied) -> bool:
    """Keep episode iff grasp was observed after a close and basket is occupied at end."""
    if not saw_grasp_after_close:
        return False
    if basket_occupied is not True:
        return False
    return True
