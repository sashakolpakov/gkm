"""Canonical public-action replay for the operational browser demonstration."""

from __future__ import annotations

from .interface import Action


CANONICAL_PICK_PLACE_ACTIONS: tuple[int, ...] = (
    int(Action.NEXT_COORDINATE),
    int(Action.NEXT_COORDINATE),
    *([int(Action.DECREASE)] * 15),
    int(Action.CLOSE_GRIPPER),
    *([int(Action.INCREASE)] * 14),
    int(Action.PREVIOUS_COORDINATE),
    int(Action.PREVIOUS_COORDINATE),
    *([int(Action.INCREASE)] * 12),
    int(Action.NEXT_COORDINATE),
    int(Action.NEXT_COORDINATE),
    *([int(Action.DECREASE)] * 14),
    int(Action.OPEN_GRIPPER),
)

LOW_CLEARANCE_COLLISION_ACTIONS: tuple[int, ...] = (
    int(Action.NEXT_COORDINATE),
    int(Action.NEXT_COORDINATE),
    *([int(Action.DECREASE)] * 15),
    int(Action.CLOSE_GRIPPER),
    *([int(Action.INCREASE)] * 7),
    int(Action.NEXT_COORDINATE),
    int(Action.INCREASE),
    int(Action.INCREASE),
)

__all__ = [
    "CANONICAL_PICK_PLACE_ACTIONS",
    "LOW_CLEARANCE_COLLISION_ACTIONS",
]
