"""Audited blank action-policy scaffold with no inherited task logic."""

from __future__ import annotations

from typing import Any


def choose_action(observation: Any) -> Any:
    """Return the next public Arena action after this scaffold is implemented."""

    raise NotImplementedError("implement the current frontier policy")
