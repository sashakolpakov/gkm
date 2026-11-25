"""Audited blank solver entrypoint; prior solution code is intentionally absent."""

from __future__ import annotations

from typing import Any

from legs import choose_action


def solve(observation: Any) -> Any:
    return choose_action(observation)
