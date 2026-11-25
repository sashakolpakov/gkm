"""Audited blank player state for a fresh contiguous lineage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlayerState:
    observations: list[Any] = field(default_factory=list)
