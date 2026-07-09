"""
ASCII rendering of cone-foraging tasks and solution traces.

Visualizes both substrates (cone_foraging v1/v2 and cone_foraging_bound v3) by
consuming the optional trace emitted by run_cone_episode / run_bound_episode.
The renderer reads a trace; it never re-implements stepping, so it cannot drift
from the executed semantics (a test pins the traced outcome to the canonical
episode result).

Symbols:
    S start    H home    * food    X hazard    @ final position
    o trail (a visited cell)    + a cell both on the trail and a feature
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cone_foraging as cf

Trace = List[Dict[str, object]]


def render_level(level: cf.ConeLevel, task: cf.TaskSpec, title: str = "") -> str:
    """ASCII map of the task layout (no solution)."""
    feature: Dict[cf.Position, str] = {}
    for food in level.food:
        feature[food] = "*"
    for hazard in level.hazards:
        feature[hazard] = "X"
    if task.requires_home:
        feature[level.home] = "H"
    feature[level.start] = "S"

    lines = []
    header = title or f"task={task.name}"
    if level.hazards:
        header += f"  (hazard X, safe radius {cf.SAFE_RADIUS})"
    lines.append(header)
    lines.append("+" + "-" * level.width + "+")
    for y in range(level.height):
        row = ["|"]
        for x in range(level.width):
            row.append(feature.get((x, y), " "))
        row.append("|")
        lines.append("".join(row))
    lines.append("+" + "-" * level.width + "+")
    return "\n".join(lines)


def render_trace(level: cf.ConeLevel, task: cf.TaskSpec, trace: Trace, title: str = "") -> str:
    """Path overlay plus an event log derived from a recorded trace."""
    positions = [event["pos"] for event in trace]
    visited = set(positions)
    final = positions[-1] if positions else level.start

    feature: Dict[cf.Position, str] = {}
    for food in level.food:
        feature[food] = "*"
    for hazard in level.hazards:
        feature[hazard] = "X"
    if task.requires_home:
        feature[level.home] = "H"
    feature[level.start] = "S"

    lines = []
    header = title or f"solution for task={task.name}"
    lines.append(header)
    lines.append("+" + "-" * level.width + "+")
    for y in range(level.height):
        row = ["|"]
        for x in range(level.width):
            cell = (x, y)
            base = feature.get(cell)
            if cell == final:
                row.append("@")
            elif base is not None:
                # A feature cell that was also traversed gets a '+'.
                row.append("+" if cell in visited and base not in ("S",) else base)
            elif cell in visited:
                row.append("o")
            else:
                row.append(" ")
        row.append("|")
        lines.append("".join(row))
    lines.append("+" + "-" * level.width + "+")

    # Event log: control-flow milestones, not every move.
    log: List[str] = []
    for event in trace:
        kind = event["kind"]
        if kind in ("start", "call", "set_focus", "return", "halt", "leg_halt", "bump"):
            detail = event["detail"]
            pos = event["pos"]
            label = {
                "start": "start",
                "call": "CALL",
                "set_focus": "SET_FOCUS",
                "return": "RETURN",
                "halt": "HALT (no rule)",
                "leg_halt": "HALT in leg (no rule)",
                "bump": "bump wall",
            }[kind]
            log.append(f"  {label} {detail} @ {pos}".rstrip())
    move_count = sum(1 for e in trace if e["kind"] == "move")
    call_count = sum(1 for e in trace if e["kind"] == "call")
    lines.append(f"moves={move_count}  calls={call_count}")
    lines.extend(log)
    return "\n".join(lines)


def render_rules(genome, library: Sequence[cf.Leg] = ()) -> str:
    """Pretty-print a solver genome and any legs it carries."""
    lines = ["controller:"]
    lines.extend(f"  {r}" for r in genome.describe())
    for leg in library:
        lines.append(f"leg {leg.name}:")
        lines.extend(f"  {r}" for r in leg.genome.describe())
    return "\n".join(lines)


def render_cone_solution(
    genome,
    library: Sequence[cf.Leg],
    level: cf.ConeLevel,
    task: cf.TaskSpec,
    max_steps: int = 44,
    title: str = "",
) -> str:
    """Run a v1/v2 cone solver with tracing and render the result."""
    trace: Trace = []
    episode = cf.run_cone_episode(genome, library, level, task, max_steps=max_steps, trace=trace)
    solved = cf.episode_solved(episode, level, task)
    head = title or f"task={task.name}  solved={solved}"
    return render_trace(level, task, trace, title=head)


def render_bound_solution(
    genome,
    library: Sequence[cf.Leg],
    level: cf.ConeLevel,
    task: cf.TaskSpec,
    max_steps: int = 44,
    title: str = "",
) -> str:
    """Run a v3 bound cone solver with tracing and render the result."""
    import cone_foraging_bound as cb

    trace: Trace = []
    episode = cb.run_bound_episode(genome, library, level, task, max_steps=max_steps, trace=trace)
    solved = cf.episode_solved(episode, level, task)
    head = title or f"task={task.name}  solved={solved}  (v3 bound)"
    return render_trace(level, task, trace, title=head)
