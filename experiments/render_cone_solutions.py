#!/usr/bin/env python3
"""Render cone-foraging tasks and witness solutions as ASCII.

A visible deliverable: for each task, print the layout and the traced witness
solution in both substrates (v1/v2 free-rebinding and v3 priced-binding). Uses
the hand-written witness legs/gluings — representability floors — so the
pictures are deterministic and reproducible.

    python3 experiments/render_cone_solutions.py
    python3 experiments/render_cone_solutions.py --task forage_then_home --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cone_foraging as cf  # noqa: E402
import cone_foraging_bound as cb  # noqa: E402
import cone_render as cr  # noqa: E402

V1_TASKS = ("forage", "homing", "forage_then_home", "flee", "forage_flee", "flee_then_home")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=V1_TASKS + ("all",), default="all")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=44)
    parser.add_argument("--substrate", choices=("v12", "bound", "both"), default="both")
    parser.add_argument("--show-rules", action="store_true")
    return parser.parse_args()


def render_one(task_name: str, seed: int, max_steps: int, substrate: str, show_rules: bool) -> None:
    task = cf.TASKS[task_name]
    level = cf.make_cone_levels(seed, 1, task)[0]
    seek = cf.witness_seek_leg()
    library = [seek]

    print("=" * 60)
    print(cr.render_level(level, task))

    if substrate in ("v12", "both"):
        controller = cf.witness_gluing(task, seek_index=0, flee_index=0) if not task.requires_safe \
            else cf.witness_gluing(task, seek_index=0, flee_index=1)
        v12_library = [seek] if not task.requires_safe else [seek, cf.witness_flee_leg()]
        print()
        print(cr.render_cone_solution(controller, v12_library, level, task, max_steps=max_steps,
                                      title=f"task={task.name}  (v1/v2 free-rebinding)"))
        if show_rules:
            print(cr.render_rules(controller, v12_library))

    if substrate in ("bound", "both"):
        bound_controller = cb.witness_bound_gluing(task, seek_index=0, flee_index=1)
        bound_library = [seek] if not task.requires_safe else [seek, cf.witness_flee_leg()]
        print()
        print(cr.render_bound_solution(bound_controller, bound_library, level, task, max_steps=max_steps,
                                       title=f"task={task.name}  (v3 priced-binding)"))
        if show_rules:
            print(cr.render_rules(bound_controller, bound_library))


def main() -> None:
    args = parse_args()
    tasks = V1_TASKS if args.task == "all" else (args.task,)
    for task_name in tasks:
        render_one(task_name, args.seed, args.max_steps, args.substrate, args.show_rules)


if __name__ == "__main__":
    main()
