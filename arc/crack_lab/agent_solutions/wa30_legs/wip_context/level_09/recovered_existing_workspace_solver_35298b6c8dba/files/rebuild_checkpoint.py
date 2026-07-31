"""Rebuild the resume checkpoint from the current replay-validated solver."""

import json
import os
from pathlib import Path

import gkm_try


class MarkLevelEight:
    def __init__(self, env):
        self.env = env
        self.moves = 0
        self.level_eight_move = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        before = self.env.levels_completed
        result = self.env.step(action, *args)
        self.moves += 1
        if before < 8 <= self.env.levels_completed:
            self.level_eight_move = self.moves
        return result


marker = [None]


def fresh_solve(env):
    wrapped = MarkLevelEight(env)
    marker[0] = wrapped
    gkm_try.m.solve(wrapped)


levels, path, error = gkm_try.A.run_program("wa30", fresh_solve)
handoff = marker[0].level_eight_move
if error is not None or levels < 9 or handoff is None:
    raise RuntimeError(
        f"fresh solver failed: levels={levels} handoff={handoff} error={error}"
    )

prefix = path[:handoff]
if not gkm_try.A.validate("wa30", prefix, 8):
    raise RuntimeError("level-8 checkpoint prefix failed replay validation")

payload = {
    "game": "wa30",
    "final_path": prefix,
    "validated": True,
}
target = Path("checkpoint.json")
temporary = Path("checkpoint.next.json")
temporary.write_text(json.dumps(payload))
os.replace(temporary, target)
print(
    "CHECKPOINT_REBUILT",
    {
        "moves": len(prefix),
        "fresh_levels": levels,
        "fresh_moves": len(path),
        "validated": True,
    },
    flush=True,
)
