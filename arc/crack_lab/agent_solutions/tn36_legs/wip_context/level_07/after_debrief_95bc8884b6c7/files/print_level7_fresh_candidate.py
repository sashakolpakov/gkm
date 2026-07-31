"""Print a fresh level-7 candidate path for preservation by the caller."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve


levels, path, error = A.run_program("tn36", solve.solve)
validated = A.validate("tn36", path, levels) if path else False
print(
    json.dumps(
        {
            "game": "tn36",
            "levels": levels,
            "path": path,
            "validated": validated,
            "error": None if error is None else str(error),
        },
        separators=(",", ":"),
    )
)
