"""Run the edited solver from a fresh arena, bypassing checkpoint replay."""

import gkm_try
from probe_cumulative import Handoffs


if __name__ == "__main__":
    levels, path, error = gkm_try.A.run_program(
        "wa30", lambda env: gkm_try.m.solve(Handoffs(env))
    )
    valid = gkm_try.A.validate("wa30", path, levels) if path else False
    print(
        "FRESH_RESULT",
        f"levels={levels}",
        f"moves={len(path)}",
        f"replay_ok={valid}",
        f"err={error}",
        flush=True,
    )
