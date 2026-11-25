"""Small reusable skills composed by per-level players."""


def follow_key_runs(env, runs):
    """Execute (key, count) runs, stopping on completion or termination."""
    level = env.levels_completed
    for action, count in runs:
        for _ in range(count):
            if env.terminal() or env.levels_completed != level:
                return
            env.step(action)
