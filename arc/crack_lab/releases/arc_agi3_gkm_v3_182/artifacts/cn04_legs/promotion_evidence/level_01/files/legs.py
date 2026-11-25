"""Reusable skills inferred from public frame and clone observations."""
from perception import arr, replay, safe_step


def mate_blue_tips(env):
    """Rotate and translate a movable piece until its blue tips meet fixed tips.

    Discover movable blue pixels with an upward clone probe. Use tip alignment
    distance as progress, then verify each complete candidate against reward.
    """
    base_level = env.levels_completed
    candidates = []
    for turns in range(4):
        rotated = replay(env, [5] * turns)
        f = arr(rotated.frame())
        moved = replay(rotated, [1])
        g = arr(moved.frame())
        blue = set(zip(*((f == 8).nonzero())))
        stationary = blue & set(zip(*((g == 8).nonzero())))
        mobile = blue - stationary
        if not mobile or len(mobile) != len(stationary):
            continue
        anchor = min(mobile)
        for target in sorted(stationary):
            dy, dx = int(target[0] - anchor[0]), int(target[1] - anchor[1])
            if dy % 3 or dx % 3:
                continue
            if {(y + dy, x + dx) for y, x in mobile} != stationary:
                continue
            path = ([5] * turns + [2 if dy > 0 else 1] * (abs(dy) // 3)
                    + [4 if dx > 0 else 3] * (abs(dx) // 3))
            candidates.append(path)
    for path in sorted(candidates, key=len):
        trial = replay(env, path)
        if trial.levels_completed <= base_level:
            continue
        for action in path:
            if env.terminal() or env.levels_completed > base_level:
                break
            safe_step(env, action)
        return
    raise RuntimeError('No verified blue-tip alignment path')
