# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
USE = 5


def repeat(env, action, times=1):
    for _ in range(times):
        if env.terminal():
            return
        env.step(action)


def up(env, times=1):
    repeat(env, UP, times)


def down(env, times=1):
    repeat(env, DOWN, times)


def left(env, times=1):
    repeat(env, LEFT, times)


def right(env, times=1):
    repeat(env, RIGHT, times)


def use(env, times=1):
    repeat(env, USE, times)


def follow_plan(env, *steps):
    for action, times in steps:
        repeat(env, action, times)


def move_and_use(env, action, steps=1):
    repeat(env, action, steps)
    use(env)


def ascend_and_use(env, steps=1):
    move_and_use(env, UP, steps)


def descend_and_use(env, steps=1):
    move_and_use(env, DOWN, steps)


def left_and_use(env, steps=1):
    move_and_use(env, LEFT, steps)


def right_and_use(env, steps=1):
    move_and_use(env, RIGHT, steps)


def service_upper_branch(
    env,
    outward_action,
    outward_steps,
    return_action,
    return_steps,
    *,
    descend_after_ascent=False,
):
    repeat(env, outward_action, outward_steps)
    ascend_and_use(env)
    if descend_after_ascent:
        down(env)
    repeat(env, return_action, return_steps)
    use(env)


def service_left_branch(env):
    service_upper_branch(env, LEFT, 4, RIGHT, 3)


def service_right_branch(env):
    service_upper_branch(env, RIGHT, 4, LEFT, 2, descend_after_ascent=True)


def enter_lower_right_lane(env):
    down(env, 3)
    right_and_use(env, 6)


def sweep_to_lower_left_switch(env):
    down(env)
    left(env, 7)
    descend_and_use(env)


def sweep_to_far_right_switch(env):
    up(env)
    right_and_use(env, 9)


def reset_on_lower_left_switch(env):
    left(env, 10)
    descend_and_use(env)


def nudge_to_center_lift(env):
    down(env)
    right(env)
    ascend_and_use(env)


def finish_right_exit(env):
    down(env, 3)
    right(env, 2)
    ascend_and_use(env)


def grab_push_release(env, face_action, push_action, push_steps=1):
    """Bump-face an adjacent box, attach with USE, push/carry it, release."""
    repeat(env, face_action)
    use(env)
    repeat(env, push_action, push_steps)
    use(env)


def yield_until_level_up(env, idle_action, cap=40):
    """Yield turns (with a harmless action) so helper agents can finish."""
    start = env.levels_completed
    for _ in range(cap):
        if env.terminal() or env.levels_completed > start:
            return
        env.step(idle_action)


def relay_box_from_west(env, carry_steps, *, depart_action=None, depart_steps=1):
    use(env)
    right(env, carry_steps)
    use(env)
    if depart_action is not None and depart_steps > 0:
        repeat(env, depart_action, depart_steps)
