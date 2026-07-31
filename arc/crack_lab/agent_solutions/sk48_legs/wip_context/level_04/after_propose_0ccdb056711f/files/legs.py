# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

UP, DOWN, RETRACT, EXTEND = 1, 2, 3, 4


def move_vertical_lanes(env, direction, lanes):
    """Move the avatar and any attached horizontal train by whole lanes."""
    for _ in range(lanes):
        if env.terminal():
            return
        env.step(direction)


def extend_tether(env, steps):
    """Extend the tether, pushing its attached train toward a new token."""
    for _ in range(steps):
        if env.terminal():
            return
        env.step(EXTEND)


def retract_tether(env, steps):
    """Retract the tether, pulling a contacted token into the train."""
    for _ in range(steps):
        if env.terminal():
            return
        env.step(RETRACT)


def reverse_row_train(env, approach_lanes, stages, final_extension):
    """Reverse a token row by staging each unwanted prefix one lane lower.

    Each stage is ``(prefix_span, wall_reach, compact_steps)``.  The extended
    train pushes the prefix down, then collects the exposed token against the
    right wall before descending to repeat on the staged prefix.
    """
    move_vertical_lanes(env, UP, approach_lanes)
    for prefix_span, wall_reach, compact_steps in stages:
        extend_tether(env, prefix_span)
        move_vertical_lanes(env, DOWN, 1)
        move_vertical_lanes(env, UP, 1)
        retract_tether(env, prefix_span)
        move_vertical_lanes(env, DOWN, 1)
        extend_tether(env, wall_reach)
        retract_tether(env, compact_steps)

    move_vertical_lanes(env, DOWN, 1)
    extend_tether(env, final_extension)


def weave_vertical_four_train(env, approach_lanes=4, thread_steps=3):
    """Weave a four-token vertical stack into its requested horizontal train.

    The lower pair is separated from the stack, hooked against the right wall,
    and compacted.  That pair then carries the third token before a vertical
    loop clears the remaining token and appends it at the tether tip.
    """
    move_vertical_lanes(env, UP, approach_lanes)
    extend_tether(env, thread_steps)
    move_vertical_lanes(env, DOWN, 2)
    retract_tether(env, 1)
    move_vertical_lanes(env, DOWN, 1)

    extend_tether(env, 4)
    retract_tether(env, 5)
    move_vertical_lanes(env, UP, 4)

    extend_tether(env, 2)
    move_vertical_lanes(env, DOWN, 3)
    retract_tether(env, 1)
    move_vertical_lanes(env, UP, 4)
    extend_tether(env, 1)
