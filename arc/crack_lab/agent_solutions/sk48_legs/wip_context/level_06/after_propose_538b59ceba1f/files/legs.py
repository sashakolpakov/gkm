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


def unweave_horizontal_pairs_to_vertical_heads(
    env, approach_lanes=5, alignment_steps=2, far_tail_reach=5
):
    """Deal a four-token row into two ordered vertical two-token trains.

    The two pair heads are aligned with their vertical rails and parked first.
    Each tail is then reached separately, lowered below the rail barriers,
    shifted one lane left, and returned beneath its parked head.
    """
    extend_tether(env, alignment_steps)
    move_vertical_lanes(env, UP, approach_lanes)

    # Peel and park the two pair heads while their tails remain staged below.
    for _ in range(2):
        retract_tether(env, 1)
        move_vertical_lanes(env, UP, 1)
        retract_tether(env, 1)
        move_vertical_lanes(env, DOWN, 1)

    # Place the nearer tail beneath the first head.
    extend_tether(env, 2)
    move_vertical_lanes(env, DOWN, 2)
    retract_tether(env, 1)
    move_vertical_lanes(env, UP, 2)
    retract_tether(env, 1)
    move_vertical_lanes(env, DOWN, 1)

    # Reach around the completed pair and place the farther tail.
    move_vertical_lanes(env, UP, 1)
    extend_tether(env, far_tail_reach)
    retract_tether(env, 1)
    move_vertical_lanes(env, DOWN, 2)
    retract_tether(env, 1)
    move_vertical_lanes(env, UP, 2)
    retract_tether(env, 1)
    move_vertical_lanes(env, DOWN, 1)
    extend_tether(env, 1)
    move_vertical_lanes(env, DOWN, 1)
    extend_tether(env, 1)
    move_vertical_lanes(env, UP, 1)


def stage_split_rows_for_lower_interleave(env):
    """Route one token from each split row into a workable lower-lane pair.

    The center block prevents a direct horizontal collection.  This ladder
    stages unwanted tokens on separate upper lanes, retrieves the first token
    below the block, adds its middle token, then lowers a second matching token
    so the two endpoints can be separated on the clear lower lane.
    """
    # Retrieve the first far-side token and expose a compatible middle token.
    extend_tether(env, 5)
    move_vertical_lanes(env, UP, 3)
    move_vertical_lanes(env, DOWN, 1)
    retract_tether(env, 2)
    move_vertical_lanes(env, DOWN, 1)
    move_vertical_lanes(env, UP, 6)
    move_vertical_lanes(env, DOWN, 1)
    extend_tether(env, 3)
    move_vertical_lanes(env, UP, 2)
    extend_tether(env, 6)
    move_vertical_lanes(env, DOWN, 7)
    retract_tether(env, 6)
    extend_tether(env, 1)
    retract_tether(env, 4)
    move_vertical_lanes(env, UP, 1)
    move_vertical_lanes(env, DOWN, 4)
    retract_tether(env, 4)
    extend_tether(env, 5)
    move_vertical_lanes(env, UP, 1)

    # Thread and compact the ordered two-token prefix.
    extend_tether(env, 2)
    retract_tether(env, 4)
    move_vertical_lanes(env, UP, 2)
    extend_tether(env, 4)
    retract_tether(env, 5)
    move_vertical_lanes(env, DOWN, 1)

    # Use that prefix as a ladder to lower the next matching endpoint.
    move_vertical_lanes(env, UP, 2)
    move_vertical_lanes(env, DOWN, 2)
    extend_tether(env, 5)
    move_vertical_lanes(env, UP, 2)
    move_vertical_lanes(env, DOWN, 1)
    retract_tether(env, 2)
    move_vertical_lanes(env, DOWN, 1)
    move_vertical_lanes(env, UP, 2)
    move_vertical_lanes(env, DOWN, 1)
    extend_tether(env, 3)
    move_vertical_lanes(env, UP, 2)
    extend_tether(env, 3)
    move_vertical_lanes(env, DOWN, 2)

    # Align the two matching endpoints on the unobstructed lower lane.
    retract_tether(env, 4)
    move_vertical_lanes(env, DOWN, 2)
    extend_tether(env, 2)


def interleave_lower_lane_train(env):
    """Insert a staged middle token between two lower-lane endpoints.

    One endpoint is parked below the lane and the other against the right
    wall.  A spare middle token is carried around the center block and parked
    between them; the first endpoint is then pushed up and the ordered triple
    is threaded from the left.
    """
    # Separate and park the two matching endpoints.
    move_vertical_lanes(env, DOWN, 1)
    retract_tether(env, 5)
    extend_tether(env, 4)
    retract_tether(env, 4)
    move_vertical_lanes(env, UP, 1)
    extend_tether(env, 6)
    retract_tether(env, 6)

    # Fetch a spare middle token from the clear top lane.
    move_vertical_lanes(env, UP, 4)
    extend_tether(env, 7)
    retract_tether(env, 4)
    move_vertical_lanes(env, DOWN, 4)
    retract_tether(env, 3)
    extend_tether(env, 5)
    retract_tether(env, 5)

    # Raise the first endpoint into the gap and thread the finished train.
    move_vertical_lanes(env, DOWN, 2)
    extend_tether(env, 5)
    move_vertical_lanes(env, UP, 1)
    retract_tether(env, 1)
    move_vertical_lanes(env, UP, 1)
    extend_tether(env, 3)
