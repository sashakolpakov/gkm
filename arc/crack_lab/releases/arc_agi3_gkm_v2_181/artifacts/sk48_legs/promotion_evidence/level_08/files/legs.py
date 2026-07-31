# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from perception import connected_components


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


def repeat_action(env, action, count):
    """Apply one opaque action a bounded number of times."""
    for _ in range(count):
        if env.terminal():
            return
        env.step(action)


def select_live_collector(env, color, edge):
    """Select the live collector of ``color`` nearest the named room edge."""
    if env.terminal():
        return
    heads = [
        blob
        for blob in connected_components(
            env.frame(), colors=(color,), min_area=16
        )
        if blob.centroid[0] < 53
    ]
    if not heads:
        raise RuntimeError(f"live collector color {color} not found")
    head = min(
        heads,
        key=lambda blob: (
            blob.centroid[1] if edge == "left" else blob.centroid[0]
        ),
    )
    env.step(6, round(head.centroid[1]), round(head.centroid[0]))


def deal_vertical_triplet_to_top_collector(env):
    """Deal a vertical triplet onto the perpendicular top collector.

    The left collector hooks each token against the right wall, carries it to
    its final row, and releases it into the growing top-owned vertical train.
    """
    # Hook the nearest token and carry it to the first top-tether slot.
    move_vertical_lanes(env, DOWN, 1)
    extend_tether(env, 6)
    retract_tether(env, 4)
    move_vertical_lanes(env, UP, 2)
    extend_tether(env, 1)
    move_vertical_lanes(env, UP, 2)
    retract_tether(env, 3)
    select_live_collector(env, 15, "top")
    repeat_action(env, DOWN, 1)
    repeat_action(env, UP, 1)
    select_live_collector(env, 6, "left")

    # The next two tokens use the first token as the top collector's prefix.
    for index in range(2):
        if index:
            select_live_collector(env, 6, "left")
        move_vertical_lanes(env, DOWN, 5)
        extend_tether(env, 6)
        retract_tether(env, 4)
        move_vertical_lanes(env, UP, 4)
        extend_tether(env, 1)
        select_live_collector(env, 15, "top")
        repeat_action(env, DOWN, 1)
        select_live_collector(env, 6, "left")
        retract_tether(env, 3)


def shuttle_row_tokens_under_vertical_train(
    env,
    stages=((1, 5, 6), (2, 6, 5), (3, 5, 3)),
):
    """Grow a left-owned row using a finished vertical train as a lift.

    Each stage is ``(right_shift, wall_reach, compact_steps)``.  The top
    collector lowers the next row token to the clear bottom lane and restores
    its vertical train.  The left collector carries its existing prefix down,
    hooks the new token at the wall, compacts the enlarged train, and returns.
    """
    # Return the left collector to the token row and restore its short tether.
    move_vertical_lanes(env, DOWN, 1)
    extend_tether(env, 1)

    for right_shift, wall_reach, compact_steps in stages:
        select_live_collector(env, 15, "top")
        repeat_action(env, EXTEND, right_shift)

        # Push the four-token column to the floor, then leave only its bottom
        # token there while returning the three-token carrier to the top.
        repeat_action(env, DOWN, 3)
        repeat_action(env, UP, 3)
        repeat_action(env, RETRACT, right_shift)

        select_live_collector(env, 6, "left")
        move_vertical_lanes(env, DOWN, 3)
        extend_tether(env, wall_reach)
        retract_tether(env, compact_steps)
        move_vertical_lanes(env, UP, 3)


def weave_shared_center_cross(env):
    """Weave a checker and one offset hub into two crossing collector trains.

    The left collector first shelves the checker into upper and lower pairs,
    hooks one lower endpoint at the far wall, and raises it beside the hub.
    The top collector then places the upper endpoints.  A short handoff between
    collectors shifts the remaining lower endpoint beneath the shared hub while
    preserving the completed perpendicular row.
    """
    # A resumed prior level can leave the left collector three lanes down with
    # one exposed tether segment.  Return that visible carry-over to the same
    # upper, fully retracted entry state used by a fresh level.
    left_heads = [
        blob
        for blob in connected_components(env.frame(), colors=(6,), min_area=16)
        if blob.centroid[0] < 53
    ]
    if not left_heads:
        raise RuntimeError("live left collector not found")
    left_head = min(left_heads, key=lambda blob: blob.centroid[1])
    carried_lanes = max(0, round((left_head.centroid[0] - 10.5) / 6))
    if carried_lanes:
        move_vertical_lanes(env, UP, carried_lanes)
        retract_tether(env, 1)

    # Split the checker and raise the near lower endpoint beside the hub.
    move_vertical_lanes(env, DOWN, 2)
    extend_tether(env, 4)
    move_vertical_lanes(env, DOWN, 2)
    retract_tether(env, 4)
    move_vertical_lanes(env, DOWN, 1)
    extend_tether(env, 6)
    retract_tether(env, 2)
    move_vertical_lanes(env, UP, 1)

    # Shift the upper pair, then lower its two tokens into their target slots.
    select_live_collector(env, 15, "top")
    repeat_action(env, DOWN, 3)
    repeat_action(env, EXTEND, 1)
    repeat_action(env, UP, 3)
    repeat_action(env, EXTEND, 1)
    repeat_action(env, DOWN, 2)
    repeat_action(env, UP, 2)
    repeat_action(env, EXTEND, 1)
    repeat_action(env, DOWN, 3)
    repeat_action(env, UP, 3)

    # Close the horizontal train around the shared hub.
    select_live_collector(env, 6, "left")
    extend_tether(env, 1)

    # Cycle the right column through the top collector, then restore the row.
    move_vertical_lanes(env, DOWN, 1)
    extend_tether(env, 2)
    retract_tether(env, 1)
    select_live_collector(env, 15, "top")
    repeat_action(env, RETRACT, 1)
    repeat_action(env, DOWN, 5)
    select_live_collector(env, 6, "left")
    retract_tether(env, 1)
    move_vertical_lanes(env, UP, 1)
    extend_tether(env, 2)


def follow_action_path(env, actions):
    """Apply a compact verified path of key and coordinate actions."""
    for action in actions:
        if env.terminal():
            return
        if isinstance(action, tuple):
            env.step(*action)
        else:
            env.step(action)


LEVEL_5_COMPACT_PATH = (
    1, 3, 3, 4, 4, 4, 1, 4, 4, 2, 3, 3, 3, 3, 3, 3, 2, 2,
    4, 4, 4, 4, 4, 4, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 3, 3,
    3, 4, 4, 4, 4, 3, 3, 3, 3, 1, 1, 1, 1, 1, 4, 4, 4, 4,
    4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4,
    4, 3, 3, 3, 3, 3, 2, 2, 4, 4, 4, 4, 4, 1, 3, 1, 4, 4,
    4,
)


def interleave_split_rows_compact(env):
    """Interleave the split-row train while preserving the level-6 entry."""
    follow_action_path(env, LEVEL_5_COMPACT_PATH)


LEVEL_6_COMPACT_PATH = (
    2, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 1, 1, 4, 1, 1, 3, 2,
    2, 2, 2, 2, 4, 4, 4, 4, 3, 3, 3, 3, 1, 1, 1, 1, 4,
    (6, 32, 4), 2, (6, 7, 16), 3, 2, 2, 2, 2, 2, 4, 4, 4, 4,
    3, 3, 3, 3, 1, 1, 1, 1, 4, (6, 32, 4), 2, (6, 7, 22), 3,
    2, (6, 32, 4), 4, 2, 2, 2, 1, 1, 1, 3, (6, 7, 28), 2, 2,
    2, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 1, 1, 1, (6, 32, 4), 4,
    4, 2, 2, 2, 1, 1, 1, 3, 3, (6, 7, 28), 2, 2, 2, 4, 1, 1,
    1, (6, 32, 4), 4, 4, 4, 2, 2, 2, (6, 7, 28), 2, 2, 2, 4,
    4, 4, 4, 4, 3, 1, 1, 1,
)


def deal_and_shuttle_cross_collectors_compact(env):
    """Deal both perpendicular trains while preserving the level-7 entry."""
    follow_action_path(env, LEVEL_6_COMPACT_PATH)


LEVEL_7_COMPACT_PATH = (
    2, 2, 4, 4, 4, 4, 2, 2, 3, 3, 2, 4, 4, 4, 3, 1,
    (6, 32, 4), 2, 2, 2, 4, 1, 1, 1, 4, 2, 1, 4, 2, 2, 2,
    1, 1, (6, 7, 34), 4, 2, 4, 4, 3, (6, 50, 4), 3, 2, 2, 2,
    2, (6, 7, 40), 3, 1, 4, 4,
)


def weave_shared_center_cross_compact(env):
    """Weave the offset shared-center cross with an exact compact exit."""
    follow_action_path(env, LEVEL_7_COMPACT_PATH)


LEVEL_8_CARDINAL_CROSS_PATH = (
    (6, 37, 58), 4, 2, 3, 1, 3,
    (6, 14, 58), 4, 4, 4,
    (6, 37, 58), 4, 2, 2, 2, 2,
    (6, 14, 58), 4, 1,
    (6, 37, 58), 2, 2, 1, 1, 1, 1, 1, 1,
    (6, 14, 58), 2, 4, 4, 3, 3,
    (6, 37, 58), 2, 2,
    (6, 14, 58), 1, 3, 3, 3,
    (6, 37, 58), 1, 1, 3, 2, 4, 4, 2, 2, 2, 2, 2,
    (6, 14, 58), 4, 4, 4, 3, 3, 2, 2, 4,
)


def weave_cardinal_cross_with_shelf_handoffs(env):
    """Weave cardinal tokens by shelving and swapping perpendicular carriers."""
    follow_action_path(env, LEVEL_8_CARDINAL_CROSS_PATH)
