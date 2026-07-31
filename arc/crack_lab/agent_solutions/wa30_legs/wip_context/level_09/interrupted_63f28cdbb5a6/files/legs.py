# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5


def walk(env, direction, steps=1):
    """Move in one cardinal direction for a fixed number of grid steps."""
    for _ in range(steps):
        if env.terminal():
            return
        env.step(direction)


def interact(env):
    """Use the context-sensitive interaction action once."""
    if not env.terminal():
        env.step(USE)


def _run_runs(env, *runs):
    """Execute compact action runs, stopping exactly at a level handoff."""
    base_level = env.levels_completed
    for action, count in runs:
        for _ in range(count):
            if env.terminal() or env.levels_completed != base_level:
                return
            env.step(action)


def fill_three_slot_target(env):
    """Carry the three surrounding blocks into a horizontal three-slot target."""
    _run_runs(
        env,
        (UP, 2), (USE, 1), (UP, 2), (USE, 1),
        (LEFT, 5), (UP, 1), (RIGHT, 1), (USE, 1),
        (RIGHT, 3), (USE, 1), (DOWN, 1), (RIGHT, 5),
        (UP, 1), (USE, 1), (DOWN, 1), (LEFT, 2), (USE, 1),
    )


def fill_bottom_slots_alongside_courier(env):
    """Fill two bottom target slots while an autonomous courier fills the rest."""
    _run_runs(
        env,
        (RIGHT, 10), (DOWN, 8), (LEFT, 1), (USE, 1),
        (UP, 1), (LEFT, 6), (USE, 1), (RIGHT, 7),
        (UP, 1), (LEFT, 1), (UP, 1), (USE, 1),
        (DOWN, 4), (LEFT, 9), (UP, 1), (USE, 1),
    )


def feed_three_blocks_to_courier(env):
    """Keep two courier ports supplied until all five target blocks are delivered."""
    _run_runs(
        env,
        (LEFT, 2), (DOWN, 2), (RIGHT, 1), (USE, 1),
        (UP, 3), (RIGHT, 14), (USE, 1), (LEFT, 3),
        (UP, 3), (RIGHT, 1), (USE, 1), (UP, 2),
        (RIGHT, 3), (USE, 1), (LEFT, 6), (DOWN, 1),
        (RIGHT, 1), (USE, 1), (DOWN, 4), (RIGHT, 6),
        (USE, 1), (LEFT, 1), (USE, 21),
    )


def feed_two_blocks_to_each_courier(env):
    """Stage two cargo blocks at each of three wall-separated courier ports."""
    _run_runs(
        env,
        (RIGHT, 1), (DOWN, 1), (USE, 1), (DOWN, 2),
        (USE, 1), (RIGHT, 1), (USE, 1), (RIGHT, 1),
        (USE, 1), (LEFT, 2), (USE, 1), (UP, 2),
        (LEFT, 1), (USE, 1), (RIGHT, 1), (UP, 2),
        (LEFT, 1), (USE, 1), (DOWN, 1), (LEFT, 1),
        (USE, 1), (RIGHT, 2), (UP, 1), (USE, 1),
        (DOWN, 2), (USE, 1), (LEFT, 1), (UP, 2),
        (RIGHT, 1), (DOWN, 1), (USE, 1), (DOWN, 3),
        (USE, 1), (UP, 4), (RIGHT, 1), (USE, 1),
        (RIGHT, 1), (USE, 1), (LEFT, 1), (USE, 9),
    )


def expedite_single_courier_with_four_blocks(env):
    """Place one cargo and shorten a courier's routes to three distant cargos."""
    _run_runs(
        env,
        (RIGHT, 1), (DOWN, 5), (LEFT, 1), (USE, 1),
        (UP, 6), (LEFT, 8), (USE, 1), (RIGHT, 9),
        (UP, 7), (LEFT, 1), (USE, 1), (DOWN, 6),
        (LEFT, 1), (USE, 1), (DOWN, 1), (RIGHT, 3),
        (UP, 6), (LEFT, 1), (USE, 1), (DOWN, 5),
        (LEFT, 3), (USE, 1), (DOWN, 1), (RIGHT, 2),
        (DOWN, 5), (RIGHT, 1), (USE, 1), (UP, 5),
        (LEFT, 10), (DOWN, 1), (USE, 1), (UP, 1),
        (USE, 14),
    )


def recover_two_blocks_across_courier_port(env):
    """Return a courier delivery, then cross its cleared port for remote cargo."""
    _run_runs(
        env,
        (UP, 7), (RIGHT, 6), (USE, 1), (RIGHT, 1),
        (USE, 1), (LEFT, 5), (UP, 2), (USE, 1),
        (DOWN, 2), (RIGHT, 6), (DOWN, 1), (RIGHT, 1),
        (USE, 1), (UP, 1), (LEFT, 7), (UP, 2), (USE, 1),
    )


def stage_and_recover_two_courier_deliveries(env):
    """Stage an off-lane cargo, dismiss its courier, and recover both deliveries."""
    _run_runs(
        env,
        (UP, 2), (RIGHT, 3), (USE, 1), (DOWN, 1),
        (USE, 1), (LEFT, 1), (USE, 2), (UP, 1),
        (RIGHT, 7), (DOWN, 1), (USE, 1), (LEFT, 2),
        (DOWN, 1), (RIGHT, 1), (USE, 1), (LEFT, 9),
        (USE, 1), (DOWN, 1), (RIGHT, 9), (USE, 1),
        (LEFT, 9), (USE, 1),
    )


def disable_competing_couriers_and_expedite_paired_depots(env):
    """Dismiss two wrong-way couriers and hand-deliver three slow remote cargos."""
    _run_runs(
        env,
        (RIGHT, 8), (DOWN, 5), (LEFT, 3), (USE, 1),
        (RIGHT, 3), (UP, 5), (LEFT, 4), (UP, 4),
        (USE, 2), (LEFT, 1), (USE, 1), (RIGHT, 7),
        (USE, 1), (DOWN, 1), (LEFT, 9), (UP, 1),
        (USE, 1), (RIGHT, 11), (UP, 1), (USE, 1),
        (DOWN, 1), (UP, 1), (LEFT, 10), (UP, 1),
        (USE, 1), (DOWN, 1), (RIGHT, 11), (UP, 1),
        (USE, 1), (DOWN, 1), (USE, 15),
    )


def service_split_depots_and_disable_thief(env):
    """Fill two local depots, stage a remote load, and dismiss its thief."""
    base_level = env.levels_completed

    # Fill the lower-right remote slot.
    walk(env, DOWN)
    walk(env, RIGHT, 6)
    walk(env, UP)
    interact(env)
    walk(env, UP, 2)
    interact(env)
    walk(env, DOWN)

    # Fill the adjacent lower-left remote slot.
    walk(env, LEFT, 2)
    walk(env, UP)
    interact(env)
    walk(env, RIGHT)
    walk(env, UP)
    interact(env)
    walk(env, DOWN)

    # Stage the upper remote cargo at its wall port.
    walk(env, LEFT, 2)
    walk(env, UP, 3)
    interact(env)
    walk(env, UP, 2)
    interact(env)
    walk(env, DOWN)

    # Meet and dismiss the competing courier.
    walk(env, DOWN, 3)
    walk(env, LEFT, 5)
    walk(env, DOWN, 2)
    walk(env, LEFT)
    interact(env)

    # Hand-deliver the final local cargo.
    walk(env, LEFT, 4)
    walk(env, UP)
    interact(env)
    walk(env, RIGHT, 7)
    walk(env, UP, 3)
    walk(env, LEFT)
    interact(env)

    while env.levels_completed == base_level and not env.terminal():
        interact(env)
