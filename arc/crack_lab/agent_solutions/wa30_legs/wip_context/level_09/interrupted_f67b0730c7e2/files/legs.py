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
    base_level = env.levels_completed

    # Carry the lower loose block beside the lower port. The repeated blocked
    # movement holds it there until the courier removes the original block.
    walk(env, LEFT, 2)
    walk(env, DOWN, 2)
    walk(env, RIGHT)
    interact(env)
    walk(env, UP, 3)
    walk(env, RIGHT, 4)
    walk(env, RIGHT, 10)
    interact(env)
    walk(env, LEFT)

    # Refill the already-cleared upper port with the central loose block.
    walk(env, LEFT, 2)
    walk(env, UP, 3)
    walk(env, RIGHT)
    interact(env)
    walk(env, UP, 2)
    walk(env, RIGHT, 3)
    interact(env)
    walk(env, LEFT)

    # Refill the lower port again with the final loose block.
    walk(env, LEFT, 5)
    walk(env, DOWN)
    walk(env, RIGHT)
    interact(env)
    walk(env, DOWN, 4)
    walk(env, RIGHT, 6)
    walk(env, DOWN)
    interact(env)
    walk(env, UP)

    # The courier now has a continuous queue; advance only until its final
    # delivery triggers the reward, without spilling actions into the next level.
    while env.levels_completed == base_level and not env.terminal():
        interact(env)


def feed_two_blocks_to_each_courier(env):
    """Stage two cargo blocks at each of three wall-separated courier ports."""
    _run_runs(
        env,
        (RIGHT, 1), (DOWN, 1), (USE, 1), (DOWN, 2), (USE, 1),
        (RIGHT, 1), (USE, 1), (RIGHT, 1), (USE, 1),
        (LEFT, 2), (USE, 1), (UP, 2), (LEFT, 1), (USE, 1),
        (RIGHT, 1), (UP, 2), (LEFT, 1), (USE, 1),
        (DOWN, 1), (LEFT, 1), (USE, 1), (RIGHT, 2),
        (UP, 1), (USE, 1), (DOWN, 2), (USE, 1),
        (LEFT, 1), (UP, 2), (RIGHT, 1), (DOWN, 1), (USE, 1),
        (DOWN, 3), (USE, 1), (UP, 4), (RIGHT, 1), (USE, 1),
        (RIGHT, 1), (USE, 1), (LEFT, 1), (USE, 9),
    )


def expedite_single_courier_with_four_blocks(env):
    """Place one cargo and shorten a courier's routes to three distant cargos."""
    base_level = env.levels_completed

    # Take the lower-left cargo from its right side and carry it straight into
    # the open target row. This supplies one slot without a courier round trip.
    walk(env, RIGHT)
    walk(env, DOWN, 5)
    walk(env, LEFT)
    interact(env)
    walk(env, UP, 6)
    walk(env, LEFT, 8)
    interact(env)

    # Bring the upper-left cargo to the corridor-side courier port.
    walk(env, RIGHT, 9)
    walk(env, UP, 7)
    walk(env, LEFT)
    interact(env)
    walk(env, DOWN, 6)
    walk(env, LEFT)
    interact(env)
    walk(env, DOWN)  # leave the staged cargo so USE cannot re-grab it

    # Queue the upper-right cargo at the same port after the first is cleared.
    walk(env, RIGHT, 3)
    walk(env, UP, 6)
    walk(env, LEFT)
    interact(env)
    walk(env, DOWN, 5)
    walk(env, LEFT, 3)
    interact(env)
    walk(env, DOWN)

    # The last boundary cargo cannot initially be approached from its right.
    # Carry it close to the target instead, where the courier's return route
    # can collect it quickly enough to beat the turn limit.
    walk(env, RIGHT, 2)
    walk(env, DOWN, 5)
    walk(env, RIGHT)
    interact(env)
    walk(env, UP, 5)
    walk(env, LEFT, 10)
    walk(env, DOWN, 2)
    interact(env)
    walk(env, UP)  # crucial: clear the cargo before using USE as a wait

    while env.levels_completed == base_level and not env.terminal():
        interact(env)


def recover_two_blocks_across_courier_port(env):
    """Return a courier delivery, then cross its cleared port for remote cargo."""
    base_level = env.levels_completed
    _run_runs(
        env,
        (UP, 7), (RIGHT, 6), (USE, 1), (RIGHT, 1), (USE, 1),
        (LEFT, 5), (UP, 2), (USE, 1), (DOWN, 2), (RIGHT, 6),
        (DOWN, 1), (RIGHT, 1), (USE, 1), (UP, 1), (LEFT, 7),
        (UP, 2),
    )
    while env.levels_completed == base_level and not env.terminal():
        interact(env)


def stage_and_recover_two_courier_deliveries(env):
    """Stage an off-lane cargo, dismiss its courier, and recover both deliveries."""
    _run_runs(
        env,
        (UP, 2), (RIGHT, 3), (USE, 1), (DOWN, 1), (USE, 1),
        (LEFT, 1), (USE, 2), (UP, 1), (RIGHT, 7), (DOWN, 1),
        (USE, 1), (LEFT, 2), (DOWN, 1), (RIGHT, 1), (USE, 1),
        (LEFT, 9), (USE, 1), (DOWN, 1), (RIGHT, 9), (USE, 1),
        (LEFT, 9), (USE, 1),
    )


def disable_competing_couriers_and_expedite_paired_depots(env):
    """Dismiss two wrong-way couriers and hand-deliver three slow remote cargos."""
    base_level = env.levels_completed
    _run_runs(
        env,
        (RIGHT, 8), (DOWN, 5), (LEFT, 3), (USE, 1),
        (RIGHT, 3), (UP, 5), (LEFT, 4), (UP, 4),
        (USE, 2), (LEFT, 1), (USE, 1),
        (RIGHT, 7), (USE, 1), (DOWN, 1),
        (LEFT, 9), (UP, 1), (USE, 1), (RIGHT, 11),
        (UP, 1), (USE, 1), (DOWN, 1),
        (UP, 1), (LEFT, 10), (UP, 1), (USE, 1), (DOWN, 1),
        (RIGHT, 11), (UP, 1), (USE, 1), (DOWN, 1),
    )
    while env.levels_completed == base_level and not env.terminal():
        interact(env)


def service_split_depots_and_disable_thief(env):
    """Service four isolated depot slots, dismiss a thief, then finish locally."""
    _run_runs(
        env,
        # Lower remote pair.
        (DOWN, 1), (RIGHT, 6), (UP, 1), (USE, 1),
        (UP, 2), (USE, 1), (DOWN, 1),
        (LEFT, 2), (UP, 1), (USE, 1), (RIGHT, 1),
        (UP, 1), (USE, 1), (DOWN, 1),
        # Upper remote pair.
        (LEFT, 2), (UP, 3), (USE, 1), (UP, 2),
        (USE, 1), (DOWN, 1),
        # Intercept and dismiss the thief.
        (DOWN, 3), (LEFT, 5), (DOWN, 2), (LEFT, 1), (USE, 1),
        # Place the two reachable local cargos.
        (LEFT, 4), (UP, 1), (USE, 1), (RIGHT, 7),
        (UP, 3), (LEFT, 1), (USE, 1),
    )
