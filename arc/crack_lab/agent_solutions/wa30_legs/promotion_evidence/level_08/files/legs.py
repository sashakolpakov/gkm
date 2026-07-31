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


def fill_three_slot_target(env):
    """Carry the three surrounding blocks into a horizontal three-slot target."""
    # Put the lower block into the target's center slot.
    walk(env, UP, 2)
    interact(env)
    walk(env, UP, 2)
    interact(env)

    # Circle around to the left of the left block, then carry it rightward
    # into the target's left slot.
    walk(env, DOWN)
    walk(env, LEFT, 5)
    walk(env, UP, 2)
    walk(env, RIGHT)  # blocked contact turns the avatar toward the block
    interact(env)
    walk(env, RIGHT, 3)
    interact(env)

    # Approach the upper-right block from below. Its carried offset remains
    # above the avatar, so one down and two left steps place it in the last slot.
    walk(env, DOWN)
    walk(env, RIGHT, 5)
    walk(env, UP)
    interact(env)
    walk(env, DOWN)
    walk(env, LEFT, 2)
    interact(env)


def fill_bottom_slots_alongside_courier(env):
    """Fill two bottom target slots while an autonomous courier fills the rest."""
    # Approach the lower loose block from its right and carry it leftward into
    # the target's bottom-right slot.
    walk(env, RIGHT, 10)
    walk(env, DOWN, 8)
    walk(env, LEFT, 2)  # the second step is blocked contact and turns left
    interact(env)
    walk(env, UP)
    walk(env, LEFT, 7)
    interact(env)

    # Reach below the upper-right block, carry it above the avatar, and route
    # one row beneath the target so the filled bottom-right slot is bypassed.
    walk(env, RIGHT, 8)
    walk(env, UP, 2)
    walk(env, LEFT)
    walk(env, UP)  # blocked contact turns upward
    interact(env)
    walk(env, DOWN, 4)
    walk(env, LEFT, 9)
    walk(env, UP)
    interact(env)


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
    interact(env)
    walk(env, LEFT)

    # The courier now has a continuous queue; advance only until its final
    # delivery triggers the reward, without spilling actions into the next level.
    while env.levels_completed == base_level and not env.terminal():
        interact(env)


def feed_two_blocks_to_each_courier(env):
    """Stage two cargo blocks at each of three wall-separated courier ports."""
    base_level = env.levels_completed

    # Send the central loose block through the bottom port.
    walk(env, RIGHT)
    walk(env, DOWN)  # blocked contact faces the block
    interact(env)
    walk(env, LEFT)
    walk(env, DOWN, 2)
    interact(env)

    # Send the lower-right block through the right port.
    walk(env, RIGHT, 2)
    interact(env)
    walk(env, UP, 3)
    walk(env, RIGHT)
    interact(env)

    # Send the lower-left block through the left port.
    walk(env, DOWN, 3)
    walk(env, LEFT, 3)
    interact(env)
    walk(env, UP, 2)
    walk(env, LEFT)
    interact(env)

    # Refill the left port once its courier has cleared the first queued block.
    walk(env, RIGHT)
    walk(env, UP, 2)
    walk(env, LEFT)
    interact(env)
    walk(env, DOWN, 2)
    walk(env, LEFT, 2)  # the second step holds the block at the port
    interact(env)

    # Turn the upper-middle block around so it can enter the bottom port.
    walk(env, RIGHT, 2)
    walk(env, UP, 2)
    interact(env)
    walk(env, DOWN, 2)
    interact(env)
    walk(env, LEFT)
    walk(env, UP, 2)
    walk(env, RIGHT)
    walk(env, DOWN)
    interact(env)
    walk(env, DOWN, 3)
    interact(env)

    # Route the final upper-right block directly into the right port.
    walk(env, UP, 4)
    walk(env, RIGHT)
    interact(env)
    walk(env, DOWN)
    walk(env, RIGHT)
    interact(env)
    walk(env, LEFT)  # leave the port so waiting cannot re-grab its cargo

    while env.levels_completed == base_level and not env.terminal():
        interact(env)


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
    walk(env, DOWN)
    interact(env)
    walk(env, UP)  # crucial: clear the cargo before using USE as a wait

    while env.levels_completed == base_level and not env.terminal():
        interact(env)


def recover_two_blocks_across_courier_port(env):
    """Return a courier delivery, then cross its cleared port for remote cargo."""
    # Meet the courier after its automatic delivery at the single wall gap.
    walk(env, UP, 7)
    walk(env, RIGHT, 6)
    interact(env)

    # Take the delivered block from the front of the remote pad and return it
    # to the lower-left target slot. Moving away finalizes the placement.
    walk(env, RIGHT, 2)
    interact(env)
    walk(env, LEFT, 6)
    walk(env, UP, 2)
    interact(env)
    walk(env, LEFT)

    # The cleared pad cell is traversable. Enter the isolated region, take its
    # remaining block from the left, and carry it horizontally through the gap.
    walk(env, DOWN, 2)
    walk(env, RIGHT, 8)
    walk(env, DOWN)
    walk(env, RIGHT)
    interact(env)
    walk(env, UP)
    walk(env, LEFT, 8)

    # Approach the target's upper-left slot without colliding with the first
    # placed block. The drop itself completes the two-block target.
    walk(env, UP, 3)
    walk(env, RIGHT)
    interact(env)


def stage_and_recover_two_courier_deliveries(env):
    """Stage an off-lane cargo, dismiss its courier, and recover both deliveries."""
    # The lower cargo is too close to the courier to claim first. While that
    # delivery runs, move the upper cargo down onto the courier's return lane.
    walk(env, UP, 2)
    walk(env, RIGHT, 3)  # blocked contact faces the upper cargo
    interact(env)
    walk(env, DOWN)
    interact(env)
    walk(env, LEFT)

    # Let the courier finish the lower route, then collect the staged cargo and
    # finish the upper route. The final wait leaves both deliveries on its pad.
    for _ in range(11):
        interact(env)

    # Approach the courier from above after delivery and dismiss it, preventing
    # it from reclaiming cargo while the avatar carries each block back.
    walk(env, UP)
    walk(env, RIGHT, 7)
    walk(env, DOWN)
    interact(env)

    # Return the upper delivery to the upper-left target.
    walk(env, LEFT, 2)
    walk(env, DOWN)
    walk(env, RIGHT)
    interact(env)
    walk(env, LEFT, 9)
    interact(env)


def disable_competing_couriers_and_expedite_paired_depots(env):
    """Dismiss two wrong-way couriers and hand-deliver two slow remote cargos."""
    base_level = env.levels_completed

    # Enter the lower room and meet its left-going courier while it is carrying
    # past the lower depot. The final left moves establish contact and facing.
    walk(env, RIGHT, 8)
    walk(env, DOWN, 5)
    walk(env, LEFT, 3)
    interact(env)

    # Cross the central corridor to the upper room. Intercept the other
    # left-going courier at the end of its return sweep, then allow its
    # dismissal animation to finish.
    walk(env, RIGHT, 3)
    walk(env, UP, 5)
    walk(env, LEFT, 5)
    walk(env, UP, 4)
    walk(env, LEFT)
    walk(env, UP)
    interact(env)
    interact(env)
    interact(env)

    # Carry the exposed bottom-right cargo of the wrong depot directly into
    # the lower row of the outlined depot.
    walk(env, LEFT)
    interact(env)
    walk(env, UP)
    walk(env, RIGHT, 8)
    interact(env)
    walk(env, DOWN)

    # Take a second cargo from below. Route one row beneath the first delivery
    # so the carried block can pass it, then lift it into a later target slot.
    walk(env, LEFT, 10)
    walk(env, UP)
    interact(env)
    walk(env, DOWN)
    walk(env, RIGHT, 11)
    walk(env, UP)
    interact(env)
    walk(env, DOWN)

    # With both competitors gone and two long transfers removed, the surviving
    # couriers monotonically empty the plain depots before the turn limit.
    while env.levels_completed == base_level and not env.terminal():
        interact(env)
    walk(env, UP)

    # Return the lower delivery to the lower-left target.
    walk(env, DOWN, 2)
    walk(env, RIGHT, 10)
    interact(env)
    walk(env, LEFT, 9)
    interact(env)
