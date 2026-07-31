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
