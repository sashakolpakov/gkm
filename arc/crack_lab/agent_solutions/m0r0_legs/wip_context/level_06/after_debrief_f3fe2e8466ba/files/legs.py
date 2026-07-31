# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.


def follow_action_sequence(env, actions):
    """Replay a verified key/coordinate route, stopping if the game terminates."""
    for action in actions:
        if env.terminal():
            return
        if isinstance(action, tuple):
            env.step(*action)
        else:
            env.step(action)


def reunite_mirrored_pair(env, route):
    """Drive a mirrored pair along a verified route until they reunite."""
    follow_action_sequence(env, route)


def reunite_helper_pinned_pair(env, route):
    """Reunite a mirrored pair using a selectable helper as a directional stop."""
    follow_action_sequence(env, route)


# Route for two horizontally mirrored avatars to ascend their separate tracks
# and converge at the shared endpoint.
MIRRORED_PAIR_ASCENT = (1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 4, 1, 4, 4, 4)


# Route through the second mirrored maze: stage the avatars through their
# separate corridors, join them, and carry the joined pair to the endpoint.
MIRRORED_PAIR_MAZE_REUNION = (
    2, 3, 3, 3, 2, 2, 2, 4, 4, 1, 4, 4,
    2, 2, 2, 2, 2, 4, 4, 2, 4, 1, 3,
)


# Clear three selectable corridor blockers so the main pair can traverse the
# central route. Coordinate actions select the group that subsequent keys move.
SELECTABLE_CORRIDOR_BLOCKER_CLEARANCE = (
    (6, 31, 15), 3, 2, 2, 2, 3, 3, 1, 1, 1, 3, 1,
    (6, 11, 19), 2, 2, 2, 2, 3,
    (6, 39, 31), 4, 4, 4,
)


def relocate_selectable_blockers(env, route):
    """Select and move blocker groups to verified out-of-corridor positions."""
    follow_action_sequence(env, route)


# Select the main pair and reunite it through the cleared central route.
SELECTABLE_PAIR_REUNION = (
    (6, 22, 46),
    1, 3, 3, 1, 1, 1, 4, 4, 4, 1, 1, 3, 3, 3, 3, 1,
    1, 1, 1, 1, 4, 4, 2, 2, 4, 4, 2, 4, 4, 1, 1,
)


# Select and place the remaining smaller agents around the reunited pair.
_SMALLER_AGENT_ASSEMBLY = (
    (6, 15, 11), (6, 7, 35), (6, 51, 31),
    3, 3, 3, 3, 1, 1, 4, 4, 4, 4, 1, 1, 1, 3, 3, 3, 3,
    (6, 15, 11), 2,
    (6, 51, 31), 3, 3, 3, 3, 1, 1,
    (6, 7, 35), 4, 1, 1, 1, 1, 1,
    (6, 35, 23), 4,
)


def assemble_smaller_agents(env):
    """Bring the selectable smaller agents to the reunited main pair."""
    follow_action_sequence(env, _SMALLER_AGENT_ASSEMBLY)


# Park the central 3x3 blocker above the left corridor, then select and reunite
# the mirrored 5x5 pair through the newly cleared passage.
CENTRAL_BLOCKER_CLEARANCE = ((6, 31, 31), 3, 3, 1)

PARKED_BLOCKER_PAIR_REUNION = (
    (6, 46, 26),
    2, 4, 4, 4, 4, 4, 1, 1,
)


# Navigate the mirrored pair through three mutually exclusive colored gate
# networks. Contacting each switch clears its matching barriers and restores
# the previous network; the route stages both avatars before each transition.
SWITCH_GATED_PAIR_REUNION = (
    3, 3, 1, 1, 1, 1, 3, 1, 4, 4, 1, 4, 4,
    1, 3, 3, 3, 3, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1,
    4, 4, 1, 1, 1, 1, 1, 3, 3, 3, 3,
)


# Use a selectable small helper as a directional stop for one member of a
# mirrored pair.  The free member crosses to a second contact, then the helper
# pins it while its partner descends into the reunion point.
_LOWER_HELPER_TO_NEXT_PIN = ((6, 19, 15), 2, 2, 2, 2, 2, 2, 2)

HELPER_PINNED_PAIR_REUNION = (
    1, 1,
    (6, 31, 43), 3, 3, 1, 3, 1, 1, 1, 1, 1,
    *_LOWER_HELPER_TO_NEXT_PIN,
    (6, 19, 19), 4, 1,
    (6, 19, 15), 4, 4, 4, 4, 4, 4,
    (6, 23, 15), 2, 2, 2, 2, 2, 2, 2, 2, 3,
    *_LOWER_HELPER_TO_NEXT_PIN,
)
