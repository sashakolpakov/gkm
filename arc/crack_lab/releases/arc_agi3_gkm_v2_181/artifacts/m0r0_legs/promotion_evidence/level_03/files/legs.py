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


# Route for two horizontally mirrored avatars to ascend their separate tracks
# and converge at the shared endpoint.
MIRRORED_PAIR_ASCENT = (1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 4, 1, 4, 4, 4)


# Route through the second mirrored maze: stage the avatars through their
# separate corridors, join them, and carry the joined pair to the endpoint.
MIRRORED_PAIR_MAZE_REUNION = (
    2, 3, 3, 3, 2, 2, 2, 4, 4, 1, 4, 4,
    2, 2, 2, 2, 2, 4, 4, 2, 4, 1, 3,
)


# Clear three selectable corridor blockers, reunite the mirrored pair, then
# assemble the smaller agents around it. Coordinate actions select a group.
SELECTABLE_BLOCKER_REUNION = (
    (6, 31, 15), 3, 2, 2, 2, 3, 3, 1, 1, 1, 3, 1,
    (6, 11, 19), 2, 2, 2, 2, 3,
    (6, 39, 31), 4, 4, 4,
    (6, 22, 46),
    1, 3, 3, 1, 1, 1, 4, 4, 4, 1, 1, 3, 3, 3, 3, 1,
    1, 1, 1, 1, 4, 4, 2, 2, 4, 4, 2, 4, 4, 1, 1,
    (6, 15, 11), (6, 7, 35), (6, 51, 31),
    3, 3, 3, 3, 1, 1, 4, 4, 4, 4, 1, 1, 1, 3, 3, 3, 3,
    (6, 15, 11), 2,
    (6, 51, 31), 3, 3, 3, 3, 1, 1,
    (6, 7, 35), 4, 1, 1, 1, 1, 1,
    (6, 35, 23), 4,
)
