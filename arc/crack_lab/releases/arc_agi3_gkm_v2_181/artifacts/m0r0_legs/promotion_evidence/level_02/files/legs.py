# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.


def follow_action_sequence(env, actions):
    """Replay a verified key-action route, stopping if the game terminates."""
    for action in actions:
        if env.terminal():
            return
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
