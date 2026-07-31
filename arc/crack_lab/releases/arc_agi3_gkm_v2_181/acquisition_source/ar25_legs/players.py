# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    repeat_action(env, 2, 10)
    repeat_action(env, 3, 5)


def play_level_2(env):
    repeat_action(env, 3, 2)
    repeat_action(env, 5, 1)
    repeat_action(env, 2, 8)


def play_level_3(env):
    # Select and align the L with its lower silhouette.
    repeat_action(env, 5, 1)
    repeat_action(env, 2, 7)
    repeat_action(env, 4, 7)

    # Select and align the T with its lower silhouette.
    repeat_action(env, 5, 1)
    repeat_action(env, 2, 5)
    repeat_action(env, 3, 12)

    # Return to the scanner and sweep upward to validate the matches.
    repeat_action(env, 5, 1)
    repeat_action(env, 1, 16)


def play_level_4(env):
    # Place the U in the upper composite target.
    repeat_action(env, 5, 1)
    repeat_action(env, 4, 7)

    # Place the vertical bar in the lower composite target.
    repeat_action(env, 5, 1)
    repeat_action(env, 4, 7)

    # Return to the scanner and sweep down to validate both matches.
    repeat_action(env, 5, 1)
    repeat_action(env, 2, 9)


def play_level_5(env):
    # Center the horizontal and vertical mirrors on the composite silhouette.
    repeat_action(env, 2, 4)
    repeat_action(env, 5, 1)
    repeat_action(env, 4, 5)

    # Select the piece and align it with its upper-left silhouette.
    repeat_action(env, 5, 1)
    repeat_action(env, 1, 7)
    repeat_action(env, 3, 10)


def play_level_6(env):
    # Center the horizontal and vertical mirrors on the four silhouettes.
    repeat_action(env, 2, 11)
    repeat_action(env, 5, 1)
    repeat_action(env, 3, 1)

    # Place the small source piece in the lower-left silhouette.
    repeat_action(env, 5, 1)
    repeat_action(env, 2, 4)
    repeat_action(env, 3, 15)

    # Place the large source piece in the lower-right silhouette.
    repeat_action(env, 5, 1)
    repeat_action(env, 2, 12)
    repeat_action(env, 3, 7)


def play_level_7(env):
    # Center the horizontal and vertical mirrors on the four silhouettes.
    repeat_action(env, 2, 2)
    repeat_action(env, 5, 1)
    repeat_action(env, 4, 9)

    # Place the small and large source pieces in the lower silhouettes.
    repeat_action(env, 5, 1)
    repeat_action(env, 1, 6)
    repeat_action(env, 3, 10)
    repeat_action(env, 5, 1)
    repeat_action(env, 1, 6)
    repeat_action(env, 4, 3)


def play_level_8(env):
    # Center the horizontal and vertical mirrors between the four silhouettes.
    repeat_action(env, 2, 6)
    repeat_action(env, 5, 1)
    repeat_action(env, 4, 9)

    # Place the large piece in the upper-left silhouette.
    repeat_action(env, 5, 1)
    repeat_action(env, 1, 7)
    repeat_action(env, 3, 9)

    # Place the small piece in the reflected upper-right silhouette.
    repeat_action(env, 5, 1)
    repeat_action(env, 1, 4)
    repeat_action(env, 4, 9)
