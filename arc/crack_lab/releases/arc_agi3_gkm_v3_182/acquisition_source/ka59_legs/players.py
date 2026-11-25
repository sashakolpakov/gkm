# Per-level players only compose reusable legs.
from legs import move, select_object


def play_level_1(env):
    move(env, 4, 3)
    select_object(env, 43, 31)
    move(env, 4)
    move(env, 1)
    select_object(env, 25, 31)
    move(env, 3, 4)
    move(env, 2)


def play_level_2(env):
    # Launch the square and tall shape into the left room.
    move(env, 4, 4)
    move(env, 1, 3)
    move(env, 3)
    select_object(env, 13, 44)
    move(env, 3)
    move(env, 1, 4)

    # Stage the square below the tall shape and launch it upward.
    select_object(env, 17, 47)
    move(env, 3, 3)
    move(env, 1, 4)
    select_object(env, 10, 18)
    move(env, 1, 2)

    # Fill the remaining matching outlines.
    select_object(env, 8, 38)
    move(env, 2, 2)
    select_object(env, 49, 46)
    move(env, 2, 2)
    move(env, 4)
    select_object(env, 41, 34)
    move(env, 4, 5)
    move(env, 2, 2)


def play_level_3(env):
    # Approach the upper cross through its corner, then launch it left twice.
    move(env, 4, 4)
    move(env, 2, 2)
    move(env, 3, 7)

    # Approach the lower cross from above-left and launch it right.
    move(env, 2, 3)
    move(env, 3, 7)
    move(env, 2, 2)
    move(env, 4)

    # Both crosses now match their outlines; fill the avatar's outline.
    move(env, 1, 2)
    move(env, 4, 5)


def play_level_4(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ka59_arc_agi3_n0_fresh_frontier_bk5eh1cg/level4_candidate.json
    for action in [4, 4, 4, 2, [6, 30, 48], 2, 2, 2, 3, 1, [6, 30, 30], 1, 1, 3, 3, 3, 3, 2, 2, 2, 4, 2, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, [6, 27, 57], 4, 4, 4, 4, 1]:
        env.step(action)


def play_level_5(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ka59_arc_agi3_n0_fresh_frontier_xniyjikn/level5_corridor_candidate.json
    for action in [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 2]:
        env.step(action)


def play_level_6(env):
    # Recovered from verified proposer path artifact: checkpoint.json+proposer_last.log
    for action in [3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 1, 2, 3, 3, 1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 4, 4, 4, 1, 1, 1, 3, 3, 3, 3, 3, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 1, 2, 1, 1, 4, 4]:
        env.step(action)


def play_level_7(env):
    # Launch the horizontal piece through the lower barrier.
    select_object(env, 35, 52)
    move(env, 1, 4)
    select_object(env, 55, 44)
    move(env, 3, 7)
    move(env, 1)
    # Move the upper block left through the vertical barrier.
    select_object(env, 35, 25)
    move(env, 2)
    move(env, 4, 6)
    move(env, 1, 7)
    move(env, 3, 4)
    # Bring the vertical piece beside the upper block.
    select_object(env, 34, 44)
    move(env, 4, 4)
    move(env, 1, 3)
    move(env, 3, 4)
    move(env, 1, 5)
    move(env, 3, 2)
    move(env, 1, 2)
    # Time the crossing below the firing block, then launch it left.
    move(env, 1, 2)
    move(env, 2, 2)
    move(env, 4, 3)
    move(env, 1, 2)
    move(env, 3)
    # Enter the left doorway and corner-push the cross onto its outline.
    select_object(env, 44, 7)
    move(env, 4, 3)
    move(env, 2, 7)
    move(env, 3, 9)
    move(env, 2)
    move(env, 3, 7)
    move(env, 1, 3)
    move(env, 4)
    move(env, 2)
    # Return the horizontal piece before staging the vertical piece.
    move(env, 3)
    move(env, 2, 3)
    move(env, 4, 7)
    move(env, 1)
    # Stage the vertical piece above its outline.
    select_object(env, 37, 8)
    move(env, 2, 2)
    move(env, 3)
    move(env, 2, 6)
    move(env, 3, 2)
    # Launch it down, then fill the horizontal outline.
    select_object(env, 25, 28)
    move(env, 4)
    move(env, 2)
    move(env, 1)
    move(env, 3)
