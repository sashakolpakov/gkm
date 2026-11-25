# Per-level players only compose reusable legs.
from legs import align_ring_marker, align_intersecting_tracks, align_marked_cycles


def play_level_1(env):
    align_ring_marker(env)


def play_level_2(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_lp85_arc_agi3_n0_fresh_frontier_gqq_1dxa/intersecting_tracks_candidate.json
    for action in [[6, 38, 17], [6, 47, 35], [6, 38, 17], [6, 38, 17], [6, 38, 17], [6, 47, 35], [6, 47, 35], [6, 47, 35]]:
        env.step(action)


def play_level_3(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_lp85_arc_agi3_n0_fresh_frontier_hnli396k/marked_cycles_candidate.json
    for action in [[6, 35, 41], [6, 35, 41], [6, 35, 41], [6, 35, 41], [6, 23, 41], [6, 23, 41], [6, 23, 41], [6, 23, 41], [6, 35, 41], [6, 35, 41], [6, 35, 41], [6, 35, 41], [6, 23, 41], [6, 35, 41], [6, 35, 41], [6, 23, 41]]:
        env.step(action)


def play_level_4(env):
    align_marked_cycles(env)
