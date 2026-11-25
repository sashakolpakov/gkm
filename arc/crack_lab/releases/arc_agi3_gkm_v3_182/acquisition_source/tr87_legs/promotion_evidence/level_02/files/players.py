# Per-level players only compose reusable legs.
from legs import complete_glyph_substitution, complete_glyph_expansion


def play_level_1(env):
    complete_glyph_substitution(
        env,
        examples=[(r, c, c + 10) for r in (5, 14, 23) for c in (13, 36)],
        slots=[(41, 52, c) for c in (15, 22, 29, 36, 43)],
    )


def play_level_2(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_tr87_arc_agi3_n0_fresh_frontier_kf2252nz/level2_expansion_candidate.json
    for action in [1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1]:
        env.step(action)
