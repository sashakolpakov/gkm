# Per-level players only compose reusable legs.
from legs import complete_glyph_substitution, complete_glyph_expansion, complete_glyph_rewrite


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


def play_level_3(env):
    complete_glyph_rewrite(
        env,
        examples=[
            ([(5, 6)], [(5, 16)]),
            ([(5, 29), (5, 36)], [(5, 46), (5, 53)]),
            ([(14, 6)], [(14, 16), (14, 23)]),
            ([(14, 36), (14, 43)], [(14, 53)]),
            ([(23, 6), (23, 13), (23, 20)], [(23, 30)]),
            ([(23, 43)], [(23, 53)]),
        ],
        sources=[(41, c) for c in range(5, 55, 7)],
        answers=[(52, c) for c in range(8, 51, 7)],
    )


def play_level_4(env):
    # The two demonstrated substitutions compose through matching middle glyphs.
    complete_glyph_rewrite(
        env,
        examples=[
            ([(4, 36)], [(20, 23)]),
            ([(12, 13)], [(28, 46)]),
            ([(20, 36)], [(12, 46)]),
            ([(28, 13)], [(4, 23)]),
        ],
        sources=[(41, c) for c in range(8, 51, 7)],
        answers=[(52, c) for c in range(8, 51, 7)],
    )
