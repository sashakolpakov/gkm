# Per-level players only compose reusable legs.
from legs import complete_glyph_substitution


def play_level_1(env):
    complete_glyph_substitution(
        env,
        examples=[(r, c, c + 10) for r in (5, 14, 23) for c in (13, 36)],
        slots=[(41, 52, c) for c in (15, 22, 29, 36, 43)],
    )
