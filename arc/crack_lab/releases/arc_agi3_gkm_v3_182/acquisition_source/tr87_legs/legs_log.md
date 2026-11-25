# Leg-library debrief log

Recurring composition patterns and repeated novelty.

Level 2: Reproduced entry with the supervisor prefix (17 moves). Public clone
probes showed actions 3/4 wrap the answer cursor, and 1/2 cycle its glyph in
opposite directions. The answer ring contained seven glyph classes. Six examples
map single source glyphs to sequences of lengths 1,3,2,2,3,1. The four lower source
glyphs expand to seven answers; setting these completed a clone at level 2.
Added complete_glyph_expansion and factored the existing verified answer-setting
loop into complete_glyph_answers, shared by substitution and expansion. Dense
progress is the number of answer slots matching their demonstrated glyph class.
Harness validation: levels=2 moves=43 replay_ok=True. The independent clone path
is preserved in level2_expansion_candidate.json; checkpoint.json was not edited.

## Level 2: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_tr87_arc_agi3_n0_fresh_frontier_kf2252nz/level2_expansion_candidate.json` and installed a thin replay player for the recovered suffix.

Level 6: Reproduced the supervisor prefix: five completed levels, 121 actions.
Public action_deltas identified the bracket cursor: 3/4 select editable groups,
1/2 cycle glyphs inversely. Twelve selections wrap in reading order; each group
has a seven-state ring. Repeated context probes verified both-glyph cycling in
paired groups and inverse actions at every selection. The three rows contain
single-to-pair expansion rules and single-to-single substitution rules. The
fixed source has three glyphs; the fixed target has six. Enumerating coupled
expansions and enforcing consistent substitutions found a candidate whose
clone earned level 6. Added complete_glyph_cascade_synthesis, reusing
complete_glyph_answers for verified execution. Dense progress is consistent
matched target prefixes, with each configured group verified by the answer leg.
Preserved bounded probes in probe6.py and probe6_synthesis.py and the independently
verified full path in level6_cascade_candidate.json. Integrated harness result:
RESULT levels=6 moves=153 replay_ok=True err=None. checkpoint.json was untouched.
