# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 3: reflected source pairs

Reproduced from the validated 26-action entry using probe_level3.py. Action 5 cycles divider, L source, then short source. Directions translate the selected object by three pixels. Moving the divider seven steps up centres the two marker pairs. Sources can cross the divider; colour 4 shows their vertical reflections. Moving the L source down seven and right seven, then the short source down five and left twelve completes the level on a clone. Reward remains 2 after aligning the first pair and becomes 3 after both pairs align. Touching sources share a colour component, so outline template tracking preserves identity.

New general legs: align_reflection_axis and guide_selected_shape_to_marker. play_level_3 composes these with switch_control. Validation: python gkm_try.py -> levels=3 moves=66 replay_ok=True err=None. reflection_pair_candidate.json preserves the full independently replayable path.

Level 5: reproduced on pristine-entry clones using probe_level5.py. Action 5 cycles horizontal divider, vertical divider, source; cardinal actions translate only the selected object along its permitted axis. Marker pixel centroid is (28,25); divider centres start at (16,10). Centring both axes then using guide_selected_shape_to_marker raises clone reward from 4 to 5. Extended align_reflection_axis with axis=1 and long-run detection for crossed dividers. play_level_5 only composes existing legs. Resumed and fresh harness runs both report levels=5 moves=116 replay_ok=True err=None. Independently validated path saved as level5_crossed_axes_candidate.json.
