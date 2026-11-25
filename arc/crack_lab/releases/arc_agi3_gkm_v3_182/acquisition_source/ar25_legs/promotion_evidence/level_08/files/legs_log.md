# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 3: reflected source pairs

Reproduced from the validated 26-action entry using probe_level3.py. Action 5 cycles divider, L source, then short source. Directions translate the selected object by three pixels. Moving the divider seven steps up centres the two marker pairs. Sources can cross the divider; colour 4 shows their vertical reflections. Moving the L source down seven and right seven, then the short source down five and left twelve completes the level on a clone. Reward remains 2 after aligning the first pair and becomes 3 after both pairs align. Touching sources share a colour component, so outline template tracking preserves identity.

New general legs: align_reflection_axis and guide_selected_shape_to_marker. play_level_3 composes these with switch_control. Validation: python gkm_try.py -> levels=3 moves=66 replay_ok=True err=None. reflection_pair_candidate.json preserves the full independently replayable path.

Level 5: reproduced on pristine-entry clones using probe_level5.py. Action 5 cycles horizontal divider, vertical divider, source; cardinal actions translate only the selected object along its permitted axis. Marker pixel centroid is (28,25); divider centres start at (16,10). Centring both axes then using guide_selected_shape_to_marker raises clone reward from 4 to 5. Extended align_reflection_axis with axis=1 and long-run detection for crossed dividers. play_level_5 only composes existing legs. Resumed and fresh harness runs both report levels=5 moves=116 replay_ok=True err=None. Independently validated path saved as level5_crossed_axes_candidate.json.

Level 6 clean reproduction: probe_level6.py reproduces entry after 116 prefix actions using public frames and clones. probe_controls.py tests selection and directional motion after existing axis alignment. probe_templates.py identifies the first selected five-tile source at pixel origin (24,51), then the disconnected eight-tile source at (9,42); its complete motion template has a unique marker/reflection embedding at (45,21) after first-source placement. Existing component-only selection captures only a fragment of the second source. Extended guide_selected_shape_to_marker with optional motion_template for disconnected thin patterns; prior defaults preserved. Player 6 only composes existing legs. probe_candidate.py verifies a 53-action clone suffix and saves level6_motion_template_candidate.json. Both normal and GKM_FRESH_REPLAY=1 harness runs report RESULT levels=6 moves=169 replay_ok=True err=None. checkpoint.json was not edited.


Level 7 clean reproduction (probe_level7.py, probe_templates7.py,
probe_cover7.py, probe_click7.py): the parent replay reaches 6 levels in
169 actions. Action 5 cycles horizontal divider, vertical divider, thin
source, broad source. Directional probes translate selected objects one
3-pixel tile. At aligned divider tile row 7 and column 12, observed colour-4
tiles equal horizontal, vertical and combined reflections of colour-5
sources, excluding source-covered tiles. Source templates contain 9 and 14
tiles. A joint translation search finds placements whose source tiles and
three reflections cover all 42 marker tiles. This plan completes level 7
on a clone; the public harness independently confirms the composed player.
Clicks on source interiors select those sources, verified at both source
positions after divider alignment. The new reusable leg enumerates marker
coverage bitsets, minimizes movement among maximum-coverage joint plans,
verifies completion on a clone, then uses the existing shape guide with
explicit target origins. players.play_level_7 only composes imported legs.
Final harness: RESULT levels=7 moves=207 replay_ok=True err=None.
Candidate path: candidate_level7_reflected_cover.json (207 public actions).
checkpoint.json was read only and was not modified.

Level 8 verified in this workspace:
- `python gkm_try.py` initially returned levels=7, moves=207, replay_ok=True.
- `probe8.py` replays the admitted checkpoint and tests valid actions on clones.
  At entry, keys 1/2 move the horizontal colour-10 divider by one 3-pixel
  tile and alter colour-4 reflections while the colour-5 sources stay fixed;
  3/4 and 7 have no visible effect there. Key 5 transfers the control dots
  to the vertical divider. A background click has no visible effect.
- `probe8_compose.py` reproduces the composed clone experiment. The original
  axis locator stops at tile axes (8,6); strongest-run localization reaches
  (11,12). Sources occupy two connected patterns of 7 and 8 tiles.
  With aligned axes, the unchanged coverage leg selects and translates both
  sources, reducing visible marker tiles from 60 to zero and raising reward
  from 7 to 8. This verifies its reflection model and click/move affordances
  in the actual finishing context.
- `play_level_8` composes alignment, selection switching, and coverage legs.
  No new leg was necessary; alignment gained an optional strongest-run mode.
- Final harness: RESULT levels=8 moves=255 replay_ok=True err=None.
  Candidate full action path: level8_reflection_candidate.json; regenerate
  and independently validate with `python verify8.py`. Checkpoint untouched.
