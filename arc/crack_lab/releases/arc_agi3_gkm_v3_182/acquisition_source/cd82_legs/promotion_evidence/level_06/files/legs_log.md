# Verified leg library

## paint_half(env, side)
- Public clone observations: the red stencil has eight orientations. Keys 1--4 steer toward cardinal directions in 45-degree increments; directly opposite steering can be inert. Key 5 paints the facing half of the black canvas white. Repeated painting preserves previously white pixels. Coordinate clicks on canvas/stencil did not change those objects; clicking the black palette square selected black for a subsequent paint.
- `probe_initial.py` reproduces independent action deltas. `probe_shapes.py` and `probe_sequences.py` reproduce context probes. `probe_orientations.py` enumerates all eight red masks and their painted row counts.
- At pristine level 1, painting upward yields 50 white cells and no reward; the reference has black above white. Rotating via 3,2,2,4 and painting via 5 increases levels_completed to 1 on a clone.
- The leg selects orientation using cloned paint mismatch against the requested canvas half. It assumes a blank black canvas larger than the black reference component, a red stencil, and white paint selected.
- Harness validation: `RESULT levels=1 moves=5 replay_ok=True err=None`.
- Independently adoptable candidate: `level1_bottom_half_candidate.json`.

Level 2 clean reproduction: probe_level2.py identified the rotating colour-2
stencil and action-5 top-half paint. probe_paint.py verified palette clicks
(34,4)->0, (40,4)->15, (46,4)->12; clicks on the canvas had no paint effect.
After paint_half(top), selecting 12 and rotating [4,2,2] before action 5
completed the level on a clone. Added finish_reference_layer: select an observed
swatch, match square reference/canvas regions, and test at most eight stencil
orientations for an exact reference match or reward increase.
Harness result: levels=2 moves=11 replay_ok=True err=None.
Saved independent adoption path: candidate_level2_reference_overlay.json.
