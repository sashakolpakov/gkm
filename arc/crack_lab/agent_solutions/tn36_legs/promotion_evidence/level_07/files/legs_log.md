# Leg-library debrief log

Recurring composition patterns and repeated novelty.

- Levels 1--4 configure or encode a segment panel and then click the largest
  color-9 submit disc. Candidate higher-order leg: `configure_then_submit`,
  preserving success gating and the optional captured-frame submit semantics.

- Levels 4--5 find a bounded lattice route, then compose that route with
  level-specific transform symbols before encoding it. Candidate higher-order
  leg: `encode_lattice_route_with_transforms`.

- Like levels 1--5, level 6 uses configure-then-submit, but repeats it across
  a bounded route plan and verifies reacquisition at each intermediate
  checkpoint. Candidate higher-order leg: `execute_reacquisition_plan`,
  parameterized by program encoding, completion, and reacquisition checks.

- Level 7 requires an ordered checkpoint column before a shaped agent can
  enter its exact socket cavity. Each checkpoint deselects the direction
  examples, so the protocol must be reselected before the next program; short
  programs terminate with empty suffix columns rather than transform padding.

- Levels 6--7 share a prepare-plan, encode-and-submit, verify-reacquisition
  loop even though their waypoint selection and route encodings differ.
  Candidate higher-order leg: `execute_reacquisition_plan`, parameterized by
  the stage list and a per-stage route encoder so protocol learning can remain
  either one-time or per-checkpoint.
