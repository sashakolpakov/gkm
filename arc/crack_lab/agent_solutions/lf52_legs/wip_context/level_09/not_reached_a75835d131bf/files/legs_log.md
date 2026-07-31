# Leg-library debrief log

Recurring composition patterns and repeated novelty.

- **Plan, verify, then replay:** the carrier solvers repeatedly search on cloned
  environments, retain a mixed sequence of key and lattice-click actions only
  after it reaches the level goal, and replay that sequence on the live
  environment. A candidate higher-order leg is
  `plan_on_clone_then_replay(env, planner)`, with the planner responsible for
  returning a verified action sequence.
