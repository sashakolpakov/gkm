# Leg-library debrief log

Recurring composition patterns and repeated novelty.

- Candidate higher-order leg: level 3 repeatedly selects a group, follows a
  verified route, then hands control to another group. A parameterized
  `follow_selected_routes(env, stages)` leg could compose an itinerary of
  `(selection, route)` stages when this pattern recurs in another level.

- Candidate higher-order leg: levels 3 and 4 both relocate selectable blockers,
  then reunite the mirrored pair through the cleared corridor. If this recurs,
  `clear_then_reunite(env, clearance_route, reunion_route)` could compose the
  two existing legs while keeping each level player declarative.
