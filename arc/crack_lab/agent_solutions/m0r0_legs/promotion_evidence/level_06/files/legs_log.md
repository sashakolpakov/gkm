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

- Candidate higher-order leg: levels 1, 2, and 5 are each a complete level made
  from one routed `reunite_mirrored_pair` call. If future players need setup or
  cleanup around that same core, a parameterized
  `complete_mirrored_reunion(env, route, before=(), after=())` leg could compose
  those phases without duplicating the reunion skill.

- Candidate higher-order leg: levels 3 and 6 both alternate coordinate-selected
  helper groups with routed main-pair motion. A
  `follow_selected_handoffs(env, stages)` leg could own the recurring
  select-route handoff composition while each stage retains its semantic route
  name.
