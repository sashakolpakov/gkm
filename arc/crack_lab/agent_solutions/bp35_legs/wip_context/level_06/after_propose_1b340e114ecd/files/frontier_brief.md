# Unverified frontier brief: bp35 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The existing solver cleanly reaches the start of level 6 and then stops because no player exists. I’m adding a tiny observational probe that reaches that state through the local solver, then reports only compact components and per-action deltas on clones.
- The fresh-path probe is still solving the five known levels; no level-6 observations have been emitted yet. I’m letting that bounded run finish rather than inspecting any stored prior artifact.
- The clone experiments identify color-9 as the avatar: action 3 moves it one band left and action 4 one band right. A bare action 6 only changes the move counter, consistent with it being coordinate interaction. The scene is on the same 6-pixel lattice as earlier levels, so I’ll first test the existing gravity-room leg on a clone and separately probe action 7 plus context-targeted clicks.
- The existing room solver does not find a route. The key new affordance is already visible: the color-12 support cells respond even when clicked remotely, while color-15 cells do not; the old leg deliberately excluded those distant support actions. I’m narrowing the mechanism with short sequences before widening the search.
- The decisive context is now verified: toggling is safe either from the open shaft at column 6, or from columns 4/5 after expanding the matching lower support. A safe toggle moves the avatar upward from band row 6 to row 4—our dense progress signal. The prior leg missed all three because its interaction radius was one band; I’ll parameterize that existing leg and test radius two rather than creating unrelated level logic.
- The widened existing search is still running within its state cap. It has not yet returned a route; if this pass exhausts, I’ll preserve the verified safe-toggle macro and search room-to-room from that denser progress frontier.
- The safe toggle reveals a narrow upper corridor with the avatar at column 6, a color-8 toggle at column 2, and a distinct color-7 object at column 0. That color-7 object is the only new candidate goal, so I’m testing the direct reachable corridor path toward it before doing any broader search.
- The next region confirms a cooperation constraint: the color-9 avatar is trapped to the right of a wall at column 2, while the persistent color-7 object occupies an open corridor on the left. Clicking color 7 has no immediate visible effect, so I’m testing the context-dependent “select, then move” affordance—exactly the kind of non-moving interaction the action model calls for.
- The “select” hypothesis is falsified: color 7 is a second gravity switch. Activating the reachable 8→7 pair returns to a similar-looking chamber with both switches gone and an opened shaft, so it is genuine structural progress rather than a loop. I’ll use that verified 14-action macro as a staged root and search only the remaining chamber, which avoids replaying thousands of irrelevant pre-switch states.
- A strong dense-progress transition is verified: after clearing the switch pair, standing in the opened shaft at column 4 and activating the aligned color-12 support scrolls into a new chamber. Column 5 does not, so this is spatially aligned support use, not arbitrary clicking. I’ll repeat that alignment test against the newly exposed supports.
- The apparent support transition was a timing coincidence: two different orderings produced the same new chamber, showing that entering the open column-4 shaft caused the scroll and the trailing click was irrelevant. In the new chamber the next barrier’s only open shaft is column 6, so I’ll follow the observable openings directly—column 4, then column 6—without unsafe support activation.
- The shaft pattern is now reproducible: entering column 4, then column 6, then column 4 advances three distinct chambers. The next chamber has a full support barrier, but its only hazard-free vertical lane is column 2; aligned supports under hazard columns kill, explaining the earlier failures. I’ll activate the column-2 support from that safe lane.

## Preserved local probes

- `probe_level6.py` (4255 bytes)
- `probe_stage.py` (2391 bytes)

Run or inspect the smallest relevant probe before writing another one.
