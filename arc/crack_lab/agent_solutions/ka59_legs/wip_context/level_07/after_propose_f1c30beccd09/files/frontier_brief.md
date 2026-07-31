# Unverified frontier brief: ka59 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace boundary, begin with `solver_index.md`, then reproduce any frontier observations through compact clone probes before editing the level-7 leg composition.
- Baseline is confirmed at level 7 (`levels=6`, 210 moves). The brief is only a hypothesis index; I’m now reproducing its geometry and action claims with the smallest preserved probes, then I’ll search for a verified sequence rather than trusting the notes.
- The fresh action probe verifies that only the two color-14 rings are selectable; arrows move the selected ring by three pixels, while both color-13 corridor agents advance every arrow turn. The three color-4 outlines are targets/transfer pieces: the large lower-left outline cannot be selected directly and must be pushed upward toward the color-11 outline. I’m using that verified three-distance measure for the bounded search.
- The search is clone-limited and hasn’t improved the dense score yet. I’m letting this bounded run finish while keeping the search space to four arrows plus selection of the other marked ring; if it stalls, I’ll reduce state keys to the movable geometry instead of expanding the raw-frame space.
- Centering both small rings is verified insufficient, including all six corridor phases. The missing subgoal is therefore the large transfer. The map shows a route for the horizontal ring beneath the lower-left large outline, so I’m probing direct contact from below and measuring whether successive upward commands displace the large outline.
- Direct access to the large lower chamber is physically blocked, so the earlier “push it from below” theory is falsified. The remaining plausible cooperative mechanic is context-dependent selection: I’m now retesting clicks on the large ring and corridor agents only after both small targets are occupied, since initial-state clicks were inert.
- Level 6 provides the decisive correction: color 11 is the large movable ring, while color 4 is its padded target. The prior search measured the static target instead of the ring, so it could never reward cooperative large-ring progress. I’m correcting the dense metric to track color 11 downward toward `(row 40, col 13)` and broadening ring switching to work even when a marker is hidden by a target.
- The corrected search is still running within its bounded state budget. Its priority now improves only when a small ring approaches its matching target or the color-11 ring actually descends, so any returned path will directly test the cooperation hypothesis.
- The corrected search confirms the large ring never moved in 40,000 target-directed states, so simply seating the small rings is another dead end. I’m narrowing to the two corridor contacts: probing both rings against the lower cycling agent from opposite sides and across all six phases, with the large ring’s displacement as the sole signal.

## Preserved local probes

- `probe_l4.py` (3341 bytes)
- `probe_l5.py` (3037 bytes)
- `probe_l6.py` (5807 bytes)
- `probe_l6_final.py` (956 bytes)
- `probe_l7.py` (2431 bytes)
- `probe_l7_focus.py` (1670 bytes)
- `probe_l7_push.py` (988 bytes)
- `probe_l7_search.py` (4469 bytes)
- `probe_l7_select.py` (1549 bytes)
- `probe_l7_state.py` (2858 bytes)
- `probe_l7_trigger.py` (1329 bytes)

Run or inspect the smallest relevant probe before writing another one.
