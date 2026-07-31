# Unverified frontier brief: lf52 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay within the current workspace and the documented harness surface. I’ll begin with the indexed file map, then reproduce only the observations needed for level 5 with compact clone probes before changing the leg library and player dispatch.
- Baseline is confirmed at four completed levels. I’m now inspecting only the indexed solver bodies plus the harness entry points needed to resume/clone level 5; the frontier notes are treated as hypotheses, not evidence.
- The reproduced level-5 frame is the same mechanic family as the prior bridge/carrier leg: peg slots and pegs, persistent bridge markers, and a key-movable bordered carrier. The one-step probes also confirm only horizontal keys materially move the upper carrier assembly in the initial state. I’ll first compose the existing general leg exactly as requested; only if replay fails will I extend the leg.
- The existing leg makes verified dense progress—two bridge transfers followed by a real capture reduce the board from two pegs to one—but its alignment search stops at 120 symbolic states before that lone peg reaches the winning configuration. This is a search-budget mismatch, not a new mechanic, so I’m testing the same reusable leg with a larger bounded alignment frontier.
- The 1,000-state clone search is too expensive in this runtime and has not finished promptly. I’m stopping that broad attempt and will tighten the symbolic key/transition search around the one-peg endgame, keeping probes within the requested budget.
- The earlier level’s verified replay confirms that reward fires on its final one-peg capture, so “one peg anywhere” is not sufficient in this configuration. The greedy leg took the only immediate capture too early. I’m upgrading that same bridge/carrier leg with an optional bounded macro search keyed only by puzzle-relevant geometry; level 5 will use it, while earlier levels keep their proven greedy route.
- The 300-state macro frontier completes but finds no win, which means plain breadth-first search is spending its budget on carrier motion. I’m narrowing around the verified decision point immediately before the premature capture to identify alternative bridge transfers and prioritize meaningful peg/bridge progress.

## Preserved local probes

- `probe_l5.py` (3536 bytes)
- `probe_level2.py` (6693 bytes)

Run or inspect the smallest relevant probe before writing another one.
