# Unverified frontier brief: ka59 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the clean-room surface, begin with the index and frontier brief if present, then reproduce only the observations needed before implementing and testing level 5.
- Baseline is confirmed at four completed levels in 139 moves. I’m now advancing a clone to level 5 and will summarize its objects and one-step action effects symbolically, using the existing perception helpers rather than raw frame dumps.
- The level-5 frame has two solid color-15 barriers, a small hollow ring, a larger hollow destination, and a separate multicolor vertical piece. Arrow actions move something immediately; coordinate action alone only changes the visible selection marker. I’m narrowing which object each coordinate selects and measuring motion while excluding the one-row replay/status strip.
- A bounded visible-state BFS found a 20-arrow candidate. It is not a plain Manhattan route: several upward commands interact with the animated multicolor obstruction/barrier state before the ring can traverse right. I’m replaying that exact sequence and nearby shortened variants on fresh clones to verify both the dense ring-position progression and the sparse level reward before editing the solver.
- The replay confirms the cooperation mechanic: early upward attempts are periodically knocked back, then the cycling corridor agents hand the ring through; 12 upward commands are necessary, while 8–11 all fail. Level 5 therefore reuses the existing general movement leg with no new leg added. I’ve added that composition and am running the full clean replay validator now.
- Implemented `play_level_5(env)` in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_ka59_budgeted_medium_screen/players.py:113), composing only the existing `move_steps` leg. Validated end-to-end: `RESULT levels=5 moves=159 replay_ok=True err=None` No changes to `legs.py` were needed.

## Preserved local probes

- `probe_l4.py` (3341 bytes)
- `probe_l5.py` (3037 bytes)

Run or inspect the smallest relevant probe before writing another one.
