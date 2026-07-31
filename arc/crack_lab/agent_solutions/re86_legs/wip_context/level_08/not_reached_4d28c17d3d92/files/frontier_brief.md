# Unverified frontier brief: re86 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed surfaces, then reproduce any inherited observations before relying on them. I’ll first establish the current level, probe level 8 on clones with compact symbolic summaries, and only then edit the reusable leg plus the level dispatcher.
- The indexed code confirms levels 1–7 are already composed from reusable legs, and there is no inherited frontier brief to trust. The baseline run is still executing; once it reaches the frontier I’ll inspect only the documented harness ranges needed to clone and summarize level 8.
- The reproduced frontier is level 8. One-step experiments identify actions 1–4 as translating the currently selected multi-cell object by three pixels; action 5 does not move it and instead shifts a two-pixel selection indicator. The board has eight generic ring targets, multiple bordered colour stations/barriers, and overlapping movable shapes—so it appears to combine the earlier selection, repainting, and deformation mechanics in a new geometry. I’m now separating each selectable shape on clones and measuring target coverage as the dense progress signal.
- The board contains exactly two selectable 13×13 outline squares. The target geometry requires one to become a 19×7 rectangle painted 11 and the other a 10×16 rectangle painted 6; neither existing leg supports two separately deformed/repainted rectangles, so a new minimal leg is warranted. I’ve bounded the clone search to one parked shape and am searching observable shape geometry—not raw “frame changed”—for a route that exactly covers each four-marker group.
- The first geometry search is broader than needed and has not produced a route within its useful time budget, so I’m narrowing it rather than spending the clone allowance blindly. The key simplification is that movement positions alone are cheap to enumerate; the more expensive four-direction shape reconstruction only needs to run at positions capable of covering a target group.
- The narrowed search reproduced the deformation mechanic cleanly: a short route changes a square from 13×13 to the required 19×7 outline while preserving its current colour. I’m now treating repainting and placement as separate verified subgoals—first reach the colour-11 station without losing that geometry, then translate the finished outline onto the matching four rings.

## Preserved local probes

- `probe_level8.py` (3591 bytes)
- `search_level8.py` (4168 bytes)

Run or inspect the smallest relevant probe before writing another one.
