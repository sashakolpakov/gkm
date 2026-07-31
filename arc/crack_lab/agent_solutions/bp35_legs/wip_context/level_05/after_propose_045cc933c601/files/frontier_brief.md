# Unverified frontier brief: bp35 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed surfaces. I’ll first establish the current level, then use bounded clone probes and keep all level-specific behavior composed through reusable legs.
- The baseline is clean: levels 1–4 solve reproducibly in 113 moves, so level 5 is the only missing dispatch. I’m now identifying the level-5 raw objects and action effects from a fresh clone, using compact component/delta summaries rather than frame dumps.
- The level-4 search is the expensive part of every probe replay, not the new level itself. I’ve isolated the probe so its eventual output will report only level-5 colors, components, and per-action deltas; subsequent experiments will reuse a shallow level-5 clone within the same run to avoid repeating that cost.
- The controlled experiment identifies the avatar unambiguously: actions 3 and 4 move the color-9/11 object one lattice column left/right; action 6 without coordinates only consumes a move, confirming it is contextual/click-like. Level 5 contains the same gravity toggle (8), removable supports (14), and shape-changing hazards (15) as level 4, plus a distinct color-7 target. That strongly suggests an earlier mechanic in a new layout, so I’m testing the existing gravity-room leg before adding any new behavior.
- A second context check shows why the first toggle appeared lethal: it is safe only after walking into columns 4 or 5. From there it lifts the avatar into the middle room; from columns 0–3 it kills. This gives a useful dense progress measure—survive each gravity phase while moving from the lower chamber toward the color-7 chamber. I also found the concrete mismatch in the reused leg: level 5 is eight columns wide, while the earlier scanner only covered seven. I’m testing that minimal generalization now.
- The widened scanner alone is not enough. After the safe toggle, the avatar lands on a removable support in the middle chamber; notably, several other supports remain in disconnected columns. The earlier leg deliberately considered only supports directly under the avatar, but this layout can require staging remote supports before later gravity phases. I’m extending the same search leg with an optional remote-support mode for level 5, preserving level 4’s narrower behavior.
- The expanded search is exploring rather than failing immediately, which confirms the new support states are reachable. I’m keeping the bound finite and will use its result to decide whether the missing affordance is remote staging or a more specific interaction; I won’t commit any speculative path unless the clone actually raises the level reward.
- The broad run exposed a performance flaw in the existing search implementation: it made a clone of an already cloned checkpoint state, causing deep-copy chains to grow with exploration. I stopped it before wasting more budget. I’m fixing the search to reconstruct directly from the shallow environment and reducing the state bound; this changes no game behavior, only clone cost.
- The clone-chain fix helped correctness but the all-support branching is still too broad for the intended probe rate. I’m treating this run as a bounded falsification test; if it does not finish promptly, I’ll replace breadth-first enumeration with phase-guided search using the verified “survive toggle and enter the next chamber” progress measure.
- The search now uses path replay only—no successor is cloned from a deep node—matching the bounded replay scaffold. The remaining cost is combinatorial, so after this finite run I’ll narrow actions by verified phase relevance rather than increase the cap.
- A key observation changes the model: color 7 disappears after the first gravity phase, even though the avatar survives. That is likely the independently affected second agent emphasized by the level design, not a passive prize. I’m now testing whether the upper-room supports must be staged before toggling so both agents survive; the dense measure becomes “advance gravity phase while retaining both colored agents.”

## Preserved local probes

- `probe_level5.py` (1382 bytes)

Run or inspect the smallest relevant probe before writing another one.
