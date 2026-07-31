# Unverified frontier brief: sk48 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed API map, then reproduce any inherited observations before probing level 5. I’ll keep all experiments bounded and symbolic, and only commit the final composed-leg solution after clone verification.
- The baseline is clean and reproducible: levels 1–4 complete in 175 moves, with replay validation passing. There is no `frontier_brief.md`, so I’m moving directly to a fresh level-5 clone and will first map the harness entry points plus compact object/action deltas.
- The fresh probes identify the controllable object and mechanics: actions 1/2 move the left-hand avatar vertically by one lane, action 3 retracts its horizontal tether, and action 4 extends it but is initially blocked. Actions 5 and 6 are contextually inert here; 7 is rejected by this harness. The level presents two three-token rows on the avatar’s lane, so I’m now decoding the requested ordering and barrier geometry before testing reusable train manipulations.
- The first bounded clone search exhausted its small frontier without a reward path, so I’m not expanding it blindly. The verified transition model shows the initial attached train is `9-9-9`, while the requested train is `8-9-8`; I’m switching to a compact logical-state progress measure (ordered attached prefix plus staged-piece positions) to target the required interleaving.
- The reusable row-reversal leg does make verified dense progress: it parks two surplus `9` tokens on separate upper lanes and leaves exactly one `9` attached at the divider. A 1,000-state continuation search from that reduced state found no completion; I’m collapsing observational keys to piece positions and widening only this promising continuation, which avoids spending states on irrelevant pixel animation.
- The widened continuation is still running past the first minute, with no reward path emitted yet. I’m letting this bounded pass finish, but I’ll cap it at the declared 5,000 symbolic states and use its result—success or exhaustion—to choose the next leg composition rather than increasing the frontier again.
- The pass has reached roughly the expected runtime for 5,000 states, so I’m checking once more for completion. If it remains nonterminal, I’ll stop it and replace the generic frontier with targeted parameter probes; the current dense-progress staging remains preserved only on clones.
- Pairwise and triple compositions of the three existing legs do not complete level 5. I’m now using the verified target itself as the dense metric: the leftmost avatar-connected train is scored by target-prefix length (`8`, then `8-9`, then `8-9-8`), target subsequence retained, and surplus attached pieces. This search starts only after the reusable reversal staging, where one useful `9` remains attached.
