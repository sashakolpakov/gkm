# Unverified frontier brief: sk48 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The targeted probe found the clean detachment in four actions: retract the final `9` to the near column, then cross vertically; the avatar moves while the `9` stays. That establishes an empty train. The remaining issue is access—the parked `9` sits before the three `8`s—so I’m verifying a small lift-and-repark maneuver to clear the target row without reordering anything accidentally.
- The row can be cleared cleanly: approach the loose near `9`, extend three times to engage it, lift it one lane, retract it to the near column, then cross back down while it remains parked. This leaves the avatar empty on the `8` row. I’m now verifying the first true dense milestone—collecting exactly one `8` as the target prefix, without dragging the whole `8` row.
- The corrected clear maneuver is verified: the target row now contains only the three `8`s, while all `9`s are parked above. Four extension steps do not yet reach the nearer `8`, so I’m measuring the exact reach/retraction pair instead of assuming the earlier-level distance transfers unchanged.
- The failed reach grid is informative: once the last controlled token is fully released, actions 3/4 no longer affect any piece, so “empty train” is not a viable collection state by itself. This is exactly a context where the otherwise inert action 6 may matter; I’m testing it now on the released avatar before abandoning that branch.
- Action 6 is inert even after release, confirming that branch is dead. I’m returning to the last useful controlled `9` and testing a tighter mechanic: retract it to create approach room, extend into the `8` chain, then retract again. If a contacted `8` moves, that gives the necessary `[9,8]` intermediate while preserving control for later reversal.
- Plain contact/retract does not move an `8`, so the missing step is likely a contextual interaction at contact rather than more reach. I’m inserting action 6 only at candidate contact distances and checking whether any `8` subsequently moves; this directly tests the affordance in the relevant context.
- A more promising physical interpretation emerges from the spacing: lift the `999` train, extend it one cell, then lower it into the single-cell gap before `888`. That should form one contiguous six-token train without pushing against the wall. I’m verifying whether a subsequent retract moves the `8`s too—this would establish the needed attachment mechanism.
- Lowering the shifted train is blocked before overlap, so adjacency itself is collision—not attachment. One unresolved possibility is a latent mode change from action 6, since it can change state without changing pixels. I’m checking post-6 action deltas at the initial, separated, and contact-ready states; this is the required second-context test for a hidden interaction.
- The post-6 deltas are identical in all tested contexts, so action 6 is a true no-op here. I’m switching to a bounded macro best-first probe with a stricter dense goal: not merely “an `8` is first on the avatar’s row,” but that the `8` also follows the avatar on a vertical test, proving it is controlled. The search expands repeated directional runs compactly and stops at 3,000 symbolic states.
- The macro search is still inside its fixed cap and has not yet emitted a better controlled-prefix state. I’ll poll it once more; unlike the earlier deep-clone BFS, this search reconstructs shallow macro paths and reports both the best dense state and exhaustion, so its negative result is still actionable.
- The bounded search found a new verified dense transition: `up, retract, down` peels one leading `9`, reducing the avatar-row sequence from `999888` to `99888`. That is simpler than the reversal leg. I’m now repeating this general peel loop to see exactly where control transfers and whether it exposes an `8` head.
- With a single controlled `9` remaining, I can isolate the exact vertical-contact experiment: align it above each of the three `8`s and push down once. This will distinguish “collision blocks” from “vertical push/swap,” without a broad search.

## Preserved local probes

- `probe_level5.py` (10202 bytes)
- `search_level5.py` (4089 bytes)

Run or inspect the smallest relevant probe before writing another one.
