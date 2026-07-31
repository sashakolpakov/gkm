# Unverified frontier brief: wa30 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- Targeted probes show the large central outline—not the two right-side pads—is the real dense objective. The moving courier fills it, but slowly; hand-carrying across the wall works, yet consumes too many turns. I’m testing a shorter reusable “feed the corridor port” strategy, staging right-side cargo at the wall gap so the courier performs only the final local delivery.
- A better timing emerged: after two useful deliveries, the competing courier reaches the avatar beside the central target. That allows dismissal without spending eighteen idle turns waiting at its exit. I’m verifying this contact and then using the saved turns for the last remote cargo.
- I found the main budget leak: the level-8 leg continues for 24 actions after its reward loop has already advanced to level 9. Those actions are provably cross-level spill—the loop condition exits only when `levels_completed` changes—so I’m removing them before designing the level-9 leg. This restores the pristine level-9 start and roughly 84 available turns.
- The targeted staging probe found the compact mechanic: place the first remote cargo at row 8, column 4. It stays available beside the avatar while also creating the exact contact geometry needed to dismiss the competing courier at turn 29. This combines staging and interception instead of paying for them separately; I’m now validating the final two deliveries from that state.
- The staged-interception sequence is verified, but two slots still remain under naïve routing. I’ve launched a narrow, score-guided 31-turn search from the verified post-dismissal clone; unlike the earlier generic BFS, it retains only 80 states per depth and ranks them by actual central-slot occupancy and cargo distance.
- The first beam hit the arena’s hard boundary while expanding the 60th action, confirming the effective level-9 allowance is 59 actions, not 60. I reduced the frontier and depth accordingly; this second run is bounded to legal states only.
- I’m testing the remaining cooperation hypothesis directly: after dismissal, move the lower-left cargo only to the courier’s return lane (10 actions), then hand-deliver the remote cargo while that courier completes the local leg in parallel. This fits exactly within the legal 59-action horizon.
- The contact geometry is now solved more efficiently: after two direct deliveries, step below the competitor, cut left across its turn, and interact. That dismisses it in six actions at turn 42—no idle interception. The avatar then sits beside the last local cargo with 17 actions remaining, enough for one final hand delivery while the helpful courier finishes its route.
- The final two loose blocks can be separated without stealing the courier’s assignment: stage the lower block on row 7, column 3 for the courier, then hand-place the upper block into the vacated center. This uses 19 actions from the post-dismissal state and preserves one turn for the courier’s delivery check.
- The real harness confirms the current leg is one delivery beyond the hard cap. I’m tightening the verified sequence rather than extending it: a small depth-5 clone search will now find the shortest possible dismissal from the two-delivery state, which is the only remaining place to recover actions without changing the solved subgoals.
- The five-action dismissal is provably shortest. The last remaining lever is preserving the center cargo while dismissing the thief; I’m searching only nine actions from the two-delivery state for a contact path that removes color 15 without opening the center cell. This is a much smaller, mechanics-specific frontier.
- The preserve-and-dismiss frontier has not produced a qualifying state yet and is nearing its bounded limit. I’ll let this bounded run finish, then use its result—or lack of one—to finalize the shortest viable route without expanding the search scope.

## Preserved local probes

- `probe9.py` (13826 bytes)

Run or inspect the smallest relevant probe before writing another one.
