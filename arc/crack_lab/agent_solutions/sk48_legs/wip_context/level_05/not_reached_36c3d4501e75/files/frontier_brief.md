# Unverified frontier brief: sk48 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The extended-contact probes rule out a latent “attach on collision, then shift” mechanic: even two horizontal runs after a genuine one-token vertical contact leave all `8`s fixed. At this point the unresolved ambiguity is the workspace’s “ordinary ladder” configuration label versus the preserved probes’ row-transfer model. I’m checking only the non-hidden workspace filenames for a clean local note or probe that names that configuration; I won’t inspect any prior transcript or hidden artifact.
- The barrier geometry points to the actual reusable abstraction: a tether ladder. Tokens that are peeled on alternating lanes can remain as joints even when they no longer move with the avatar; frame-only “attached/not attached” reasoning was too coarse. I’m tracing the minimal ladder primitive—extend above the block, retract the tip, then descend—at each alignment, including the visible tether span.
- The ladder trace shows retraction can preserve vertically staged joints even when no visible tether component remains, so I’m running a bounded best-first macro search keyed by both frame and recent tether operations. Its dense score rewards verified cross-block prerequisites—mixed vertical staging, rightward reach, and exact vertical contacts—and stops immediately on either reward or the first moved `8`.
- The search is still within its fixed 6,000-state cap; output is buffered, but the process remains active. If it does not return a transfer at the first progress boundary, I’ll stop it and seed the same search from the strongest verified staged states instead of enlarging the cap.
- The best-first run exposed a stronger verified frontier before I stopped it: `retract, up, extend×5` preserves one `9` on the left while placing two carried `9`s directly above the two rightmost `8`s. That is the first state with two simultaneous cross-block contacts. I’m seeding focused two-macro probes from it, including action 6 in that exact context, instead of resuming the expensive general queue.
- Two-macro expansion from the simultaneous-contact state yields only 31 visible configurations and no transfer, so ordinary movement alone is exhausted locally. I’m testing action 6 as a latent mode switch at the two states where it would matter—fully retracted/empty tether and simultaneous contacts—by comparing the complete next-action fingerprints before and after 6.
- The physical model now explains the transfer challenge: level 2 could lower an empty extended tether onto a row, but level 5 needs its three `9`s as temporary joints to get the tether across the block, then shed them without shortening it. Rather than widen replay BFS, I’m running bounded deterministic macro rollouts from the verified two-contact frontier; each rollout is cheap on one live clone and stops on reward or any moved `8`.
- The bounded rollouts found the first real transfer. A reproducible path pushes two `8`s down while leaving the wall `8` in place: `[(25,48), (31,36), (31,42)]`. This validates the ladder route and gives a dense subgoal. I’m tracing that path into meaningful runs, then I’ll continue from the transferred state toward the actual `898` reward and minimize only after it succeeds.
- The transfer trace is now interpretable: it builds a three-step ladder, then the final two downward moves push `8,8` into the lower lane while producing rows `9 / 998 / 88`. Random continuation has already produced a visible `898` row, but reward did not fire, proving geometry alone is insufficient—the selected row must also be the tether-controlled train. The continuation search is therefore checking the real reward, not accepting that visual false positive.
- The visual `898` state is precise but uncontrolled: the last upward push modifies only one vertical column, leaving the target row jammed above the avatar while extra tokens remain staged. I’m mapping the short action runs from this state now; the next subgoal is to make that `898` row move with the tether, which is the missing condition indicated by the absent reward.
- The `898` state reveals the correct next anchor: the only horizontally controlled object is the extra lower `8`; action 4 pushes it toward the right wall while the visible target row remains fixed. I’m seeding bounded reward rollouts directly from this much closer state so exploration no longer has to rediscover the ladder or the `898` arrangement.
- The candidate-seeded search has passed 200 bounded rollouts without reward. I’m letting this fixed run finish because each rollout explores a deep sequence on one clone; if it exhausts, I’ll switch from random continuation to a targeted “collect lower 8 against the wall, then ladder the far row” probe based on the newly verified controlled object.

## Preserved local probes

- `probe_level5.py` (10202 bytes)
- `probe_level5_candidate.py` (2018 bytes)
- `probe_level5_coordinate.py` (2751 bytes)
- `probe_level5_empty_below.py` (5705 bytes)
- `probe_level5_empty_collect.py` (2899 bytes)
- `probe_level5_ladder.py` (1509 bytes)
- `probe_level5_routes.py` (4998 bytes)
- `probe_level5_strong.py` (2333 bytes)
- `probe_level5_transfer_path.py` (1280 bytes)
- `probe_level5_vertical.py` (2251 bytes)
- `probe_prior_mechanics.py` (1887 bytes)
- `search_level5.py` (4089 bytes)
- `search_level5_bfs.py` (1408 bytes)
- `search_level5_eight.py` (2902 bytes)
- `search_level5_random.py` (3164 bytes)
- `search_level5_transfer.py` (3602 bytes)

Run or inspect the smallest relevant probe before writing another one.
