# Unverified frontier brief: vc33 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay within the workspace and begin from the indexed API surface, then reproduce any prior observations before relying on them. I’ll keep probes bounded and symbolic, and put all reusable mechanics in `legs.py` with `play_level_4` only composing those legs.
- The baseline is reproduced: `levels=3`, `moves=33`, replay valid. The frontier contains no level-4 observations, so I’m now creating a level-4 clone through the harness and reducing it to component/control signatures rather than dumping frames.
- The level is another coordinate-control configuration, but not the same relay: six buttons split into paired controls. A verified single-step effect is that the left `4/11` platform moves exactly three rows up or down; its cyan marker begins 18 rows below the fixed cyan marker. I’m testing the direct six-step alignment hypothesis first, then I’ll check whether the right-side controls gate or reshape the target.
- The existing alignment leg cannot solve this configuration within its bounded search because the first upward move consumes a local resource; repeated presses stall. I’m switching to the required dense-progress decomposition: search only for the next strict reduction in marker-row gap, commit that path on a clone, and repeat. This will reveal the necessary recharge/relay pattern without assuming it.
- The right-hand buttons only conserve and relay three-row “air units” among their three reservoirs; they do not affect the left pair in the tested sequences. That makes marker alignment impossible using buttons alone unless the two color-1 barrier pieces provide the cross-wall interaction. I’m now testing those visible objects directly with action 6 instead of treating color 9 as the only clickable affordance.
- Clicks on the barrier pieces and markers are verified no-ops. The visible mechanics form two conservative hydraulic subsystems, so the likely win is a particular combined configuration rather than immediate marker motion. I’m doing one bounded clone BFS over the six verified buttons, keyed only by observable pixels below the changing header, to test all reachable configurations without inventing a hidden interaction.
- The six-button state graph is fully exhausted at 532 observable configurations and contains no reward, so a button-only plan is ruled out. Since a no-op coordinate click can still be a hidden select/attach action, I’m now comparing two-step contexts—object then button, button then object, and object pairs—against their one-step baselines.
- No two-step selection effect appears in the initial state. I’m extending the finite hydraulic-state sweep with context-sensitive clicks on every visible non-button component at each reachable configuration. This specifically checks whether a barrier, marker, or chamber becomes interactive only at a pressure/height threshold.
- A threshold interaction is now reproduced: two downward hydraulic moves turn the first color-1 barrier into color 12; clicking that active barrier transfers the `4/11` platform from the left chamber into the adjacent one. So the platform is the controllable avatar, barriers are context-sensitive transfer gates, and the task is cooperative reachability across chambers. I’m searching one verified subgoal at a time—advance the avatar one chamber, then reduce its height gap—using only buttons plus visibly active gates.

## Preserved local probes

- `probe_level4.py` (13644 bytes)

Run or inspect the smallest relevant probe before writing another one.
