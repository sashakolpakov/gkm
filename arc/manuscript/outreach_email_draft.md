Subject: ARC-AGI-3 self-improvement artifact: request for feedback / collaboration

Dear Professor [NAME],

I am writing to ask whether you would be willing to look at a short one-page summary of a project we are developing around ARC-AGI-3.

The project is called GKM. The core idea is a verifier-driven self-improvement loop for interactive ARC agents: a coding proposer grows solver structure from the raw game interface, the simulator verifies progress by replay, and each new structure is charged by marginal description length. In other words, we are not only asking whether an agent solves levels, but how much new reusable structure it had to invent to do so.

This differs from executable-world-model agents in a specific way. Those systems primarily optimize around building a predictive simulator and planning through it. GKM treats a world model as only one possible useful structure: a promoted artifact can be a probe, perception routine, BFS, planner, reusable leg, literal replay path, or world model. The selection rule is replay-verified reward improvement plus marginal complexity cost. The attached one-pager highlights the resulting `ls20` sawtooth profile: marginal complexity drops when legs transfer and spikes when the game introduces a new mechanic. That profile is the evidence that the system is tracking reusable structure, not only solving levels.

Our current promoted artifacts solve two local ARC-AGI-3 preview games:

- `wa30`: 9/9 levels, replay-validated, 596 actions.
- `ls20`: 7/7 levels, replay-validated, 393 actions.

The comparison that makes this worth discussing is `wa30`: the closest executable-world-model baseline I found reports 7/9 on `wa30` with GPT-5.4 and 4/9 with GPT-5.5, while solving `ls20` 7/7. GKM is not claiming an official leaderboard result or a compute-matched win yet. The point is different: the promoted artifacts preserve WIP snapshots, replay validation, marginal complexity accounting, charged literal paths, and refactored reusable legs, so the solver-growth process is auditable.

I am attaching a one-page summary. What I would most value is not a courtesy acknowledgment, but concrete guidance on whether this is worth pushing into a collaboration-grade evaluation. In particular, I would appreciate help with any of the following:

1. Feedback on whether the artifact discipline is scientifically credible for ARC-AGI-3.
2. Advice on the right compute-matched comparison against graph exploration and executable-world-model agents.
3. An introduction to ARC-AGI-3 organizers or researchers who could review the leakage/audit protocol.
4. Potential collaboration, compute/model access, or funding routes for a systematic public-game benchmark run.

If you think the direction is flawed, I would also appreciate a blunt technical objection. If you think it is promising, I would be grateful for a short call or for a pointer to the person who would be most useful to contact next.

Best regards,

[YOUR NAME]
