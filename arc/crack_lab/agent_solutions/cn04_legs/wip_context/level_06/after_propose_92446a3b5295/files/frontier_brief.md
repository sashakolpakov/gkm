# Unverified frontier brief: cn04 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay within the workspace and the exposed harness surface. I’ll begin with `solver_index.md`, then reproduce any observations from `frontier_brief.md` before relying on them.
- The frontier brief contains only hypotheses; I won’t treat them as facts. I’m reproducing the baseline first, then I’ll inspect the smallest preserved symbolic probes and the documented API ranges they depend on.
- The reproduced frame confirms six occupied components: the selected 126-cell staging figure plus four separate black figures and a top boundary. Actions 1–4 translate the selected figure, 5 rotates it, and coordinate action 6 changes selection state; clicking any of the three green markers locks the same staging figure into black. I’m now reproducing the preserved component-reduction search, whose dense metric is the number of disconnected occupied components.
- The bounded search has now reproduced four verified subgoals, reducing occupied components from 6 → 5 → 4 → 3 → 2. The remaining search is slower because only one small disconnected figure remains; I’m letting this bounded probe finish before converting the verified action sequence into reusable leg calls.
- The preserved search stalls only at the final merge, not because the dense measure was wrong: it reaches one 513-cell assembly plus a 26-cell top component. I’m isolating that state and measuring its bounding boxes and legal one-step effects, then I’ll search the final placement with a compact state key instead of repeating flood-fill work at every node.
- The final-component hypothesis is now falsified: after selecting the only movable subfigure in the 513-cell assembly, a compact 20,000-state search finds no route to the top bar. That bar is therefore a boundary/reference, not another piece to join. I’m switching the dense measure from connectivity alone to arrangement correctness by reproducing the exact level-5 pre-win transition.
- The level-5 transition confirms that reward is tied to a valid final interlock, not merely proximity. At the current level-6 frontier, the figures are touching but the accumulated occupied area changed during intermediate rotations, so this is an invalid assembly. I’m enumerating the distinct selectable subfigures inside that merged shape and testing whether a short corrective move can reach reward before discarding the frontier.
- Four of the six distinct selections have no corrective route within 8,000 compact states; the search is checking the last two. The enumeration also reveals the concrete defect: two figures have lost one 3×3 cell and another has gained cells, so the connectivity path involved invalid collisions. If the remaining checks fail, I’ll restart from the clean level-6 frame and search only area-preserving placements.
- All local corrections failed, so I’ve discarded the corrupted frontier. The clean search will now reject every transition that changes the total occupied area, use component count only as a staged progress signal, and require the real reward at the end. This directly prevents the cell-loss/cell-growth failure observed above.
- The area-preserving constraint works, but the first clean merge `[2, 2, 5]` is a dead end. I’m now backtracking over distinct clean first placements and asking only whether each admits a second clean merge; this is a bounded high-level branch search, not stepwise brute force over the whole level.
- The level-1 trace exposed the real measurement bug: color `0` is not empty—it is part of a figure in that level. Only the dominant background is empty. Excluding `0` created false component reductions and the entire corrupted frontier. I’m correcting the symbolic probe to count every non-background cell, then rerunning the bounded staged search from a clean clone.

## Preserved local probes

- `probe_branch.py` (2446 bytes)
- `probe_clean_connect.py` (2832 bytes)
- `probe_connect.py` (2560 bytes)
- `probe_final_bfs.py` (978 bytes)
- `probe_finish.py` (1946 bytes)
- `probe_goal.py` (981 bytes)
- `probe_l6.py` (2302 bytes)
- `probe_prior.py` (1539 bytes)
- `probe_search.py` (2359 bytes)
- `probe_selections.py` (1953 bytes)
- `wip_glue_notes.md` (293 bytes)

Run or inspect the smallest relevant probe before writing another one.
