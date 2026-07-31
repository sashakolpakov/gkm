# Unverified frontier brief: sp80 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The conditioned search is running normally and has not hit a reward in its first portion. Every tested state satisfies socket projection compatibility and safe object identity, so a miss here meaningfully narrows the mechanic rather than repeating invalid overlap branches.
- The conditioned socket matcher exhausted without reward, which supports the frontier’s original “continuous projection cover” observation instead. There is one important gap-free cover family that prior probes did not jointly test: the upper turn remains at its initial row, the vertical bar bridges rows 20–31, the marked bar follows immediately, and the lower turn closes the cover. I’m testing that compact family with the bar’s column joined to the central projection.
- That single cover family also misses, so I’m broadening only within the verified invariant: all four movable x-projections must form one component containing the central top/bottom ports, and all four y-projections must form one component spanning the side-port band. This rejects disconnected random states before any clone work and samples only safe, continuous-cover arrangements.
- A useful reduction falls out of the verified projection invariant: only 132 row tuples make the four movable y-intervals a gap-free cover spanning all three side ports. I’m exhaustively testing those tuples with the central x-chain and four adjacent vertical-bar columns, choosing unoccluded selection pixels symbolically for each overlap case.
- The full gap-free level-6 row cover also misses, so connectivity alone is not the acceptance rule. I’m isolating the ordering rule on level 2, where the winning x-projections are known and all three object rows can be enumerated cheaply without identity ambiguity. The resulting accepted row orders will tell us exactly how source direction constrains depth.
- The level-2 row enumeration reveals the actual direction rule: pieces overlapping the source projection are energized directly; only pieces outside it require an upstream feeder in the source direction. Transferring that to level 6 gives a new compact topology: both turn pieces sit on the source column, the mirrored turn feeds the vertical bar horizontally, that bar’s 12-pixel span bridges both left-side sockets, the other turn serves the right socket, and the marked horizontal bar preserves the downward path.
- The bridge-expander family is still running without a hit. It has passed the central placements and is now checking the outward turn/bar orderings; all branches retain the marked horizontal bar on the direct downward source projection.
- The level-2 ordering result also exposes why the bridge family could fail despite correct rows: both turns were competing for the same narrow source column before it was widened. I’m moving the marked horizontal bar above them so it first expands the downward beam, then placing the two turns on separate columns; one feeds the right socket, the mirrored one feeds the vertical expander for both left sockets.
- One final topology remains untested as a joint state: the mirrored turn on the upper-left port, the ordinary turn on the right port, the vertical bar immediately below the upper turn so its span reaches the lower-left port, and the marked bar staged as the bottom carrier. I’m checking that exact arrangement across its valid along-ray positions.
- The lower-level feeder relation points to a sharper junction condition: turn pieces sit just outside the horizontal feeder’s projection, not inside it. I’m testing the corresponding level-6 layout—marked bar first widens the downward beam, mirrored turn just left of its edge, ordinary turn just right, and the vertical bar carries the left branch across the two left sockets.
- The outside-edge junction family is still running and has not rewarded in its first half. The remaining branches vary the vertical expander’s span and offset while preserving the widened source split.
- The widened-beam family is exhausted. I’m now closing the one remaining hole in the preserved search matrix: the exact upper-turn / marked-middle / lower-turn rows with the vertical carrier below them. Prior central-stack work used the carrier above the marked bar; this sweep fixes it below and exhausts all three lateral junctions.

## Preserved local probes

- `probe_axis_ranges.py` (1883 bytes)
- `probe_central_stack.py` (3763 bytes)
- `probe_chain_candidates.py` (1824 bytes)
- `probe_constraints.py` (4143 bytes)
- `probe_exact_l6.py` (2761 bytes)
- `probe_l1_map.py` (979 bytes)
- `probe_l2_acceptance.py` (2569 bytes)
- `probe_l2_rows.py` (1608 bytes)
- `probe_l5_relations.py` (1722 bytes)
- `probe_l6.py` (9720 bytes)
- `probe_l6_conditioned.py` (4068 bytes)
- `probe_l6_connected.py` (2572 bytes)
- `probe_l6_junctions.py` (3442 bytes)
- `probe_l6_ycovers.py` (3287 bytes)
- `probe_search_l6.py` (4397 bytes)
- `probe_snapshots.py` (1466 bytes)

Run or inspect the smallest relevant probe before writing another one.
