# Unverified frontier brief: lf52 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The symbolic map exposes two special two-cell gate blocks (`?`) at the bottom of the upper board that the peg-hole parser ignored. I’m identifying their raw color and exact component bounds; they are the likely ladder/transfer affordance controlling entry from the carrier shaft.
- The ignored gate blocks are color 15—the persistent bridges already modeled by the existing bridge/carrier leg. In the far-right staged state, the movable color-8 piece sits at `(30,28)`, a static bridge is at `(24,28)`, and `(18,28)` is a hole. I’m verifying the direct leap across that bridge; this is the missing connection from the shaft into the upper board.
- The bridge leap is confirmed and exposes the shaft’s color-12 landing at `(30,28)`. The transported peg remains immediately below at `(36,28)`. I’m now testing Up after the bridge vacates: the peg should advance onto that landing and may then cross the static bridge into the upper board.
- The color-12 cell exposed at `(30,28)` is the empty carrier itself, not a floor tile. It blocks the peg’s carrier below. The next required subgoal is therefore clearing that empty carrier sideways at the upper junction; I’m testing horizontal keys immediately after the bridge disembarks.
- After entry, Down separates the carriers: the empty bridge-carrier stops at row 36 while the peg-carrier returns to row 42. That should free the horizontal track for the lower carrier. I’m testing “down, move horizontally, down” to clear the shaft and recycle the empty carrier under the other static gate.
- The split works: after one Down, action 3 drives the peg’s lower carrier left while the bridge and its empty carrier stay with the upper world. I’m scanning the finite horizontal stops and pressing Up to find the next shaft where the surviving peg can be raised; this is now a deterministic carrier-routing problem.
- The carrier scan now gives an exact upper-board solution. At horizontal stop 3, the peg carrier rises to `(30,28)` directly beneath a static bridge, so it can leap into `(18,28)`. From there a four-macro sequence captures the native upper peg and returns the survivor to the carrier. I’m replaying that full subgoal and checking the peg count after every macro.
- The upper subgoal is fully verified: global peg count drops by one and the survivor exits onto its carrier. The movable bridge can remain upstairs; the lower board has two native pegs, so the transported survivor is sufficient to solve it as an ordinary three-peg board. I’m descending that peg carrier and locating the lower-board docking position.
- No immediate lower-board capture appears at any horizontal stop, so the survivor carrier is not yet on the lower board’s row/entry. I’m tracing only carrier and peg coordinates while driving right to identify the lower shaft, then I’ll apply the same lift-and-static-bridge pattern used upstairs.
- The maze geometry resolves the lower route: the occupied right carrier can climb to row 30 and traverse the horizontal corridor. The earlier probe let it run past intermediate stops without checking macros. I’m rescanning that corridor one stop at a time with a complete move detector that treats color 8 and color 15 as persistent bridge pieces.
- The row-30 corridor is definitely the upper-board docking rail, but it may also turn downward into the lower board at one of its horizontal stops. I’m scanning each stop and applying Down stepwise, checking for a new bridge/peg macro after every descent. This directly tests the lower reachability branch the earlier long loops blurred together.
- At corridor stop 4 the bridge carrier can descend to row 42 on the far side of the central barrier—this is the first confirmed lower-region reachability. I’m now moving that descended carrier back along the lower corridor and checking every stop for a bridge/peg leap; this is the decisive loop-around-the-wall test.

## Preserved local probes

- `probe_level6.py` (1119 bytes)
- `probe_level6_focus.py` (14424 bytes)
- `probe_level6_macros.py` (13153 bytes)
- `probe_level6_search.py` (5292 bytes)

Run or inspect the smallest relevant probe before writing another one.
