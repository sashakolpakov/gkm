# Unverified frontier brief: re86 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin with the indexed signatures/ranges, then reproduce any frontier observations with bounded level-7 clones before editing the leg composition.
- Baseline is clean at 6 levels/306 moves. I’m now reproducing the inherited level-7 claims from fresh clones—first the action/selection model, then each shape’s target geometry—using the smallest preserved probes and compact symbolic output.
- The fresh board confirms three selectable movers on a 3-cell lattice and a contextual deformation/paint mechanic. One inherited candidate already reaches both small-cross targets but has the wrong paint; I’m treating “target coverage + correct color” as the dense score and searching only that selected-object state, with bounded caps, before combining shapes.
- The small-cross search is now one cell from satisfying both targets while retaining color 9; the prior heuristic had treated finite arms as infinite axes, so I corrected that scoring error. I’m letting the bounded compact-state search finish, then I’ll replay any found route on a fresh clone before using it.
- The compact search has exhausted the near-goal basin without yet producing the final cell, which suggests that repair may require temporarily moving away from the target or changing paint again. The cap remains fixed; I’m not expanding it blindly.
- A reproduced geometric check overturns the inherited assignment: the large cross’s native spans already match the two color-9 markers exactly at center `(18,39)`, while the small cross is the one that must be extended for the four color-8 markers. I’m testing this alternative assignment now; it sharply reduces unnecessary deformation.
- The apparent “one-cell defect” is confirmed as an observation artifact: repainting a shape to the marker’s own color makes the covered endpoint indistinguishable from the underlying center. The mover-size constraint also confirms the original assignment after all: the 37-cell cross can span the two color-9 targets, while only the 55-cell cross has enough material to span all four color-8 axes.
- The large-cross search has reached color 8 with three of the four axes geometrically aligned and only a six-pixel residual to the last marker. This is now a genuinely bounded near-solution, not sparse frame-change chasing; I’m keeping the existing cap and will replay the best complete route only if it reaches zero.
- The focused suffix search confirms the remaining large-cross problem is specifically redistributing six cells to the right arm; simple translation does not improve it. I’m using the central fixture interaction as the only remaining search dimension, with no cap increase.
- One fixture maneuver successfully moved three cells from the left side to the right arm, reducing the residual from six to three. I’m testing whether the same reusable maneuver composes once more from that new state; if it does, the large-cross route becomes a compact deterministic leg.
- The large-cross route is now complete and independently reconstructed: two applications of the same fixture maneuver produce the four-point geometry, and an up/down paint pass changes it to color 8; the three “missing” endpoint pixels after painting are exactly the three same-color marker centers, confirming coverage rather than damage. I’m applying the same exact-cell method to the outline’s three targets.

## Preserved local probes

- `probe.py` (1257 bytes)
- `probe10.py` (1214 bytes)
- `probe11.py` (1105 bytes)
- `probe12.py` (1127 bytes)
- `probe13.py` (3590 bytes)
- `probe14.py` (1425 bytes)
- `probe15.py` (1874 bytes)
- `probe16.py` (1699 bytes)
- `probe17.py` (3009 bytes)
- `probe18.py` (3091 bytes)
- `probe19.py` (1455 bytes)
- `probe2.py` (1981 bytes)
- `probe20.py` (880 bytes)
- `probe21.py` (1500 bytes)
- `probe22.py` (1187 bytes)
- `probe23.py` (1663 bytes)
- `probe24.py` (1675 bytes)
- `probe25.py` (1508 bytes)
- `probe26.py` (1215 bytes)
- `probe27.py` (1840 bytes)
- `probe28.py` (1602 bytes)
- `probe29.py` (1191 bytes)
- `probe3.py` (1408 bytes)
- `probe30.py` (1537 bytes)
- `probe31.py` (1853 bytes)
- `probe32.py` (1311 bytes)
- `probe33.py` (1461 bytes)
- `probe34.py` (2648 bytes)
- `probe35.py` (1370 bytes)
- `probe36.py` (1027 bytes)
- `probe37.py` (1607 bytes)
- `probe38.py` (1628 bytes)
- `probe39.py` (1598 bytes)
- `probe4.py` (1994 bytes)
- `probe40.py` (2252 bytes)
- `probe41.py` (1864 bytes)
- `probe42.py` (1807 bytes)
- `probe43.py` (1239 bytes)
- `probe44.py` (1378 bytes)
- `probe45.py` (1928 bytes)
- `probe46.py` (3463 bytes)
- `probe47.py` (1797 bytes)
- `probe48.py` (6744 bytes)
- `probe49.py` (1880 bytes)
- `probe5.py` (2400 bytes)
- `probe50.py` (3581 bytes)
- `probe51.py` (1474 bytes)
- `probe52.py` (2969 bytes)
- `probe53.py` (3505 bytes)
- `probe54.py` (2024 bytes)
- `probe55.py` (3001 bytes)
- `probe56.py` (2595 bytes)
- `probe57.py` (1120 bytes)
- `probe58.py` (3033 bytes)
- `probe59.py` (1554 bytes)
- `probe6.py` (1996 bytes)
- `probe60.py` (3142 bytes)
- `probe61.py` (1460 bytes)
- `probe62.py` (2879 bytes)
- `probe7.py` (1661 bytes)
- `probe8.py` (1791 bytes)
- `probe9.py` (1589 bytes)
- `probe_level7.py` (5686 bytes)

Run or inspect the smallest relevant probe before writing another one.
