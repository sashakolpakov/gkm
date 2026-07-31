# Unverified frontier brief: re86 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside this workspace and the documented `gkm_try.py`/`perception.py` surface. I’ll begin with the compact index, reproduce any inherited observations with bounded clone probes, then implement level 7 as a composition of reusable legs and verify through `gkm_try.py`.
- Baseline is confirmed at 6 levels in 306 moves with a valid replay. I’m now narrowing the prior level-7 work to the smallest probes that expose object selection, motion/interaction effects, and the proposed target assignment; each useful claim will be rerun on a fresh clone before it informs the solver.
- The fresh survey confirms five action affordances: actions 1–4 translate the selected object on a 3-cell lattice, and action 5 cycles among exactly three shapes. The dense objective is now concrete: repaint and deform/position a small cross onto two color-9 centers, an outline onto three color-11 centers, and a large cross onto four color-8 centers. I’m testing those as independent selected-object searches, scoring both correct paint and geometric coverage rather than waiting for reward.
- The small-cross route is now verified geometrically: it first lengthens the right arm to span both target axes, then makes a safe loop through the color-9 station and returns without disturbing that placement. One visual endpoint becomes indistinguishable from its same-colored underlying marker, so I’m reconstructing occupancy from motion rather than treating unchanged pixels as absence. The outline search is using exact reconstructed cells for the same reason.
- The outline’s deformation is genuinely contextual: an immediate reverse can alter the shape after it catches on the central fixture, so pruning “undo” moves was excluding valid mechanics. I’ve corrected that and expanded the geometry search under a fixed 60,000-state cap; its best reconstruction is currently within two lattice cells of the three-point pattern.
- The capped search is still running but has not improved beyond the six-pixel residual, which is evidence that the assumed outline-to-color-11 assignment may be wrong or the scoring model is missing a non-rectangular occupancy. I’ll let the bounded run finish, then test the alternative shape/target assignments rather than enlarging the search blindly.

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
- `probe6.py` (1996 bytes)
- `probe7.py` (1661 bytes)
- `probe8.py` (1791 bytes)
- `probe9.py` (1589 bytes)
- `probe_level7.py` (5686 bytes)

Run or inspect the smallest relevant probe before writing another one.
