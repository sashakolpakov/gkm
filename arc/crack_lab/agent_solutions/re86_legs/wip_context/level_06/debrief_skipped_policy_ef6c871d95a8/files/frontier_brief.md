# Unverified frontier brief: re86 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly within the workspace and the documented harness surface. I’ll begin with `solver_index.md`, then reproduce any needed observations through compact clone probes before changing the level-6 player.
- The frontier brief contains no trusted mechanics, only probe pointers, so I’m treating level 6 as fresh. I’m running the harness now to confirm the exact frontier, then I’ll inspect only the smallest relevant preserved probes and reproduce their claims.
- The fresh experiments identify two selectable avatars: a large color-11 hollow square centered at `(48,15)` and a color-9 cross centered at `(15,48)`. Actions 1–4 translate the selected object by three pixels; action 5 switches selection. The eight ring markers split by matching colors, suggesting the dense measure “matching marker centers currently covered,” but later probes indicate the shapes deform against the central fixture, so I’m reproducing that mechanic next.
- The deformation is now reproducible: pressing a shape against the central color-1 fixture transfers extent from the blocked side to the opposite side while preserving its total span. That lets the square become the exact tall rectangle covering all four color-11 markers. For the cross, the target geometry is centered at `(9,12)` with asymmetric arms reaching the four color-9 markers; I’m testing a constructive two-axis deformation route rather than expanding blind BFS.
- The constructive cross route exposed an important correction: marker rings are true barriers, and the selected black pixel is a material point rather than a reliable geometric center after deformation. A hand-derived straight-line placement therefore isn’t trustworthy. I’m switching to bounded observational search from the already verified square subgoal, with the real level reward as the terminal test and a larger—but still finite—state/depth envelope.

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
- `probe5.py` (2400 bytes)
- `probe6.py` (1996 bytes)
- `probe7.py` (1661 bytes)
- `probe8.py` (1791 bytes)
- `probe9.py` (1589 bytes)

Run or inspect the smallest relevant probe before writing another one.
