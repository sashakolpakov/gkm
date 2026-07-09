# cone/ — the colimit-cone program core

This directory holds the substrate-agnostic core of the **colimit-cone program**:
learn the *cone* (a compiled, verifiable structure over goal atoms) rather than a
monolithic policy or functor. The full program document — motivation, falsification
criteria, experiment log — is [`COLIMIT_CONE_APPROACH.md`](../COLIMIT_CONE_APPROACH.md)
at the repository root.

## Modules

- [`cone_foraging.py`](cone_foraging.py) — the founding cone substrate: grid
  foraging with observation atoms, where cone legs are discovered and composed.
- [`cone_foraging_bound.py`](cone_foraging_bound.py) — sibling substrate probing
  the free-energy bound (report: [`cone_bound_report.md`](cone_bound_report.md)).
- [`cone_goal_induction.py`](cone_goal_induction.py) — the substrate-agnostic
  free-energy core for inferring hidden objectives from scalar reward
  (report: [`cone_goal_induction_report.md`](cone_goal_induction_report.md)).
- [`cone_method.py`](cone_method.py) / [`cone_method_foraging.py`](cone_method_foraging.py) —
  the cone *method* layer over goal induction, and its foraging instantiation.
- [`cone_render.py`](cone_render.py) — rendering of cone solutions.

The ARC-AGI-3 lift of this machinery (the connector, scene atoms, leg discovery on
live games) lives in [`arc/`](../arc/ARC.md) — those modules import this core.

## Runnable experiments

```bash
python3 cone/run_colimit_cone_foraging.py   # report: colimit_cone_foraging_report.md
python3 cone/run_cone_bound.py              # report: cone_bound_report.md
python3 cone/run_cone_goal_induction.py     # report: cone_goal_induction_report.md
python3 cone/run_cone_leg_robustness.py
python3 cone/run_method_foraging.py
python3 cone/render_cone_solutions.py
```

## Tests

```bash
python -m pytest cone/ -q
```
