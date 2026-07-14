# Colimit-Cone Core

This subject contains the substrate-independent cone experiments used elsewhere in
the repository. A cone is an executable composition of partial, testable behaviors
over a set of goal atoms. The experiments study when such components can be reused
and verified instead of replacing a policy wholesale.

The categorical language is a formal organization of compatible partial policies.
The union of a compatible family is its colimit in the corresponding poset category.
That elementary result does not by itself establish empirical transfer; the evidence
must come from executable factorization, replay, and description accounting in a
specific substrate.

## Entry Points

- [`CONE.md`](CONE.md): detailed module and experiment guide.
- [`../COLIMIT_CONE_APPROACH.md`](../COLIMIT_CONE_APPROACH.md): program statement and
  falsification criteria.
- [`cone_foraging.py`](cone_foraging.py): founding foraging substrate.
- [`cone_goal_induction.py`](cone_goal_induction.py): goal induction from scalar
  reward.
- [`cone_method.py`](cone_method.py): substrate-independent method layer.

Run the subject tests from the repository root:

```bash
python -m pytest cone/ -q
```
