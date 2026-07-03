import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "arc" / "crack_lab"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from object_mdl import ObjectMDL, ObjectTransition


def transition(action, near, outcome):
    return ObjectTransition(
        action=action,
        anchor_delta=(0, 0),
        near_before=near,
        target_bucket=2,
        outcome=outcome,
    )


class ObjectMDLTests(unittest.TestCase):
    def test_richer_context_wins_when_near_object_controls_outcome(self):
        learner = ObjectMDL(rewrite_cost=0.0, delta=0.5)
        for _ in range(8):
            learner.assess(transition(5, ((9, 0, 0, -1),), ("attach",)))
            learner.assess(transition(5, (), ("noop",)))
        self.assertIn("near", learner.active.fields)

    def test_repeated_transition_becomes_less_surprising(self):
        learner = ObjectMDL()
        event = transition(1, (), ("move",))
        first = learner.assess(event).surprise_bits
        second = learner.assess(event).surprise_bits
        self.assertLess(second, first)


if __name__ == "__main__":
    unittest.main()
