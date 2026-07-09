import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "arc" / "crack_lab"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from universal_connector import BindingPacket, BoundObjective
from universal_crack import (
    EvolvedAttachmentLeg,
    induce_attachment_objective,
)


class UniversalBoundaryTests(unittest.TestCase):
    def test_level_objective_rebinds_instead_of_reusing_old_footprint(self):
        def objective_for(frame):
            marker = int(frame[0, 0])

            def rebind(next_frame):
                return objective_for(next_frame)

            return BoundObjective(
                name="relative",
                verb="test",
                potential=lambda arr, marker=marker: abs(int(arr[0, 0]) - marker),
                rebind=rebind,
            )

        first = np.asarray([[2]])
        second = np.asarray([[9]])
        packet = BindingPacket(
            actions=(1,),
            anchor=object(),
            grid=object(),
            movement={},
            effect_candidates=(1,),
            objectives=[objective_for(first)],
            source="test",
        )

        rebound = packet.at_level(second)
        self.assertEqual(packet.objectives[0].potential(second), 7.0)
        self.assertEqual(rebound.objectives[0].potential(second), 0.0)

    def test_attachment_objective_is_composed_from_learned_roles(self):
        class Obj:
            def __init__(self, color, size, cell):
                self.color = color
                self.size = size
                self.cell = cell

        class Anchor:
            color = 9

            @staticmethod
            def locate(arr, grid, prev_cell=None):
                return (0, 0)

        frame = np.zeros((3, 3), dtype=np.uint8)
        base = BoundObjective(
            name="target",
            verb="transport",
            potential=lambda arr: 0.0,
            rebind=lambda arr: base,
            target_cells=frozenset({(2, 2)}),
        )
        packet = BindingPacket(
            actions=(1, 2, 3, 4, 5),
            anchor=Anchor(),
            grid=object(),
            movement={1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)},
            effect_candidates=(5,),
            objectives=[base],
            source="test",
        )
        leg = EvolvedAttachmentLeg(
            trigger_action=5,
            driver_actions=(1, 2, 3, 4),
            source_color=4,
            source_size=12,
            status_color=3,
            status_size=12,
            evidence=2,
            free_energy=0.4,
            name="attachment",
        )

        states = {
            id(frame): [Obj(4, 12, (0, 0)), Obj(4, 12, (2, 2))],
        }
        moved = frame.copy()
        states[id(moved)] = [Obj(4, 12, (2, 1)), Obj(3, 12, (2, 2))]

        from unittest.mock import patch
        with patch("universal_crack.objects",
                   side_effect=lambda arr, grid, colors=None: [
                       obj for obj in states[id(arr)]
                       if colors is None or obj.color in colors
                   ]):
            objective = induce_attachment_objective(
                frame, packet, base, leg)
            self.assertIsNotNone(objective)
            self.assertLess(
                objective.potential(moved),
                objective.potential(frame),
            )
            self.assertEqual(
                objective.source,
                "l1-induced+connector-verified",
            )


if __name__ == "__main__":
    unittest.main()
