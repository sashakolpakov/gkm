import unittest

import arc_agi3_adapter as arc
import arc_goal_induction as ag


CANDS = (2, 3, 5)
SEEDS = tuple(range(12))


class GameTests(unittest.TestCase):
    def test_collect_removes_only_collectible(self) -> None:
        game = ag.make_goal_game(1, ("clear@2",), CANDS)
        # colour 2 is collectible; 3 and 5 are not.
        self.assertIn(2, game.collect_colors)
        self.assertNotIn(3, game.collect_colors)
        before = len(game.remaining(2))
        # Walk the avatar onto a colour-2 cell.
        target = game.remaining(2)[0]
        game.avatar = (target[0] - 1, target[1]) if target[0] > 0 else (target[0] + 1, target[1])
        # Step toward it.
        dx = target[0] - game.avatar[0]
        game.step(arc.GameAction.ACTION2 if dx > 0 else arc.GameAction.ACTION4)
        self.assertEqual(len(game.remaining(2)), before - 1)

    def test_scene_atoms_bounded(self) -> None:
        game = ag.make_goal_game(2, ("clear@2", "avoid@5"), CANDS)
        feats = ag.scene_atoms(game, CANDS)
        for value in feats.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class InductionTests(unittest.TestCase):
    OBJECTIVES = [
        ("clear@2",),
        ("avoid@5",),
        ("clear@2", "clear@3"),
        ("clear@2", "avoid@5"),
        ("clear@3", "avoid@5"),
    ]

    def test_objectives_induced_exactly(self) -> None:
        for obj in self.OBJECTIVES:
            task = ag.HiddenArcTask(_objective=obj, candidate_colors=CANDS, seeds=SEEDS)
            res = ag.induce_arc_goal(task, list(range(6)), lam=0.05, budget=8)
            self.assertEqual(set(res.inferred_goal), set(obj), f"{obj}: got {res.inferred_goal}")

    def test_compiled_cone_solves_holdout(self) -> None:
        for obj in self.OBJECTIVES:
            task = ag.HiddenArcTask(_objective=obj, candidate_colors=CANDS, seeds=SEEDS)
            res = ag.induce_arc_goal(task, list(range(6)), lam=0.05, budget=8)
            phases = ag.goal_to_cone(res.inferred_goal)
            solved = task.solved_fraction(phases, list(range(6, 12)))
            self.assertEqual(solved, 1.0, f"{obj}: holdout {solved}")

    def test_distractor_colour_not_selected(self) -> None:
        # Colour 3 present but irrelevant; the agent must not include it.
        task = ag.HiddenArcTask(_objective=("clear@2",), candidate_colors=CANDS, seeds=SEEDS)
        res = ag.induce_arc_goal(task, list(range(6)), lam=0.05, budget=8)
        self.assertNotIn("clear@3", res.inferred_goal)
        self.assertNotIn("avoid@3", res.inferred_goal)


class CorrespondenceTests(unittest.TestCase):
    def test_seek_clears_collectible_colour(self) -> None:
        # A bare seek@2 cone clears all colour-2 objects on a clear@2 game.
        task = ag.HiddenArcTask(_objective=("clear@2",), candidate_colors=CANDS, seeds=SEEDS)
        results = task.evaluate([("seek", 2)], list(range(6)))
        self.assertTrue(all(atoms["clear@2"] > 0.99 for atoms, _score in results))

    def test_flee_avoids_hazard_colour(self) -> None:
        task = ag.HiddenArcTask(_objective=("avoid@5",), candidate_colors=CANDS, seeds=SEEDS)
        results = task.evaluate([("flee", 5)], list(range(6)))
        self.assertTrue(all(score > 0.99 for _atoms, score in results))


if __name__ == "__main__":
    unittest.main()
