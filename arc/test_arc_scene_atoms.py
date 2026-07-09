import unittest

import arc_agi3_adapter as arc
import arc_goal_induction as ag
import arc_scene_atoms as sa


CANDS = (2, 3, 5)
SEEDS = tuple(range(10))


class FrameMeasurementTests(unittest.TestCase):
    def test_count_and_centroids(self) -> None:
        frame = [
            [0, 2, 0, 0],
            [0, 0, 0, 2],
            [4, 0, 0, 0],
        ]
        self.assertEqual(sa.count_color(frame, 2), 2)
        self.assertEqual(sa.avatar_cell(frame, 4), (0, 2))
        self.assertEqual(sa.colors_present(frame), [2, 4])

    def test_atom_evaluators(self) -> None:
        # Two SEPARATE colour-2 objects (non-adjacent so they don't merge into
        # one component), avatar colour 4 at (0,0).
        f0 = [[4, 2, 0, 2]]
        f1 = [[4, 0, 0, 2]]  # one collected -> one remains at (3,0)
        clear = sa.Atom("clear", 2)
        self.assertAlmostEqual(clear.evaluate(f0, f1, 4), 0.5)
        avoid = sa.Atom("avoid", 2)
        # nearest colour-2 now at (3,0), distance 3; avoid = min(1, 3/SAFE_RADIUS)
        self.assertAlmostEqual(avoid.evaluate(f0, f1, 4), min(1.0, 3 / sa.SAFE_RADIUS))


class AvatarDiscoveryTests(unittest.TestCase):
    def test_discovers_default_avatar(self) -> None:
        def factory(seed):
            return ag.make_goal_game(seed, ("clear@2",), CANDS)
        self.assertEqual(sa.discover_avatar_color(factory, 0), 4)

    def test_discovers_nonstandard_avatar_colour(self) -> None:
        # The avatar is whatever object the actions move — not hardcoded to 4.
        def factory(seed):
            return ag.GoalGame(
                width=10, height=10, avatar_color=7, avatar=(5, 5),
                objects=[((1, 1), 2), ((8, 8), 3)], collect_colors=frozenset({2}),
                initial_counts={2: 1, 3: 1},
            )
        self.assertEqual(sa.discover_avatar_color(factory, 0), 7)


class VocabularyDiscoveryTests(unittest.TestCase):
    def test_discovers_colours_and_prunes(self) -> None:
        objective = ("clear@2",)

        def factory(seed):
            return ag.make_goal_game(seed, objective, CANDS)

        explore = list(ag.probe_phase_sets(CANDS).values())
        vocab = sa.discover_vocabulary(factory, lambda g, ph: ag.run_cone(g, ph), explore, list(SEEDS))
        self.assertEqual(vocab.avatar_color, 4)
        self.assertEqual(vocab.colors, [2, 3, 5])
        # Some atoms must survive and some must be pruned (constant ones).
        self.assertGreater(len(vocab.atoms), 0)
        self.assertGreater(len(vocab.pruned), 0)
        # Pruned atoms have sub-threshold variance; kept atoms meet it.
        for atom in vocab.pruned:
            self.assertLess(vocab.variances[atom.name], 0.02)
        for atom in vocab.atoms:
            self.assertGreaterEqual(vocab.variances[atom.name], 0.02)


class DiscoverAndInduceTests(unittest.TestCase):
    OBJECTIVES = [("clear@2",), ("avoid@5",), ("clear@2", "avoid@5")]

    def test_raw_frame_pipeline_recovers_goal_and_solves(self) -> None:
        seeds = tuple(range(12))
        for obj in self.OBJECTIVES:
            result, vocab = ag.discover_and_induce(obj, CANDS, seeds, list(range(6)), lam=0.05, budget=8)
            phases = ag.goal_to_cone(result.inferred_goal)
            task = ag.HiddenArcTask(_objective=obj, candidate_colors=tuple(vocab.colors),
                                    seeds=seeds, vocabulary=vocab)
            solved = task.solved_fraction(phases, list(range(6, 12)))
            # The compiled cone must solve the held-out instances; the inferred
            # atom set should match the objective (clear/avoid distinguish here).
            self.assertEqual(solved, 1.0, f"{obj}: holdout {solved}")
            self.assertEqual(set(result.inferred_goal), set(obj), f"{obj}: got {result.inferred_goal}")

    def test_avatar_not_assumed_in_pipeline(self) -> None:
        # The pipeline must discover the avatar, not be told it.
        result, vocab = ag.discover_and_induce(("clear@2",), CANDS, tuple(range(12)),
                                               list(range(6)), lam=0.05, budget=8)
        self.assertEqual(vocab.avatar_color, 4)
        self.assertNotIn(vocab.avatar_color, vocab.colors)


if __name__ == "__main__":
    unittest.main()
