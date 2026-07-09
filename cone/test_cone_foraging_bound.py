import unittest

import cone_foraging as cf
import cone_foraging_bound as cb


class BoundWitnessTests(unittest.TestCase):
    def test_inline_witness_solves_forage_then_home(self) -> None:
        task = cf.TASKS["forage_then_home"]
        genome = cb.witness_inline(task)
        for level in cf.make_cone_levels(51, 6, task):
            ep = cb.run_bound_episode(genome, [], level, task)
            self.assertTrue(cf.episode_solved(ep, level, task))

    def test_cone_witness_solves_forage_then_home(self) -> None:
        task = cf.TASKS["forage_then_home"]
        library = [cf.witness_seek_leg()]
        controller = cb.witness_bound_gluing(task, seek_index=0)
        for level in cf.make_cone_levels(52, 6, task):
            ep = cb.run_bound_episode(controller, library, level, task)
            self.assertTrue(cf.episode_solved(ep, level, task))
            self.assertEqual(ep.collected, ep.total_food)
            self.assertEqual(ep.final_position, level.home)
            self.assertEqual(ep.dynamic_calls, 2)

    def test_cone_witness_solves_flee_then_home(self) -> None:
        task = cf.TASKS["flee_then_home"]
        library = [cf.witness_seek_leg(), cf.witness_flee_leg()]
        controller = cb.witness_bound_gluing(task, seek_index=0, flee_index=1)
        for level in cf.make_cone_levels(53, 6, task):
            ep = cb.run_bound_episode(controller, library, level, task)
            self.assertTrue(cf.episode_solved(ep, level, task))

    def test_flee_witness_moves_away_from_hazard(self) -> None:
        # The bound flee witness must INCREASE distance to the hazard, not seek
        # it. Caught by the renderer: the seek-bound flee was walking onto X.
        task = cf.TASKS["flee"]
        library = [cf.witness_seek_leg(), cf.witness_flee_leg()]
        controller = cb.witness_bound_gluing(task, seek_index=0, flee_index=1)
        for level in cf.make_cone_levels(54, 6, task):
            ep = cb.run_bound_episode(controller, library, level, task)
            self.assertTrue(cf.episode_solved(ep, level, task))
            self.assertGreaterEqual(cf.final_hazard_distance(ep, level), cf.SAFE_RADIUS)


class BoundSemanticsTests(unittest.TestCase):
    def test_channel_dispatch_precedence(self) -> None:
        # State 0 has a FOOD rule (move RIGHT toward food at E) and a HOME rule.
        # FOOD has lower channel index, so the FOOD rule must win when both match.
        right = cf.MOVE_NAMES.index("RIGHT")
        up = cf.MOVE_NAMES.index("UP")
        east = cf.OBS_LABELS.index("E")
        north = cf.OBS_LABELS.index("N")
        genome = cb.BoundGenome(
            state_count=1,
            rules=[
                cb.BoundRule(0, cf.FOOD_CHANNEL, east, (right,), 0),
                cb.BoundRule(0, cf.HOME_CHANNEL, north, (up,), 0),
            ],
        )
        task = cf.TASKS["forage"]
        # Food to the east, home to the north: both rules would match; food wins.
        level = cf.ConeLevel(width=5, height=5, start=(1, 2), food=((3, 2),), home=(1, 0))
        ep = cb.run_bound_episode(genome, [], level, task, max_steps=10)
        self.assertEqual(ep.collected, 1)

    def test_no_rule_halts(self) -> None:
        task = cf.TASKS["forage"]
        level = cf.ConeLevel(width=5, height=5, start=(1, 2), food=((3, 2),), home=(0, 0))
        ep = cb.run_bound_episode(cb.BoundGenome(1, []), [], level, task, max_steps=10)
        self.assertTrue(ep.halted)
        self.assertEqual(ep.steps, 0)

    def test_no_set_focus_in_inline_actions(self) -> None:
        for action in cb.bound_inline_actions():
            self.assertFalse(cf.is_set_focus(action))
        for action in cb.bound_controller_actions(1):
            self.assertFalse(cf.is_set_focus(action))


class BoundAccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.leg = cf.witness_seek_leg()
        self.library = [self.leg]

    def test_binding_is_priced(self) -> None:
        # One CALL rule: rule_overhead + call_cost + binding_cost.
        glue = cb.BoundGenome(
            state_count=2,
            rules=[cb.BoundRule(0, cf.FOOD_CHANNEL, cf.ANY_OBS, (cf.call_action(0, cf.FOOD_CHANNEL),), 1)],
        )
        priced = cb.bound_genome_complexity(glue, self.library, "shared", call_cost=0.5, binding_cost=0.5)
        free = cb.bound_genome_complexity(glue, self.library, "shared", call_cost=0.5, binding_cost=0.0)
        self.assertAlmostEqual(priced, 1.0 + 0.5 + 0.5)
        self.assertAlmostEqual(free, 1.0 + 0.5)
        self.assertAlmostEqual(priced - free, 0.5)

    def test_shared_charges_leg_once(self) -> None:
        task = cf.TASKS["forage_then_home"]
        glue = cb.witness_bound_gluing(task, seek_index=0)
        shared = cb.bound_cone_complexity([glue], self.library, "shared", call_cost=0.5, binding_cost=0.5)
        no_share = cb.bound_cone_complexity([glue], self.library, "no_share", call_cost=0.5, binding_cost=0.5)
        # Two CALL rules: 2*(1 + 0.5 + 0.5) = 4 of controller; shared adds def once.
        self.assertAlmostEqual(shared, 4.0 + cf.leg_def_complexity(self.leg))
        # no_share adds def per call (twice), no separate library term.
        self.assertAlmostEqual(no_share, 4.0 + 2 * cf.leg_def_complexity(self.leg))


class OrFactorTests(unittest.TestCase):
    """The headline v3 claim, accounting mode: within a SINGLE two-phase task,
    the cone (seek leg + 2 priced calls) is cheaper than the inline solver that
    must duplicate the seek body per channel."""

    def test_within_task_factoring_pays(self) -> None:
        task = cf.TASKS["forage_then_home"]
        leg = cf.witness_seek_leg()
        inline = cb.witness_inline(task)
        cone = cb.witness_bound_gluing(task, seek_index=0)
        inline_c = cb.bound_cone_complexity([inline], [], "inline")
        shared_c = cb.bound_cone_complexity([cone], [leg], "shared")
        no_share_c = cb.bound_cone_complexity([cone], [leg], "no_share")
        # Falsification criterion 1: shared must beat inline within one task.
        self.assertLess(shared_c, inline_c)
        # Falsification criterion 3: no_share must not beat inline.
        self.assertGreaterEqual(no_share_c, inline_c)

    def test_single_phase_does_not_factor(self) -> None:
        # Control: forage alone has one channel, no duplication; the cone for a
        # single phase is NOT cheaper than inline.
        task = cf.TASKS["forage"]
        leg = cf.witness_seek_leg()
        inline = cb.witness_inline(task)
        cone = cb.witness_bound_gluing(task, seek_index=0)
        inline_c = cb.bound_cone_complexity([inline], [], "inline")
        shared_c = cb.bound_cone_complexity([cone], [leg], "shared")
        self.assertGreaterEqual(shared_c, inline_c)


class BoundEvolutionSmokeTests(unittest.TestCase):
    def test_evolve_inline_forage_improves(self) -> None:
        task = cf.TASKS["forage"]
        levels = cf.make_cone_levels(61, 4, task)
        result = cb.evolve_bound_task(
            task, levels, cb.bound_inline_actions(), [], "inline", 0.001,
            seed=3, population_size=30, generations=10,
        )
        self.assertLess(result.train_loss, 1.0)


if __name__ == "__main__":
    unittest.main()
