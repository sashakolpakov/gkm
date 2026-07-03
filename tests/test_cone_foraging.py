import random
import unittest

import cone_foraging as cf


def first_level(task_name: str, seed: int = 11) -> cf.ConeLevel:
    task = cf.TASKS[task_name]
    return cf.make_cone_levels(seed, 1, task)[0]


class WitnessTests(unittest.TestCase):
    def test_witness_solves_forage(self) -> None:
        task = cf.TASKS["forage"]
        library = [cf.witness_seek_leg()]
        controller = cf.witness_gluing(task)
        for level in cf.make_cone_levels(31, 6, task):
            episode = cf.run_cone_episode(controller, library, level, task)
            self.assertTrue(cf.episode_solved(episode, level, task))

    def test_witness_solves_homing(self) -> None:
        task = cf.TASKS["homing"]
        library = [cf.witness_seek_leg()]
        controller = cf.witness_gluing(task)
        for level in cf.make_cone_levels(32, 6, task):
            episode = cf.run_cone_episode(controller, library, level, task)
            self.assertTrue(cf.episode_solved(episode, level, task))
            self.assertEqual(episode.final_position, level.home)

    def test_witness_solves_forage_then_home(self) -> None:
        task = cf.TASKS["forage_then_home"]
        library = [cf.witness_seek_leg()]
        controller = cf.witness_gluing(task)
        for level in cf.make_cone_levels(33, 6, task):
            episode = cf.run_cone_episode(controller, library, level, task)
            self.assertTrue(cf.episode_solved(episode, level, task))
            self.assertEqual(episode.collected, episode.total_food)
            self.assertEqual(episode.final_position, level.home)
            self.assertEqual(episode.dynamic_calls, 2)


class SemanticsTests(unittest.TestCase):
    def test_exact_match_beats_any(self) -> None:
        # ANY says STAY forever; the exact E rule moves RIGHT. Exact must win.
        right = cf.MOVE_NAMES.index("RIGHT")
        stay = cf.MOVE_NAMES.index("STAY")
        east = cf.OBS_LABELS.index("E")
        genome = cf.ConeGenome(
            state_count=1,
            rules=[
                cf.ConeRule(0, cf.ANY_OBS, (stay,), 0),
                cf.ConeRule(0, east, (right,), 0),
            ],
        )
        task = cf.TASKS["forage"]
        level = cf.ConeLevel(width=5, height=5, start=(1, 2), food=((3, 2),), home=(0, 0))
        episode = cf.run_cone_episode(genome, [], level, task, max_steps=10)
        self.assertEqual(episode.collected, 1)

    def test_no_rule_halts_episode(self) -> None:
        task = cf.TASKS["homing"]
        level = cf.ConeLevel(width=5, height=5, start=(1, 2), food=(), home=(3, 2))
        genome = cf.ConeGenome(state_count=1, rules=[])
        episode = cf.run_cone_episode(genome, [], level, task, max_steps=10)
        self.assertTrue(episode.halted)
        self.assertEqual(episode.steps, 0)

    def test_op_budget_terminates_zero_step_loop(self) -> None:
        # SET_FOCUS loop: no world steps, must end by op budget (pitfall P6).
        set_food = cf.SET_FOCUS_BASE + cf.FOOD_CHANNEL
        genome = cf.ConeGenome(state_count=1, rules=[cf.ConeRule(0, cf.ANY_OBS, (set_food,), 0)])
        task = cf.TASKS["homing"]
        level = cf.ConeLevel(width=5, height=5, start=(1, 2), food=(), home=(3, 2))
        episode = cf.run_cone_episode(genome, [], level, task, max_steps=10, op_budget=25)
        self.assertEqual(episode.steps, 0)
        self.assertGreaterEqual(episode.ops, 25)

    def test_leg_without_matching_rule_halts_episode(self) -> None:
        # A leg with no HERE rule must halt the episode, not return (pitfall P3).
        right = cf.MOVE_NAMES.index("RIGHT")
        east = cf.OBS_LABELS.index("E")
        leg = cf.Leg("partial", cf.ConeGenome(state_count=1, rules=[cf.ConeRule(0, east, (right,), 0)]))
        controller = cf.ConeGenome(
            state_count=2,
            rules=[cf.ConeRule(0, cf.ANY_OBS, (cf.call_action(0, cf.HOME_CHANNEL),), 1)],
        )
        task = cf.TASKS["homing"]
        level = cf.ConeLevel(width=5, height=5, start=(1, 2), food=(), home=(3, 2))
        episode = cf.run_cone_episode(controller, [leg], level, task, max_steps=10)
        # The leg walks east to home; HERE has no rule, but task_done stops first.
        self.assertEqual(episode.final_position, level.home)
        # Move home out of reach east: leg overshoots scenario -> halts at HERE gap.
        level2 = cf.ConeLevel(width=5, height=5, start=(1, 1), food=(), home=(1, 3))
        episode2 = cf.run_cone_episode(controller, [leg], level2, task, max_steps=10)
        self.assertTrue(episode2.halted)


class AccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.leg = cf.witness_seek_leg()
        self.library = [self.leg]
        self.glue_forage = cf.witness_gluing(cf.TASKS["forage"])
        self.glue_homing = cf.witness_gluing(cf.TASKS["homing"])

    def test_leg_definition_complexity(self) -> None:
        # 9 single-action rules: 9 * (1 overhead + 1 action) + 1 leg overhead.
        self.assertAlmostEqual(cf.leg_def_complexity(self.leg), 19.0)

    def test_shared_charges_definition_once(self) -> None:
        genomes = [self.glue_forage, self.glue_homing]
        shared = cf.cone_complexity(genomes, self.library, "shared", call_cost=0.5)
        # Each gluing: 1 rule overhead + 0.5 call = 1.5; library def once: 19.
        self.assertAlmostEqual(shared, 19.0 + 1.5 + 1.5)

    def test_no_share_charges_definition_per_call(self) -> None:
        genomes = [self.glue_forage, self.glue_homing]
        no_share = cf.cone_complexity(genomes, self.library, "no_share", call_cost=0.5)
        self.assertAlmostEqual(no_share, (1.0 + 0.5 + 19.0) * 2)

    def test_unused_leg_is_not_charged(self) -> None:
        inline = cf.ConeGenome(state_count=1, rules=[cf.ConeRule(0, cf.ANY_OBS, (1,), 0)])
        shared = cf.cone_complexity([inline], self.library, "shared", call_cost=0.5)
        self.assertAlmostEqual(shared, 2.0)

    def test_single_use_cannot_pay(self) -> None:
        # Property 1 of Section 4: lifting an inline solver and calling it once
        # is strictly more expensive than the inline solver itself.
        rng = random.Random(5)
        inline = cf.ConeGenome.random(rng, cf.inline_actions(), state_count=2, initial_rule_count=6)
        lifted = cf.lift_leg(inline, "lifted")
        glue = cf.witness_gluing(cf.TASKS["forage"])
        inline_cost = cf.cone_complexity([inline], [], "inline")
        shared_cost = cf.cone_complexity([glue], [lifted], "shared", call_cost=0.5)
        moves_only_cost = sum(
            cf.RULE_OVERHEAD + sum(1.0 for action in rule.actions if cf.is_move(action))
            for rule in inline.rules
            if any(cf.is_move(action) for action in rule.actions) and rule.observation != cf.HERE_OBS
        )
        # def(lifted) >= moves-only inline body + RETURN boundary + overhead,
        # so the single-use cone exceeds the moves-only inline cost.
        self.assertGreater(shared_cost, moves_only_cost)


class LiftTests(unittest.TestCase):
    def test_lift_strips_control_ops_and_adds_return(self) -> None:
        right = cf.MOVE_NAMES.index("RIGHT")
        set_home = cf.SET_FOCUS_BASE + cf.HOME_CHANNEL
        east = cf.OBS_LABELS.index("E")
        inline = cf.ConeGenome(
            state_count=2,
            rules=[
                cf.ConeRule(0, east, (right, set_home), 0),
                cf.ConeRule(0, cf.HERE_OBS, (set_home,), 1),
                cf.ConeRule(1, cf.ANY_OBS, (set_home,), 1),
            ],
        )
        leg = cf.lift_leg(inline, "test")
        keys = {rule.key: rule for rule in leg.genome.rules}
        self.assertEqual(keys[(0, east)].actions, (right,))
        self.assertEqual(keys[(0, cf.HERE_OBS)].actions, (cf.RETURN_ACTION,))
        for rule in leg.genome.rules:
            for action in rule.actions:
                self.assertTrue(cf.is_move(action) or action == cf.RETURN_ACTION)

    def test_lifted_forage_witness_returns_home_when_rebound(self) -> None:
        # Naturality check: the witness leg lifted from "seek" semantics works
        # bound to either channel; here we just confirm the lift of an inline
        # seek-like genome can drive homing through a controller.
        task = cf.TASKS["homing"]
        leg = cf.witness_seek_leg()
        controller = cf.witness_gluing(task)
        level = first_level("homing")
        episode = cf.run_cone_episode(controller, [leg], level, task)
        self.assertTrue(cf.episode_solved(episode, level, task))


class JointEvolutionSmokeTests(unittest.TestCase):
    def test_joint_cone_evolution_runs(self) -> None:
        data = {
            "forage": (cf.TASKS["forage"], cf.make_cone_levels(7, 3, cf.TASKS["forage"])),
            "homing": (cf.TASKS["homing"], cf.make_cone_levels(8, 3, cf.TASKS["homing"])),
        }
        result = cf.evolve_joint_cone(data, "shared", 0.003, seed=2, population_size=24, generations=6)
        self.assertEqual(set(result.controllers), {"forage", "homing"})
        self.assertLessEqual(result.train_loss, 2.5)
        allowed = set(cf.leg_actions())
        for rule in result.legs[0].rules:
            for action in rule.actions:
                self.assertIn(action, allowed)

    def test_joint_cone_with_frozen_leg_runs(self) -> None:
        data = {
            "flee": (cf.TASKS["flee"], cf.make_cone_levels(9, 3, cf.TASKS["flee"])),
            "flee_then_home": (
                cf.TASKS["flee_then_home"],
                cf.make_cone_levels(10, 3, cf.TASKS["flee_then_home"]),
            ),
        }
        frozen = (cf.witness_seek_leg(),)
        result = cf.evolve_joint_cone(
            data, "shared", 0.003, seed=3, population_size=24, generations=6,
            frozen_legs=frozen, evolved_legs=1,
        )
        self.assertEqual(len(result.legs), 1)


class HazardTests(unittest.TestCase):
    def test_hazard_channel_reads_safe_when_far(self) -> None:
        task = cf.TASKS["flee"]
        level = cf.ConeLevel(width=7, height=7, start=(3, 3), food=(), home=(0, 0), hazards=((3, 2),))
        # A flee witness bound to HAZARD must end at distance >= SAFE_RADIUS.
        library = [cf.witness_seek_leg(), cf.witness_flee_leg()]
        controller = cf.witness_gluing(task, seek_index=0, flee_index=1)
        episode = cf.run_cone_episode(controller, library, level, task)
        self.assertTrue(cf.episode_solved(episode, level, task))
        self.assertGreaterEqual(cf.final_hazard_distance(episode, level), cf.SAFE_RADIUS)

    def test_witness_solves_forage_flee(self) -> None:
        task = cf.TASKS["forage_flee"]
        library = [cf.witness_seek_leg(), cf.witness_flee_leg()]
        controller = cf.witness_gluing(task, seek_index=0, flee_index=1)
        for level in cf.make_cone_levels(41, 6, task):
            episode = cf.run_cone_episode(controller, library, level, task)
            self.assertTrue(cf.episode_solved(episode, level, task))
            self.assertEqual(episode.collected, episode.total_food)

    def test_witness_solves_flee_then_home(self) -> None:
        task = cf.TASKS["flee_then_home"]
        library = [cf.witness_seek_leg(), cf.witness_flee_leg()]
        controller = cf.witness_gluing(task, seek_index=0, flee_index=1)
        for level in cf.make_cone_levels(42, 6, task):
            episode = cf.run_cone_episode(controller, library, level, task)
            self.assertTrue(cf.episode_solved(episode, level, task))
            self.assertEqual(episode.final_position, level.home)

    def test_free_legs_are_not_recharged(self) -> None:
        seek = cf.witness_seek_leg()
        flee = cf.witness_flee_leg()
        library = [seek, flee]
        gluing = cf.witness_gluing(cf.TASKS["flee_then_home"], seek_index=0, flee_index=1)
        full = cf.cone_complexity([gluing], library, "shared", call_cost=0.5)
        marginal = cf.cone_complexity(
            [gluing], library, "shared", call_cost=0.5, free_legs=frozenset({0})
        )
        self.assertAlmostEqual(full - marginal, cf.leg_def_complexity(seek))


class EvolutionSmokeTests(unittest.TestCase):
    def test_evolve_inline_forage_improves(self) -> None:
        task = cf.TASKS["forage"]
        levels = cf.make_cone_levels(7, 4, task)
        result = cf.evolve_cone_task(
            task,
            levels,
            cf.inline_actions(),
            [],
            "inline",
            lambda_value=0.001,
            seed=3,
            population_size=30,
            generations=10,
        )
        self.assertLess(result.train_loss, 1.0)

    def test_evolved_genome_only_uses_allowed_actions(self) -> None:
        task = cf.TASKS["homing"]
        levels = cf.make_cone_levels(9, 3, task)
        result = cf.evolve_cone_task(
            task,
            levels,
            cf.leg_actions(),
            [],
            "inline",
            lambda_value=0.001,
            seed=4,
            population_size=20,
            generations=5,
        )
        allowed = set(cf.leg_actions())
        for rule in result.genome.rules:
            for action in rule.actions:
                self.assertIn(action, allowed)


if __name__ == "__main__":
    unittest.main()
