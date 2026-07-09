import unittest

import cone_foraging as cf
import cone_goal_induction as gi


class FeatureTests(unittest.TestCase):
    def test_outcome_features_bounded(self) -> None:
        task = cf.TASKS["forage_then_home"]
        for level in cf.make_cone_levels(29, 6, task):
            controller, library = gi.compile_goal(("food", "home"))
            import cone_foraging_bound as cb
            ep = cb.run_bound_episode(controller, library, level, task)
            f = gi.outcome_features(ep, level)
            for key in gi.FEATURES:
                self.assertGreaterEqual(f[key], 0.0)
                self.assertLessEqual(f[key], 1.0)

    def test_hidden_reward_one_when_goal_met(self) -> None:
        # A cone that achieves the goal should earn reward close to 1.
        task = cf.TASKS["forage"]
        levels = cf.make_cone_levels(29, 6, task)
        env = gi.HiddenTask(task, levels)
        controller, library = gi.compile_goal(("food",))
        results = env.evaluate(controller, library, list(range(6)))
        self.assertTrue(all(r > 0.99 for _f, r in results))


class InductionTests(unittest.TestCase):
    CLEAN = ["forage", "homing", "forage_then_home", "flee", "forage_flee"]

    def test_clean_tasks_induced_exactly(self) -> None:
        for name in self.CLEAN:
            task = cf.TASKS[name]
            levels = cf.make_cone_levels(29, 10, task)
            env = gi.HiddenTask(task, levels)
            result = gi.induce_active(env, list(range(6)), lam=0.05, budget=7)
            self.assertEqual(
                set(result.inferred_goal), set(gi.task_goal_features(task)),
                f"{name}: inferred {result.inferred_goal}",
            )

    def test_compiled_goal_solves_holdout(self) -> None:
        for name in self.CLEAN:
            task = cf.TASKS[name]
            levels = cf.make_cone_levels(29, 12, task)
            env = gi.HiddenTask(task, levels)
            result = gi.induce_active(env, list(range(6)), lam=0.05, budget=7)
            controller, library = gi.compile_goal(result.inferred_goal)
            solved = env.solved_fraction(controller, library, list(range(6, 12)))
            self.assertEqual(solved, 1.0, f"{name}: holdout solved {solved}")

    def test_parsimony_lambda_sweep_on_confounded_task(self) -> None:
        # flee_then_home: home implies safe (home generated outside hazard
        # radius). Small lambda recovers the full goal; larger lambda prefers
        # the simpler equally-rewarded goal. Either choice solves the task.
        task = cf.TASKS["flee_then_home"]
        levels = cf.make_cone_levels(29, 10, task)
        env = gi.HiddenTask(task, levels)
        observations = []
        for _name, (controller, library) in gi.probe_cones().items():
            observations += env.evaluate(controller, library, list(range(6)))
        small, _ = gi.induce_goal(observations, lam=0.005)
        large, _ = gi.induce_goal(observations, lam=0.05)
        self.assertEqual(set(small), {"home", "safe"})
        self.assertEqual(set(large), {"home"})
        # Both compile to a cone that solves the task on held-out levels.
        for goal in (small, large):
            controller, library = gi.compile_goal(goal)
            self.assertEqual(env.solved_fraction(controller, library, list(range(6, 10))), 1.0)

    def test_active_loop_is_sample_efficient(self) -> None:
        # Single-feature goals should be identified in one round on this set.
        task = cf.TASKS["homing"]
        levels = cf.make_cone_levels(29, 10, task)
        env = gi.HiddenTask(task, levels)
        result = gi.induce_active(env, list(range(6)), lam=0.05, budget=7)
        self.assertLessEqual(result.rounds, 3)
        self.assertGreater(result.probe_episodes, 0)


if __name__ == "__main__":
    unittest.main()
