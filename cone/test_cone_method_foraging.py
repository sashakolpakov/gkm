import unittest

import cone_foraging as cf
import cone_method as cm
import cone_method_foraging as cmf


class MethodDrivesForagingTests(unittest.TestCase):
    """A GKM (a cone constructed from legs and selected by free energy), driven
    through the general cone_method interface, solves these foraging tasks with
    no hand-coded policy. Scope: this validates the GKM selection loop on the
    foraging connector only — not a claim about other substrates."""

    TASKS = ["forage", "homing", "forage_then_home", "forage_flee", "flee_then_home"]

    def _solve(self, task_name):
        task = cf.TASKS[task_name]
        levels = cf.make_cone_levels(29, 8, task)
        env = cmf.ForagingEnvironment(task=task, levels=levels, library=cmf.goal_library())
        plans = [[("CONE", subset)] for subset, _g in cmf.candidate_cones()]
        result = cm.induce_goal_over_env(env, plans, lam=0.05, max_goal_size=2)
        chosen, _fe = cm.select_plan_by_free_energy(env, plans, result.inferred_goal, lam=0.02)
        return result.inferred_goal, chosen[0][1], env.solved_fraction(chosen[0][1])

    def test_method_solves_every_task(self) -> None:
        for name in self.TASKS:
            _goal, _cone, solved = self._solve(name)
            self.assertEqual(solved, 1.0, f"{name}: method-selected cone solved {solved}")

    def test_induced_goal_matches_task(self) -> None:
        # The induced goal recovers the task's required features (the parsimony
        # confound on flee_then_home is acceptable — home implies safe).
        for name in ("forage", "homing", "forage_then_home", "forage_flee"):
            goal, _cone, _solved = self._solve(name)
            self.assertEqual(set(goal), set(cmf.gi.task_goal_features(cf.TASKS[name])),
                             f"{name}: induced {goal}")

    def test_connector_actions_are_cones(self) -> None:
        task = cf.TASKS["forage"]
        env = cmf.ForagingEnvironment(task=task, levels=cf.make_cone_levels(1, 2, task),
                                      library=cmf.goal_library())
        env.reset()
        acts = env.actions()
        self.assertTrue(all(a[0] == "CONE" for a in acts))
        self.assertIsInstance(env, cm.Environment)


if __name__ == "__main__":
    unittest.main()
