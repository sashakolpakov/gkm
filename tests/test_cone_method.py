import unittest

import cone_method as cm
import arc_agi3_adapter as arc


class MockEnvironment:
    """A tiny substrate-free Environment: actions toggle hidden boolean
    features; the reward is the mean of a hidden target subset. Proves the
    method is general — it induces the goal with no ARC or foraging at all."""

    def __init__(self, target):
        self.target = set(target)
        self.state = {f: 0.0 for f in ("a", "b", "c")}

    def reset(self):
        self.state = {f: 0.0 for f in ("a", "b", "c")}

    def actions(self):
        return [("set", f) for f in ("a", "b", "c")]

    def apply(self, action):
        self.state[action[1]] = 1.0

    def features(self):
        return dict(self.state)

    def reward(self):
        return sum(self.state[f] for f in self.target) / len(self.target) if self.target else 1.0

    def done(self):
        return False


class GeneralityTests(unittest.TestCase):
    def test_environment_protocol(self) -> None:
        env = MockEnvironment(target=("a",))
        self.assertIsInstance(env, cm.Environment)  # runtime_checkable protocol

    def test_method_induces_goal_on_mock(self) -> None:
        # Single-feature and two-feature hidden goals, recovered by the method
        # with zero substrate knowledge — only the abstract Environment.
        for target in (("a",), ("b",), ("a", "c")):
            env = MockEnvironment(target=target)
            plans = [[], [("set", "a")], [("set", "b")], [("set", "c")],
                     [("set", "a"), ("set", "c")], [("set", "a"), ("set", "b")]]
            result = cm.induce_goal_over_env(env, plans, lam=0.05, max_goal_size=2)
            self.assertEqual(set(result.inferred_goal), set(target), f"target {target}: got {result.inferred_goal}")

    def test_select_plan_by_free_energy(self) -> None:
        env = MockEnvironment(target=("a", "b"))
        plans = [[], [("set", "a")], [("set", "a"), ("set", "b")]]
        plan, _f = cm.select_plan_by_free_energy(env, plans, goal=("a", "b"), lam=0.01)
        # The plan that sets both a and b best satisfies the goal.
        self.assertEqual(set(a[1] for a in plan), {"a", "b"})

    def test_feature_keys_from_env(self) -> None:
        env = MockEnvironment(target=("a",))
        self.assertEqual(cm.feature_keys(env), ["a", "b", "c"])


class ArcConnectorTests(unittest.TestCase):
    """The ARC connector conforms to the method's Environment contract and
    provides actions — constructed offline (no network until reset/step)."""

    def test_arc_environment_is_environment(self) -> None:
        conn = arc.ArcEnvironment("wa30-ee6fef47", api_key="dummy")
        self.assertIsInstance(conn, cm.Environment)
        # Full id is preserved (required for the action loop; the bare short
        # code is rejected by ACTION). reset() resolves a short code to full.
        self.assertEqual(conn.env.game_id, "wa30-ee6fef47")

    def test_actions_and_features_shapes(self) -> None:
        # Drive the connector against a synthetic frame without network by
        # injecting a Snapshot directly.
        conn = arc.ArcEnvironment("ls20", api_key="dummy")
        frame = [[0, 0, 0, 0], [0, 3, 0, 2], [0, 0, 0, 0], [4, 0, 0, 0]]
        conn._snap = arc.Snapshot(frames=[frame], score=0.0, state=arc.GameState.IN_PROGRESS)
        conn.env.available_actions = [1, 2, 3, 4, 5, 6]
        acts = conn.actions()
        self.assertTrue(any(a[0] == "KEY" for a in acts))
        self.assertTrue(any(a[0] == "CLICK" for a in acts))  # 6 available -> click candidates
        feats = conn.features()
        self.assertIn("present@3", feats)
        for v in feats.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        self.assertEqual(conn.reward(), 0.0)
        self.assertFalse(conn.done())


if __name__ == "__main__":
    unittest.main()
