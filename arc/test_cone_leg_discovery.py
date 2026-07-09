import unittest

import arc_agi3_adapter as arc
import cone_foraging as cf
import cone_leg_discovery as cld
import cone_leg_composition as clc


def make_stub(seed: int = 7, w: int = 12, h: int = 12):
    return lambda: arc.StubNavigationGame.random(seed, width=w, height=h)


class LegDiscoveryTests(unittest.TestCase):
    def test_detect_avatar_on_stub(self) -> None:
        self.assertEqual(cld.detect_avatar(make_stub(), [1, 2, 3, 4]), 4)

    def test_learns_directional_legs_above_random_floor(self) -> None:
        res = cld.discover_effect_legs(make_stub(), trials=20, seed=3)
        self.assertEqual(res.avatar_color, 4)
        # All four cardinal legs are reliably controllable on open navigation.
        self.assertEqual(len(res.legs), 4)
        directions = {leg.direction for leg in res.legs}
        self.assertEqual(directions, {(0, -1), (1, 0), (0, 1), (-1, 0)})
        # Every learned leg is perfectly consistent (never moves the wrong way)
        # and clears the random control floor.
        for leg in res.legs:
            self.assertEqual(leg.consistency, 1.0)
            self.assertGreater(leg.consistency, res.random_consistency)
        self.assertLess(res.random_consistency, 1.0)  # random is not consistently directional

    def test_no_avatar_no_legs(self) -> None:
        # A degenerate env whose actions never change the frame: nothing is
        # controllable, so no avatar and no legs are learned (honest null).
        class Frozen:
            available_actions = [1, 2, 3, 4]
            def __init__(self):
                self._f = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]  # static colour-5 blob
            def reset(self):
                return arc.Snapshot(frames=[self._f], score=0.0, state=arc.GameState.IN_PROGRESS)
            def step(self, action, x=None, y=None):
                return arc.Snapshot(frames=[self._f], score=0.0, state=arc.GameState.IN_PROGRESS)
        res = cld.discover_effect_legs(lambda: Frozen(), trials=8, seed=0)
        self.assertIsNone(res.avatar_color)
        self.assertEqual(res.legs, [])


class LegCompositionTests(unittest.TestCase):
    def test_composed_cone_over_learned_legs_wins(self) -> None:
        res = cld.discover_effect_legs(make_stub(), trials=20, seed=3)
        snap, steps, reached = clc.run_composed_seek(
            make_stub()(), res.legs, goal_color=2, avatar_color=res.avatar_color, max_steps=80)
        self.assertEqual(snap.state, arc.GameState.WIN)
        self.assertTrue(reached)

    def test_matches_hand_built_witness(self) -> None:
        # The learned-leg cone and the hand-built witness both solve the stub.
        res = cld.discover_effect_legs(make_stub(), trials=20, seed=3)
        learned, _, _ = clc.run_composed_seek(
            make_stub()(), res.legs, goal_color=2, avatar_color=4, max_steps=80)
        witness = arc.run_seek_leg_on_game(
            make_stub()(), cf.witness_seek_leg(), goal_color=2, avatar_color=4, max_steps=80)
        self.assertEqual(learned.state, arc.GameState.WIN)
        self.assertEqual(witness.state, arc.GameState.WIN)


class InteractStub:
    """Tiny env: avatar (colour 7) moves canonically; ACTION5 toggles a separate
    object (colour 3) between two cells — a non-avatar scene change. Used to test
    interaction-leg discovery hermetically (no local game)."""
    available_actions = [1, 2, 3, 4, 5]

    def __init__(self):
        self.reset()

    def reset(self):
        self.ax, self.ay, self.toggled = 3, 3, False
        return self._snap()

    def step(self, action, x=None, y=None):
        d = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}.get(int(action))
        if d:
            nx, ny = self.ax + d[0], self.ay + d[1]
            if 0 <= nx < 7 and 0 <= ny < 7:
                self.ax, self.ay = nx, ny
        elif int(action) == 5:
            self.toggled = not self.toggled
        return self._snap()

    def _snap(self):
        f = [[0] * 7 for _ in range(7)]
        f[self.ay][self.ax] = 7                 # avatar
        f[1][0 if self.toggled else 6] = 3      # the toggled (non-avatar) object
        return arc.Snapshot(frames=[f], score=0.0, state=arc.GameState.IN_PROGRESS)


class InteractionLegTests(unittest.TestCase):
    def test_discovers_action5_interaction_above_move_floor(self) -> None:
        legs, floor = cld.discover_interaction_legs(lambda: InteractStub(), trials=12, seed=1)
        actions = {lg.action for lg in legs}
        self.assertIn(5, actions)                       # ACTION5 toggles a non-avatar object
        leg5 = next(lg for lg in legs if lg.action == 5)
        self.assertGreater(leg5.efficacy, floor)        # beats the move-control floor
        self.assertEqual(floor, 0.0)                    # moves cause no non-avatar change


if __name__ == "__main__":
    unittest.main()
