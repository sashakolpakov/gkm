import unittest

import cone_foraging as cf
import cone_foraging_bound as cb
import arc_agi3_adapter as arc


class SceneFunctorTests(unittest.TestCase):
    def test_connected_components(self) -> None:
        frame = [
            [0, 0, 0, 0],
            [0, 3, 3, 0],
            [0, 3, 0, 2],
            [0, 0, 0, 0],
        ]
        comps3 = arc.connected_components(frame, 3)
        self.assertEqual(len(comps3), 1)
        self.assertEqual(set(comps3[0]), {(1, 1), (2, 1), (1, 2)})
        comps2 = arc.connected_components(frame, 2)
        self.assertEqual(comps2, [((3, 2),)])

    def test_extract_scene_avatar_heuristic(self) -> None:
        frame = [
            [0, 0, 0, 0],
            [0, 4, 0, 0],  # avatar: singleton colour 4
            [0, 0, 0, 2],  # goal: singleton colour 2
            [0, 0, 0, 0],
        ]
        scene = arc.extract_scene(frame, avatar_color=4)
        self.assertIsNotNone(scene.avatar)
        self.assertEqual(scene.avatar.color, 4)
        self.assertEqual(scene.avatar.centroid, (1, 1))
        self.assertEqual(scene.colors_present(), [2, 4])

    def test_scene_delta_detects_move(self) -> None:
        before = arc.extract_scene([[4, 0], [0, 2]], avatar_color=4)
        after = arc.extract_scene([[0, 4], [0, 2]], avatar_color=4)
        delta = arc.scene_delta(before, after)
        self.assertIn(4, delta.moved)
        self.assertEqual(delta.moved[4], ((0, 0), (1, 0)))


class CorrespondenceTests(unittest.TestCase):
    def test_slot_observation_is_azimuth(self) -> None:
        # Avatar at (1,1), goal colour 2 to the east at (3,1).
        frame = [
            [0, 0, 0, 0],
            [0, 4, 0, 2],
            [0, 0, 0, 0],
        ]
        scene = arc.extract_scene(frame, avatar_color=4)
        self.assertEqual(arc.slot_observation(scene, 2), cf.OBS_LABELS.index("E"))

    def test_slot_observation_here_when_no_target(self) -> None:
        frame = [[0, 4, 0]]
        scene = arc.extract_scene(frame, avatar_color=4)
        # No colour-2 object => that channel reads HERE (nothing to seek).
        self.assertEqual(arc.slot_observation(scene, 2), cf.HERE_OBS)

    def test_scene_to_cone_level_projection(self) -> None:
        frame = [
            [4, 0, 0],
            [0, 0, 0],
            [0, 0, 2],
        ]
        scene = arc.extract_scene(frame, avatar_color=4)
        level = arc.scene_to_cone_level(scene, goal_color=2)
        self.assertEqual(level.start, (0, 0))
        self.assertEqual(level.home, (2, 2))


class EndToEndTests(unittest.TestCase):
    def test_seek_leg_solves_stub_game(self) -> None:
        # The channel-blind witness seek leg, bound to the goal colour, should
        # drive the avatar to the goal across seeded fixtures.
        leg = cf.witness_seek_leg()
        wins = 0
        for seed in range(10):
            game = arc.StubNavigationGame.random(seed)
            final = arc.run_seek_leg_on_game(game, leg, goal_color=game.goal_color)
            if final.state == arc.GameState.WIN:
                wins += 1
        self.assertEqual(wins, 10)

    def test_cone_witness_solves_projected_level(self) -> None:
        # The bound-cone witness (gluing + seek leg) solves the projected level,
        # i.e. the SAME free-energy substrate runs on an ARC-shaped scene.
        game = arc.StubNavigationGame.random(3)
        game.reset()
        scene = arc.extract_scene(game.render(), avatar_color=game.avatar_color)
        level = arc.scene_to_cone_level(scene, goal_color=game.goal_color)
        task = cf.TASKS["homing"]
        controller = cb.witness_bound_gluing(task, seek_index=0)
        episode = cb.run_bound_episode(
            controller, [cf.witness_seek_leg()], level, task, max_steps=64
        )
        self.assertTrue(cf.episode_solved(episode, level, task))


class ArcEnvParsingTests(unittest.TestCase):
    """Offline tests of the live-client logic — no network, no committed real
    data. They check the parsing/normalization against synthetic payloads of
    the verified shape."""

    def test_normalize_game_id_preserves_version(self) -> None:
        # The full id is required for the action loop, so normalization keeps the
        # version suffix (only lowercases/trims).
        self.assertEqual(arc.normalize_game_id("wa30-ee6fef47"), "wa30-ee6fef47")
        self.assertEqual(arc.normalize_game_id("WA30-EE6FEF47"), "wa30-ee6fef47")
        self.assertEqual(arc.normalize_game_id("LS20"), "ls20")

    def test_resolve_full_game_id(self) -> None:
        games = [{"game_id": "ls20-9607627b", "title": "LS20"},
                 {"game_id": "wa30-ee6fef47", "title": "WA30"}]
        # short code -> full id (offline; games list supplied)
        self.assertEqual(arc.resolve_full_game_id("ls20", games), "ls20-9607627b")
        # already-full id passes through; unknown short code falls back to itself
        self.assertEqual(arc.resolve_full_game_id("wa30-ee6fef47", games), "wa30-ee6fef47")
        self.assertEqual(arc.resolve_full_game_id("zz99", games), "zz99")

    def test_snapshot_parses_frame_list_and_actions(self) -> None:
        env = arc.ArcEnv("wa30-ee6fef47", api_key="dummy")  # constructing needs no network
        self.assertEqual(env.game_id, "wa30-ee6fef47")  # full id preserved
        payload = {
            "frame": [[[0, 1], [2, 0]], [[0, 0], [2, 1]]],  # two settling grids
            "state": "NOT_FINISHED", "available_actions": [1, 2, 3, 4, 5],
            "levels_completed": 2, "win_levels": 7, "guid": "g-123",
        }
        snap = env._snapshot_from_payload(payload)
        self.assertEqual(snap.frame, [[0, 0], [2, 1]])  # latest grid
        self.assertEqual(env.available_actions, [1, 2, 3, 4, 5])
        # NOT_FINISHED maps onto the running state; no `score` field => reward is
        # levels_completed; win progress carried through.
        self.assertEqual(snap.state, arc.GameState.IN_PROGRESS)
        self.assertEqual(snap.score, 2.0)
        self.assertEqual((snap.levels_completed, snap.win_levels), (2, 7))

    def test_env_requires_key(self) -> None:
        import os
        saved = os.environ.pop("ARC_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError):
                arc.ArcEnv("wa30")
        finally:
            if saved is not None:
                os.environ["ARC_API_KEY"] = saved


class LocalArcEnvTests(unittest.TestCase):
    """Local-play tests via the arc_agi toolkit. They SKIP (not fail) when the
    toolkit or the pre-downloaded game files are absent, so the suite stays
    hermetic and portable."""

    def _require_local_ls20(self):
        import importlib.util
        from pathlib import Path
        if importlib.util.find_spec("arc_agi") is None:
            self.skipTest("arc_agi toolkit not installed")
        env_dir = Path(__file__).resolve().parents[1] / "environment_files"
        if not any(env_dir.glob("ls20/*/metadata.json")) if env_dir.exists() else True:
            self.skipTest("ls20 not downloaded under environment_files/ (run normal mode once)")
        return str(env_dir)

    def test_local_offline_reset_and_step(self) -> None:
        env_dir = self._require_local_ls20()
        env = arc.LocalArcEnv("ls20", operation_mode="offline", environments_dir=env_dir)
        snap = env.reset()
        # Same Snapshot contract as the online client, on real local frames.
        self.assertEqual(snap.state, arc.GameState.IN_PROGRESS)
        self.assertEqual((len(snap.frame), len(snap.frame[0])), (64, 64))
        self.assertIsInstance(snap.frame[0][0], int)  # numpy coerced to python int
        self.assertEqual(snap.win_levels, 7)
        self.assertEqual(env.available_actions, [1, 2, 3, 4])
        s = env.step(arc.GameAction.ACTION1)
        self.assertEqual(s.state, arc.GameState.IN_PROGRESS)


if __name__ == "__main__":
    unittest.main()
