import unittest

import cone_foraging as cf
import cone_foraging_bound as cb
import cone_render as cr


class TraceFidelityTests(unittest.TestCase):
    """The trace must not drift from executed semantics: the traced final
    position and collected count must match a non-traced run."""

    def test_v12_trace_matches_episode(self) -> None:
        task = cf.TASKS["forage_then_home"]
        library = [cf.witness_seek_leg()]
        controller = cf.witness_gluing(task, seek_index=0)
        for level in cf.make_cone_levels(71, 4, task):
            plain = cf.run_cone_episode(controller, library, level, task)
            trace = []
            traced = cf.run_cone_episode(controller, library, level, task, trace=trace)
            self.assertEqual(plain.final_position, traced.final_position)
            self.assertEqual(plain.collected, traced.collected)
            # The last traced position equals the episode's final position.
            self.assertEqual(trace[-1]["pos"], traced.final_position)

    def test_bound_trace_matches_episode(self) -> None:
        task = cf.TASKS["forage_then_home"]
        library = [cf.witness_seek_leg()]
        controller = cb.witness_bound_gluing(task, seek_index=0)
        for level in cf.make_cone_levels(72, 4, task):
            plain = cb.run_bound_episode(controller, library, level, task)
            trace = []
            traced = cb.run_bound_episode(controller, library, level, task, trace=trace)
            self.assertEqual(plain.final_position, traced.final_position)
            self.assertEqual(trace[-1]["pos"], traced.final_position)

    def test_trace_records_calls(self) -> None:
        task = cf.TASKS["forage_then_home"]
        library = [cf.witness_seek_leg()]
        controller = cb.witness_bound_gluing(task, seek_index=0)
        level = cf.make_cone_levels(73, 1, task)[0]
        trace = []
        cb.run_bound_episode(controller, library, level, task, trace=trace)
        calls = [e for e in trace if e["kind"] == "call"]
        self.assertEqual(len(calls), 2)  # seek food, then seek home


class RenderTests(unittest.TestCase):
    def test_render_level_has_markers(self) -> None:
        task = cf.TASKS["forage_then_home"]
        level = cf.make_cone_levels(74, 1, task)[0]
        text = cr.render_level(level, task)
        self.assertIn("S", text)
        self.assertIn("H", text)
        self.assertIn("*", text)
        # Grid is bordered to the declared width.
        self.assertIn("+" + "-" * level.width + "+", text)

    def test_render_solution_marks_final_and_solves(self) -> None:
        task = cf.TASKS["forage"]
        library = [cf.witness_seek_leg()]
        controller = cf.witness_gluing(task, seek_index=0)
        level = cf.make_cone_levels(75, 1, task)[0]
        text = cr.render_cone_solution(controller, library, level, task)
        self.assertIn("@", text)
        self.assertIn("moves=", text)

    def test_render_hazard_layout(self) -> None:
        task = cf.TASKS["flee"]
        level = cf.make_cone_levels(76, 1, task)[0]
        text = cr.render_level(level, task)
        self.assertIn("X", text)
        self.assertIn("safe radius", text)


if __name__ == "__main__":
    unittest.main()
