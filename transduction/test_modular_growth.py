import copy
import json
import tempfile
import unittest
from pathlib import Path

from audit_growth import equality_patterns
from cofibration import verify_certificate
from growth_benchmark import (FRAGMENTS, acquire_word, candidate_payload, candidates,
                              dataset, evaluate, expression, mechanical, parse_proposal,
                              prompt_for, replay, response_schema, run, target)
from modular_machine import Library, Machine, call, expanded_size, structural_metrics, unshare


class ModularGrowthTests(unittest.TestCase):
    def setUp(self):
        self.library = Library.empty()
        for index, (name, width) in enumerate(FRAGMENTS, 1):
            word, _ = acquire_word(dataset(name, 1, "train", width))
            self.library, _ = self.library.attach(f"F{index:03}", word, kind="fragment", width=width)

    def test_seed_fragments_are_learned_and_generalize(self):
        self.library.verify()
        for index, (name, width) in enumerate(FRAGMENTS, 1):
            self.assertTrue(evaluate(self.library, f"F{index:03}", dataset(name, 11, "hidden", width))["exact"])

    def test_equality_pattern_generator_has_bell_number_counts(self):
        expected = (1, 1, 2, 5, 15, 52, 203, 877, 4140)
        for length, count in enumerate(expected):
            patterns = list(equality_patterns(length))
            self.assertEqual(len(patterns), count)
            self.assertEqual(len(set(patterns)), count)
            for pattern in patterns:
                for index, value in enumerate(pattern):
                    self.assertLessEqual(value, 1 + max(pattern[:index], default=-1))

    def test_calls_really_return_to_following_call(self):
        program = expression("seq", "F004", "F004")
        library, square = self.library.attach("M01", program)
        result = Machine(library).run("M01", [1, 2, 3, 4], trace_limit=100)
        self.assertEqual(result["output"], [2, 1, 4, 3])
        self.assertEqual(result["calls"], result["returns"])
        self.assertEqual(result["visits"]["F004"], 2)
        self.assertEqual([r[:2] for r in result["trace"]],
                         [["enter", "M01"], ["enter", "F004"], ["return", "F004"],
                          ["enter", "F004"], ["return", "F004"], ["return", "M01"]])
        self.assertEqual(square.boundary.states, 2)  # shared entry AND return, one callee
        self.assertEqual(len(square.boundary.rules), 3)
        self.assertEqual(verify_certificate(square.certificate(), self.library.graph).graph, library.graph)
        self.assertEqual(tuple(range(self.library.graph.states)), square.from_left.states)
        self.assertEqual(library.graph.rules[:len(self.library.graph.rules)], self.library.graph.rules)

    def test_return_to_loop_and_tail_handles_odd_inputs(self):
        library, _ = self.library.attach("M01", expression("while", "F001", n=1))
        library, _ = library.attach("M02", expression("while_then", "F004", "M01", n=2))
        self.assertTrue(evaluate(library, "M02", dataset("swap_pairs", 11, "hidden"))["exact"])
        self.assertTrue(evaluate(library, "M01", dataset("copy_all", 11, "hidden"))["exact"])
        self.assertFalse(evaluate(library, "M02", dataset("swap_pairs", 11, "train"), disabled=("F004",))["exact"])
        library.verify()

    def test_guards_and_pipeline_are_executed(self):
        library, _ = self.library.attach("M01", expression("while", "F003", n=1))
        library, _ = library.attach("M02", expression("while_if", "F002", "F001", n=1))
        library, _ = library.attach("M03", expression("pipe", "M01", "M02"))
        self.assertEqual(Machine(library).run("M03", [1, 2, 2, 3])["output"], [1, 2, 3])
        self.assertTrue(evaluate(library, "M02", dataset("dedupe_runs", 11, "hidden"))["exact"])
        self.assertEqual(Machine(library).run("M03", [1, 2])["max_call_depth"], 3)
        with self.assertRaises(ValueError):
            library.attach("bad", expression("pipe", "F001", "M02"))

    def test_window_prevents_eos_noop_from_reading_callers_next_token(self):
        library = Library.empty()
        library, _ = library.attach("F", {"op": "word", "actions": [1, 0, 1]}, kind="fragment", width=1)
        library, _ = library.attach("M", expression("seq", "F", "F"))
        self.assertEqual(Machine(library).run("M", [7, 9])["output"], [7, 9])

    def test_local_registers_are_not_leaked(self):
        library = Library.empty()
        library, _ = library.attach("F", {"op": "word", "actions": [30, 10, 0]}, kind="fragment", width=1)
        library, _ = library.attach("M", expression("while", "F", n=1))
        self.assertEqual(Machine(library).run("M", [7, 9])["output"], [])

    def test_execution_limits_and_interface_contracts_fail_closed(self):
        library, _ = self.library.attach("M", expression("while", "F003", n=1))
        machine = Machine(library)
        self.assertEqual(machine.run("M", [1] * 8, output_limit=8)["error"], "output/buffer limit")
        self.assertEqual(machine.run("M", [1] * 8, step_limit=4)["error"], "step limit")
        self.assertEqual(machine.run("M", [1], depth_limit=1)["error"], "call depth limit")
        self.assertEqual(machine.run("F004", [1])["error"], "fragment window underflow")
        bad, _ = self.library.attach("bad", expression("call", "F001"))
        self.assertIn("unread", Machine(bad).run("bad", [1, 2])["error"])
        bad, _ = Library.empty().attach("F", {"op": "word", "actions": [1]}, kind="fragment", width=1)
        self.assertIn("consume", Machine(bad).run("F", [1])["error"])

    def test_content_and_callee_versions_are_immutable(self):
        library, _ = self.library.attach("M", expression("while", "F001", n=1))
        payload = json.loads(json.dumps(library.payload()))
        payload["cells"]["M"]["program"]["n"] = 2
        with self.assertRaisesRegex(ValueError, "content"):
            Library.read(payload)
        payload = json.loads(json.dumps(library.payload()))
        payload["cells"]["M"]["dependencies"]["F001"] = "wrong"
        with self.assertRaises(ValueError):
            Library.read(payload)
        with self.assertRaises(ValueError):
            library.attach("M", call("F001"))

    def test_certificate_cannot_change_old_body(self):
        _, square = self.library.attach("M", expression("while", "F001", n=1))
        certificate = copy.deepcopy(square.certificate())
        certificate["pushout"]["states"] += 1
        with self.assertRaises(ValueError):
            verify_certificate(certificate, self.library.graph)

    def test_shared_schema_and_enumerator_have_identical_shapes(self):
        schema = response_schema(self.library)
        seen = set()
        for payload in candidates(self.library):
            self.library.attach("M", parse_proposal(self.library, payload))
            seen.add(payload["shape"])
        self.assertEqual(seen, set(schema["properties"]["shape"]["enum"]) - {"pipe"})
        payload = candidate_payload(self.library, "while", ["F001"], 1)
        for key, invalid in (("n", True), ("b", "F002"), ("library_hash", "bad"), ("a", "M99")):
            with self.assertRaises(ValueError):
                parse_proposal(self.library, {**payload, key: invalid})

    def test_prompt_does_not_reveal_task_name_or_hidden_examples(self):
        prompt = prompt_for(self.library, dataset("dedupe_runs", 1, "train"))
        self.assertNotIn("dedupe_runs", json.dumps(prompt))
        self.assertNotIn("hidden_examples", prompt)
        pools = [{t for pair in dataset("guarded_pipeline", 1, split) for seq in pair for t in seq}
                 for split in ("train", "validation", "hidden", "stress")]
        for index, pool in enumerate(pools):
            for other in pools[:index]:
                self.assertTrue(pool.isdisjoint(other))

    def test_reference_order_is_not_commutative(self):
        source = (1, 2, 2, 3, 4, 4, 4)
        self.assertNotEqual(target("swap_then_dedupe", source), target("dedupe_then_swap", source))

    def test_composition_size_diagnostic_counts_sharing_not_execution(self):
        library, _ = self.library.attach("M01", expression("while", "F001", n=1))
        library, _ = library.attach("M02", expression("pipe", "M01", "M01"))
        self.assertGreater(expanded_size(library, "M02"), structural_metrics(library, "M02")["reachable_shared_term_units"])
        other, entry = unshare(library, "M02")
        self.assertEqual(structural_metrics(other, entry)["library_stored_term_units"], expanded_size(library, "M02"))
        a, b = Machine(library).run("M02", [1, 2, 3]), Machine(other).run(entry, [1, 2, 3])
        for key in ("output", "cursor", "ok", "steps", "calls", "returns"):
            self.assertEqual(a[key], b[key])

    def test_ablation_can_leave_redundant_piece_successful(self):
        library, _ = self.library.attach("M01", expression("while", "F001", n=1))
        library, _ = library.attach("M02", expression("seq", "M01", "M01"))
        # Eliding both whole-input copy calls should fail for a nonempty source;
        # eliding only the redundant final *empty window* fragment is different.
        self.assertFalse(Machine(library).run("M02", [1], disabled=("M01",))["ok"])
        # On empty input the same intervention succeeds: ablation does not force failure.
        self.assertTrue(Machine(library).run("M02", [], disabled=("M01",))["ok"])

    def test_mechanical_search_proposes_returning_glue(self):
        proposal, effort = mechanical(self.library, "M", dataset("copy_all", 11, "train"), budget=100)
        self.assertTrue(effort["exact_found"])
        library, _ = self.library.attach("M", parse_proposal(self.library, proposal))
        self.assertTrue(evaluate(library, "M", dataset("copy_all", 11, "hidden"))["exact"])

    def test_small_run_freezes_then_independently_replays(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "run"
            rows = run(folder, seeds=(1,), arms=("mechanical",), stages=1)
            self.assertTrue(rows[0]["admitted"])
            self.assertTrue(rows[0]["stress"]["exact"])
            self.assertEqual(replay(folder)["verified_pushouts"], 6)


if __name__ == "__main__":
    unittest.main()
