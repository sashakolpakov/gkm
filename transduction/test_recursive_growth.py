import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import z3

from cofibration import verify_certificate
from recursive_benchmark import (DEFAULT_BINDINGS, TRANSFER_BINDINGS, acquire_basis,
                                 pairs_for, replay, run, specification, witness)
from recursive_machine import Machine, RecursiveLibrary, node, parse
from recursive_search import Synthesizer, mechanical, prompt, score
from audit_recursive import audit


class RecursiveGrowthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = acquire_basis(Path(cls.temporary.name) / "basis")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_old_basis_is_unchanged_and_new_words_are_acquired(self):
        self.assertEqual(self.base.base.cells["F001"]["program"]["actions"], [1, 0])
        self.assertEqual(self.base.base.cells["F004"]["program"]["actions"], [10, 0, 1, 0, 30])
        self.assertEqual(set(self.base.base.cells), {"F001", "F002", "F003", "F004", "F005", "C01", "C02", "C03"})

    def test_private_witnesses_use_only_the_shared_grammar(self):
        for family in ("right_parity", "two_contexts"):
            spec = specification(family)
            body = witness(family)
            self.assertEqual(parse(dict(library_hash=self.base.digest, nodes=body), self.base, spec["bits"], spec["maximum"]), body)
            lib, _ = self.base.attach("R01", body, spec["bits"])
            for split in ("train", "validation", "hidden", "stress"):
                result = score(lib, "R01", pairs_for(family, 1, split), DEFAULT_BINDINGS)
                self.assertTrue(result["exact"], (family, split, [r for r in result["records"] if not r["exact"]][:1]))

    def test_pushout_preserves_exact_old_graph_and_cell_hashes(self):
        built, square = self.base.attach("R01", witness("right_parity"), 1)
        rebuilt = verify_certificate(json.loads(json.dumps(square.certificate())), self.base.graph)
        self.assertEqual(rebuilt.graph, built.graph)
        self.assertEqual(built.graph.rules[:len(self.base.graph.rules)], self.base.graph.rules)
        self.assertEqual(built.base.payload(), self.base.base.payload())
        self.assertEqual(RecursiveLibrary.read(json.loads(json.dumps(built.payload()))).digest, built.digest)

    def test_reuse_is_execution_of_same_recursive_cell_with_new_bindings(self):
        first, _ = self.base.attach("R01", witness("right_parity"), 1)
        adapter = [node("invoke", cell="R01", arg=0, a=1, bindings=TRANSFER_BINDINGS), node("ret")]
        second, square = first.attach("R02", adapter, 1)
        self.assertEqual(first.cells["R01"], second.cells["R01"])
        self.assertEqual(second.cells["R02"]["dependencies"]["R01"], first.cells["R01"]["hash"])
        self.assertGreater(square.boundary.states, 0)
        pairs = pairs_for("right_parity", 2, "hidden")
        result = score(second, "R02", pairs, DEFAULT_BINDINGS)
        self.assertTrue(result["exact"])
        self.assertTrue(all(r["visits"].get("R01", 0) > 0 for r in result["records"]))
        self.assertTrue(all(r["calls"] == r["returns"] for r in result["records"]))
        self.assertEqual(score(second, "R02", pairs, DEFAULT_BINDINGS, disabled=("R01",))["correct"], 0)
        self.assertTrue(score(second, "R01", pairs_for("right_parity", 1, "hidden"), DEFAULT_BINDINGS)["exact"])

    def test_context_is_restored_on_return_not_accumulated_globally(self):
        lib, _ = self.base.attach("R01", witness("right_parity"), 1)
        # Root(left internal(two leaves), right leaf): the left child's right
        # call must not toggle the root's own context for its subsequent call.
        source = [9, 9, 8, 8, 1, 2, 3, 4, 5, 6]
        result = Machine(lib).run("R01", source, DEFAULT_BINDINGS, trace_limit=100)
        self.assertEqual(result["output"], [1, 2, 4, 3, 6, 5])
        self.assertTrue(result["ok"])
        self.assertEqual(result["trace"][-1][3], 0)

    def test_progress_and_consumption_are_hard_gates(self):
        lib, _ = self.base.attach("Bad", [node("recur", arg=0, a=1), node("ret")], 1)
        result = Machine(lib).run("Bad", [1, 2], DEFAULT_BINDINGS)
        self.assertFalse(result["ok"])
        self.assertIn("strict input progress", result["error"])
        lib, _ = self.base.attach("Short", [node("ret")], 1)
        self.assertIn("unread", Machine(lib).run("Short", [1, 2], DEFAULT_BINDINGS)["error"])

    def test_callbacks_cannot_change_consumption_width_or_old_body(self):
        lib, _ = self.base.attach("R01", witness("right_parity"), 1)
        with self.assertRaises(ValueError):
            lib.attach("R02", [node("invoke", cell="R01", arg=0, a=1, bindings=("F001", "F004")), node("ret")], 1)
        payload = copy.deepcopy(lib.payload())
        payload["cells"]["R01"]["nodes"][3]["arg"] = 1
        with self.assertRaises(ValueError):
            RecursiveLibrary.read(payload)

    def test_no_ordinary_cycles_boolean_indices_or_unknown_fields(self):
        for body in ([node("call", cell="F001", a=0)],
                     [node("call", cell="F001", a=True), node("ret")],
                     [{**node("ret"), "tree_parser": True}]):
            with self.assertRaises(ValueError):
                self.base.attach("Bad", body, 1)

    def test_training_and_hidden_token_pools_are_disjoint(self):
        train = pairs_for("right_parity", 1, "train")
        hidden = pairs_for("right_parity", 1, "hidden")
        self.assertFalse({x for s, _ in train for x in s} & {x for s, _ in hidden for x in s})
        self.assertGreater(max(len(s) for s, _ in hidden), max(len(s) for s, _ in train))

    def test_prompt_does_not_reveal_target_or_traversal_skeleton(self):
        request = prompt(self.base, pairs_for("right_parity", 1, "train"), 1, 12, DEFAULT_BINDINGS)
        self.assertEqual(request["retained_prefix_procedures"], {})
        text = json.dumps(request)
        for forbidden in ("right_parity", "left_mask", "right_mask", "truth", "reference", "witness"):
            self.assertNotIn(forbidden, text)

    def test_smt_discovers_a_tiny_program_and_matches_graph_execution(self):
        pairs = [((1, 2), (1, 2)), ((5, 7), (5, 7))]
        body, effort = mechanical(self.base, pairs, 1, 2, DEFAULT_BINDINGS, seconds=10)
        self.assertIsNotNone(body, effort)
        lib, _ = self.base.attach("R01", body, 1)
        self.assertTrue(score(lib, "R01", pairs, DEFAULT_BINDINGS)["exact"])

    def test_mechanical_reuse_search_finds_correct_binding_without_target(self):
        first, _ = self.base.attach("R01", witness("right_parity"), 1)
        body, effort = mechanical(first, pairs_for("right_parity", 2, "train"), 1, 12,
                                  DEFAULT_BINDINGS, seconds=10)
        self.assertEqual(body[0]["op"], "invoke")
        self.assertEqual(body[0]["bindings"], list(TRANSFER_BINDINGS))
        self.assertEqual(effort["smt_candidates"], 0)

    def test_symbolic_stack_matches_recursive_graph_semantics(self):
        body = witness("right_parity")
        synth = Synthesizer(self.base, 1, len(body), DEFAULT_BINDINGS, time.monotonic() + 15)
        for i, n in enumerate(body):
            synth.solver.add(synth.op[i] == synth.OPS.index(n["op"]))
            arg = synth.names.index(n["cell"]) if n["op"] == "call" else n["arg"] or 0
            synth.solver.add(synth.arg[i] == arg,
                             synth.a[i] == (n["a"] or 0),
                             synth.b[i] == (n["b"] or 0))
        source = [9, 9, 8, 8, 1, 2, 3, 4, 5, 6]
        synth.add_example(source, [1, 2, 4, 3, 6, 5])
        actual, status = synth.candidate()
        self.assertEqual(actual, body, status)

    def test_full_freeze_replay_and_continuation_audit(self):
        # Fixture proposers test the harness, never count as a search success.
        def fixture(library, pairs, bits, maximum, folder, seconds):
            from recursive_benchmark import save
            request = dict(library=library.payload(), pairs=pairs, bits=bits, maximum=maximum,
                           bindings=DEFAULT_BINDINGS, seconds=seconds)
            save(folder / "mechanical-request.json", request)
            body = ([node("invoke", cell="R01", arg=0, a=1, bindings=TRANSFER_BINDINGS), node("ret")]
                    if library.cells else witness("right_parity"))
            save(folder / "mechanical-response.json", dict(nodes=body, effort={"fixture": True}))
            return body, {"fixture": True}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with patch("recursive_benchmark.mechanical_process", fixture):
                rows = run(output, arms=("mechanical",), families=("right_parity",), seconds=1)
            self.assertTrue(all(row["admitted"] and row["hidden"]["exact"] and row["stress"]["exact"] for row in rows))
            self.assertTrue(rows[1]["transfer"]["calls_retained_traversal"])
            self.assertEqual(replay(output)["verified_pushouts"], 2)
            report = audit(output, maximum_leaves=5, assignments=2)
            self.assertEqual(report["admitted_conditions"], 2)
            self.assertTrue(report["all_exact"])


if __name__ == "__main__":
    unittest.main()
