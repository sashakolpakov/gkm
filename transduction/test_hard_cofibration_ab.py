import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

from audit_frontier import _specialize_body, exact_extension
import cofibration_ab as original
import hard_cofibration_ab as hard
from frontier_search import verify_derivation
from interface_benchmark import seed_library
from interface_machine import attach_proposal
from interface_search import definition, prompt, reuse_search, score
from interface_machine import parse


class HardCofibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.setrecursionlimit(10000)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = seed_library(Path(cls.temporary.name) / "basis")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_harder_task_is_in_unchanged_search_grammar(self):
        for phase in (1, 2):
            raw, origin = hard.witness(self.base, hard.FAMILIES[0], phase)
            self.assertEqual(len(origin["edits"]), 3)
            self.assertTrue(verify_derivation(self.base, raw, origin, "W"))
            library, _ = attach_proposal(self.base, raw, "W")
            for split, count in (("train", 44), ("validation", 44), ("hidden", 40), ("stress", 13)):
                with self.subTest(phase=phase, split=split):
                    pairs = hard.pairs_for(hard.FAMILIES[0], phase, split, 3)
                    self.assertEqual(len(pairs), count)
                    self.assertTrue(score(library, "W", pairs)["exact"])

    def test_private_control_reuses_immutable_helper(self):
        raw, _ = hard.witness(self.base, hard.FAMILIES[0], 1, "T1")
        first, _ = attach_proposal(self.base, raw, "T1")
        raw, effort = reuse_search(first, hard.pairs_for(hard.FAMILIES[0], 2, "train", 3), "T2", 10)
        self.assertIsNotNone(raw, effort)
        second, _ = attach_proposal(first, raw, "T2")
        self.assertEqual(first.cells["T1H"], second.cells["T1H"])
        hidden = hard.pairs_for(hard.FAMILIES[0], 2, "hidden", 3)
        self.assertTrue(score(second, "T2", hidden)["exact"])
        self.assertEqual(score(second, "T2", hidden, disabled=("T1H",))["correct"], 0)

    def test_oracle_depends_on_both_child_counts_and_has_full_frames(self):
        tree = hard.previous.materialize(((None, None), None), random.Random(7), 0)
        expected = (tuple(reversed(tree["pre"]))
            + hard.reference(tree["left"], hard.FAMILIES[0], 1)[0]
            + tree["mid"][:1] + tree["right"]["pair"] + tree["post"][:1])
        output, leaves = hard.reference(tree, hard.FAMILIES[0], 1)
        self.assertEqual((output, leaves), (expected, 3))
        self.assertEqual(hard.encode(tree, hard.FAMILIES[0]), hard.previous.encode(tree, "framed"))

    def test_model_gets_no_task_label_solution_or_mechanical_hint(self):
        request = prompt(self.base, hard.pairs_for(hard.FAMILIES[0], 1, "train", 3), "T1")
        self.assertIsNone(request["reuse_probe"])
        for key in ("family", "reference", "witness", "edits", "signature"):
            self.assertNotIn(key, request)
        pools = [{v for source, _ in hard.pairs_for(hard.FAMILIES[0], 1, split, 3) for v in source}
                 for split in ("train", "validation", "hidden", "stress")]
        self.assertTrue(all(not a & b for i, a in enumerate(pools) for b in pools[i + 1:]))

    def test_same_engine_separate_budget_and_restore(self):
        frontier, sources = original.frontier, original.SOURCES
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "run"
            plan = hard.prepare(output, seeds=(3,))
            self.assertEqual((plan["seconds"], plan["replies"], plan["candidate_limit"]), (1200, 10, 32000000))
            self.assertIs(original.frontier, frontier)
            self.assertEqual(original.SOURCES, sources)
            with hard.configured_engine() as experiment:
                loaded, _ = experiment.load_plan(output)
                self.assertEqual(json.loads(json.dumps(plan)), loaded)
                with patch.object(experiment, "model_search", return_value=(None, dict(attempts=[], stopped="time_limit"))):
                    rows = experiment.run_arm(output, "codex")
                self.assertEqual([r["admitted"] for r in rows], [False, False])
                self.assertIn("blocked", rows[1]["status"])
                with self.assertRaises(FileNotFoundError):
                    experiment.replay(output)
            self.assertIs(original.frontier, frontier)
            self.assertEqual(original.SOURCES, sources)

    def test_let_extension_preserves_effect_order_and_recursive_binding(self):
        old = definition("Old", [], "B", parse(
            "(if eq (seq (call F002) (call F002) (let left (call self) "
            "(let right (call self) (xor left right)))) (seq (call C01) true))"))
        new = definition("New", [("emit", "F")], "B", parse(
            "(if eq (seq (call F002) (call F002) (let left (call self emit) "
            "(let right (call self emit) (xor left right)))) (seq (call emit) true))"))
        library, _ = self.base.attach(old)
        library, _ = library.attach(new)
        proof = exact_extension("Old", library.cells["Old"], "New", library.cells["New"], library.signatures())
        self.assertEqual(proof["fixed_arguments"], {"emit": "(ref C01)"})
        bad = {**library.cells["New"], "body": new["body"].replace("(call self emit)", "(call self (ref C02))", 1)}
        self.assertIsNone(exact_extension("Old", library.cells["Old"], "New", bad, library.signatures()))
        capture = definition("Capture", [("arg", "B"), ("emit", "F")], "B",
                             parse("(let oldArg true arg)"))
        with self.assertRaisesRegex(ValueError, "capture"):
            _specialize_body(capture, (0,), {"emit": ("ref", "C01")}, ["oldArg"])


if __name__ == "__main__":
    unittest.main()
