import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from audit_composition import audit
from composition_benchmark import (BOUNDS, DEFAULT_BASIS, FAMILIES, make_task, maximum_lengths,
                                   pairs_for, reference, replay, run)
from composition_search import (CELLS, FastEvaluator, attach_term, decode, encode, enumerate_terms,
                                graph_score, leaves, load_basis, mechanical, normalize, prompt,
                                schema, syntax_counts)
from cofibration import verify_certificate
from modular_machine import Library, Machine
from growth_benchmark import run as grow_basis


class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tests must also work in a fresh checkout without ignored run artifacts.
        cls.temporary = tempfile.TemporaryDirectory()
        cls.basis_folder = Path(cls.temporary.name) / "basis"
        grow_basis(cls.basis_folder, seeds=(1,), arms=("mechanical",), stages=6)
        cls.library = load_basis(cls.basis_folder)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_basis_excludes_previous_composition_solutions(self):
        self.assertEqual(list(self.library.cells)[-6:], list(CELLS))
        self.assertNotIn("M07", self.library.cells)
        self.assertEqual(len(self.library.cells), 11)

    def test_combinatorial_counts_are_explicit(self):
        self.assertEqual(syntax_counts(6), [6, 72, 1728, 51840, 1741824, 62705664])

    def test_enumeration_is_increasing_size_and_has_no_duplicates(self):
        terms = list(enumerate_terms(3))
        self.assertEqual(len(terms), len(set(terms)))
        self.assertEqual([leaves(t) for t in terms], sorted(leaves(t) for t in terms))
        self.assertTrue(all(normalize(t) == t for t in terms))
        self.assertTrue(all(decode(encode(t, self.library.digest), self.library.digest, 3) == t for t in terms))

    def test_normalization_preserves_reference_and_intermediate_order(self):
        a, b, c = (("call", cell) for cell in ("M02", "M03", "M06"))
        terms = [("pipe", ("pipe", a, b), c), ("if", ("if", a, b), ("if", c, a)), ("if", a, a)]
        for term in terms:
            for length in range(5):
                for source in itertools.product((0, 1), repeat=length):
                    self.assertEqual(reference(term, source), reference(normalize(term), source))

    def test_normalization_covers_all_small_raw_trees(self):
        canonical = set(enumerate_terms(3))
        atoms = [("call", cell) for cell in CELLS]
        pairs = [(op, a, b) for op in ("pipe", "if") for a in atoms for b in atoms]
        for op in ("pipe", "if"):
            for a in atoms:
                for b in pairs:
                    self.assertIn(normalize((op, a, b)), canonical)
                    self.assertIn(normalize((op, b, a)), canonical)

    def test_schema_and_parser_allow_general_nested_trees(self):
        term = ("pipe", ("if", ("call", "M03"), ("call", "M06")),
                ("pipe", ("call", "M02"), ("call", "M04")))
        encoded = encode(term, self.library.digest)
        self.assertEqual(decode(encoded, self.library.digest, 4), normalize(term))
        self.assertEqual(schema(self.library.digest, 4)["properties"]["nodes"]["maxItems"], 7)
        for mutate in (lambda p: p.update(library_hash="bad"),
                       lambda p: p.update(root=True),
                       lambda p: p["nodes"][0].update(cell="M07"),
                       lambda p: p["nodes"][-1].update(left=len(p["nodes"]) - 1),
                       lambda p: p["nodes"][-1].update(right=p["nodes"][-1]["left"])):
            bad = copy.deepcopy(encoded)
            mutate(bad)
            with self.assertRaises(ValueError):
                decode(bad, self.library.digest, 4)
        with self.assertRaises(ValueError):
            decode(encoded, self.library.digest, 3)

    def test_lifting_executes_nested_operand_via_unchanged_return_interfaces(self):
        term = ("pipe", ("if", ("call", "M03"), ("call", "M06")),
                ("pipe", ("call", "M02"), ("call", "M04")))
        built, certificates = attach_term(self.library, term)
        self.assertGreater(len(certificates), 1)
        previous = self.library.graph
        for record in certificates:
            previous = verify_certificate(record["certificate"], previous).graph
        self.assertEqual(previous, built.graph)
        self.assertEqual(built.graph.rules[:len(self.library.graph.rules)], self.library.graph.rules)
        self.assertEqual({name: built.cells[name] for name in self.library.cells}, self.library.cells)
        result = Machine(built).run("Q", [1, 1, 2, 3, 3, 4], trace_limit=128)
        self.assertEqual(result["output"], list(reference(term, [1, 1, 2, 3, 3, 4])))
        self.assertEqual(result["calls"], result["returns"])
        self.assertTrue(any(name.startswith("QH") for name in result["visits"]))
        Library.read(json.loads(json.dumps(built.payload())))

    def test_fast_evaluator_agrees_with_executable_quotients(self):
        evaluator = FastEvaluator(self.library, cache_limit=50)
        terms = list(enumerate_terms(3))
        # Broad deterministic sample of both constructors and split orders.
        for term in terms[::17]:
            built, _ = attach_term(self.library, term)
            for source in ((), (0,), (0, 1), (0, 0, 1, 2, 2), (0, 1, 2, 0, 0, 1, 2)):
                result = Machine(built).run("Q", source)
                self.assertTrue(result["ok"])
                self.assertEqual(tuple(result["output"]), evaluator.run(term, source))
                self.assertEqual(evaluator.run(term, source), reference(term, source))
        self.assertLessEqual(len(evaluator.cache), 50)

    def test_original_input_fingerprint_is_not_used_as_pipeline_congruence(self):
        # Copy and pair-swap agree on constant original inputs, but the cache
        # still distinguishes their behavior on every different intermediate.
        evaluator = FastEvaluator(self.library)
        copy_term, swap = ("call", "M01"), ("call", "M03")
        self.assertEqual(evaluator.run(copy_term, (1, 1)), evaluator.run(swap, (1, 1)))
        self.assertNotEqual(evaluator.run(copy_term, (1, 2)), evaluator.run(swap, (1, 2)))

    def test_generated_tasks_are_reproducible_and_within_supplied_basis(self):
        for bound in BOUNDS:
            for seed, family in FAMILIES:
                task = make_task(seed, family, bound)
                self.assertEqual(task, make_task(seed, family, bound))
                self.assertEqual(leaves(task["target"]), bound)
                self.assertLessEqual(maximum_lengths(task["target"], 512)[1], 4096)
                self.assertTrue(FastEvaluator(self.library).score(task["target"], pairs_for(task, "train"))["exact"])

    def test_model_gets_examples_not_target_or_family(self):
        task = make_task(7, "pipeline", 4)
        request = prompt(self.library, pairs_for(task, "train"), 4)
        self.assertEqual(set(request), {"instructions", "max_call_leaves", "library_hash", "library", "training_examples", "previous_training_feedback"})
        self.assertNotIn("target", request)
        self.assertNotIn("family", request)
        self.assertNotIn("M07", request["library"])

    def test_search_bounds_and_accepted_result(self):
        task = make_task(7, "pipeline", 2)
        term, effort = mechanical(self.library, pairs_for(task, "train"), 2, budget=1000)
        self.assertIsNotNone(term)
        self.assertEqual(effort["stop_reason"], "training_exact")
        built, _ = attach_term(self.library, term)
        self.assertTrue(graph_score(built, "Q", pairs_for(task, "validation"))["exact"])
        term, effort = mechanical(self.library, pairs_for(task, "train"), 2, budget=1)
        self.assertIsNone(term)
        self.assertEqual(effort["evaluated"], 1)
        self.assertEqual(effort["stop_reason"], "candidate_limit")

    def test_tiny_frozen_experiment_replays_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "experiment"
            rows = run(folder, self.basis_folder, bounds=(2,), families=((7, "pipeline"),), arms=("mechanical",), budget=1000)
            self.assertTrue(rows[0]["admitted"])
            self.assertTrue(rows[0]["hidden"]["exact"])
            self.assertTrue(rows[0]["stress"]["exact"])
            result = replay(folder, reproduce_search=True)
            self.assertEqual(result["admitted_conditions_replayed"], 1)
            exhaustive = audit(folder, maximum=4)
            self.assertEqual(exhaustive["patterns_per_task"], 24)
            self.assertTrue(exhaustive["rows"][0]["exact"])

    def test_budget_failure_does_not_mean_missing_primitives(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "experiment"
            rows = run(folder, self.basis_folder, bounds=(2,), families=((7, "pipeline"),), arms=("mechanical",), budget=1)
            self.assertFalse(rows[0]["admitted"])
            self.assertEqual(rows[0]["hidden"]["correct"], 0)
            self.assertEqual(rows[0]["hidden"]["total"], 48)
            witness = json.loads((folder / "pipeline-s7-l2" / "private-witness-check.json").read_text())
            self.assertTrue(witness["all_exact"])
            self.assertEqual(replay(folder, reproduce_search=True)["nonadmitted_conditions"], 1)
            self.assertFalse(audit(folder, maximum=2)["rows"][0]["audited"])


if __name__ == "__main__":
    unittest.main()
