import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cofibration import verify_certificate
from interface_benchmark import (FAMILIES, model_search, pairs_for, replay, run,
                                 save, search_stage, seed_library, witness)
from interface_machine import InterfaceLibrary, Machine, attach_proposal, parse
from interface_search import (GROWTH_CONTRACT, abstract_pair, bool_terms, definition, discover,
                              lifted_skeleton, mechanical, prompt, reuse_search, score,
                              substitute, test_ast as ast_score)


class InterfaceGrowthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.setrecursionlimit(10000)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = seed_library(Path(cls.temporary.name) / "basis")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_unknown_interfaces_start_without_callbacks_or_context(self):
        self.assertTrue(all(c["params"] == [] and c["returns"] == "U" for c in self.base.cells.values()))
        self.assertEqual(self.base.base.cells["F004"]["program"]["actions"], [10, 0, 1, 0, 30])

    def test_arbitrary_signatures_are_chosen_not_two_predefined_slots(self):
        for params, returns, body in (([], "U", "unit"), ([("x", "B")], "B", "x"),
                                     ([("f", "F")], "U", "(call f)"),
                                     ([("f", "F"), ("b", "B"), ("g", "F")], "U", "(if b (call f) (call g))")):
            self.base.attach(definition("H", params, returns, parse(body)))

    def test_expressibility_controls_are_correct_but_not_search_results(self):
        for family in FAMILIES:
            for stage in (1, 2):
                library, _ = attach_proposal(self.base, witness(self.base, family, stage, "W"), "W")
                for split in ("train", "validation", "hidden", "stress"):
                    result = score(library, "W", pairs_for(family, stage, split))
                    self.assertTrue(result["exact"], (family, stage, split))

    def test_returned_left_summary_drives_right_and_caller_scope_is_restored(self):
        library, _ = attach_proposal(self.base, witness(self.base, "returned", 1, "W"), "W")
        # Root(left internal(two leaves), right leaf): the right subtree starts
        # with false, because two leaves were consumed, not odd branch depth.
        result = Machine(library).run("W", [9, 9, 8, 8, 1, 2, 3, 4, 5, 6], trace_limit=100)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["output"], [1, 2, 4, 3, 5, 6])
        returns = [t[3] for t in result["trace"] if t[0] == "return" and t[1] == "WH"]
        self.assertIn(False, returns)
        self.assertIn(True, returns)

    def test_old_cells_survive_actual_pushouts(self):
        built, certificates = attach_proposal(self.base, witness(self.base, "returned", 1, "W"), "W")
        graph = self.base.graph
        for cert in certificates:
            graph = verify_certificate(json.loads(json.dumps(cert)), graph).graph
        self.assertEqual(graph, built.graph)
        self.assertEqual(self.base.graph.rules, built.graph.rules[:len(self.base.graph.rules)])
        self.assertEqual(self.base.cells, {n: built.cells[n] for n in self.base.cells})
        self.assertEqual(InterfaceLibrary.read(json.loads(json.dumps(built.payload(), sort_keys=True))).digest, built.digest)

    def test_tampered_body_interface_and_graph_are_rejected(self):
        built, _ = attach_proposal(self.base, witness(self.base, "returned", 1, "W"), "W")
        for mutate in (lambda p: p["cells"]["WH"].update(returns="U"),
                       lambda p: p["cells"]["W"].update(body="unit"),
                       lambda p: p["graph"]["rules"].pop()):
            value = copy.deepcopy(json.loads(json.dumps(built.payload())))
            mutate(value)
            with self.assertRaises(ValueError):
                InterfaceLibrary.read(value)

    def test_type_scope_and_token_literal_rejections(self):
        for body in ("(call F004 true)", "(if true unit false)", "(let x true (let x false unit))",
                     "(call self)", "(call missing)", "17", "__import__('os')"):
            if body == "(call self)":
                library, _ = self.base.attach(definition("H", [], "U", parse(body)))
                self.assertIn("progress", Machine(library).run("H", [1, 2])["error"])
            else:
                with self.assertRaises(ValueError):
                    self.base.attach(dict(name="H", params=[], returns="U", body=body))

    def test_discovery_has_exact_specializations_not_only_example_fit(self):
        found = discover(self.base)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["params"], ["f0"])
        for evidence in found[0]["specializations"]:
            self.assertTrue(evidence["exact_syntax"])
            self.assertIn(evidence["recovered_body"], [c["body"] for c in self.base.cells.values()])

    def test_mechanical_lifting_can_express_new_return_and_inherited_interfaces(self):
        abstract = discover(self.base)[0]
        proto = lifted_skeleton(abstract, 1, 2, "B")
        # Fill by role/available variables to check search coverage, not to
        # supply this assignment to a benchmark proposer.
        choices = []
        for hole in proto["holes"]:
            env = hole["available"]
            if hole["role"] == "recursive_argument":
                desired = "b0" if len(env) == 1 else ("xor", "b0", env[-1])
            elif hole["role"] == "procedure_selection":
                desired = "b0"
            else:
                desired = ("xor", env[-2], env[-1]) if len(env) > 2 else "true"
            self.assertIn(desired, proto["domains"][hole["index"]])
            choices.append(desired)
        body = substitute(proto["body"], choices)
        raw = dict(library_hash=self.base.digest, definitions=[
            definition("H", proto["params"], "B", body),
            definition("T", [], "U", parse("(seq (call H false (ref C01) (ref F004)) unit)"))])
        built, _ = attach_proposal(self.base, raw, "T")
        self.assertTrue(score(built, "T", pairs_for("returned", 1, "hidden"))["exact"])

    def test_actual_mechanical_acquisition_and_later_transfer(self):
        raw, effort = mechanical(self.base, pairs_for("local", 1, "train"), "T1", seconds=5)
        self.assertIsNotNone(raw, effort)
        first, _ = attach_proposal(self.base, raw, "T1")
        raw2, effort2 = reuse_search(first, pairs_for("local", 2, "train"), "T2", seconds=5)
        self.assertIsNotNone(raw2, effort2)
        second, _ = attach_proposal(first, raw2, "T2")
        self.assertEqual(first.cells["T1H"], second.cells["T1H"])
        self.assertTrue(score(second, "T2", pairs_for("local", 2, "hidden"))["exact"])
        self.assertEqual(score(second, "T2", pairs_for("local", 2, "hidden"), disabled=("T1H",))["correct"], 0)

    def test_graph_runtime_matches_ast_search(self):
        from interface_search import runtime
        machine = runtime(self.base)
        raw = witness(self.base, "returned", 1, "W")
        self.assertTrue(ast_score(machine, raw["definitions"], "W", pairs_for("returned", 1, "train"))["exact"])

    def test_prompt_does_not_leak_reference_or_private_family(self):
        request = prompt(self.base, pairs_for("returned", 1, "train"), "T1")
        text = json.dumps(request)
        for forbidden in ("reference(tree", "left_summary", "effects_for", "returned\"", "interface_hint(family)"):
            self.assertNotIn(forbidden, text)
        self.assertNotIn("interface_hint", request)
        self.assertEqual(request["growth_contract"], GROWTH_CONTRACT)
        self.assertEqual(request["prior_tasks"], [])
        self.assertTrue({x for s, _ in pairs_for("local", 1, "train") for x in s}.isdisjoint(
            {x for s, _ in pairs_for("local", 1, "hidden") for x in s}))

    def test_full_mechanical_freeze_and_fresh_replay(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("interface_benchmark.search_process", self.direct_search):
                rows = run(Path(td) / "run", arms=("mechanical",), families=("local",), seconds=5)
            self.assertTrue(all(r["admitted"] for r in rows))
            self.assertTrue(rows[1]["genuine_transfer"])
            self.assertEqual(rows[1]["selected_by"], "host_reuse")
            self.assertEqual(replay(Path(td) / "run")["verified_pushouts"], 3)

    @staticmethod
    def direct_search(library, pairs, root, folder, seconds, *, phase, history=None, reuse=None):
        save(folder / "search-request.json", dict(library=library.payload(), pairs=pairs, root=root,
             seconds=seconds, phase=phase, growth_contract=GROWTH_CONTRACT, history=history or [], reuse=reuse))
        candidate, effort = (reuse_search if phase == "reuse" else mechanical)(library, pairs, root, seconds)
        save(folder / "search-response.json", dict(candidate=candidate, effort=effort))
        return candidate, effort

    def test_shared_reuse_precedes_and_can_skip_either_proposer(self):
        library, _ = attach_proposal(self.base, witness(self.base, "local", 1, "T1"), "T1")
        for arm in ("mechanical", "codex"):
            with tempfile.TemporaryDirectory() as td, patch("interface_benchmark.search_process", self.direct_search), \
                    patch("interface_benchmark.model_search", side_effect=AssertionError("model must not run")):
                raw, effort, selected_by = search_stage(library, pairs_for("local", 2, "train"),
                                                       "T2", Path(td), 5, arm, [])
                self.assertEqual(selected_by, "host_reuse")
                self.assertEqual(effort["proposer"], {})
                self.assertEqual([d["name"] for d in raw["definitions"]], ["T2"])

    def test_thin_root_contract_does_not_prescribe_helper_signature_or_force_transfer(self):
        specialized = dict(name="H", params=[], returns="U", body=self.base.cells["S0"]["body"])
        raw = dict(library_hash=self.base.digest, definitions=[specialized,
                   definition("T", [], "U", parse("(call H)"))])
        attach_proposal(self.base, raw, "T")  # Specialized helper allowed, not called reusable yet.
        for body in (self.base.cells["S0"]["body"], "(call self)", "(let f (ref T) (call f))"):
            raw = dict(library_hash=self.base.digest, definitions=[definition("T", [], "U", parse(body))])
            with self.assertRaisesRegex(ValueError, "task root must compose"):
                attach_proposal(self.base, raw, "T")

    def test_self_procedure_reference_compiles_without_external_dependency(self):
        library, _ = self.base.attach(definition("H", [], "U", parse("(let f (ref H) (call f))")))
        self.assertEqual(library.cells["H"]["dependencies"], {})
        library.verify()
        self.assertIn("progress", Machine(library).run("H", [1, 2])["error"])

    def test_model_gets_training_feedback_not_private_validation(self):
        bad = dict(library_hash=self.base.digest, definitions=[definition("T1", [], "U", "unit")])
        good = witness(self.base, "local", 1, "T1")
        requests = []
        def answer(request, *args, **kwargs):
            requests.append(request)
            return bad if len(requests) == 1 else good
        with tempfile.TemporaryDirectory() as td, patch("interface_benchmark.propose", answer):
            raw, effort = model_search(self.base, pairs_for("local", 1, "train"), "T1", Path(td), 5)
        self.assertEqual(raw, good)
        self.assertEqual(len(effort["attempts"]), 2)
        self.assertEqual(requests[1]["feedback"]["proposal"], bad)
        self.assertEqual(len(requests[1]["feedback"]["counterexamples"]), 2)
        visible_sources = [list(p[0]) for p in pairs_for("local", 1, "train")]
        self.assertTrue(all(c["source"] in visible_sources for c in requests[1]["feedback"]["counterexamples"]))

    def test_codex_acquisition_and_host_transfer_replay_keep_distinct_provenance(self):
        raw = witness(self.base, "local", 1, "T1")
        def answer(request, folder, **kwargs):
            save(folder / "request.json", request)
            events = [dict(type="turn.started"),
                      dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(raw))),
                      dict(type="turn.completed", usage={})]
            (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            return raw
        with tempfile.TemporaryDirectory() as td, patch("interface_benchmark.search_process", self.direct_search), \
                patch("interface_benchmark.propose", side_effect=answer) as model:
            rows = run(Path(td) / "run", arms=("codex",), families=("local",), seconds=5)
            self.assertEqual(model.call_count, 1)
            self.assertEqual([r["selected_by"] for r in rows], ["codex", "host_reuse"])
            self.assertTrue(rows[1]["genuine_transfer"])
            self.assertEqual(replay(Path(td) / "run")["verified_pushouts"], 3)


if __name__ == "__main__":
    unittest.main()
