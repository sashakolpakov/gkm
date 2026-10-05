import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audit_frontier import audit, exact_extension
from frontier_benchmark import (FAMILIES, pairs_for, replay, replay_batch, run,
                                run_batch, sources, witness)
from frontier_search import (insert_effects, insertion_sites, mechanical,
                             verify_derivation)
from interface_benchmark import model_search, save, seed_library
from interface_machine import Machine, attach_proposal, parse, typecheck
from interface_search import (GROWTH_CONTRACT, definition, discover, lifted_skeleton,
                              prompt, reuse_search, score)


class FrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.setrecursionlimit(10000)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = seed_library(Path(cls.temporary.name) / "basis")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_every_task_has_a_solution_in_expanded_mechanical_grammar(self):
        for family in FAMILIES:
            for stage in (1, 2):
                with self.subTest(family=family, stage=stage):
                    raw, origin = witness(self.base, family, stage)
                    self.assertTrue(verify_derivation(self.base, raw, origin, "W"))
                    built, _ = attach_proposal(self.base, raw, "W")
                    for split in ("train", "validation", "hidden", "stress"):
                        self.assertTrue(score(built, "W", pairs_for(family, stage, split))["exact"], split)

    def test_edited_prototypes_preserve_types_and_original_results(self):
        abstract = discover(self.base)[0]
        for spec in ((0, 2, "U"), (0, 3, "B"), (1, 2, "B")):
            proto = lifted_skeleton(abstract, *spec)
            for site in insertion_sites(proto, self.base.signatures()):
                changed = insert_effects(proto, [site])
                env = dict(changed["params"])
                env.update({"HOLE" + str(i): "B" for i in range(len(changed["domains"]))})
                sigs = {**self.base.signatures(), "H": ([t for _, t in changed["params"]], changed["returns"])}
                self.assertEqual(typecheck(changed["body"], env, sigs, "H"), changed["returns"])

    def test_private_witnesses_are_not_supplied_to_model(self):
        request = prompt(self.base, pairs_for("framed", 1, "train"), "T1")
        self.assertEqual(request["growth_contract"], GROWTH_CONTRACT)
        for key in ("family", "signature", "edits", "reference", "witness"):
            self.assertNotIn(key, request)
        public = {n for p, _ in pairs_for("framed", 1, "train") for n in p}
        hidden = {n for p, _ in pairs_for("framed", 1, "hidden") for n in p}
        self.assertTrue(public.isdisjoint(hidden))

    def test_each_coverage_helper_transfers_without_a_body_change(self):
        for family in FAMILIES:
            raw, _ = witness(self.base, family, 1, "T1")
            first, _ = attach_proposal(self.base, raw, "T1")
            second_raw, effort = reuse_search(first, pairs_for(family, 2, "train"), "T2", 5)
            self.assertIsNotNone(second_raw, (family, effort))
            second, _ = attach_proposal(first, second_raw, "T2")
            self.assertEqual(first.cells["T1H"], second.cells["T1H"])
            hidden = pairs_for(family, 2, "hidden")
            self.assertTrue(score(second, "T2", hidden)["exact"])
            self.assertEqual(score(second, "T2", hidden, disabled=("T1H",))["correct"], 0)

    def test_mechanical_control_still_succeeds(self):
        raw, effort = mechanical(self.base, pairs_for("control", 1, "train"), "T1", seconds=5)
        self.assertIsNotNone(raw, effort)
        self.assertTrue(verify_derivation(self.base, raw, effort["derivation"], "T1"))
        self.assertEqual(effort["derivation"]["edits"], [])

    def test_changed_edit_or_filling_fails_derivation(self):
        raw, origin = witness(self.base, "framed", 1)
        tampered = copy.deepcopy(origin)
        tampered["edits"][0]["path"] = [999]
        with self.assertRaises(ValueError):
            verify_derivation(self.base, raw, tampered, "W")
        tampered = copy.deepcopy(origin)
        tampered["filling"][0] = "eq"
        with self.assertRaises(ValueError):
            verify_derivation(self.base, raw, tampered, "W")

    def test_no_admitted_programs_is_not_a_successful_behavioral_audit(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            save(output / "protocol.json", dict(source_hashes=sources()))
            save(output / "frozen.json", dict(rows=[dict(admitted=False)]))
            result = audit(output, maximum_leaves=2, assignments=1)
            self.assertEqual(result["admitted_roots"], 0)
            self.assertIsNone(result["all_exact"])

    def test_exact_interface_extension_preserves_recursive_specialization(self):
        old = definition("Old", [("mode", "B")], "U", parse(
            "(if eq (seq (call F002) (call F002) (call self true) (call self false)) "
            "(if mode (call F004) (call C01)))"))
        new = definition("New", [("s", "B"), ("a", "F"), ("b", "F")], "U", parse(
            "(if eq (seq (call F002) (call F002) (call self true a b) (call self false a b)) "
            "(if s (call a) (call b)))"))
        library, _ = self.base.attach(old)
        library, _ = library.attach(new)
        proof = exact_extension("Old", library.cells["Old"], "New", library.cells["New"], library.signatures())
        self.assertEqual(proof["fixed_arguments"], {"a": "(ref F004)", "b": "(ref C01)"})
        self.assertEqual(proof["retained_arguments"], {"s": "mode"})
        for replacement in ("(call self true b a)", "(call self true (seq (call F002) a) b)"):
            bad = {**library.cells["New"], "body": new["body"].replace("(call self true a b)", replacement)}
            self.assertIsNone(exact_extension("Old", library.cells["Old"], "New", bad, library.signatures()))

    @staticmethod
    def direct_reuse(library, pairs, root, folder, seconds, *, phase, history=None):
        assert phase == "reuse"
        save(folder / "search-request.json", dict(library=library.payload(), pairs=pairs, root=root,
            seconds=seconds, phase=phase, growth_contract=GROWTH_CONTRACT, history=history or [], reuse=None))
        raw, effort = reuse_search(library, pairs, root, seconds)
        save(folder / "search-response.json", dict(candidate=raw, effort=effort))
        return raw, effort

    def test_full_mechanical_freeze_and_replay(self):
        def direct(library, pairs, root, folder, seconds, history, reuse, *, candidate_limit):
            save(folder / "request.json", dict(library=library.payload(), pairs=pairs, root=root,
                seconds=seconds, growth_contract=GROWTH_CONTRACT, history=history, reuse=reuse,
                candidate_limit=candidate_limit))
            raw, effort = mechanical(library, pairs, root, seconds, candidate_limit=candidate_limit)
            save(folder / "response.json", dict(candidate=raw, effort=effort))
            return raw, effort
        with tempfile.TemporaryDirectory() as td, patch("frontier_benchmark.worker", direct), \
                patch("frontier_benchmark.search_process", self.direct_reuse):
            output = Path(td) / "run"
            rows = run(output, ("control",), ("mechanical",), seconds=5)
            self.assertTrue(rows[1]["genuine_transfer"])
            self.assertEqual(replay(output)["verified_pushouts"], 3)

    def test_more_replies_enable_a_third_attempt_without_changing_feedback(self):
        bad = dict(library_hash=self.base.digest, definitions=[definition("T1", [], "U", "unit")])
        good, _ = witness(self.base, "framed", 1, "T1")
        pairs = pairs_for("framed", 1, "train")
        for limit, expected in ((2, None), (6, good)):
            requests = []
            def answer(request, *args, **kwargs):
                requests.append(request)
                return good if len(requests) == 3 else bad
            with tempfile.TemporaryDirectory() as td, patch("interface_benchmark.propose", answer):
                raw, effort = model_search(self.base, pairs, "T1", Path(td), 5,
                                           max_replies=limit, log_progress=True)
                progress = json.loads((Path(td) / "progress.json").read_text())
            self.assertEqual(raw, expected)
            self.assertEqual(len(requests), min(limit, 3))
            self.assertEqual(progress["attempts"], effort["attempts"])
            self.assertEqual(effort["stopped"], "reply_limit" if limit == 2 else "solved")
            for request in requests[1:]:
                self.assertEqual(request["feedback"]["proposal"], bad)
                self.assertEqual(len(request["feedback"]["counterexamples"]), 2)
                self.assertTrue(all(c["source"] in [list(p[0]) for p in pairs]
                                    for c in request["feedback"]["counterexamples"]))

    def test_fixed_trials_reset_libraries_and_replay_every_outcome(self):
        bad = dict(library_hash=self.base.digest, definitions=[definition("T1", [], "U", "unit")])
        def answer(request, folder, **kwargs):
            self.assertEqual(request["library_hash"], self.base.digest)
            self.assertEqual(request["prior_tasks"], [])
            save(folder / "request.json", request)
            events = [dict(type="turn.started"),
                      dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(bad))),
                      dict(type="turn.completed", usage={})]
            (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            return bad
        with tempfile.TemporaryDirectory() as td, patch("frontier_benchmark.search_process", self.direct_reuse), \
                patch("interface_benchmark.propose", side_effect=answer) as model:
            output = Path(td) / "batch"
            records = run_batch(output, [2, 3], 2, families=["control"], arms=["codex"],
                                seconds=5, replies=3, candidate_limit=17)
            self.assertEqual(model.call_count, 12)
            self.assertEqual(len(records), 4)
            self.assertTrue(all(not r["admitted"] for t in records for r in t["rows"]))
            self.assertTrue(replay_batch(output)["all_trials_replayed"])
            summary = json.loads((output / "batch-summary.json").read_text())
            summary["trials"].pop()
            save(output / "batch-summary.json", summary)
            with self.assertRaisesRegex(ValueError, "trial omitted"):
                replay_batch(output)

    def test_model_receipts_and_host_transfer_replay(self):
        raw, _ = witness(self.base, "infix", 1, "T1")
        def answer(request, folder, **kwargs):
            save(folder / "request.json", request)
            events = [dict(type="turn.started"),
                      dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(raw))),
                      dict(type="turn.completed", usage={})]
            (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            return raw
        with tempfile.TemporaryDirectory() as td, patch("frontier_benchmark.search_process", self.direct_reuse), \
                patch("interface_benchmark.propose", side_effect=answer) as model:
            output = Path(td) / "run"
            rows = run(output, ("infix",), ("codex",), seconds=5)
            self.assertEqual(model.call_count, 1)
            self.assertEqual(rows[1]["selected_by"], "host_reuse")
            self.assertTrue(rows[1]["genuine_transfer"])
            self.assertEqual(replay(output)["verified_pushouts"], 3)


if __name__ == "__main__":
    unittest.main()
