import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cofibration_ab as experiment
from analyze_cofibration_ab import analyze, recursive, supported
from frontier_benchmark import witness
from interface_benchmark import save
from interface_machine import InterfaceLibrary
from interface_search import GROWTH_CONTRACT, reuse_search
from modular_machine import digest


class CofibrationABTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.setrecursionlimit(10000)

    @staticmethod
    def reuse(library, pairs, root, folder, seconds, *, phase, history=None):
        save(folder / "search-request.json", dict(library=library.payload(), pairs=pairs, root=root,
            seconds=seconds, phase=phase, growth_contract=GROWTH_CONTRACT, history=history or [], reuse=None))
        raw, effort = reuse_search(library, pairs, root, seconds)
        save(folder / "search-response.json", dict(candidate=raw, effort=effort))
        return raw, effort

    @staticmethod
    def worker(library, pairs, root, folder, seconds, history, reuse, *, candidate_limit):
        save(folder / "request.json", dict(library=library.payload(), pairs=pairs, root=root,
            seconds=seconds, history=history, reuse=reuse, growth_contract=GROWTH_CONTRACT,
            candidate_limit=candidate_limit))
        raw, origin = witness(library, "infix", 1, root)
        effort = dict(derivation=origin, candidates=1, stopped="solved")
        save(folder / "response.json", dict(candidate=raw, effort=effort))
        return raw, effort

    def test_model_authors_every_attachment_with_no_host_binder(self):
        requests = []
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "ab"
            experiment.prepare(output, seeds=[31], seconds=10, tasks=[("infix", 1), ("infix", 2)])
            base = InterfaceLibrary.read(json.loads((output / "basis/corpus.json").read_text())["library"])
            library = base
            def propose(request, folder, **kwargs):
                nonlocal library
                from interface_machine import attach_proposal
                requests.append(request)
                self.assertIsNone(request["reuse_probe"])
                self.assertEqual(request["library_hash"], library.digest)
                raw, _ = reuse_search(library, request["training_examples"], request["root"], 5)
                if raw is None:
                    raw, _ = witness(library, "infix", 1, request["root"])
                save(folder / "request.json", request)
                events = [dict(type="turn.started"),
                    dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(raw))),
                    dict(type="turn.completed", usage={})]
                (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
                library, _ = attach_proposal(library, raw, request["root"])
                return raw
            with patch("interface_benchmark.propose", propose), \
                    patch("cofibration_ab.frontier.search", side_effect=AssertionError("host search in model arm")):
                model = experiment.run_arm(output, "codex")
            self.assertEqual(len(requests), 2)
            self.assertTrue(all(r["selected_by"] == "codex" for r in model))
            self.assertFalse((output / "seed-31/codex/task-2/reuse").exists())
            with patch("frontier_benchmark.search_process", self.reuse), patch("frontier_benchmark.worker", self.worker), \
                    patch("interface_benchmark.propose", side_effect=AssertionError("model in mechanical arm")):
                mechanical = experiment.run_arm(output, "mechanical")
            self.assertEqual([r["selected_by"] for r in mechanical], ["mechanical", "host_reuse"])
            report = experiment.replay(output)
            self.assertEqual(report["verified_pushouts"], 6)
            self.assertTrue(all(r["admitted"] and r["audit"]["exact"] for r in report["rows"]))
            self.assertTrue(all(r["verified_reuse"] for r in report["rows"] if r["task"] == 2))
            measured = analyze(output)
            self.assertEqual(measured["arms"]["codex"]["fully_verified"], 2)
            self.assertEqual(measured["certificates"]["promoted"], 6)
            self.assertTrue(all(r["necessary_recursive_reuse"] for r in measured["rows"] if r["task"] == 2))
            (output / "seed-31/codex/task-2/reuse").mkdir()
            with self.assertRaisesRegex(ValueError, "mechanical glue"):
                experiment.replay(output)

    def test_failure_blocks_prefix_and_never_becomes_zero_complexity(self):
        with tempfile.TemporaryDirectory() as td, patch("cofibration_ab.model_search",
                return_value=(None, dict(attempts=[], stopped="time_limit"))) as model:
            output = Path(td) / "ab"
            experiment.prepare(output, seeds=[31, 32], tasks=[("infix", 1), ("infix", 2)])
            rows = experiment.run_arm(output, "codex")
            self.assertEqual(model.call_count, 2)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(not r["admitted"] and r["marginal_nodes"] is None for r in rows))
            self.assertTrue(all("blocked" in r["status"] for r in rows if r["task"] == 2))
            with self.assertRaises(FileNotFoundError):
                experiment.replay(output)  # Other arm has not frozen; no private evaluation.
            with self.assertRaisesRegex(ValueError, "completed arm"):
                experiment.run_arm(output, "codex")

    def test_manifest_cannot_enable_model_arm_host_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "ab"
            experiment.prepare(output, seeds=[31])
            m = json.loads((output / "plan.json").read_text())
            m["plan"]["model_arm_host_reuse"] = True
            m["hash"] = digest(m["plan"])
            save(output / "plan.json", m)
            with self.assertRaisesRegex(ValueError, "separation changed"):
                experiment.load_plan(output)

    def test_post_run_diagnostic_requires_all_checks_and_actual_recursion(self):
        self.assertFalse(supported(dict(admitted=False)))
        self.assertFalse(supported(dict(admitted=True, hidden=dict(exact=True))))
        self.assertTrue(supported(dict(admitted=True, **{k: dict(exact=True)
                                                       for k in ("hidden", "stress", "audit")})))
        self.assertTrue(recursive("(seq (call F002) (call self f))", "H"))
        self.assertTrue(recursive("(call H f)", "H"))
        self.assertFalse(recursive("(call OlderHelper f)", "H"))


if __name__ == "__main__":
    unittest.main()
