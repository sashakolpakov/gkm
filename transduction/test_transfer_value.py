import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import transfer_value as study
from frontier_benchmark import materialize, reference as old_reference, witness
from interface_benchmark import save, seed_library
from interface_machine import attach_proposal, parse, render
from interface_search import GROWTH_CONTRACT, brief, prompt, reuse_search, score
from modular_machine import digest


class TransferValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.setrecursionlimit(10000)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = seed_library(Path(cls.temporary.name) / "basis")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def candidate(self, library, root, effect):
        raw, _ = witness(library, "framed", 1, root)
        term = parse(raw["definitions"][-1]["body"])
        raw["definitions"][-1]["body"] = render((term[0], term[1],
            ("ref", {0: "C01", 1: "F004", 2: "C02", 3: "C03", 4: "C04"}[effect]), *term[3:]))
        return raw

    def inputs(self):
        first_raw = self.candidate(self.base, "T1", 0)
        first, first_certs = attach_proposal(self.base, first_raw, "T1")
        second_raw, _ = reuse_search(first, study.pairs_for(2, "train", 1), "T2")
        second, second_certs = attach_proposal(first, second_raw, "T2")
        stages = [dict(candidate=raw, library=lib.payload(), certificates=certs,
                       row=dict(new_expression_nodes=study.node_cost(raw)))
                  for raw, lib, certs in ((first_raw, first, first_certs), (second_raw, second, second_certs))]
        return [dict(trial=dict(seed=1, repeat=1, directory="seed-1-trial-1"), base=self.base.payload(),
                     stages=stages, extensions=[])]

    @staticmethod
    def direct_reuse(library, pairs, root, folder, seconds, *, phase, history=None):
        assert phase == "reuse"
        save(folder / "search-request.json", dict(library=library.payload(), pairs=pairs, root=root,
            seconds=seconds, phase=phase, growth_contract=GROWTH_CONTRACT, history=history or [], reuse=None))
        raw, effort = reuse_search(library, pairs, root, seconds)
        save(folder / "search-response.json", dict(candidate=raw, effort=effort))
        return raw, effort

    def test_oracle_agrees_with_original_tasks(self):
        rng = random.Random(8)
        for form in (None, (None, None), ((None, None), None)):
            tree = materialize(form, rng, 3000)
            for stage, effect in ((1, 0), (2, 2)):
                self.assertEqual(study.reference(tree, effect), old_reference(tree, "framed", stage)[0])

    def test_same_inputs_novel_effects_and_disjoint_splits(self):
        panels = [study.pairs_for(e, "train", 21) for _, e in study.TASKS]
        self.assertTrue(all([s for s, _ in p] == [s for s, _ in panels[0]] for p in panels))
        self.assertEqual(len({digest(p) for p in panels}), 3)
        pools = [{v for s, _ in study.pairs_for(1, split, 21) for v in s}
                 for split in ("train", "validation", "hidden", "stress")]
        self.assertTrue(all(a.isdisjoint(b) for i, a in enumerate(pools) for b in pools[i + 1:]))
        request = prompt(self.base, panels[0], "T3")
        self.assertNotIn("effect", request)
        self.assertNotIn("reference", request)

    def test_all_lineages_and_all_tasks_in_fixed_schedule(self):
        inputs = self.inputs()
        second = copy.deepcopy(inputs[0])
        second["trial"]["directory"] = "another"
        tasks = study.schedule([*inputs, second], [21, 22])
        self.assertEqual(len(tasks), 30)
        self.assertEqual(len({t["directory"] for t in tasks}), 30)
        self.assertEqual(sum(t["arm"] == "cold" for t in tasks), 6)

    def test_witness_is_only_a_private_test_fixture(self):
        for _, effect in study.TASKS:
            raw = self.candidate(self.base, "T3", effect)
            library, _ = attach_proposal(self.base, raw, "T3")
            for split in ("train", "validation", "hidden", "stress"):
                self.assertTrue(score(library, "T3", study.pairs_for(effect, split, 21))["exact"])

    def test_whole_study_replays_and_rejects_omitted_outcome(self):
        inputs, requests = self.inputs(), []
        def propose(request, folder, **kwargs):
            requests.append(request)
            self.assertEqual(request["library_hash"], self.base.digest)
            self.assertEqual(request["prior_tasks"], [])
            root = request["root"]
            effect = dict(study.TASKS)[int(root[1:])]
            raw = self.candidate(self.base, root, effect)
            save(folder / "request.json", request)
            events = [dict(type="turn.started"),
                dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(raw))),
                dict(type="turn.completed", usage={})]
            (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            return raw
        with tempfile.TemporaryDirectory() as td, patch.object(study, "prior_inputs", return_value=inputs), \
                patch.object(study, "search_process", self.direct_reuse), \
                patch("frontier_benchmark.search_process", self.direct_reuse), \
                patch("interface_benchmark.propose", propose):
            output = Path(td) / "run"
            result = study.run(output, Path(td), seeds=[21], seconds=5, replies=2)
            self.assertEqual(len(requests), 3)
            self.assertEqual(len(result["rows"]), 9)
            self.assertTrue(all(r["admitted"] and r["audit"]["exact"] for r in result["rows"]))
            self.assertEqual(study.replay(output), result)
            frozen = json.loads((output / "frozen.json").read_text())
            frozen["rows"].pop()
            frozen["hash"] = digest(frozen["rows"])
            save(output / "frozen.json", frozen)
            with self.assertRaisesRegex(ValueError, "outcome changed"):
                study.replay(output)

    def test_failed_cold_controls_have_no_complexity_value(self):
        with tempfile.TemporaryDirectory() as td, patch.object(study, "prior_inputs", return_value=self.inputs()), \
                patch.object(study, "search_process", self.direct_reuse), \
                patch("frontier_benchmark.search_process", self.direct_reuse), \
                patch("frontier_benchmark.model_search", return_value=(None, dict(attempts=[], stopped="time_limit"))):
            output = Path(td) / "run"
            result = study.run(output, Path(td), seeds=[21], seconds=5, replies=2)
            failures = [r for r in result["rows"] if r["arm"] == "cold"]
            self.assertEqual(len(failures), 3)
            self.assertTrue(all(not r["admitted"] and r["marginal_nodes"] is None for r in failures))
            self.assertTrue(all("audit" not in r and "hidden" not in r for r in failures))

    def test_changed_interface_or_body_invalidates_attachment(self):
        raw = self.candidate(self.base, "T3", 1)
        library, certs = attach_proposal(self.base, raw, "T3")
        bad = copy.deepcopy(raw)
        bad["definitions"][-1]["body"] = "unit"
        with self.assertRaisesRegex(ValueError, "attachment"):
            study.certificate_replay(self.base, bad, library.payload(), certs, "T3")


if __name__ == "__main__":
    unittest.main()
