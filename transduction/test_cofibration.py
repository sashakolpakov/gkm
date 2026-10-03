import itertools
import json
import unittest

from cofibration import Graph, Morphism, attach, parse_attachment, pushout, verify_certificate
from codex_glue import DISABLED_HOST_NOTICE, command, extract
from pattern_fsa import MOVE_RIGHT, OBS_TOKEN, PatternRule, WRITE_CURRENT, stream_primitives
from run_cofibration_experiment import evaluate, make_task, necessary_reuse, proposal_prompt, search_rigid, seed_machine


class CofibrationTests(unittest.TestCase):
    def setUp(self):
        self.primitives = stream_primitives(10)
        self.library = Graph(1, (PatternRule(0, OBS_TOKEN, (WRITE_CURRENT, MOVE_RIGHT), 0),),
                             self.primitives)

    def test_pushout_shares_boundary_states_and_edges_exactly(self):
        boundary = self.library
        right = Graph(2, self.library.rules + (PatternRule(1, OBS_TOKEN, (WRITE_CURRENT,), 0),),
                      self.primitives)
        inclusion = Morphism((0,), (0,))
        result = pushout(boundary, self.library, right, inclusion, inclusion)
        self.assertEqual(result.graph, right)
        self.assertEqual(result.from_right, Morphism((0, 1), (0, 1)))

    def test_universal_map_is_unique_for_every_small_cocone(self):
        # Exhaust the mediators for this finite graph example, not just an identity check.
        boundary = Graph(1, (), self.primitives)
        left = Graph(2, (), self.primitives)
        right = Graph(2, (), self.primitives)
        result = pushout(boundary, left, right, Morphism((1,), ()), Morphism((0,), ()))
        target = Graph(2, (), self.primitives)
        for lm in itertools.product(range(2), repeat=2):
            for rm in itertools.product(range(2), repeat=2):
                if lm[1] != rm[0]:
                    with self.assertRaises(ValueError):
                        result.mediate(target, Morphism(lm, ()), Morphism(rm, ()))
                    continue
                mediator = result.mediate(target, Morphism(lm, ()), Morphism(rm, ()))
                candidates = [candidate for candidate in itertools.product(range(2), repeat=3)
                              if tuple(candidate[i] for i in result.from_left.states) == lm
                              and tuple(candidate[i] for i in result.from_right.states) == rm]
                self.assertEqual(candidates, [mediator.states])

    def test_equal_parallel_edges_are_not_silently_merged(self):
        boundary = Graph(1, (), self.primitives)
        result = pushout(boundary, self.library, self.library,
                         Morphism((0,), ()), Morphism((0,), ()))
        self.assertEqual(len(result.graph.rules), 2)
        with self.assertRaises(ValueError):
            result.graph.genome()

    def test_attachment_preserves_old_halts_and_runs(self):
        square, entry = attach(self.library, 1, (PatternRule(0, OBS_TOKEN, (WRITE_CURRENT,), 1),), (0,))
        self.assertEqual(square.from_left.states, (0,))
        pairs = [(tokens, tokens) for length in range(5)
                 for tokens in itertools.product((7, 8), repeat=length)]
        self.assertEqual(evaluate(self.library, 0, pairs), evaluate(square.graph, 0, pairs))
        self.assertEqual(evaluate(square.graph, entry, [((7, 8), (7, 7, 8))], 1)["reuse_cases"], 1)

    def test_cannot_fill_an_old_missing_rule(self):
        with self.assertRaisesRegex(ValueError, "retained"):
            attach(self.library, 1, (PatternRule(1, 0, (WRITE_CURRENT,), 0),), (0,))

    def test_wrong_hash_and_coerced_values_fail(self):
        payload = {"library_hash": self.library.digest, "fresh_states": 1,
                   "ports": [0], "rules": [{"state": 0, "observation": 1,
                                              "actions": [1], "next_state": 1}]}
        parse_attachment(payload, self.library)
        bad = json.loads(json.dumps(payload))
        bad["library_hash"] = "wrong"
        with self.assertRaises(ValueError):
            parse_attachment(bad, self.library)
        payload["rules"][0]["actions"] = [True]
        with self.assertRaises(ValueError):
            parse_attachment(payload, self.library)

    def test_noninjective_boundary_rejected(self):
        boundary = Graph(2, (), self.primitives)
        with self.assertRaisesRegex(ValueError, "injective"):
            pushout(boundary, self.library, self.library,
                    Morphism((0, 0), ()), Morphism((0, 0), ()))

    def test_signature_change_rejected(self):
        from pattern_fsa import register_primitives
        other = Graph(1, (), register_primitives(10, 1))
        with self.assertRaisesRegex(ValueError, "signature"):
            Morphism((0,), ()).validate(Graph(1, (), self.primitives), other)

    def test_certificate_roundtrip_and_tamper_rejection(self):
        square, _ = attach(self.library, 1, (PatternRule(0, 1, (WRITE_CURRENT,), 1),), (0,))
        certificate = json.loads(json.dumps(square.certificate()))
        self.assertEqual(verify_certificate(certificate, self.library), square)
        certificate["maps"]["from_right"]["states"] = [0, 1]
        with self.assertRaises(ValueError):
            verify_certificate(certificate, self.library)

    def test_edge_cocone_can_identify_distinct_edges_but_mediator_is_forced(self):
        empty = Graph(0, (), self.primitives)
        square = pushout(empty, self.library, self.library, Morphism((), ()), Morphism((), ()))
        mapping = Morphism((0,), (0,))
        mediator = square.mediate(self.library, mapping, mapping)
        self.assertEqual(mediator, Morphism((0, 0), (0, 0)))

    def test_mechanical_acquisition_and_real_reuse(self):
        anchor, count = seed_machine(make_task("copy", 7), self.primitives)
        self.assertGreater(count, 1)
        task = make_task("duplicate_first", 7)
        best, _ = search_rigid(anchor, task, 128)
        _, payload, square, entry, _ = best
        fresh, fresh_entry = parse_attachment(json.loads(json.dumps(payload)), anchor)
        self.assertEqual(fresh.graph.digest, square.graph.digest)
        self.assertEqual(fresh_entry, entry)
        result = evaluate(fresh.graph, entry, task.test_pairs, anchor.states)
        self.assertEqual(result["exact"], 1)
        self.assertEqual(result["reuse_cases"], 48)

    def test_prompt_has_no_validation_or_hidden_examples(self):
        task = make_task("swap", 7)
        prompt = proposal_prompt(self.library, task, [])
        self.assertEqual(prompt["training_examples"], task.train_pairs)
        self.assertNotIn("test_pairs", prompt)
        self.assertNotIn("val_pairs", prompt)
        self.assertNotIn("task", prompt)

    def test_codex_requires_completion_and_rejects_tools(self):
        answer = {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}}
        with self.assertRaises(RuntimeError):
            extract([answer])
        self.assertEqual(extract([answer, {"type": "turn.completed", "usage": {}}]), ({}, {}))
        with self.assertRaises(RuntimeError):
            extract([{"type": "item.completed", "item": {"type": "command_execution"}},
                     answer, {"type": "turn.completed"}])
        args = command("/tmp/neutral", "/tmp/schema.json")
        for flag in ("--ignore-user-config", "--ephemeral", "--output-schema", "read-only", "gpt-5.6-sol"):
            self.assertIn(flag, args)

    def test_known_disabled_host_notice_only_allowed_before_turn(self):
        notice = {"type": "item.completed", "item": {"type": "error", "message": DISABLED_HOST_NOTICE}}
        answer = {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}}
        self.assertEqual(extract([notice, {"type": "turn.started"}, answer,
                                  {"type": "turn.completed"}])[0], {})
        with self.assertRaises(RuntimeError):
            extract([{"type": "turn.started"}, notice, answer, {"type": "turn.completed"}])

    def test_reuse_is_counterfactually_necessary_not_required_on_empty_suffix(self):
        # Some inputs legitimately stop at an old port without executing a rule.
        square, entry = attach(self.library, 1, (PatternRule(0, 1, (MOVE_RIGHT,), 1),), (0,))
        pairs = [((7,), ()), ((7, 8), (8,))]
        self.assertEqual(evaluate(square.graph, entry, pairs, 1)["reuse_cases"], 1)
        self.assertTrue(necessary_reuse(square.graph, entry, pairs, 1))
        # A call into old states is not enough when it changes no task result.
        self.assertFalse(necessary_reuse(square.graph, entry, pairs[:1], 1))


if __name__ == "__main__":
    unittest.main()
