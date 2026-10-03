import unittest

from glue_ab import (ACTIONS, advance, check_common_grammar, mechanical, payload_for,
                     prompt_for, response_schema, task_for)
from pattern_fsa import PatternRule, register_primitives
from cofibration import Graph
from run_cofibration_experiment import evaluate


class PairedGlueTests(unittest.TestCase):
    def setUp(self):
        self.library = Graph(1, (PatternRule(0, 1, (1, 0), 0),), register_primitives(48, 2))

    def test_both_arms_have_exactly_one_rule_and_same_action_vocabulary(self):
        check_common_grammar(payload_for((1,), self.library), self.library)
        for invalid in ((2,), (1,) * 9):
            with self.assertRaises(ValueError):
                check_common_grammar(payload_for(invalid, self.library), self.library)
        payload = payload_for((1,), self.library)
        payload["fresh_states"] = 2
        with self.assertRaises(ValueError):
            check_common_grammar(payload, self.library)
        result = response_schema(self.library)["properties"]
        self.assertEqual(result["rules"]["maxItems"], 1)
        self.assertEqual(result["fresh_states"]["enum"], [1])
        rule = result["rules"]["items"]["properties"]
        self.assertEqual(rule["state"]["enum"], [0])
        self.assertEqual(rule["actions"]["items"]["enum"], list(ACTIONS))

    def test_advance_rejects_only_irreparable_written_prefix(self):
        pairs = (((3, 4), (4, 3)),)
        initial = ((0, None, None, 0),)
        self.assertIsNone(advance(initial, 1, pairs))
        stored = advance(initial, 10, pairs)
        self.assertEqual(stored, ((0, 3, None, 0),))
        moved = advance(stored, 0, pairs)
        self.assertEqual(advance(moved, 1, pairs), ((1, 3, None, 1),))
        self.assertEqual(advance(initial, 30, pairs), initial)

    def test_mechanical_search_finds_glue_from_examples(self):
        task = task_for("duplicate_first", 11)
        candidate, effort = mechanical(self.library, task.train_pairs, budget=32)
        square, entry = check_common_grammar(candidate, self.library)
        self.assertTrue(effort["exact_found"])
        self.assertEqual(evaluate(square.graph, entry, task.test_pairs)["exact"], 1)

    def test_paired_task_split_is_disjoint_and_prompt_has_only_training(self):
        task = task_for("reverse_first_three", 23)
        pools = [{token for source, target in split for token in source + target}
                 for split in (task.train_pairs, task.val_pairs, task.test_pairs)]
        for i in range(3):
            for j in range(i):
                self.assertTrue(pools[i].isdisjoint(pools[j]))
        prompt = prompt_for(self.library, task.train_pairs)
        self.assertEqual(set(prompt), {"instructions", "library_hash", "library", "training_examples"})
        self.assertEqual(prompt["training_examples"], task.train_pairs)


if __name__ == "__main__":
    unittest.main()
