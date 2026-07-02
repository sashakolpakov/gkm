import unittest

from run_bongard_sparse_classifier import (
    CONFIGS,
    PREDICT_TRUE,
    ClassifierGenome,
    ClassifierRule,
    evaluate,
    exhaustive_split,
    merge_splits,
    misclassified_split,
)
from run_bongard_symbolic_baseline import CONCEPTS, Split, make_problem
from pattern_fsa import (
    MOVE_LEFT,
    MOVE_RIGHT,
    OBS_EOS,
    OBS_TOKEN,
    PRIMITIVE_SETS,
    register_match_observation,
    register_store_action,
)


class BongardSparseClassifierTests(unittest.TestCase):
    def test_first_equals_last_validation_contains_middle_repeat_counterexamples(self):
        concept = next(item for item in CONCEPTS if item.name == "first_equals_last")
        problem = make_problem(
            concept,
            seed=211,
            train_count=8,
            validation_count=6,
            hidden_count=16,
            objects_per_split=10,
            min_length=2,
            max_length=5,
            counterexample_train=True,
        )

        hard_negatives = [
            example
            for example in problem.validation.negatives + problem.hidden_test.negatives
            if len(example) >= 3 and example[0] in example[1:-1] and example[0] != example[-1]
        ]

        self.assertGreater(len(hard_negatives), 0)

    def test_exhaustive_probe_labels_every_configured_concept(self):
        concepts = {concept.name: concept for concept in CONCEPTS}
        for config in CONFIGS:
            concept = concepts[config.concept]
            probe = exhaustive_split(concept, object_pool=(0, 1, 2), min_length=2, max_length=4)
            with self.subTest(concept=config.concept):
                self.assertGreater(len(probe.positives), 0)
                self.assertGreater(len(probe.negatives), 0)
                self.assertTrue(all(concept.predicate(example) for example in probe.positives))
                self.assertTrue(all(not concept.predicate(example) for example in probe.negatives))


    def test_merge_splits_deduplicates_archive_examples(self):
        left = Split(
            positives=((0, 1),),
            negatives=((0, 1, 2),),
        )
        right = Split(
            positives=((0, 1), (2, 3)),
            negatives=((0, 1, 2), (2, 3, 4)),
        )

        merged = merge_splits(left, right)

        self.assertEqual(merged.positives, ((0, 1), (2, 3)))
        self.assertEqual(merged.negatives, ((0, 1, 2), (2, 3, 4)))

    def test_archive_finds_misclassified_first_last_counterexamples(self):
        concept = next(item for item in CONCEPTS if item.name == "first_equals_last")
        primitives = PRIMITIVE_SETS["bidirectional_compare"](40, register_count=1)
        genome = ClassifierGenome(state_count=3, rules=[])
        archive = exhaustive_split(concept, object_pool=(0, 1, 2), min_length=2, max_length=4)

        misses = misclassified_split(genome, archive, primitives, max_steps=16, max_examples=6)

        self.assertGreater(len(misses.positives), 0)
        self.assertEqual(len(misses.negatives), 0)
        self.assertTrue(all(concept.predicate(example) for example in misses.positives))

    def test_first_equals_last_is_representable_by_bidirectional_compare(self):
        concept = next(item for item in CONCEPTS if item.name == "first_equals_last")
        primitives = PRIMITIVE_SETS["bidirectional_compare"](40, register_count=1)
        match_r0 = register_match_observation((0,))
        genome = ClassifierGenome(
            state_count=3,
            rules=[
                ClassifierRule(0, OBS_TOKEN, (register_store_action(0),), 1),
                ClassifierRule(1, OBS_TOKEN, (MOVE_RIGHT,), 1),
                ClassifierRule(1, match_r0, (MOVE_RIGHT,), 1),
                ClassifierRule(1, OBS_EOS, (MOVE_LEFT,), 2),
                ClassifierRule(2, match_r0, (PREDICT_TRUE,), 2),
            ],
        )
        probe = exhaustive_split(concept, object_pool=(30, 31, 32), min_length=2, max_length=6)

        result = evaluate(genome, probe, primitives, lambda_value=0.0, max_steps=32)

        self.assertEqual(result.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
