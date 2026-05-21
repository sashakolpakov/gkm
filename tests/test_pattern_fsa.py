import random
import unittest

from pattern_fsa import (
    HALT,
    MOVE_RIGHT,
    OBS_EOS,
    OBS_TOKEN,
    PatternGenome,
    PatternLambdaRecord,
    PatternRule,
    WRITE_CURRENT,
    compare_primitives,
    evaluate_genome,
    evolve_solver,
    make_object_task,
    normalized_edit_distance,
    register_match_observation,
    register_primitives,
    register_store_action,
    register_write_action,
    run_transducer,
    stream_primitives,
    validation_elbow,
)


class PatternFsaTests(unittest.TestCase):
    def test_normalized_edit_distance(self):
        self.assertEqual(normalized_edit_distance((1, 2, 3), (1, 2, 3)), 0.0)
        self.assertEqual(normalized_edit_distance((), (1, 2)), 1.0)
        self.assertGreater(normalized_edit_distance((1, 2), (2, 3)), 0.0)

    def test_hand_built_swap_solver_generalizes_to_foreign_objects(self):
        object_count = 24
        primitives = register_primitives(object_count, register_count=1)
        rules = [
            PatternRule(state=0, observation=OBS_TOKEN, actions=(register_store_action(0), MOVE_RIGHT), next_state=1),
            PatternRule(state=1, observation=OBS_TOKEN, actions=(WRITE_CURRENT, register_write_action(0), HALT), next_state=1),
        ]
        genome = PatternGenome(state_count=2, alphabet_size=object_count, rules=rules)
        task = make_object_task(
            "swap",
            seed=3,
            train_examples=3,
            val_examples=3,
            test_examples=3,
            train_objects=8,
            val_objects=8,
            test_objects=8,
            length=2,
        )

        train_eval = evaluate_genome(genome, task.train_pairs, primitives, lambda_value=0.0)
        val_eval = evaluate_genome(genome, task.val_pairs, primitives, lambda_value=0.0)
        test_eval = evaluate_genome(genome, task.test_pairs, primitives, lambda_value=0.0)

        self.assertEqual(train_eval.exact_match_rate, 1.0)
        self.assertEqual(val_eval.exact_match_rate, 1.0)
        self.assertEqual(test_eval.exact_match_rate, 1.0)
        self.assertEqual(test_eval.loss, 0.0)

    def test_stream_primitives_can_express_copy_solver(self):
        object_count = 12
        primitives = stream_primitives(object_count)
        rules = [
            PatternRule(state=0, observation=OBS_TOKEN, actions=(WRITE_CURRENT, MOVE_RIGHT), next_state=0),
            PatternRule(state=0, observation=OBS_EOS, actions=(HALT,), next_state=0),
        ]
        genome = PatternGenome(state_count=1, alphabet_size=object_count, rules=rules)

        run = run_transducer(genome, (10, 3, 11), primitives)

        self.assertEqual(run.output, (10, 3, 11))
        self.assertTrue(run.halted)

    def test_primitive_tiers_are_distinct(self):
        object_count = 4
        stream = stream_primitives(object_count)
        register = register_primitives(object_count, register_count=1)
        compare = compare_primitives(object_count, register_count=1)

        self.assertFalse(stream.allows(register_store_action(0), object_count))
        self.assertTrue(register.allows(register_store_action(0), object_count))
        self.assertFalse(register.compare_registers)
        self.assertTrue(compare.compare_registers)

    def test_comparison_observation_can_branch_on_register_match(self):
        object_count = 12
        primitives = compare_primitives(object_count, register_count=1)
        rules = [
            PatternRule(
                state=0,
                observation=OBS_TOKEN,
                actions=(WRITE_CURRENT, register_store_action(0), MOVE_RIGHT),
                next_state=1,
            ),
            PatternRule(state=1, observation=register_match_observation((0,)), actions=(HALT,), next_state=1),
            PatternRule(state=1, observation=OBS_TOKEN, actions=(WRITE_CURRENT, HALT), next_state=1),
            PatternRule(state=1, observation=OBS_EOS, actions=(HALT,), next_state=1),
        ]
        genome = PatternGenome(state_count=2, alphabet_size=object_count, rules=rules)

        same = run_transducer(genome, (7, 7), primitives)
        different = run_transducer(genome, (7, 8), primitives)

        self.assertEqual(same.output, (7,))
        self.assertEqual(different.output, (7, 8))

    def test_evolution_smoke_returns_valid_solver(self):
        task = make_object_task("copy", seed=4, train_examples=4, val_examples=2, test_examples=2, length=3)
        primitives = stream_primitives(task.alphabet_size)

        best, history, train_eval, val_eval = evolve_solver(
            task=task,
            primitives=primitives,
            seed=5,
            generations=2,
            population_size=12,
            state_count=2,
            initial_rule_count=4,
            max_rules=12,
            max_rule_length=2,
            report_every=0,
        )

        self.assertEqual(len(history), 3)
        self.assertEqual(best.alphabet_size, task.alphabet_size)
        self.assertGreaterEqual(train_eval.loss, 0)
        self.assertGreaterEqual(val_eval.loss, 0)

    def test_object_task_has_disjoint_validation_and_hidden_test_objects(self):
        task = make_object_task(
            "swap",
            seed=9,
            train_examples=4,
            val_examples=2,
            test_examples=2,
            train_objects=5,
            val_objects=5,
            test_objects=5,
            length=2,
        )

        self.assertEqual(len(task.train_pairs), 4)
        self.assertEqual(len(task.val_pairs), 2)
        self.assertEqual(len(task.test_pairs), 2)
        train_symbols = {symbol for pair in task.train_pairs for seq in pair for symbol in seq}
        val_symbols = {symbol for pair in task.val_pairs for seq in pair for symbol in seq}
        test_symbols = {symbol for pair in task.test_pairs for seq in pair for symbol in seq}
        self.assertTrue(train_symbols.isdisjoint(val_symbols))
        self.assertTrue(train_symbols.isdisjoint(test_symbols))
        self.assertTrue(val_symbols.isdisjoint(test_symbols))

    def test_validation_elbow_selects_loss_complexity_compromise(self):
        records = [
            PatternLambdaRecord(
                lambda_value=0.0,
                train_loss=0.0,
                val_loss=0.05,
                train_exact_match=1.0,
                val_exact_match=1.0,
                train_free_energy=0.0,
                val_free_energy=0.05,
                complexity=100.0,
                active_rules=20,
                encoded_rules=20,
            ),
            PatternLambdaRecord(
                lambda_value=0.002,
                train_loss=0.05,
                val_loss=0.08,
                train_exact_match=0.8,
                val_exact_match=0.8,
                train_free_energy=0.09,
                val_free_energy=0.12,
                complexity=20.0,
                active_rules=5,
                encoded_rules=5,
            ),
            PatternLambdaRecord(
                lambda_value=0.006,
                train_loss=0.4,
                val_loss=0.5,
                train_exact_match=0.2,
                val_exact_match=0.2,
                train_free_energy=0.46,
                val_free_energy=0.56,
                complexity=10.0,
                active_rules=2,
                encoded_rules=2,
            ),
        ]

        selected = validation_elbow(records)

        self.assertEqual(selected.lambda_value, 0.002)
        self.assertTrue(selected.selected)
        self.assertLess(selected.elbow_score, records[0].elbow_score)

    def test_mutation_keeps_sparse_unique_rule_keys(self):
        rng = random.Random(7)
        primitives = compare_primitives(5, register_count=2)
        genome = PatternGenome.random(
            rng,
            state_count=3,
            alphabet_size=5,
            primitives=primitives,
            initial_rule_count=6,
            max_rule_length=3,
        )

        mutated = genome.mutate(
            rng,
            primitives,
            rate=1.0,
            max_rule_length=3,
            max_rules=20,
            max_states=4,
        )

        self.assertEqual(len({rule.key for rule in mutated.rules}), len(mutated.rules))
        self.assertTrue(all(1 <= len(rule.actions) <= 3 for rule in mutated.rules))


if __name__ == "__main__":
    unittest.main()
