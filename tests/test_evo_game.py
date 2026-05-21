import random
import unittest

from evo_game import (
    ACTION_NAMES,
    COMPLEXITY_MODES,
    Genome,
    Level,
    Rule,
    STAY_ACTION,
    evaluate,
    evolve,
    export_policy_code,
    free_energy_function,
    lambda_sweep,
    loss_function,
    make_levels,
    observation,
    run_episode,
)


class EvoGameTests(unittest.TestCase):
    def test_observation_points_toward_nearest_food(self):
        self.assertEqual(observation((3, 3), [(5, 1)]), 2)  # NE
        self.assertEqual(observation((3, 3), [(3, 6)]), 7)  # S

    def test_episode_collects_adjacent_food_with_simple_policy(self):
        rules = [Rule(state=0, previous_action=STAY_ACTION, observation=5, action=1, next_state=0)]
        genome = Genome(state_count=1, rules=rules)
        level = Level(width=5, height=5, start=(2, 2), food=((3, 2),))

        episode = run_episode(genome, level, max_steps=4)

        self.assertEqual(episode.collected, 1)
        self.assertEqual(episode.path[1], (3, 2))

    def test_mutation_preserves_genome_shape(self):
        rng = random.Random(1)
        genome = Genome.random(rng, state_count=3, max_rule_length=3)
        mutated = genome.mutate(rng, rate=1.0, max_rule_length=3)

        self.assertEqual(mutated.state_count, 3)
        self.assertGreater(len(mutated.rules), 0)
        self.assertTrue(all(0 <= rule.action < len(ACTION_NAMES) for rule in mutated.rules))
        self.assertTrue(all(1 <= len(rule.actions) <= 3 for rule in mutated.rules))
        self.assertTrue(all(0 <= rule.next_state < 3 for rule in mutated.rules))
        self.assertEqual(len({rule.key for rule in mutated.rules}), len(mutated.rules))

    def test_rule_sequence_executes_multiple_moves_and_costs_more(self):
        rules = [Rule(state=0, previous_action=STAY_ACTION, observation=5, actions=(1, 1), next_state=0)]
        single = [Rule(state=0, previous_action=STAY_ACTION, observation=5, action=1, next_state=0)]
        macro_genome = Genome(state_count=1, rules=rules)
        single_genome = Genome(state_count=1, rules=single)
        level = Level(width=5, height=5, start=(1, 2), food=((3, 2),))

        episode = run_episode(macro_genome, level, max_steps=4)
        macro_eval = evaluate(macro_genome, [level])
        single_eval = evaluate(single_genome, [level])

        self.assertEqual(episode.collected, 1)
        self.assertEqual(episode.path[2], (3, 2))
        self.assertEqual(macro_eval.table_complexity, single_eval.table_complexity + 1)

    def test_evolution_returns_valid_history_and_policy_export(self):
        train_levels = make_levels(seed=11, count=3, width=5, height=5, food_count=2)
        val_levels = make_levels(seed=12, count=2, width=5, height=5, food_count=2)

        _initial, best, history, train_eval, val_eval = evolve(
            seed=3,
            generations=3,
            population_size=20,
            state_count=2,
            mutation_rate=0.08,
            max_steps=60,
            train_levels=train_levels,
            val_levels=val_levels,
            report_every=0,
        )

        self.assertEqual(len(history), 4)
        self.assertGreaterEqual(train_eval.mean_collected, 0)
        self.assertGreaterEqual(val_eval.mean_collected, 0)
        exported = export_policy_code(best)
        self.assertIn("def act", exported)

    def test_evaluation_reports_complexity(self):
        rng = random.Random(2)
        genome = Genome.random(rng, state_count=2)
        levels = make_levels(seed=20, count=2, width=5, height=5, food_count=2)

        result = evaluate(genome, levels, complexity_weight=0.1, max_steps=80)

        self.assertEqual(result.free_energy, free_energy_function(genome, levels, 0.1, max_steps=80))
        self.assertEqual(result.loss, loss_function(genome, levels, max_steps=80))
        self.assertEqual(result.max_steps, 80)
        self.assertGreater(result.complexity, 0)
        self.assertGreaterEqual(result.active_states, 1)
        self.assertGreaterEqual(result.active_rules, 0)
        self.assertEqual(result.complexity_mode, "table")
        self.assertIn(result.complexity_mode, COMPLEXITY_MODES)
        self.assertGreaterEqual(result.table_complexity, result.active_complexity)
        self.assertGreaterEqual(result.table_rules, result.active_rules)

    def test_complexity_modes_distinguish_unused_table_from_active_behavior(self):
        rules = [
            Rule(state=state, previous_action=STAY_ACTION, observation=obs, action=4, next_state=0)
            for state in range(4)
            for obs in range(9)
        ]
        rules[5] = Rule(state=0, previous_action=STAY_ACTION, observation=5, action=1, next_state=0)
        genome = Genome(state_count=4, rules=rules)
        level = Level(width=5, height=5, start=(2, 2), food=((3, 2),))

        active = evaluate(genome, [level], complexity_mode="active")
        table = evaluate(genome, [level], complexity_mode="table")
        pruned = evaluate(genome, [level], complexity_mode="pruned")
        mixed = evaluate(genome, [level], complexity_mode="mixed")

        self.assertEqual(active.active_states, 1)
        self.assertEqual(active.active_rules, 1)
        self.assertEqual(table.table_states, 4)
        self.assertEqual(table.table_rules, 36)
        self.assertEqual(pruned.reachable_states, 1)
        self.assertEqual(pruned.reachable_rules, 9)
        self.assertEqual(table.table_complexity, 72)
        self.assertEqual(active.active_complexity, 2)
        self.assertEqual(pruned.pruned_complexity, 18)
        self.assertLess(active.complexity, table.complexity)
        self.assertLess(active.complexity, pruned.complexity)
        self.assertLess(active.complexity, mixed.complexity)
        self.assertLess(mixed.complexity, table.complexity)

    def test_lambda_sweep_returns_per_lambda_records(self):
        train_levels = make_levels(seed=31, count=2, width=5, height=5, food_count=2)
        val_levels = make_levels(seed=32, count=2, width=5, height=5, food_count=2)

        best, records, history, train_eval, val_eval = lambda_sweep(
            seed=5,
            lambda_values=[0.0, 0.1],
            optimizer="genetic",
            generations=1,
            population_size=10,
            state_count=2,
            mutation_rate=0.05,
            complexity_weight=0.0,
            complexity_mode="mixed",
            max_steps=80,
            hyperopt_evals=5,
            train_levels=train_levels,
            val_levels=val_levels,
            report_every=0,
        )

        self.assertEqual(len(records), 2)
        self.assertGreater(len(history), 0)
        self.assertEqual(best.state_count, 2)
        self.assertGreaterEqual(train_eval.loss, 0)
        self.assertGreaterEqual(val_eval.loss, 0)
        self.assertGreaterEqual(records[0].complexity_variance, 0)
        self.assertEqual(records[0].complexity_mode, "mixed")
        self.assertEqual(train_eval.max_steps, 80)
        self.assertGreaterEqual(records[0].table_complexity, records[0].active_complexity)


if __name__ == "__main__":
    unittest.main()
