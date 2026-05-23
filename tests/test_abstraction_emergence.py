import unittest
from types import SimpleNamespace

from experiments.run_abstraction_emergence import (
    CORE_ABSTRACTION,
    make_tasks,
    run_scenario,
    select_condition_result,
)


class AbstractionEmergenceTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            seed=173,
            train_count=18,
            validation_count=12,
            hidden_count=32,
            max_inline_atoms=4,
            max_solver_atoms=2,
            max_macro_atoms=3,
            call_cost=0.35,
            macro_overhead=1.0,
            rule_overhead=1.0,
        )

    def test_single_task_does_not_select_library_macro(self):
        args = self.args()
        tasks = make_tasks(("solid_loop_curve",), args.seed, args.train_count, args.validation_count, args.hidden_count)
        result = select_condition_result(
            "single",
            tasks,
            "shared",
            lambda_values=(0.005,),
            max_inline_atoms=args.max_inline_atoms,
            max_solver_atoms=args.max_solver_atoms,
            max_macro_atoms=args.max_macro_atoms,
            call_cost=args.call_cost,
            macro_overhead=args.macro_overhead,
            rule_overhead=args.rule_overhead,
        )

        self.assertIsNone(result.macro)
        self.assertEqual(result.hidden_loss, 0.0)

    def test_multi_task_selects_shared_core_abstraction(self):
        args = self.args()
        tasks = make_tasks(
            ("solid_loop_curve", "solid_loop_thin", "solid_loop_symmetric"),
            args.seed,
            args.train_count,
            args.validation_count,
            args.hidden_count,
        )
        result = select_condition_result(
            "multi",
            tasks,
            "shared",
            lambda_values=(0.005,),
            max_inline_atoms=args.max_inline_atoms,
            max_solver_atoms=args.max_solver_atoms,
            max_macro_atoms=args.max_macro_atoms,
            call_cost=args.call_cost,
            macro_overhead=args.macro_overhead,
            rule_overhead=args.rule_overhead,
        )

        self.assertIsNotNone(result.macro)
        self.assertEqual(result.macro.atoms, CORE_ABSTRACTION)
        self.assertEqual(result.hidden_loss, 0.0)

    def test_no_share_ablation_does_not_select_macro_on_multi_task_support(self):
        args = self.args()
        tasks = make_tasks(
            ("solid_loop_curve", "solid_loop_thin", "solid_loop_symmetric"),
            args.seed,
            args.train_count,
            args.validation_count,
            args.hidden_count,
        )
        result = select_condition_result(
            "multi",
            tasks,
            "no_share",
            lambda_values=(0.005,),
            max_inline_atoms=args.max_inline_atoms,
            max_solver_atoms=args.max_solver_atoms,
            max_macro_atoms=args.max_macro_atoms,
            call_cost=args.call_cost,
            macro_overhead=args.macro_overhead,
            rule_overhead=args.rule_overhead,
        )

        self.assertIsNone(result.macro)
        self.assertEqual(result.hidden_loss, 0.0)

    def test_unrelated_or_control_does_not_select_macro(self):
        args = self.args()
        tasks = make_tasks(("curve_or_thin",), args.seed, args.train_count, args.validation_count, args.hidden_count)
        result = select_condition_result(
            "or_control",
            tasks,
            "shared",
            lambda_values=(0.005,),
            max_inline_atoms=args.max_inline_atoms,
            max_solver_atoms=args.max_solver_atoms,
            max_macro_atoms=args.max_macro_atoms,
            call_cost=args.call_cost,
            macro_overhead=args.macro_overhead,
            rule_overhead=args.rule_overhead,
        )

        self.assertIsNone(result.macro)
        self.assertEqual(result.hidden_loss, 0.0)

    def test_or_factor_selects_macro_from_repeated_inline_branches(self):
        args = self.args()
        tasks = make_tasks(("solid_loop_curve_or_thin",), args.seed, args.train_count, args.validation_count, args.hidden_count)
        result = select_condition_result(
            "or_factor",
            tasks,
            "shared",
            lambda_values=(0.005,),
            max_inline_atoms=args.max_inline_atoms,
            max_solver_atoms=args.max_solver_atoms,
            max_macro_atoms=args.max_macro_atoms,
            call_cost=args.call_cost,
            macro_overhead=args.macro_overhead,
            rule_overhead=args.rule_overhead,
        )

        self.assertIsNotNone(result.macro)
        self.assertEqual(result.macro.atoms, CORE_ABSTRACTION)
        self.assertEqual(result.hidden_loss, 0.0)

    def test_or_factor_no_share_ablation_does_not_select_macro(self):
        args = self.args()
        tasks = make_tasks(("solid_loop_curve_or_thin",), args.seed, args.train_count, args.validation_count, args.hidden_count)
        result = select_condition_result(
            "or_factor",
            tasks,
            "no_share",
            lambda_values=(0.005,),
            max_inline_atoms=args.max_inline_atoms,
            max_solver_atoms=args.max_solver_atoms,
            max_macro_atoms=args.max_macro_atoms,
            call_cost=args.call_cost,
            macro_overhead=args.macro_overhead,
            rule_overhead=args.rule_overhead,
        )

        self.assertIsNone(result.macro)
        self.assertEqual(result.hidden_loss, 0.0)

    def test_learned_library_lowers_transfer_complexity(self):
        args = self.args()
        results = run_scenario(args, "multi", lambda_values=(0.005,))
        by_key = {(result.scenario, result.condition): result for result in results}
        inline_transfer = by_key[("multi_transfer", "inline")]
        shared_transfer = by_key[("multi_transfer", "shared")]
        no_share_transfer = by_key[("multi_transfer", "no_share")]

        self.assertLess(shared_transfer.complexity, inline_transfer.complexity)
        self.assertLess(shared_transfer.complexity, no_share_transfer.complexity)
        self.assertEqual(shared_transfer.hidden_loss, 0.0)

    def test_or_factor_library_lowers_transfer_complexity(self):
        args = self.args()
        results = run_scenario(args, "or_factor", lambda_values=(0.005,))
        by_key = {(result.scenario, result.condition): result for result in results}
        inline_transfer = by_key[("or_factor_transfer", "inline")]
        shared_transfer = by_key[("or_factor_transfer", "shared")]
        no_share_transfer = by_key[("or_factor_transfer", "no_share")]

        self.assertLess(shared_transfer.complexity, inline_transfer.complexity)
        self.assertLess(shared_transfer.complexity, no_share_transfer.complexity)
        self.assertEqual(shared_transfer.hidden_loss, 0.0)

if __name__ == "__main__":
    unittest.main()
