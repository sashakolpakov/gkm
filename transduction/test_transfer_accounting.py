import unittest

from analyze_transfer_value import payback, supported


class TransferAccountingTests(unittest.TestCase):
    def test_extension_is_charged_before_payback(self):
        result = payback([30, 40, 50], [7, 7, 7], 45)
        self.assertEqual(result["cumulative_future_savings"], [23, 56, 99])
        self.assertEqual(result["pays_back_at_task"], 4)

    def test_missing_control_or_failed_retained_task_is_not_free(self):
        for cold, retained in (([None, 100, 100], [7, 7, 7]), ([30, 40, 50], [7, None, 7])):
            result = payback(cold, retained, 45)
            self.assertIsNone(result["pays_back_at_task"])
            self.assertIsNone(result["cumulative_future_savings"][-1])

    def test_savings_can_be_negative(self):
        self.assertEqual(payback([4, 4, 4], [7, 7, 7], 45)["cumulative_future_savings"], [-3, -6, -9])

    def test_admission_without_private_success_is_not_counted(self):
        row = dict(admitted=True, hidden=dict(exact=True), stress=dict(exact=True), audit=dict(exact=True))
        self.assertTrue(supported(row))
        for key in ("hidden", "stress", "audit"):
            self.assertFalse(supported({**row, key: dict(exact=False)}))
        self.assertFalse(supported(dict(admitted=False)))


if __name__ == "__main__":
    unittest.main()
