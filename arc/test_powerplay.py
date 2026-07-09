import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "arc" / "crack_lab"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from powerplay import description_length


class PowerPlayTests(unittest.TestCase):
    def test_description_length_prices_longer_programs(self):
        guard = ((), 2)
        self.assertLess(
            description_length((1,), 5, guard),
            description_length((1, 2, 3), 5, guard),
        )


if __name__ == "__main__":
    unittest.main()
