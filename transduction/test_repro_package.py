import json
from pathlib import Path
import tempfile
import unittest

import repro_package as repro


class ReproPackageTests(unittest.TestCase):
    def test_new_audit_keeps_all_old_results(self):
        repro.compare_report({"auditor_sha256": "old", "rows": [{"correct": 4}]},
                             {"auditor_sha256": "new", "rows": [{"correct": 4, "new_check": True}]})
        with self.assertRaises(ValueError):
            repro.compare_report({"correct": 4}, {"correct": 3})
        with self.assertRaises(ValueError):
            repro.compare_report({"correct": 4}, {})
        with self.assertRaises(ValueError):
            repro.compare_report([1, 2], [1])

    def test_checksums_detect_change_addition_and_unsafe_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data.json").write_text("{}")
            repro.seal(root)
            self.assertEqual(repro.checksums(root), 1)
            (root / "extra.json").write_text("{}")
            with self.assertRaises(ValueError):
                repro.checksums(root)
            (root / "SHA256SUMS").write_text("a" * 64 + "  ../outside\n")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                repro.checksums(root)

    def test_credentials_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "env.local").write_text("placeholder")
            with self.assertRaisesRegex(ValueError, "credential"):
                list(repro.payload_files(root))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "link").symlink_to("missing")
            with self.assertRaisesRegex(ValueError, "symlink"):
                list(repro.payload_files(root))

    def test_nested_checksum_file_is_not_exempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data.json").write_text("{}")
            repro.seal(root)
            (root / "nested").mkdir()
            (root / "nested/SHA256SUMS").write_text("unexpected")
            with self.assertRaises(ValueError):
                repro.checksums(root)

    def test_results_must_stay_outside_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "outside"):
                repro.verify(root, root / "results")


if __name__ == "__main__":
    unittest.main()
