import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import register_artifacts as artifact


class RegisterArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "run"
        self.config = replace(artifact.historical.CONFIGS[0], generations=1, population=6)
        artifact.prepare(self.output, [self.config])

    def generate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return artifact.run_condition(self.output, 0)

    def test_unchanged_runner_and_resume(self):
        expected, rules = artifact.historical.run_config(self.config)
        actual = self.generate()
        self.assertEqual(expected, actual["row"])
        self.assertEqual(rules, actual["exported_rules"])
        self.assertEqual(actual, self.generate())
        summary = artifact.finalize(self.output)
        self.assertEqual(summary, artifact.replay_all(self.output))
        self.assertEqual(summary["candidate_programs_verified"], 4)
        self.assertEqual(len(actual["executions"]["hidden"]), 6)

    def test_changed_checkpoint_rejected(self):
        self.generate()
        path = self.output / "condition-00/lambda-1.json"
        value = json.loads(path.read_text())
        value["result"]["train"]["loss"] += 1
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            artifact.replay_all(self.output)

    def test_missing_candidate_does_not_start_search(self):
        with self.assertRaisesRegex(ValueError, "missing candidate"):
            artifact.replay_all(self.output)

    def test_changed_hidden_result_rejected(self):
        self.generate()
        path = self.output / "condition-00/selection.json"
        value = json.loads(path.read_text())
        value["result"]["executions"]["hidden"][0]["run"]["output"].append(999)
        value["hash"] = artifact.digest(value["result"])
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "fresh selection"):
            artifact.replay_all(self.output)


if __name__ == "__main__":
    unittest.main()
