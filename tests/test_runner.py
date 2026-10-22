"""Verify the server plan keeps E0 settings and separates E1 interventions."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_code"))
from run_e1 import ROOT, build_commands


class RunnerTests(unittest.TestCase):
    def test_plan_preserves_baseline_and_single_factor_modes(self):
        commands = build_commands("data root with spaces", "E0 best.pt", "outputs with spaces", "python")
        self.assertEqual(len(commands), 10)
        trains = [command for command in commands if command[1].endswith("train.py")]
        baseline = json.loads((ROOT / "experiments/E0/config.json").read_text())["args"]
        for command in trains:
            for key in ("epochs", "batch_size", "lr", "weight_decay", "seed", "workers", "attention"):
                flag = "--" + key.replace("_", "-")
                self.assertEqual(command[command.index(flag) + 1], str(baseline[key]))
            self.assertEqual(command[command.index("--data-root") + 1], "data root with spaces")
        self.assertIn("--pos-weight", trains[0])
        self.assertNotIn("--pos-weight", trains[1])
        self.assertEqual(trains[1][trains[1].index("--loss") + 1], "focal")
        self.assertNotIn("--focal-alpha", trains[1])
        evaluations = [command for command in commands if command[1].endswith("evaluate.py")]
        self.assertEqual(len(evaluations), 3)
        for command in evaluations:
            self.assertEqual(command[command.index("--device") + 1], "cuda")
            self.assertEqual(command[command.index("--batch-size") + 1], "64")


if __name__ == "__main__":
    unittest.main()
