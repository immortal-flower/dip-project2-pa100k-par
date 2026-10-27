"""Check case evidence stays attached to the correct sample and label."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_code"))
import e1_cases


class CaseTests(unittest.TestCase):
    def test_improvement_and_regression_export_and_alignment_guard(self):
        scratch = Path(__file__).resolve().parents[1] / "runs"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="case_test_", dir=scratch) as temp:
            root = Path(temp)
            paths = np.array(["a.jpg", "b.jpg"])
            attrs = np.array(["boots", "holding", "lower_stripe", "age_lt18"])
            targets = np.array([[1, 0, 0, 0], [0, 0, 0, 0]])
            before = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
            after = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])
            for name in paths:
                Image.new("RGB", (64, 128), "gray").save(root / name)
            for name, pred in (("old", before), ("new", after)):
                np.savez(root / f"{name}.npz", paths=paths, attributes=attrs,
                         targets=targets, predictions=pred, probabilities=pred * .8 + .1)
            argv = ["e1_cases.py", "--e0", str(root / "old.npz"), "--e1", str(root / "new.npz"),
                    "--name", "E1a", "--data-root", str(root), "--out", str(root / "cases")]
            with patch.object(sys, "argv", argv):
                e1_cases.main()
            cases = json.loads((root / "cases" / "cases.json").read_text())
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0]["path"], "a.jpg")
            self.assertEqual(cases[0]["change"], "improvement")
            self.assertEqual(cases[1]["change"], "regression")
            for case in cases:
                with Image.open(root / "cases" / case["figure"]) as img:
                    self.assertEqual(img.size, (760, 390))
            np.savez(root / "new.npz", paths=paths[::-1], attributes=attrs,
                     targets=targets, predictions=after, probabilities=after * .8 + .1)
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "Unmatched"):
                e1_cases.main()


if __name__ == "__main__":
    unittest.main()
