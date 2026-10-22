"""Opt-in CPU integration: real images, tiny subsets, no pretrained downloads.

Not scientific E1 results. Run with PA100K_SMOKE_ROOT set to the raw data root.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter_code"))
import train
from dataset import PA100KDataset
from model import PAResNet


@unittest.skipUnless(os.environ.get("PA100K_SMOKE_ROOT"), "Set PA100K_SMOKE_ROOT for real-image smoke test")
class TrainingSmoke(unittest.TestCase):
    def test_three_loss_modes_write_checkpoints_and_metrics(self):
        torch.set_num_threads(2)
        def small_dataset(*args, **kwargs):
            ds = PA100KDataset(*args, **kwargs)
            ds.samples = ds.samples[:4]
            return ds
        def small_model(*args, **kwargs):
            return PAResNet(*args, **kwargs, pretrained=False)
        (ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="e1_smoke_", dir=ROOT / "runs") as temp:
            for mode, switches in (("bce", []), ("weighted", ["--pos-weight"]),
                                   ("focal", ["--loss", "focal"])):
                output = Path(temp) / mode
                argv = ["train.py", "--attributes", str(ROOT / "attributes.yaml"),
                        "--data-root", os.environ["PA100K_SMOKE_ROOT"],
                        "--epochs", "1", "--batch-size", "2", "--workers", "0",
                        "--out", str(output)] + switches
                with patch.object(sys, "argv", argv), patch.object(train, "PA100KDataset", small_dataset), \
                     patch.object(train, "PAResNet", small_model):
                    train.main()
                self.assertTrue((output / "best.pt").is_file())
                self.assertTrue((output / "last.pt").is_file())
                result = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
                self.assertEqual(result["samples"], 4)
                self.assertEqual(len(result["attributes"]), 26)
                self.assertEqual(result["best_epoch"], 1)
                checkpoint = torch.load(output / "best.pt", weights_only=True)
                self.assertEqual(checkpoint["complete_metrics"],
                                 json.loads((output / "summary.json").read_text(encoding="utf-8"))["best_complete_metrics"])


if __name__ == "__main__":
    unittest.main()
