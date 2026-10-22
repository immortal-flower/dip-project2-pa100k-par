"""Numerical contracts for the E1 loss and evaluation changes."""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_code"))
from dataset import load_spec, _ordered_labels, CSV_COLUMNS
from evaluate import AttributeMetrics, decode_outputs, evaluate
from losses import focal_loss_with_logits, group_losses
from compare_e1 import validate, FIXED


class LossTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec(str(Path(__file__).resolve().parents[1] / "attributes.yaml"))
        self.outputs = {g["name"]: torch.randn(4, len(g["attrs"]), requires_grad=True)
                        for g in self.spec["groups"]}
        self.targets = torch.randint(2, (4, 26)).float()

    def test_focal_gamma_zero_equals_bce(self):
        x = torch.tensor([-1000., -2., 0., 2., 1000.], requires_grad=True)
        y = torch.tensor([1., 0., 1., 1., 0.])
        self.assertTrue(torch.equal(focal_loss_with_logits(x, y, 0),
                                    F.binary_cross_entropy_with_logits(x, y)))
        loss = focal_loss_with_logits(x, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_easy_samples_are_downweighted_and_alpha_definition(self):
        x, y = torch.tensor([3., -3.]), torch.tensor([1., 0.])
        self.assertLess(focal_loss_with_logits(x, y), focal_loss_with_logits(x, y, 0))
        self.assertAlmostEqual(focal_loss_with_logits(torch.zeros(2), y, 0, 0.25).item(),
                               0.5 * torch.log(torch.tensor(2.)).item(), places=6)

    def test_softmax_unchanged_and_bce_backward_compatible(self):
        original = group_losses(self.outputs, self.targets, self.spec)
        focal = group_losses(self.outputs, self.targets, self.spec, loss_mode="focal")
        for group in self.spec["groups"]:
            name, idx = group["name"], group["indices"]
            if group["type"] == "softmax":
                self.assertTrue(torch.equal(original[name], focal[name]))
            else:
                expected = F.binary_cross_entropy_with_logits(self.outputs[name], self.targets[:, idx])
                self.assertTrue(torch.equal(original[name], expected))
        sum(focal.values()).backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in self.outputs.values()))

    def test_pos_weight_only_affects_binary_groups(self):
        base = group_losses(self.outputs, self.targets, self.spec)
        pw = torch.full((26,), 3.)
        weighted = group_losses(self.outputs, self.targets, self.spec, pw)
        for g in self.spec["groups"]:
            if g["type"] == "softmax":
                self.assertTrue(torch.equal(base[g["name"]], weighted[g["name"]]))
        with self.assertRaises(ValueError):
            group_losses(self.outputs, self.targets, self.spec, pw, loss_mode="focal")


class MetricTests(unittest.TestCase):
    def test_hand_calculated_confusion_and_batch_invariance(self):
        spec = {"num_attrs": 2, "attr2idx": {"a": 0, "b": 1},
                "groups": [{"name": "both", "attrs": ["a", "b"]}]}
        target = torch.tensor([[1, 0], [1, 1], [0, 0], [0, 1]])
        pred = torch.tensor([[1, 1], [0, 1], [1, 0], [0, 1]])
        metric = AttributeMetrics(spec)
        metric.update(pred, target)
        result = metric.result()
        self.assertEqual(result["attributes"]["a"]["TP"], 1)
        self.assertEqual(result["attributes"]["a"]["FP"], 1)
        self.assertEqual(result["attributes"]["a"]["FN"], 1)
        self.assertEqual(result["attributes"]["a"]["TN"], 1)
        self.assertAlmostEqual(result["attributes"]["a"]["mA"], .5)
        self.assertAlmostEqual(result["attributes"]["b"]["precision"], 2 / 3)
        self.assertAlmostEqual(result["macro"]["mA"], .625)
        self.assertAlmostEqual(result["instance"]["f1"], (2 / 3 + 2 / 3 + 0 + 1) / 4)
        batched = AttributeMetrics(spec)
        batched.update(pred[:1], target[:1])
        batched.update(pred[1:], target[1:])
        self.assertEqual(result, batched.result())

    def test_missing_class_has_null_ma(self):
        spec = {"num_attrs": 1, "attr2idx": {"a": 0},
                "groups": [{"name": "one", "attrs": ["a"]}]}
        metric = AttributeMetrics(spec)
        metric.update(torch.zeros(2, 1), torch.zeros(2, 1))
        result = metric.result()
        self.assertIsNone(result["macro"]["mA"])
        self.assertEqual(result["macro"]["precision"], 0)

    def test_softmax_onehot_and_legacy_return_contract(self):
        spec = {"num_attrs": 3, "attr2idx": {"a": 0, "b": 1, "c": 2},
                "groups": [{"name": "g", "type": "softmax", "indices": [0, 1, 2],
                            "attrs": ["a", "b", "c"]}]}
        outputs = {"g": torch.tensor([[5., 1., 0.], [5., 1., 0.]])}
        pred, probs = decode_outputs(outputs, spec)
        self.assertTrue(torch.equal(pred.sum(1), torch.ones(2)))
        target = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        class Fixed(torch.nn.Module):
            def forward(self, _):
                return outputs
        loader = [(torch.zeros(2, 1), target)]
        old = evaluate(Fixed(), loader, spec, "cpu")
        new = evaluate(Fixed(), loader, spec, "cpu", return_details=True)
        self.assertEqual(old, new[:2])
        self.assertEqual(old[0]["g"], .5)
        self.assertEqual(new[2]["attributes"]["c"]["AttrAcc"], 1)

    def test_csv_gender_and_column_mapping(self):
        columns = {column: 0 for column in CSV_COLUMNS.values()}
        columns.update(Female=1, Back=1, AgeLess18=1, HandBag=1)
        from dataset import ATTR_ORDER
        labels = dict(zip(ATTR_ORDER, _ordered_labels(columns)))
        self.assertEqual(labels["male"], 0)
        self.assertEqual(labels["back"], 1)
        self.assertEqual(labels["age_lt18"], 1)
        self.assertEqual(labels["handbag"], 1)
        self.assertEqual(labels["backpack"], 0)

    def test_comparison_rejects_confounded_runs(self):
        import copy
        args = {key: 1 for key in FIXED}
        args["attention"] = "none"
        args["pos_weight"] = False
        base = {"checkpoint_args": args, "split": "val", "samples": 10000,
                "label_file_sha256": "same", "evaluation_environment": {
                    "device": "cuda", "torch": "2.1.2", "torchvision": "0.16.2",
                    "Pillow": "test", "batch_size": 64}}
        records = {"E0": base, "E1a": copy.deepcopy(base), "E1b": copy.deepcopy(base)}
        records["E1a"]["checkpoint_args"]["pos_weight"] = True
        records["E1b"]["checkpoint_args"].update(loss="focal", focal_gamma=2., focal_alpha=None)
        validate(records)
        records["E1a"]["checkpoint_args"]["lr"] = 123
        with self.assertRaisesRegex(ValueError, "confounded setting lr"):
            validate(records)
        records["E1a"]["checkpoint_args"]["lr"] = 1
        records["E1a"]["evaluation_environment"]["device"] = "cpu"
        with self.assertRaisesRegex(ValueError, "evaluation environment differs"):
            validate(records)


if __name__ == "__main__":
    unittest.main()
