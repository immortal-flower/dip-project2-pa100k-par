import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_code"))
from model import PAResNet, SEBlock  # noqa: E402


class ResidualSETest(unittest.TestCase):
    def test_gate_formula_and_range(self):
        torch.manual_seed(42)
        x = torch.randn(4, 32)
        standard = SEBlock(32, reduction=4)
        centered = SEBlock(32, reduction=4, centered=True)
        residual = SEBlock(32, reduction=4, residual_alpha=.25)
        centered.load_state_dict(standard.state_dict(), strict=True)
        residual.load_state_dict(standard.state_dict(), strict=True)
        gate = standard.gate(x)
        self.assertTrue(torch.allclose(standard(x), x * gate))
        self.assertTrue(torch.allclose(centered(x), x * (2 * gate)))
        self.assertTrue(torch.allclose(
            residual(x), x * (1 + .25 * (2 * gate - 1))))
        factor = 1 + .25 * (2 * gate - 1)
        self.assertGreater(float(factor.detach().min()), .75)
        self.assertLess(float(factor.detach().max()), 1.25)

    def test_old_checkpoint_state_structure_unchanged(self):
        spec = {"groups": [{"name": "upper", "attrs": ["short_sleeve",
                                                  "long_sleeve"]},
                           {"name": "accessory", "attrs": ["hat", "glasses"]}]}
        torch.manual_seed(42)
        centered = PAResNet(spec, attention="se_group_center",
                            pretrained=False)
        residual = PAResNet(spec, attention="se_group_residual025",
                            pretrained=False)
        residual.load_state_dict(centered.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in centered.parameters()),
                         sum(p.numel() for p in residual.parameters()))
        x = torch.randn(2, 512)
        for name in centered.group_attention:
            gate = centered.group_attention[name].gate(x)
            expected = x * (1 + .25 * (2 * gate - 1))
            self.assertTrue(torch.allclose(
                residual.group_attention[name](x), expected))


if __name__ == "__main__":
    unittest.main()
