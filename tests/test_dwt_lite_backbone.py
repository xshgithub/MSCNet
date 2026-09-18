import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DWTLiteBackboneTests(unittest.TestCase):
    def test_dwt_lite_backbone_api_is_registered(self):
        dwt_source = (ROOT / "models" / "dqdetr" / "dwt_new.py").read_text(encoding="utf-8")
        backbone_source = (ROOT / "models" / "dqdetr" / "backbone.py").read_text(encoding="utf-8")

        self.assertIn("class DWTLite", dwt_source)
        self.assertIn("class DSFusionLite", dwt_source)
        self.assertIn("class MultiBackboneLite", dwt_source)
        self.assertIn("def build_dwt_backbone_lite", dwt_source)
        self.assertIn("dwt_resnet50_lite", backbone_source)
        self.assertIn("build_dwt_backbone_lite", backbone_source)

    def test_dwt_lite_uses_bottleneck_and_depthwise_fusion(self):
        dwt_source = (ROOT / "models" / "dqdetr" / "dwt_new.py").read_text(encoding="utf-8")

        self.assertIn("bottleneck_channels", dwt_source)
        self.assertIn("groups=hidden_channels", dwt_source)
        self.assertIn("groups=self.channels", dwt_source)


if __name__ == "__main__":
    unittest.main()