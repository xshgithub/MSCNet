import unittest

import numpy as np

from tools.visualize_multibackbone_feats import colorize_heatmap, feature_to_map, normalize_to_uint8

try:
    import torch
except ModuleNotFoundError:
    torch = None


class VisualizeMultiBackboneFeatureTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_feature_to_map_averages_channels_and_removes_batch(self):
        feature = torch.tensor(
            [[
                [[1.0, 3.0], [5.0, 7.0]],
                [[3.0, 5.0], [7.0, 9.0]],
            ]]
        )

        result = feature_to_map(feature)

        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(torch.equal(result, torch.tensor([[2.0, 4.0], [6.0, 8.0]])))

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_normalize_to_uint8_handles_constant_maps(self):
        result = normalize_to_uint8(torch.ones(2, 3) * 4.0)

        self.assertEqual(result.dtype, "uint8")
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.max(), 0)
        self.assertEqual(result.min(), 0)

    def test_colorize_heatmap_returns_rgb_uint8_image(self):
        gray = np.array([[0, 128, 255]], dtype=np.uint8)

        result = colorize_heatmap(gray)

        self.assertEqual(result.dtype, "uint8")
        self.assertEqual(result.shape, (1, 3, 3))


if __name__ == "__main__":
    unittest.main()
