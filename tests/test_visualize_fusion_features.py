import unittest

import numpy as np

from tools.visualize_fusion_features import (
    assemble_compare_grid,
    assemble_wavelet_grid,
    FusionFeatureHook,
    normalize_to_uint8,
    parse_stages,
    reduce_feature_to_map,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


class VisualizeFusionFeatureUtilityTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_reduce_feature_to_map_uses_first_batch_and_abs_channel_mean(self):
        feature = torch.tensor(
            [[
                [[-1.0, 3.0], [5.0, -7.0]],
                [[3.0, -5.0], [-7.0, 9.0]],
            ]]
        )

        result = reduce_feature_to_map(feature)

        self.assertTrue(torch.equal(result, torch.tensor([[2.0, 4.0], [6.0, 8.0]])))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_normalize_to_uint8_handles_constant_maps(self):
        result = normalize_to_uint8(torch.ones(2, 3) * 4.0)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.max(), 0)
        self.assertEqual(result.min(), 0)

    def test_parse_stages_accepts_all_or_unique_ints(self):
        self.assertIsNone(parse_stages(["all"], 4))
        self.assertEqual(parse_stages(["2", "0", "2"], 4), [0, 2])

    def test_parse_stages_rejects_out_of_range_stage(self):
        with self.assertRaisesRegex(ValueError, "outside available range"):
            parse_stages(["4"], 4)

    def test_assemble_compare_grid_returns_expected_shape(self):
        heatmaps = {
            "dwt_before": [np.zeros((4, 5, 3), dtype=np.uint8), np.ones((4, 5, 3), dtype=np.uint8)],
            "dwt_after": [np.zeros((4, 5, 3), dtype=np.uint8), np.ones((4, 5, 3), dtype=np.uint8)],
            "fusion_before": [np.zeros((4, 5, 3), dtype=np.uint8), np.ones((4, 5, 3), dtype=np.uint8)],
            "fused_after": np.ones((4, 5, 3), dtype=np.uint8),
        }

        grid = assemble_compare_grid(heatmaps, ["a", "b"])

        self.assertEqual(grid.shape, (16, 10, 3))

    def test_assemble_wavelet_grid_returns_expected_shape(self):
        heatmaps = {
            "H_f0": np.zeros((4, 5, 3), dtype=np.uint8),
            "H_f1": np.ones((4, 5, 3), dtype=np.uint8),
            "H_f2": np.ones((4, 5, 3), dtype=np.uint8) * 2,
        }

        grid = assemble_wavelet_grid(heatmaps)

        self.assertEqual(grid.shape, (4, 15, 3))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_fusion_feature_hook_captures_wavelet_components_from_dwt_module(self):
        class FakeDWT(torch.nn.Module):
            def forward(self, inputs):
                self.last_wavelet_components = {
                    "H_f0": inputs[0] + 1,
                    "H_f1": inputs[0] + 2,
                    "H_f2": inputs[0] + 3,
                }
                return inputs

        class FakeFusion(torch.nn.Module):
            def forward(self, inputs):
                return inputs[0]

        class FakeMultiBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dwt = torch.nn.ModuleList([FakeDWT()])
                self.Fusion = torch.nn.ModuleList([FakeFusion()])

        multibackbone = FakeMultiBackbone()
        inputs = [torch.zeros(1, 1, 2, 2)]

        with FusionFeatureHook(multibackbone, [0]) as capture:
            outputs = multibackbone.dwt[0](inputs)
            multibackbone.Fusion[0](outputs)

        components = capture.records[0]["wavelet_components"]
        self.assertTrue(torch.equal(components["H_f0"], torch.ones(1, 1, 2, 2)))
        self.assertTrue(torch.equal(components["H_f1"], torch.ones(1, 1, 2, 2) * 2))
        self.assertTrue(torch.equal(components["H_f2"], torch.ones(1, 1, 2, 2) * 3))


if __name__ == "__main__":
    unittest.main()
