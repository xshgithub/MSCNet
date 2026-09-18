import unittest

import torch

from models.dqdetr.dwt_new import DWT


class RecordingWavelet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.autocast_enabled = None
        self.input_dtype = None

    def forward(self, x):
        self.autocast_enabled = torch.is_autocast_enabled(x.device.type)
        self.input_dtype = x.dtype

        b, c, _, _ = x.shape
        low = x.new_zeros(b, c, 4, 4)
        high_l1 = x.new_zeros(b, c, 3, 8, 8)
        high_l2 = x.new_zeros(b, c, 3, 4, 4)
        return low, (high_l1, high_l2)


class DWTAmpTests(unittest.TestCase):
    def test_wavelet_forward_runs_with_autocast_disabled(self):
        dwt = DWT([4, 4], 4)
        dwt.xfm = RecordingWavelet()
        inputs = [
            torch.randn(1, 4, 16, 16),
            torch.randn(1, 4, 16, 16),
        ]

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            dwt(inputs)

        self.assertFalse(dwt.xfm.autocast_enabled)
        self.assertEqual(dwt.xfm.input_dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
