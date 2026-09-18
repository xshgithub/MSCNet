import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DSFusion(nn.Module):
    """Adaptive Fusion Module

    Based on aligned spatial structure features, dynamically evaluates
    structural consistency of each modality at different spatial positions,
    generates local similarity feature maps, and adaptively assigns
    per-position fusion weights: enhances complementary fusion in regions
    of high structural consistency, suppresses redundant interference in
    regions of low consistency.
    """
    def __init__(self, channels, num_backbones):
        super().__init__()
        self.channels = channels
        self.num_backbones = num_backbones

        def make_branch():
            return nn.Sequential(
                nn.Conv2d(self.channels, self.channels,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            )
        self.convs = nn.ModuleList(
            [make_branch() for _ in range(self.num_backbones)]
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channels * self.num_backbones, self.channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.DSConv = nn.Sequential(
            nn.Conv2d(self.channels, self.channels // 4,
                      kernel_size=3, stride=1, padding=1, groups=4),
            nn.BatchNorm2d(self.channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels // 4, self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

        self.LN1 = nn.BatchNorm2d(self.channels)
        self.LN2 = nn.BatchNorm2d(self.channels)

        # Local structural consistency evaluator:
        # generates spatial (H x W) consistency maps that assess how
        # structurally similar each modality is at every spatial position
        self.consistency_conv = nn.Sequential(
            nn.Conv2d(self.channels * self.num_backbones,
                      self.channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.num_backbones,
                      kernel_size=3, padding=1),
        )

        # Spatial adaptive weight generator:
        # produces per-position, per-modality fusion weights
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(self.channels, self.channels // 4,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels // 4, self.num_backbones,
                      kernel_size=3, padding=1),
        )

        total_ch = self.channels * self.num_backbones
        self.gn = nn.GroupNorm(min(8, total_ch), total_ch)

    def forward(self, inputs: List):
        B, C, H, W = inputs[0].shape

        # 1. Concatenate aligned multi-modal features and fuse
        stack = torch.stack(inputs, dim=1)
        stack = stack.permute(0, 2, 1, 3, 4)
        stack = stack.reshape(B, len(inputs) * C, H, W)

        fused = self.DSConv(self.conv1(stack))

        # 2. Compute local structural consistency maps
        #    Evaluate per-position structural similarity across modalities
        consistency = self.consistency_conv(stack)
        consistency = torch.sigmoid(consistency)  # B, num_backbones, H, W

        # 3. Per-modality branch processing guided by consistency maps
        #    Enhance features where structure is consistent, suppress elsewhere
        new_inputs = []
        for i in range(len(inputs)):
            new_inputs.append(
                inputs[i] + consistency[:, i:i + 1] * self.convs[i](inputs[i])
            )
        inputs = new_inputs

        # 4. Spatial adaptive fusion weights
        #    Higher weight in high-consistency regions (complementary),
        #    lower weight in low-consistency regions (redundancy suppression)
        spatial_weights = self.spatial_gate(fused)
        spatial_weights = torch.sigmoid(spatial_weights)
        spatial_weights = spatial_weights / (
            spatial_weights.sum(dim=1, keepdim=True) + 1e-6
        )  # B, num_backbones, H, W

        # 5. Weighted fusion with spatial-level weights
        outputs = inputs[0].new_zeros(B, C, H, W)
        for i in range(len(inputs)):
            weight = spatial_weights[:, i:i + 1]
            outputs = outputs + weight * inputs[i]

        # 6. L2 normalization for training stability
        norm = outputs.norm(dim=1, keepdim=True)
        outputs = outputs / (norm + 1e-6)
        return outputs
