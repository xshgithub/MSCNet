import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from pytorch_wavelets import DWTForward


class DWT(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.xfm = DWTForward(J=2, wave='haar')

        self.DSConv = nn.Sequential(
            nn.Conv2d(in_channels_list[0] * 3, in_channels_list[0],
                      kernel_size=3, stride=1, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_list[0] * 6, in_channels_list[0], kernel_size=1),
            nn.BatchNorm2d(in_channels_list[0]),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels_list[0] * 3, in_channels_list[0], kernel_size=1),
            nn.BatchNorm2d(in_channels_list[0]),
            nn.ReLU(inplace=True),
        )
        self.guidance_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, in_channels_list[i],
                          kernel_size=3, padding=1),
                nn.Tanh(),
            ) for i in range(len(in_channels_list))
        ])

    def forward(self, inputs):
        H_f0, (H_f1, H_f2) = self.xfm(inputs[0].float())
        B, C, _, H1, W1 = H_f1.shape
        H_f1 = H_f1.view(B, -1, H1, W1)
        H2, W2 = H_f2.shape[-2:]
        H_f2 = H_f2.view(B, -1, H2, W2)

        H_f1 = torch.cat((self.maxpool(H_f1), self.avgpool(H_f1)), dim=1)

        if H_f1.shape[-1] != W2 and H_f1.shape[-2] != H2:
            H_f1 = F.interpolate(H_f1, (H2, W2), mode='bilinear')
        H_f1 = self.conv1(H_f1)
        H_f2 = self.conv2(H_f2)

        H_f = torch.stack([H_f0, H_f1, H_f2], dim=1)
        H_f = H_f.permute(0, 2, 1, 3, 4)
        H_f = H_f.reshape(B, 3 * C, H2, W2)
        H_f = self.DSConv(H_f)
        H_f = F.interpolate(H_f, inputs[0].size()[-2:],
                            mode='bilinear', align_corners=True)

        outputs = []
        for i, x in enumerate(inputs):
            guidance = self.guidance_convs[i](H_f)
            outputs.append(x + guidance * x)

        return outputs
