from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward, DWTInverse

from util.misc import NestedTensor


def check(name, x):
    if isinstance(x, List):
        for xs in x:
            if isinstance(xs, NestedTensor):
                xs = xs.tensors
            if torch.isnan(xs).any() or torch.isinf(xs).any():
                print(f"NaN/Inf at {name}")
                print(xs)
                raise ValueError("Found NaN")
    else:
        if isinstance(x, NestedTensor):
            x = x.tensors
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"NaN/Inf at {name}")
            print(x)
            raise ValueError("Found NaN")


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 鍗风Н"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 鍗风Н"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4  # 杈撳嚭閫氶亾鏁版墿寮犲€嶆暟 (ResNet50/101/152 浣跨敤 Bottleneck)

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        # 1x1 闄嶇淮
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        # 3x3 鍗风Н
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        # 1x1 鍗囩淮
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,
                 channels=None):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group

        # Stem 閮ㄥ垎 (7x7 conv + maxpool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ResNet 鍥涗釜 stage
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.layers = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        # 鍒嗙被澶?
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 鍒濆鍖栨潈閲?
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward_layer(self, layer_idx, x):
        if layer_idx == 0:
            x = self.stem(x)

        x = self.layers[layer_idx](x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x


def _lite_hidden_channels(channels, bottleneck_channels=256):
    return max(16, min(channels // 4, bottleneck_channels))


def _wavelet_forward_float32(xfm, x):
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        return xfm(x.float())


class DWT(nn.Module):
    def __init__(self, in_channels_list, out_channels, bottleneck_channels=256):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.xfm = DWTForward(J=2, wave='haar')
        self.capture_wavelet_components = False
        self.last_wavelet_components = None
        hidden_channels = _lite_hidden_channels(out_channels, bottleneck_channels)
        high_channels = in_channels_list[0]

        self.DSConv = nn.Sequential(
            nn.Conv2d(high_channels * 3, high_channels * 3,
                      kernel_size=3, stride=1, padding=1,
                      groups=high_channels * 3, bias=False),
            nn.BatchNorm2d(high_channels * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_channels * 3, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(high_channels * 6, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, high_channels, kernel_size=1),
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(high_channels * 3, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, high_channels, kernel_size=1),
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True),
        )
        self.guidance_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                          padding=1, groups=hidden_channels, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, in_channels_list[i], kernel_size=1),
                nn.Tanh(),
            ) for i in range(len(in_channels_list))
        ])

    def forward(self, inputs):
        H_f0, (H_f1, H_f2) = _wavelet_forward_float32(self.xfm, inputs[0])
        B, C, _, H1, W1 = H_f1.shape
        H_f1 = H_f1.view(B, -1, H1, W1)
        H2, W2 = H_f2.shape[-2:]
        H_f2 = H_f2.view(B, -1, H2, W2)

        H_f1 = torch.cat((self.maxpool(H_f1), self.avgpool(H_f1)), dim=1)

        if H_f1.shape[-1] != W2 and H_f1.shape[-2] != H2:
            H_f1 = F.interpolate(H_f1, (H2, W2), mode='bilinear')
        H_f1 = self.conv1(H_f1)
        H_f2 = self.conv2(H_f2)

        if self.capture_wavelet_components:
            self.last_wavelet_components = {
                "H_f0": H_f0.detach(),
                "H_f1": H_f1.detach(),
                "H_f2": H_f2.detach(),
            }
        else:
            self.last_wavelet_components = None

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


class DSFusion(nn.Module):
    def __init__(self, channels, num_backbones, bottleneck_channels=256):
        super().__init__()
        self.channels = channels
        self.num_backbones = num_backbones
        hidden_channels = _lite_hidden_channels(channels, bottleneck_channels)

        def make_branch():
            return nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3,
                          stride=1, padding=1, groups=self.channels, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            )
        self.convs = nn.ModuleList(
            [make_branch() for _ in range(self.num_backbones)]
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channels * self.num_backbones, hidden_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.DSConv = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3,
                      stride=1, padding=1, groups=self.channels, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

        self.LN1 = nn.BatchNorm2d(self.channels)
        self.LN2 = nn.BatchNorm2d(self.channels)

        self.consistency_conv = nn.Sequential(
            nn.Conv2d(self.channels * self.num_backbones, hidden_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.num_backbones, kernel_size=1),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(self.channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.num_backbones, kernel_size=1),
        )

        total_ch = self.channels * self.num_backbones
        self.gn = nn.GroupNorm(min(8, total_ch), total_ch)

    def forward(self, inputs: List):
        B, C, H, W = inputs[0].shape

        stack = torch.stack(inputs, dim=1)
        stack = stack.permute(0, 2, 1, 3, 4)
        stack = stack.reshape(B, len(inputs) * C, H, W)

        fused = self.DSConv(self.conv1(stack))

        consistency = self.consistency_conv(stack)
        consistency = torch.sigmoid(consistency)  # B, num_backbones, H, W

        new_inputs = []
        for i in range(len(inputs)):
            new_inputs.append(
                inputs[i] + consistency[:, i:i + 1] * self.convs[i](inputs[i])
            )
        inputs = new_inputs

        spatial_weights = self.spatial_gate(fused)
        spatial_weights = torch.sigmoid(spatial_weights)
        spatial_weights = spatial_weights / (
            spatial_weights.sum(dim=1, keepdim=True) + 1e-6
        )  # B, num_backbones, H, W

        outputs = inputs[0].new_zeros(B, C, H, W)
        for i in range(len(inputs)):
            weight = spatial_weights[:, i:i + 1]
            outputs = outputs + weight * inputs[i]

        norm = outputs.norm(dim=1, keepdim=True)
        outputs = outputs / (norm + 1e-6)
        return outputs


class MultiBackbone(nn.Module):
    def __init__(self, backbones, channels_per_layer, return_dwt_feats=False,
                 bottleneck_channels=256):
        super().__init__()
        self.return_dwt_feats = return_dwt_feats
        self.backbones = nn.ModuleList(backbones)
        self.num_layers = len(channels_per_layer[0])
        self.dwt = nn.ModuleList([
            DWT(
                [channels_per_layer[backbone][i] * Bottleneck.expansion for backbone in range(len(backbones))],
                max([channels_per_layer[backbone][i] * Bottleneck.expansion for backbone in range(len(backbones))]),
                bottleneck_channels=bottleneck_channels,
            )
            for i in range(self.num_layers)
        ])
        self.Fusion = nn.ModuleList([
            DSFusion(channels_per_layer[0][i] * Bottleneck.expansion,
                         len(backbones), bottleneck_channels=bottleneck_channels)
            for i in range(self.num_layers)
        ])

    def forward(self, x_list):

        inputs = x_list
        outputs = []
        dwt_before = []
        dwt_after = []

        for l in range(self.num_layers):
            layer_outs = []
            for i, backbone in enumerate(self.backbones):
                out = backbone.forward_layer(l, inputs[i])
                layer_outs.append(out)
            if self.return_dwt_feats:
                dwt_before.append([x.detach() for x in layer_outs])
            inputs = self.dwt[l](layer_outs)
            if self.return_dwt_feats:
                dwt_after.append([x.detach() for x in inputs])
            res = self.Fusion[l](inputs)
            outputs.append(res)
        if self.return_dwt_feats:
            return outputs, {
                "dwt_before": dwt_before,
                "dwt_after": dwt_after,
            }
        return outputs


def build_dwt_backbone(num_classes=2, modalities=None, bottleneck_channels=256):
    if modalities is None:
        modalities = ['sla', 'ssh', 'sst', 'chl']
    if len(modalities) not in (2, 3, 4):
        raise ValueError("dwt_resnet50_lite expects 2, 3, or 4 modalities")

    backbones = [
        ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        for _ in modalities
    ]
    channels_per_layer = [[64, 128, 256, 512] for _ in modalities]
    return MultiBackbone(
        backbones,
        channels_per_layer,
        bottleneck_channels=bottleneck_channels,
    )

if __name__ == '__main__':
    model = build_dwt_backbone(2)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: {}".format(total_params))
    x1 = torch.randn(1, 3, 640, 640)
    x2 = torch.randn(1, 3, 640, 640)
    x3 = torch.randn(1, 3, 640, 640)
    x4 = torch.randn(1, 3, 640, 640)
    outputs = model([x1, x2, x3, x4])
    print(model)


