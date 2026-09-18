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
    """3x3 卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4  # 输出通道数扩张倍数 (ResNet50/101/152 使用 Bottleneck)

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        # 1x1 降维
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        # 3x3 卷积
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        # 1x1 升维
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

        # Stem 部分 (7x7 conv + maxpool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ResNet 四个 stage
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.layers = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
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


class DWT(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.xfm = DWTForward(J=2, wave='haar')
        # self.conv = nn.Conv2d(in_channels_list[0] * 3, in_channels_list[0] * 3, kernel_size=3, stride=2, padding=1)
        self.DSConv = nn.Sequential(
            nn.Conv2d(in_channels_list[0] * 3, in_channels_list[0], kernel_size=3, stride=1, padding=1, groups=4),
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

    def forward(self, inputs):
        H_f0, (H_f1, H_f2) = self.xfm(inputs[0].float())
        B, C, _, H1, W1 = H_f1.shape
        H_f1 = H_f1.view(B, -1, H1, W1)
        H2, W2 = H_f2.shape[-2:]
        H_f2 = H_f2.view(B, -1, H2, W2)
        # H_f2 = F.interpolate(H_f2.view(B, -1, H2, W2), (H1, W1), mode='bilinear', align_corners=True)
        # H_f1 = self.conv(H_f1)
        H_f1 = torch.cat((self.maxpool(H_f1), self.avgpool(H_f1)), dim=1)
        # H_f2 = torch.cat((self.maxpool(H_f2), self.avgpool(H_f2)), dim=1)

        if H_f1.shape[-1] != W2 and H_f1.shape[-2] != H2:
            H_f1 = F.interpolate(H_f1, (H2, W2), mode='bilinear')
        H_f1 = self.conv1(H_f1)
        H_f2 = self.conv2(H_f2)
        # H_f = torch.cat([H_f0, H_f1, H_f2], dim=1)
        H_f = torch.stack([H_f0, H_f1, H_f2], dim=1)
        H_f = H_f.permute(0, 2, 1, 3, 4)
        H_f = H_f.reshape(B, 3 * C, H2, W2)
        H_f = self.DSConv(H_f)
        H_f = F.interpolate(H_f, inputs[0].size()[-2:], mode='bilinear', align_corners=True)

        alpha = 1.0 / len(inputs)
        outputs = [x + alpha * torch.tanh(H_f).detach() * x for x in inputs]
        # outputs = [x + H_f.detach() * x for x in inputs]

        return outputs


# class DSFusion(nn.Module):
#     def __init__(self, channels, num_backbones):
#         super().__init__()
#         self.channels = channels
#         self.num_backbones = num_backbones
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(self.channels * self.num_backbones, self.channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(self.channels),
#             nn.ReLU(inplace=True),
#         )
#         self.LN1 = nn.LayerNorm(self.channels)
#         self.LN2 = nn.LayerNorm(self.channels)
#         self.DSConv = nn.Sequential(
#             nn.Conv2d(self.channels, self.channels // 4, kernel_size=3, stride=1, padding=1, groups=4),
#             nn.BatchNorm2d(self.channels // 4),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.channels // 4, self.channels, kernel_size=1),
#             nn.BatchNorm2d(self.channels),
#             nn.ReLU(inplace=True),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.SE = nn.Sequential(
#             nn.Linear(self.channels, self.channels // 4, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.channels // 4, self.channels, bias=False),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, inputs: torch.Tensor):
#         inputs = self.conv1(inputs)
#         inputs = self.LN1(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         tmp = self.DSConv(inputs)
#         tmp = self.LN2(tmp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         gate = self.avgpool(inputs).squeeze(-1).squeeze(-1)
#         gate = self.SE(gate)[:, :, None, None]
#         outputs = inputs + gate * tmp
#         return outputs

class DSFusion(nn.Module):
    def __init__(self, channels, num_backbones):
        super().__init__()
        self.channels = channels
        self.num_backbones = num_backbones
        def make_branch():
            return nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            )
        self.convs = nn.ModuleList([make_branch() for _ in range(self.num_backbones)])
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channels * self.num_backbones, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

        self.DSConv = nn.Sequential(
            nn.Conv2d(self.channels, self.channels // 4, kernel_size=3, stride=1, padding=1, groups=4),
            nn.BatchNorm2d(self.channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels // 4, self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        # self.LN1 = nn.LayerNorm(self.channels)
        # self.LN2 = nn.LayerNorm(self.channels)
        self.LN1 = nn.BatchNorm2d(self.channels)
        self.LN2 = nn.BatchNorm2d(self.channels)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.SE = nn.Sequential(
            nn.Linear(self.channels, self.channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // 4, self.channels, bias=False),
            nn.Sigmoid(),
        )

        total_ch = self.channels * self.num_backbones
        groups = min(32, total_ch)
        while total_ch % groups != 0:
            groups -= 1
        # self.gn = nn.GroupNorm(groups, total_ch)
        self.gn = nn.GroupNorm(8, total_ch)

        self.gate = nn.Conv2d(self.channels * self.num_backbones, self.num_backbones, kernel_size=1)

    def forward(self, inputs: List):
        B, C, H, W = inputs[0].shape

        stack = torch.stack(inputs, dim=1)
        stack = stack.permute(0, 2, 1, 3, 4)
        stack = stack.reshape(B, len(inputs) * C, H, W)

        stack = self.conv1(stack)
        stack = self.DSConv(stack)
        stack_gate = self.avgpool(stack).squeeze(-1).squeeze(-1)
        stack_gate = self.SE(stack_gate)[:, :, None, None]

        new_inputs = []
        for i in range(len(inputs)):
            new_inputs.append(inputs[i] + stack_gate * self.convs[i](inputs[i]))
        inputs = new_inputs

        gate_logit = self.gate(self.gn(torch.cat(inputs, dim=1)))
        gate_logit = F.adaptive_avg_pool2d(gate_logit, (1, 1)).view(B, self.num_backbones)
        # gate_logit = gate_logit.clamp(-20.0, 20.0)

        # gates = F.softmax(gate_logit, dim=1).view(B, self.num_backbones, 1, 1)
        gates = torch.sigmoid(gate_logit)
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-6)
        gates = gates.view(B, self.num_backbones, 1, 1)

        outputs = inputs[0].new_zeros(B, C, H, W)
        for i in range(len(inputs)):
            weight = gates[:, i:i+1, :, :]
            outputs = outputs + weight * inputs[i]

        norm = outputs.norm(dim=1, keepdim=True)
        outputs = outputs / (norm + 1e-6)
        return outputs


class ConcatFusion(nn.Module):
    def __init__(self, channels, num_backbones):
        super().__init__()
        self.channels = channels
        self.num_backbones = num_backbones
        self.fusion = nn.Sequential(
            nn.Conv2d(self.channels * num_backbones, self.channels, 1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        inputs = torch.cat(inputs, dim=1)
        outputs = self.fusion(inputs)
        outputs = outputs / (outputs.norm(dim=1, keepdim=True) + 1e-6)
        return outputs


class AvgFusion(nn.Module):
    def __init__(self, channels, num_backbones, reduction=16, use_attention=True):
        super().__init__()

        self.channels = channels
        self.use_attention = use_attention

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.weight_net = nn.Sequential(
            nn.Linear(channels * 4, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, 4)
        )

        if use_attention:
            self.channel_att = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, inputs):
        F1, F2, F3, F4 = inputs
        B, C, H, W = F1.shape

        g1 = self.gap(F1).view(B, C)
        g2 = self.gap(F2).view(B, C)
        g3 = self.gap(F3).view(B, C)
        g4 = self.gap(F4).view(B, C)

        g = torch.cat([g1, g2, g3, g4], dim=1)

        alpha = self.weight_net(g)
        alpha = F.softmax(alpha, dim=1)

        a1 = alpha[:, 0].view(B, 1, 1, 1)
        a2 = alpha[:, 1].view(B, 1, 1, 1)
        a3 = alpha[:, 2].view(B, 1, 1, 1)
        a4 = alpha[:, 3].view(B, 1, 1, 1)

        F_fuse = a1 * F1 + a2 * F2 + a3 * F3 + a4 * F4

        if self.use_attention:
            att = self.channel_att(F_fuse)
            F_fuse = F_fuse * att + F_fuse

        return F_fuse


class MultiBackbone(nn.Module):
    def __init__(self, backbones, channels_per_layer):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.num_layers = len(channels_per_layer[0])
        self.dwt = nn.ModuleList([
            # DWT(
            #     [channels_per_layer[backbone][i]*Bottleneck.expansion for backbone in range(len(backbones))],
            #     max([channels_per_layer[backbone][i]*Bottleneck.expansion for backbone in range(len(backbones))])
            # )
            nn.Identity()
            for i in range(self.num_layers)
        ])
        # self.Fusion = nn.ModuleList([
        #     DSFusion(channels_per_layer[0][i]*Bottleneck.expansion, len(backbones)) for i in range(self.num_layers)
        # ])

        self.Fusion = nn.ModuleList([
            ConcatFusion(channels_per_layer[0][i]*Bottleneck.expansion, len(backbones)) for i in range(self.num_layers)
        ])

        # self.Fusion = nn.ModuleList([
        #     AvgFusion(channels_per_layer[0][i]*Bottleneck.expansion, len(backbones)) for i in range(self.num_layers)
        # ])

    # def forward(self, x_list):
    #     inputs = x_list
    #     outputs = []
    #
    #     for l in range(self.num_layers):
    #         layer_outs = []
    #         for i, backbone in enumerate(self.backbones):
    #             out = backbone.forward_layer(l, inputs[i])
    #             layer_outs.append(out)
    #         inputs = self.dwt[l](layer_outs)
    #         B, C, H, W = inputs[0].shape
    #         stack = torch.stack(inputs, dim=1)
    #         stack = stack.permute(0, 2, 1, 3, 4)
    #         stack = stack.reshape(B, len(inputs) * C, H, W)
    #         res = self.Fusion[l](stack)
    #         outputs.append(res)
    #
    #     return outputs
    def forward(self, x_list):

        inputs = x_list
        outputs = []

        for l in range(self.num_layers):
            layer_outs = []
            for i, backbone in enumerate(self.backbones):
                out = backbone.forward_layer(l, inputs[i])
                layer_outs.append(out)
            inputs = self.dwt[l](layer_outs)
            inputs = [i.clone() for i in inputs]
            res = self.Fusion[l](inputs)
            outputs.append(res)

        return outputs


def build_dwt_backbone(num_classes=2, modalities=['sla', 'ssh', 'sst', 'chl']):

    sla = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    ssh = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    sst = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    chl = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

    if len(modalities) == 4:
        channels_per_layer = [[64, 128, 256, 512],
                              [64, 128, 256, 512],
                              [64, 128, 256, 512],
                              [64, 128, 256, 512],]
        model = MultiBackbone([sla, ssh, sst, chl], channels_per_layer)

    elif len(modalities) == 3:
        channels_per_layer = [[64, 128, 256, 512],
                              [64, 128, 256, 512],
                              [64, 128, 256, 512],]
        model = MultiBackbone([sla, ssh, sst], channels_per_layer)

    elif len(modalities) == 2:
        channels_per_layer = [[64, 128, 256, 512],
                              [64, 128, 256, 512],]
        model = MultiBackbone([sla, ssh], channels_per_layer)
    return model


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

