# -*- coding: utf-8 -*-
"""
Model definitions for layered line detection with heatmap regression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

try:
    import timm
except ImportError:
    timm = None


class DenoiseBlock(nn.Module):
    def __init__(self, channels=1, kernel_size=5, sigma=1.0, learnable=True):
        super().__init__()
        padding = kernel_size // 2

        grid = torch.arange(kernel_size) - kernel_size // 2
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

        self.filter = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.filter.weight.data = weight
        self.filter.weight.requires_grad = learnable

    def forward(self, x):
        return self.filter(x)


def _normalize_backbone_name(name):
    return name.lower().replace("-", "_")


class TimmFeatureBackbone(nn.Module):
    def __init__(self, model_name, pretrained=False, in_ch=1, out_indices=(2, 4)):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for MobileNetV4 backbones")

        self.model = timm.create_model(
            model_name,
            features_only=True,
            pretrained=pretrained,
            in_chans=in_ch,
            out_indices=out_indices,
        )
        feature_info = list(self.model.feature_info)
        self.low_level_channels = feature_info[out_indices[0]]["num_chs"]
        self.high_level_channels = feature_info[out_indices[1]]["num_chs"]

    def forward(self, x):
        low_level_feat, high_level_feat = self.model(x)
        return low_level_feat, high_level_feat


def build_backbone(name="mobilenet_v3_large", pretrained=False, in_ch=1):
    name = _normalize_backbone_name(name)

    if name == "mobilenetv4_conv_medium":
        backbone = TimmFeatureBackbone(name, pretrained=pretrained, in_ch=in_ch, out_indices=(2, 4))
        return (
            backbone,
            nn.Identity(),
            backbone.low_level_channels,
            backbone.high_level_channels,
        )

    if name == "mobilenet_v3_large":
        mobilenet = models.mobilenet_v3_large(pretrained=pretrained)
        features = list(mobilenet.features.children())
        low_level_channels = 40
        high_level_channels = 960

        first_conv = features[0][0]
        features[0][0] = nn.Conv2d(
            in_channels=in_ch,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        low_level_features = nn.Sequential(*features[:5])
        high_level_features = nn.Sequential(*features[5:])

    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return low_level_features, high_level_features, low_level_channels, high_level_channels


def soft_argmax(heatmaps: Tensor, temperature: float = 1.0) -> Tensor:
    B, L, H, W = heatmaps.shape
    device = heatmaps.device

    heatmaps = heatmaps / temperature
    max_v, _ = heatmaps.max(dim=2, keepdim=True)
    heatmaps = heatmaps - max_v
    probs = F.softmax(heatmaps, dim=2)

    y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    coords = (probs * y_coords).sum(dim=2)
    return coords


def generate_gaussian_heatmap(coords, valid_mask, image_height, image_width, sigma=2.0):
    B, L, W = coords.shape
    device = coords.device

    y_grid = torch.arange(image_height, device=device, dtype=torch.float32).view(1, 1, image_height, 1)
    x_grid = torch.arange(image_width, device=device, dtype=torch.float32).view(1, 1, 1, image_width)

    coords_expanded = coords.unsqueeze(2)
    valid_expanded = valid_mask.unsqueeze(2)

    y_diff = y_grid - coords_expanded
    gaussian = torch.exp(-(y_diff ** 2) / (2 * sigma ** 2))
    gaussian = gaussian * valid_expanded

    gaussian_sum = gaussian.sum(dim=2, keepdim=True) + 1e-8
    gaussian = gaussian / gaussian_sum

    return gaussian


class FeatureFusionModule(nn.Module):
    def __init__(self, high_level_channels, low_level_channels):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(high_level_channels + low_level_channels, low_level_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, high_level_feat, low_level_feat):
        _, _, h, w = low_level_feat.shape
        high_upsampled = F.interpolate(high_level_feat, size=(h, w), mode='bilinear', align_corners=False)
        fused = torch.cat([low_level_feat, high_upsampled], dim=1)
        return self.fusion_conv(fused)


class LineCoordNetFPN(nn.Module):
    def __init__(self, backbone='mobilenet_v3_large', num_lines=3, pretrained=False, use_heatmap=True, sigma=2.0):
        super().__init__()
        self.num_lines = num_lines
        self.use_heatmap = use_heatmap
        self.sigma = sigma
        self.denoise = DenoiseBlock(channels=1, kernel_size=5, sigma=1.0, learnable=True)

        self.low_level_features, self.high_level_features, low_level_channels, high_level_channels = build_backbone(
            backbone, pretrained, in_ch=1
        )

        self.feature_fusion = FeatureFusionModule(high_level_channels, low_level_channels)
        feat_channels = low_level_channels

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_lines, 1),
        )

    def forward(self, x):
        x = self.denoise(x)
        low_level_feat = self.low_level_features(x)
        if isinstance(low_level_feat, (list, tuple)):
            low_level_feat, high_level_feat = low_level_feat
        else:
            high_level_feat = self.high_level_features(low_level_feat)

        feat = self.feature_fusion(high_level_feat, low_level_feat)

        heatmaps = self.heatmap_head(feat)
        heatmaps = F.interpolate(heatmaps, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)

        y_px = soft_argmax(heatmaps, temperature=1.0)
        return y_px, heatmaps
