"""Self-built non-heatmap networks: ordered curves and nested tissue regions."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


WEIGHTS = {'resnet18': 'resnet18-f37072fd.pth',
           'mobilenet_v3_large': 'mobilenet_v3_large-8738ca79.pth',
           'mobilenet_v3_small': 'mobilenet_v3_small-047dcff4.pth'}


class Encoder(nn.Module):
    def __init__(self, name, assets=None):
        super().__init__()
        self.name = name
        base = getattr(models, name)(weights=None)
        if assets is not None:
            path = Path(assets) / WEIGHTS[name]
            base.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        if name == 'resnet18':
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.stages = nn.ModuleList([base.layer1, base.layer2, base.layer3, base.layer4])
            self.channels = [64, 128, 256, 512]
        else:
            self.features = base.features
            if name == 'mobilenet_v3_large':
                self.indices, self.channels = [3, 6, 12, 16], [24, 40, 112, 960]
            else:
                self.indices, self.channels = [1, 3, 8, 12], [16, 24, 48, 576]

    def forward(self, x):
        outputs = []
        if self.name == 'resnet18':
            x = self.stem(x)
            for stage in self.stages:
                x = stage(x); outputs.append(x)
        else:
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self.indices:
                    outputs.append(x)
        return outputs


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.SiLU(inplace=True))


class BoundaryNet(nn.Module):
    def __init__(self, method='ordered_curves', backbone='resnet18', assets=None):
        super().__init__()
        self.method = method
        self.backbone = backbone
        self.encoder = Encoder(backbone, assets)
        self.lateral = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in self.encoder.channels])
        self.refine = nn.ModuleList([block(64, 64) for _ in range(4)])
        self.regions = nn.Sequential(block(66, 64), nn.Conv2d(64, 2, 1))
        if method == 'ordered_curves':
            self.reduce = nn.Conv2d(64, 16, 1)
            self.context = nn.Sequential(*[
                nn.Sequential(nn.Conv1d(128, 128, 3, padding=d, dilation=d),
                              nn.GroupNorm(8, 128), nn.GELU()) for d in (1, 2, 4)])
            self.position = nn.Parameter(torch.randn(1, 64, 128) * 0.01)
            layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1, activation='gelu',
                                               batch_first=True, norm_first=True)
            self.attention = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.curves = nn.Sequential(nn.Conv1d(128, 64, 1), nn.GELU(), nn.Conv1d(64, 3, 1))
        elif method != 'ordinal_regions':
            raise ValueError(method)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        h, w = image.shape[-2:]
        x = (image.expand(-1, 3, -1, -1) - self.mean) / self.std
        encoded = self.encoder(x)
        x = self.refine[-1](self.lateral[-1](encoded[-1]))
        for i in (2, 1, 0):
            x = self.refine[i](self.lateral[i](encoded[i]) +
                               F.interpolate(x, size=encoded[i].shape[-2:], mode='bilinear', align_corners=False))
        b, _, fh, fw = x.shape
        yy = torch.linspace(0, 1, fh, device=x.device, dtype=x.dtype).view(1, 1, fh, 1).expand(b, 1, fh, fw)
        xx = torch.linspace(0, 1, fw, device=x.device, dtype=x.dtype).view(1, 1, 1, fw).expand(b, 1, fh, fw)
        logits = self.regions(torch.cat([x, yy, xx], 1)).float()
        upper = logits[:, :1].sigmoid()
        lower = upper * logits[:, 1:].sigmoid()
        regions = torch.cat([upper, lower], 1)  # P(pixel is below each boundary), nested by construction.
        if self.method == 'ordered_curves':
            sequence = F.adaptive_avg_pool2d(self.reduce(x), (8, 64)).flatten(1, 2)
            sequence = self.context(sequence).transpose(1, 2) + self.position
            sequence = self.attention(sequence).transpose(1, 2)
            gap_logits = F.interpolate(self.curves(sequence).float(), size=w, mode='linear', align_corners=False)
            fractions = gap_logits.softmax(dim=1)
            y = torch.cumsum(fractions, 1)[:, :2]
        else:
            # Integral of the survival mask gives the expected boundary height.
            # No per-column bright-peak search and no Gaussian heatmap supervision.
            y = (1 - regions).mean(2)
            y = F.interpolate(y, size=w, mode='linear', align_corners=False)
        return {'y': y * (h - 1), 'regions': regions}


def build_model(method, backbone, assets=None):
    return BoundaryNet(method, backbone, assets)
