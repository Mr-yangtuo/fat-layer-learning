"""Four independently adapted boundary representations; no old heatmap head is reused."""
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
        base = getattr(models, name)(weights=None)
        if assets is not None:
            base.load_state_dict(torch.load(Path(assets)/WEIGHTS[name], map_location='cpu', weights_only=True))
        self.name = name
        if name == 'resnet18':
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.stages = nn.ModuleList([base.layer1, base.layer2, base.layer3, base.layer4])
            self.channels = [64, 128, 256, 512]
        else:
            self.features = base.features
            self.indices, self.channels = ([3, 6, 12, 16], [24, 40, 112, 960]) if name.endswith('large') else ([1, 3, 8, 12], [16, 24, 48, 576])

    def forward(self, x):
        outputs = []
        if self.name == 'resnet18':
            x = self.stem(x)
            for stage in self.stages:
                x = stage(x); outputs.append(x)
        else:
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self.indices: outputs.append(x)
        return outputs


def separable(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
                         nn.Conv2d(cin, cout, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())


class Decoder(nn.Module):
    def __init__(self, channels, width, concatenate=False):
        super().__init__()
        self.concatenate = concatenate
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in channels])
        self.top = separable(width, width)
        self.fuse = nn.ModuleList([separable(width * (2 if concatenate else 1), width) for _ in range(3)])

    def forward(self, features):
        x = self.top(self.lateral[3](features[3]))
        for i in (2, 1, 0):
            skip = self.lateral[i](features[i])
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = self.fuse[i](torch.cat([x, skip], 1) if self.concatenate else x+skip)
        return x


def spline_basis(controls, width=256, degree=3):
    """Clamped uniform B-spline basis; nonnegative and partition of unity."""
    knots = torch.cat([torch.zeros(degree+1), torch.linspace(0, 1, controls-degree+1)[1:-1], torch.ones(degree+1)])
    x = torch.linspace(0, 1, width).clamp_max(1-1e-6)
    basis = ((x[None] >= knots[:-1,None]) & (x[None] < knots[1:,None])).float()
    for d in range(1, degree+1):
        terms = []
        for i in range(len(knots)-d-1):
            a = knots[i+d]-knots[i]; b = knots[i+d+1]-knots[i+1]
            first = (x-knots[i])/a*basis[i] if a > 0 else torch.zeros_like(x)
            second = (knots[i+d+1]-x)/b*basis[i+1] if b > 0 else torch.zeros_like(x)
            terms.append(first+second)
        basis = torch.stack(terms)
    basis[:,-1] = 0; basis[-1,-1] = 1
    return basis


def regions_from_y(y, height=64):
    y = F.interpolate(y, size=64, mode='linear', align_corners=False)
    grid = (torch.arange(height, device=y.device, dtype=y.dtype)+.5)/height
    return torch.sigmoid((grid[None,None,:,None]-y[:,:,None,:])/.02)


class BoundaryNet(nn.Module):
    def __init__(self, method, backbone='resnet18', assets=None, settings=None):
        super().__init__()
        settings = settings or {}
        self.method = method
        self.encoder = Encoder(backbone, assets)
        width = int(settings.get('width', 64))
        self.decoder = Decoder(self.encoder.channels, width, concatenate=method=='signed_distance')
        self.register_buffer('mean', torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).view(1,3,1,1))
        if method == 'signed_distance':
            self.field_head = nn.Sequential(separable(width+2, width), nn.Conv2d(width, 2, 1))
            self.temperature = float(settings.get('distance_temperature', .06))
        elif method == 'spline_queries':
            controls = int(settings.get('controls', 16))
            self.query = nn.Parameter(torch.randn(1, controls, width)*.02)
            self.position = nn.Conv2d(2, width, 1)
            self.attention = nn.ModuleList([nn.MultiheadAttention(width, 4, dropout=.1, batch_first=True) for _ in range(2)])
            self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
            self.ffns = nn.ModuleList([nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width)) for _ in range(2)])
            self.control_head = nn.Linear(width, 3)
            self.register_buffer('basis', spline_basis(controls))
        elif method == 'iterative_refine':
            self.reduce = nn.Conv2d(width, 16, 1)
            self.initial = nn.Sequential(nn.Conv1d(128, width, 3, padding=1), nn.SiLU(), nn.Conv1d(width, 3, 1))
            self.sample_projection = nn.Conv1d(width*10+3, width, 1)
            self.update = nn.Sequential(nn.Conv1d(width, width, 5, padding=2, groups=width),
                                        nn.GroupNorm(8,width), nn.SiLU(), nn.Conv1d(width, width, 1),
                                        nn.SiLU(), nn.Conv1d(width, 3, 1))
            self.steps = int(settings.get('refine_steps', 3))
            self.radius = float(settings.get('sample_radius', .04))
        elif method == 'axial_distribution':
            self.reduce = nn.Conv2d(width, 8, 1)
            # Explicit dropout uses the checkpointed PyTorch RNG. cuDNN RNN dropout
            # keeps an opaque state that cannot be restored from CUDA RNG alone.
            self.sequence_dropout = nn.Dropout(.1)
            self.sequence = nn.GRU(128, width, num_layers=2, bidirectional=True, batch_first=True, dropout=0.)
            self.bins = int(settings.get('bins', 128))
            self.classifier = nn.Conv1d(width*2, 2*self.bins, 1)
        else:
            raise ValueError(method)

    def forward(self, image):
        h, w = image.shape[-2:]
        x = (image.expand(-1,3,-1,-1)-self.mean)/self.std
        x = self.decoder(self.encoder(x))
        b, c, fh, fw = x.shape
        yy = (torch.arange(fh,device=x.device,dtype=x.dtype)+.5)/fh
        xx = torch.linspace(0,1,fw,device=x.device,dtype=x.dtype)
        position = torch.stack([yy[:,None].expand(fh,fw), xx[None].expand(fh,fw)],0)[None].expand(b,-1,-1,-1)
        result = {'method': self.method}
        if self.method == 'signed_distance':
            field = self.field_head(torch.cat([x,position],1)).float().tanh()
            grid = yy.float()[None,None,:,None]
            weights = torch.softmax(-field.abs()/self.temperature, dim=2)
            raw = ((grid-field)*weights).sum(2).clamp(0,1)
            raw = F.interpolate(raw,size=w,mode='linear',align_corners=False)
            y = raw.sort(dim=1).values
            result.update(field=field, raw_y=raw*(h-1))
            regions = torch.sigmoid(field/.025)
        elif self.method == 'spline_queries':
            memory = F.adaptive_avg_pool2d(x+self.position(position),(16,64)).flatten(2).transpose(1,2)
            q = self.query.expand(b,-1,-1)
            for attn, norm, ffn in zip(self.attention,self.norms,self.ffns):
                q = norm(q+attn(q,memory,memory,need_weights=False)[0]);q=q+ffn(q)
            control = self.control_head(q).float().softmax(-1).cumsum(-1)[...,:2].transpose(1,2)
            basis = self.basis if w==256 else spline_basis(control.shape[-1],w).to(control)
            y = control@basis
            result['controls'] = control*(h-1)
            regions = regions_from_y(y)
        elif self.method == 'iterative_refine':
            logits = self.initial(F.adaptive_avg_pool2d(self.reduce(x),(8,64)).flatten(1,2)).float()
            trajectory = []
            for step in range(self.steps+1):
                curves = logits.softmax(1).cumsum(1)[:,:2]
                trajectory.append(F.interpolate(curves,size=w,mode='linear',align_corners=False)*(h-1))
                if step == self.steps: break
                offsets = torch.linspace(-self.radius,self.radius,5,device=x.device)
                rows = curves[:,:,None,:]+offsets[None,None,:,None]
                cols = torch.linspace(-1,1,64,device=x.device)[None,None,None,:].expand_as(rows)
                sample_grid = torch.stack([cols,rows*2-1],-1).reshape(b,10,64,2)
                sampled = F.grid_sample(x.float(),sample_grid,mode='bilinear',padding_mode='border',align_corners=True)
                sequence = torch.cat([sampled.reshape(b,c*10,64),curves,curves[:,1:]-curves[:,:1]],1)
                logits = logits+.5*self.update(F.silu(self.sample_projection(sequence))).float().tanh()
            y = trajectory[-1]/(h-1)
            result['trajectory'] = trajectory
            regions = regions_from_y(y)
        else:
            seq = F.adaptive_avg_pool2d(self.reduce(x),(16,64)).flatten(1,2).transpose(1,2)
            seq,_ = self.sequence(self.sequence_dropout(seq.float()))
            logits = self.classifier(seq.transpose(1,2)).float()
            logits = F.interpolate(logits,size=w,mode='linear',align_corners=False).reshape(b,2,self.bins,w)
            probs = logits.softmax(2)
            heights = torch.linspace(0,1,self.bins,device=x.device)[None,None,:,None]
            raw = (probs*heights).sum(2)
            y = raw.sort(dim=1).values
            result.update(distribution_logits=logits,raw_y=raw*(h-1))
            regions = F.interpolate(probs.cumsum(2),size=(64,64),mode='bilinear',align_corners=False)
        result.update(y=y*(h-1),regions=regions)
        return result


def build_model(method, backbone, assets=None, settings=None):
    return BoundaryNet(method, backbone, assets, settings)
