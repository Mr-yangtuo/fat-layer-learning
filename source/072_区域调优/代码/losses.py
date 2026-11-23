import torch
from torch.nn import functional as F


def masked_mean(values, valid):
    return (values * valid).sum() / valid.sum().clamp_min(1)


def targets(y, valid, shape, height=256):
    h, w = shape
    yy = F.interpolate(y, size=w, mode='linear', align_corners=False) / (height - 1)
    vv = F.interpolate(valid.float(), size=w, mode='nearest') > 0.5
    # Region pixels represent bin centers, so area integral agrees with normalized coordinate.
    grid = (torch.arange(h, device=y.device).float() + 0.5) / h
    # Anti-aliased step targets, approximately one source pixel transition.
    region = ((grid.view(1,1,h,1) - yy[:,:,None]) * h + 0.5).clamp(0,1)
    return region, vv[:,:,None].expand(-1,-1,h,-1)


def supervised(output, y, valid, settings=None):
    settings = settings or {}
    pred = output['y'].float()
    weights = torch.tensor(settings.get('line_weights', [1.,1.]), device=y.device).view(1,2,1)
    coord = masked_mean(F.smooth_l1_loss(pred, y, reduction='none', beta=settings.get('coord_beta',2.0)), valid*weights)
    adjacent = valid[:,:,1:] & valid[:,:,:-1]
    slope = masked_mean(F.smooth_l1_loss(pred.diff(dim=-1), y.diff(dim=-1), reduction='none', beta=1.0), adjacent)
    both = valid.all(1)
    thickness = masked_mean(F.smooth_l1_loss(pred[:,1]-pred[:,0], y[:,1]-y[:,0], reduction='none', beta=2), both)
    regions = output['regions'].float().clamp(1e-5, 1-1e-5)
    target, mask = targets(y, valid, regions.shape[-2:])
    # Nested probabilities are products of sigmoids, so a single-logit BCE is not applicable.
    # Compute probability BCE explicitly in FP32 outside AMP (PyTorch forbids BCE under autocast).
    with torch.autocast(device_type=regions.device.type, enabled=False):
        bce = masked_mean(F.binary_cross_entropy(regions, target.float(), reduction='none'), mask)
    mono_mask = mask[:,:,1:] & mask[:,:,:-1]
    monotonic = masked_mean(F.relu(-regions.diff(dim=2)), mono_mask)
    return (coord + settings.get('slope_weight',.25)*slope + settings.get('thickness_weight',.15)*thickness
            + settings.get('region_weight',2.)*bce + .5*monotonic)


def distillation(student, teacher, y, valid):
    # Confidence gates are based only on the TRAINING annotation after identical augmentation.
    # Bad teacher points do not pull a supervised student toward the teacher's mistake.
    weight = torch.exp(-(teacher['y'] - y).abs().detach() / 8) * valid
    coord = masked_mean(F.smooth_l1_loss(student['y'], teacher['y'].detach(), reduction='none', beta=2), weight)
    s = student['regions'].float().clamp(1e-5,1-1e-5)
    t = teacher['regions'].detach().float().clamp(1e-5,1-1e-5)
    gate = F.interpolate(weight, size=s.shape[-1], mode='linear', align_corners=False)[:,:,None].expand_as(s)
    kl = t*(t.log()-s.log()) + (1-t)*((1-t).log()-(1-s).log())
    return coord + masked_mean(kl, gate)
