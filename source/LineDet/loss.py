# -*- coding: utf-8 -*-
"""Loss functions for layered line detection with heatmap regression."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import generate_gaussian_heatmap


def coord_mse_loss(pred_y, gt_y, valid_mask):
    diff = (pred_y - gt_y).pow(2).masked_select(valid_mask)
    if diff.numel() == 0:
        return torch.tensor(0.0, device=pred_y.device)
    return diff.mean()


def heatmap_mse_loss(pred_heatmaps, gt_heatmaps, valid_mask):
    B, L, H, W = pred_heatmaps.shape
    valid_expanded = valid_mask.unsqueeze(2).expand(-1, -1, H, -1)
    diff = (pred_heatmaps - gt_heatmaps).pow(2) * valid_expanded
    valid_count = valid_expanded.sum()
    if valid_count == 0:
        return torch.tensor(0.0, device=pred_heatmaps.device)
    return diff.sum() / valid_count


def heatmap_bce_loss(pred_heatmaps, gt_heatmaps, valid_mask):
    B, L, H, W = pred_heatmaps.shape
    valid_expanded = valid_mask.unsqueeze(2).expand(-1, -1, H, -1)
    pred_probs = torch.sigmoid(pred_heatmaps)
    bce_loss = F.binary_cross_entropy(pred_probs, gt_heatmaps, reduction='none') * valid_expanded
    valid_count = valid_expanded.sum()
    if valid_count == 0:
        return torch.tensor(0.0, device=pred_heatmaps.device)
    return bce_loss.sum() / valid_count


def topology_loss(y, margin=0.0):
    if y.size(1) < 2:
        return torch.tensor(0.0, device=y.device)
    diffs = y[:, :-1] - y[:, 1:] + margin
    return F.relu(diffs).mean()


def smoothness_loss(y):
    if y.size(-1) < 3:
        return torch.tensor(0.0, device=y.device)
    y_l, y_c, y_r = y[..., :-2], y[..., 1:-1], y[..., 2:]
    return (y_l - 2 * y_c + y_r).abs().mean()


def tv_loss(y):
    """一阶总变分损失，抑制相邻像素差突变。"""
    if y.size(-1) < 2:
        return torch.tensor(0.0, device=y.device)
    return (y[..., 1:] - y[..., :-1]).abs().mean()


def false_region_smooth(pred_valid_logits, valid_mask, eps=1e-6):
    """只在 valid_mask == False 的区域做平滑正则化。"""
    mask_false = (valid_mask == 0).float()
    p = torch.sigmoid(pred_valid_logits)
    if p.shape[-1] < 2:
        return torch.tensor(0., device=p.device)

    diff = p[..., 1:] - p[..., :-1]
    mask_adj = mask_false[..., 1:] * mask_false[..., :-1]
    denom_adj = mask_adj.sum()
    if denom_adj.item() > 0:
        return ((diff ** 2) * mask_adj).sum() / (denom_adj + eps)
    return torch.tensor(0., device=p.device)


class HeatmapCompositeLoss(nn.Module):
    """基于 heatmap 的复合损失函数。"""
    def __init__(self, lambda_heatmap=1.0, lambda_coord=0.1, lambda_topo=0.5, lambda_smooth=0.01,
                 heatmap_loss_type='bce', sigma=2.0):
        super().__init__()
        self.lambda_heatmap = lambda_heatmap
        self.lambda_coord = lambda_coord
        self.lambda_topo = 1e-2
        self.lambda_smooth = 0.1
        self.lambda_tv = 0.05
        self.heatmap_loss_type = heatmap_loss_type
        self.sigma = 3.0

    def forward(self, pred_y, gt_y, valid_mask, pred_heatmaps):
        B, L, H, W = pred_heatmaps.shape

        gt_heatmaps = generate_gaussian_heatmap(gt_y, valid_mask, H, W, self.sigma)

        if self.heatmap_loss_type == 'mse':
            pred_heatmap_probs = F.softmax(pred_heatmaps, dim=2)
            Lheatmap = heatmap_mse_loss(pred_heatmap_probs, gt_heatmaps, valid_mask)
        elif self.heatmap_loss_type == 'bce':
            Lheatmap = heatmap_bce_loss(pred_heatmaps, gt_heatmaps, valid_mask)
        else:
            raise ValueError(f"Unsupported heatmap loss type: {self.heatmap_loss_type}")

        Lcoord = coord_mse_loss(pred_y, gt_y, valid_mask)
        Ltopo = topology_loss(pred_y)
        Lsmooth = smoothness_loss(pred_y)
        Ltv = tv_loss(pred_y)

        total = (self.lambda_heatmap * Lheatmap +
                 self.lambda_coord * Lcoord +
                 self.lambda_topo * Ltopo +
                 self.lambda_smooth * Lsmooth +
                 self.lambda_tv * Ltv)

        return total, {
            'total': total,
            'heatmap': Lheatmap.detach(),
            'coord': Lcoord.detach(),
            'topo': Ltopo.detach(),
            'smooth': Lsmooth.detach(),
            'tv': Ltv.detach(),
        }


class CompositeLoss(nn.Module):
    """保持兼容的非 heatmap 复合损失。"""
    def __init__(self, lambda_coord=1.0, lambda_topo=0.5, lambda_smooth=0.01, lambda_dice=1.0):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_topo = 1e-4
        self.lambda_smooth = 1e-4
        self.DiceCELossFunc = nn.Module()

    def forward(self, pred_y, gt_y, valid_mask):
        Lcoord = coord_mse_loss(pred_y, gt_y, valid_mask)
        Ltopo = topology_loss(pred_y)
        Lsmooth = smoothness_loss(pred_y)
        total = Lcoord + self.lambda_topo * Ltopo + self.lambda_smooth * Lsmooth
        return total, {
            'total': total,
            'coord': Lcoord.detach(),
            'topo': Ltopo.detach(),
            'smooth': Lsmooth.detach(),
        }


@torch.no_grad()
def line_mae(pred_y, gt_y, valid_mask):
    """计算线条的平均绝对误差。"""
    diff = (pred_y - gt_y).abs().masked_select(valid_mask)
    if diff.numel() == 0:
        return float('nan')
    return diff.mean().item()
