# -*- coding: utf-8 -*-
"""Main training script for layered line detection with heatmap support."""
import os
import argparse
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from alive_progress import alive_bar
from tabulate import tabulate
import mlflow
import mlflow.pytorch

from LineDet.model import LineCoordNetFPN
from LineDet.dataset import LineDataset, default_collate
from LineDet.loss import CompositeLoss, HeatmapCompositeLoss, line_mae
from LineDet.utils import infer_lines, visualize_pred_gt, visualize_heatmaps


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_split_summary(path, args, train_set, val_set):
    train_sources = {os.path.realpath(item) for item in train_set.img_paths}
    val_sources = {os.path.realpath(item) for item in val_set.img_paths}
    payload = {
        'split_mode': args.split_mode,
        'split_manifest': os.path.realpath(args.split_manifest) if args.split_manifest else None,
        'train_items': len(train_set),
        'val_items': len(val_set),
        'train_unique_source_images': len(train_sources),
        'val_unique_source_images': len(val_sources),
        'source_image_overlap': len(train_sources & val_sources),
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Split summary: {payload}")


def threshold_accuracy(pred, gt, thresholds=(2, 3, 5)):
    """pred, gt: Tensor of shape (B, L, W)。按像素误差阈值统计准确率。"""
    errors = torch.abs(pred - gt)
    metrics = {}
    for t in thresholds:
        acc = (errors <= t).float().mean().item() * 100  # 百分比
        metrics[f"A@{t}"] = acc
    return metrics


def hausdorff_distance(pred, gt, percentile=95):
    """pred, gt: Tensor of shape (B, L, W)。返回 Hausdorff 距离和 HD95。"""
    B, L, W = pred.shape
    hd_list, hd95_list = [], []

    for b in range(B):
        for l in range(L):
            p = pred[b, l].view(-1)
            g = gt[b, l].view(-1)

            diff = torch.abs(p[:, None] - g[None, :])
            d1 = diff.min(dim=1)[0]  # 每个 pred 点到最近 gt 点
            d2 = diff.min(dim=0)[0]  # 每个 gt 点到最近 pred 点
            hd = torch.max(torch.max(d1), torch.max(d2)).item()
            both = torch.cat([d1, d2], dim=0)
            hd95 = torch.quantile(both, percentile / 100.0).item()
            hd_list.append(hd)
            hd95_list.append(hd95)

    return {
        "Hausdorff": sum(hd_list) / len(hd_list),
        "HD95": sum(hd95_list) / len(hd95_list),
    }


def train_one_epoch(model, loader, optimizer, loss_fn, device, use_heatmap=True):
    model.train()
    total_loss = 0

    with alive_bar(len(loader), force_tty=True, title="Training") as bar:
        for batch in loader:
            img = batch['image'].to(device)
            gt_y = batch['y_px'].to(device)
            valid = batch['valid'].to(device)

            optimizer.zero_grad()

            if use_heatmap:
                gt_heatmaps = batch['heatmaps'].to(device)
                pred_y, pred_heatmaps = model(img)
                loss, loss_dict = loss_fn(pred_y, gt_y, valid, pred_heatmaps)
            else:
                pred_y = model(img)
                loss, loss_dict = loss_fn(pred_y, gt_y, valid)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            bar()

    return total_loss / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device, use_heatmap=True):
    model.eval()
    logs = []

    with alive_bar(len(loader), force_tty=True, title="Evaluating") as bar:
        for batch in loader:
            img = batch['image'].to(device)
            gt_y = batch['y_px'].to(device)
            valid = batch['valid'].to(device)

            if use_heatmap:
                gt_heatmaps = batch['heatmaps'].to(device)
                pred_y, pred_heatmaps = model(img)
                loss, loss_dict = loss_fn(pred_y, gt_y, valid, pred_heatmaps)
            else:
                pred_y = model(img)
                loss, loss_dict = loss_fn(pred_y, gt_y, valid)

            mae = line_mae(pred_y, gt_y, valid)
            TA = threshold_accuracy(pred_y, gt_y)
            HD = hausdorff_distance(pred_y, gt_y)

            log_entry = {
                'loss': loss.item(),
                'mae': mae,
                'coord_loss': loss_dict['coord'].item(),
                'topo_loss': loss_dict['topo'].item(),
                'smooth_loss': loss_dict['smooth'].item(),
                **TA,
                **HD,
            }
            if use_heatmap and 'heatmap' in loss_dict:
                log_entry['heatmap_loss'] = loss_dict['heatmap'].item()

            logs.append(log_entry)
            bar()

    out = {k: float(np.nanmean([d[k] for d in logs])) for k in logs[0].keys()}
    return out


def parse_img_size(s):
    """Accept '256' (square) or 'W,H' (e.g. '516,594')."""
    s = str(s).strip()
    if "," in s:
        w, h = s.split(",")
        return int(w.strip()), int(h.strip())
    v = int(s)
    return v, v


def main():
    parser = argparse.ArgumentParser(description='line-style detection training')
    parser.add_argument('--img_size', type=str, default='256', help='Input image size: "256" or "W,H" (e.g. 516,594)')
    parser.add_argument('--num_lines', type=int, default=2, help='Number of lines')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--backbone', default='mobilenet_v3_large', help='mobilenet_v3_large | mobilenetv4_conv_medium')
    parser.add_argument('--backbone_pretrained', action='store_true', default=False, help='Use torchvision pretrained backbone weights')
    parser.add_argument('--save_dir', default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--use_heatmap', action='store_true', default=True, help='Use heatmap architecture')
    parser.add_argument('--heatmap_sigma', type=float, default=2.0, help='Sigma for heatmap generation')
    parser.add_argument('--data_root', default='/home/zlg/zlg/00000腹部数据集（仅仅腹部+有垫片+无垫片）/无垫/', help='Data root directory (images/annotations or subdirs)')
    parser.add_argument('--split_mode', choices=['train100', 'train90'], default='train100')
    parser.add_argument('--split_manifest', default=None, help='Directory containing meta.json, train.jsonl and val.jsonl')
    parser.add_argument('--seed', type=int, default=20260925, help='Random seed shared by both comparison runs')
    parser.add_argument('--skip_export', action='store_true', help='Skip PTE export and runtime validation after training')
    args = parser.parse_args()
    args.img_size = parse_img_size(args.img_size)
    if args.split_mode == 'train90' and not args.split_manifest:
        parser.error('--split_manifest is required for train90')
    if args.split_mode == 'train100' and args.split_manifest:
        parser.error('--split_manifest must not be used with train100')
    set_global_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)

    train_set = LineDataset(args.data_root, mode='train',
                            img_size=args.img_size,
                            sigma=args.heatmap_sigma,
                            split_manifest=args.split_manifest)
    val_set = LineDataset(args.data_root, mode='val',
                          img_size=args.img_size,
                          sigma=args.heatmap_sigma,
                          split_manifest=args.split_manifest)
    write_split_summary(os.path.join(args.save_dir, 'split_summary.json'), args, train_set, val_set)

    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=default_collate,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=default_collate,
        generator=val_generator,
    )

    model = LineCoordNetFPN(
        backbone=args.backbone,
        num_lines=args.num_lines,
    ).to(args.device)
    print(model)

    if args.use_heatmap:
        loss_fn = HeatmapCompositeLoss(sigma=args.heatmap_sigma).to(args.device)
    else:
        loss_fn = CompositeLoss().to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_mae = float('inf')
    best_epoch = 0
    best_metrics = None
    best_state_dict = None
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)

        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, args.device, args.use_heatmap)
        ev = eval_one_epoch(model, val_loader, loss_fn, args.device, args.use_heatmap)

        headers = ['Train Loss', 'Val Loss', 'MAE (px)', 'Coord Loss', 'Topo Loss', 'Smooth Loss', 'A@2', 'A@3', 'A@5', 'Hausdorff', 'HD95']
        table_data = [[
            f"{tr_loss:.4f}",
            f"{ev['loss']:.4f}",
            f"{ev['mae']:.2f}",
            f"{ev['coord_loss']:.4f}",
            f"{ev['topo_loss']:.4f}",
            f"{ev['smooth_loss']:.4f}",
            f"{ev['A@2']:.4f}",
            f"{ev['A@3']:.4f}",
            f"{ev['A@5']:.4f}",
            f"{ev['Hausdorff']:.4f}",
            f"{ev['HD95']:.4f}",
        ]]
        if args.use_heatmap and 'heatmap_loss' in ev:
            headers.append('Heatmap Loss')
            table_data[0].append(f"{ev['heatmap_loss']:.4f}")

        print(tabulate(table_data, headers=headers, tablefmt='grid'))

        try:
            metrics = {
                "train.loss": float(tr_loss),
                "val.loss": float(ev['loss']),
                "val.mae_px": float(ev['mae']) if not np.isnan(ev['mae']) else None,
                "loss.coord": float(ev['coord_loss']),
                "loss.topo": float(ev['topo_loss']),
                "loss.smooth": float(ev['smooth_loss']),
            }
            if args.use_heatmap and 'heatmap_loss' in ev:
                metrics["loss.heatmap"] = float(ev['heatmap_loss'])
            metrics = {k: v for k, v in metrics.items() if v is not None and np.isfinite(v)}
            mlflow.log_metrics(metrics, step=epoch)
        except Exception as e:
            print(f"[Warn] MLflow metric logging failed: {e}")

        if ev['mae'] < best_mae:
            best_mae = ev['mae']
            best_epoch = epoch
            best_metrics = {
                'epoch': epoch,
                'best_metric': 'mae',
                'best_metric_value': float(best_mae),
                'val_mae': float(best_mae),
                'val_loss': float(ev['loss']),
                'train_loss': float(tr_loss),
                'split_mode': args.split_mode,
            }
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'mae': best_mae,
                'loss': ev['loss'],
                'args': args,
            }, os.path.join(args.save_dir, 'best.pth'))
            with open(os.path.join(args.save_dir, 'best_metrics.json'), 'w', encoding='utf-8') as handle:
                json.dump(best_metrics, handle, ensure_ascii=False, indent=2)
            print(f"Best model saved with MAE: {best_mae:.2f} px")
            try:
                mlflow.log_metric("best.mae_px", float(best_mae))
                mlflow.log_metric("best.epoch", int(epoch))
            except Exception as e:
                print(f"[Warn] MLflow best metric logging failed: {e}")

        if epoch % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'loss': ev['loss'],
                'args': args,
            }, os.path.join(args.save_dir, f'epoch_{epoch}.pth'))

        for idx, batch in enumerate(val_loader):
            if idx >= 5:
                break
            img = batch['image'][:1].to(args.device)
            out = infer_lines(model, img, use_heatmap=args.use_heatmap)
            save_path = os.path.join(log_dir, f"epoch_{epoch}_val_pred_gt_{idx}.png")
            heatmaps = out.get('heatmaps', None)
            visualize_pred_gt(
                img.cpu(),
                batch['y_px'][:1].cpu(),
                out['y_px'][:1].cpu(),
                heatmaps=heatmaps[:1].cpu() if heatmaps is not None else None,
                save_path=save_path,
            )
            if args.use_heatmap and heatmaps is not None:
                heatmap_save_path = os.path.join(log_dir, f"epoch_{epoch}_heatmaps_{idx}.png")
                visualize_heatmaps(
                    heatmaps[:1].cpu(),
                    gt_y_px=batch['y_px'][:1].cpu(),
                    pred_y_px=out['y_px'][:1].cpu(),
                    save_path=heatmap_save_path,
                )

    print("\nTraining completed!")
    print(f"Best validation MAE: {best_mae:.2f} px")
    print(f"Best epoch: {best_epoch}")

    if args.skip_export:
        print('Skipping PTE export and runtime validation (--skip_export). Training checkpoints are saved.')
    else:
        print('Loading best model to export')
        print('status:', model.load_state_dict(best_state_dict))
        from LineDet.export import export_pte_model
        export_pte_model(model, args.save_dir, img_size=args.img_size)

    try:
        if os.path.isdir(args.save_dir):
            mlflow.log_artifacts(args.save_dir)
            best_pte = os.path.join(args.save_dir, "best.pte")
            if os.path.isfile(best_pte):
                mlflow.log_artifact(best_pte, artifact_path="exports")
        if os.path.isdir(log_dir):
            mlflow.log_artifacts(log_dir)
        mlflow.pytorch.log_model(model, artifact_path="pytorch-model", registered_model_name="LineDet")
    except Exception as e:
        print(f"[Warn] MLflow artifact/model logging failed: {e}")
    finally:
        mlflow.end_run()


if __name__ == '__main__':
    main()
