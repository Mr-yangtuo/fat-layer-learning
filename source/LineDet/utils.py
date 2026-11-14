import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def smooth_y_px(y_px, kernel_size=5, sigma=1.5):
    if isinstance(y_px, np.ndarray):
        y_px = torch.from_numpy(y_px)
    orig_dim = y_px.dim()
    if orig_dim == 1:
        y_px = y_px.unsqueeze(0).unsqueeze(0)
    elif orig_dim == 2:
        y_px = y_px.unsqueeze(0)
    B, L, W = y_px.shape
    half = kernel_size // 2
    gauss = torch.arange(kernel_size, device=y_px.device, dtype=torch.float32) - half
    gauss = torch.exp(-(gauss ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    gauss = gauss.view(1, 1, -1).repeat(L, 1, 1)
    y_pad = F.pad(y_px, (half, half), mode='replicate')
    y_smooth = F.conv1d(y_pad, gauss, groups=L)
    if orig_dim == 1:
        return y_smooth.squeeze(0).squeeze(0)
    elif orig_dim == 2:
        return y_smooth.squeeze(0)
    return y_smooth


@torch.no_grad()
def infer_lines(model, image, use_heatmap=False, smooth=True, smooth_kernel=5, smooth_sigma=1.5):
    """使用模型推理线条，返回 {'y_px', 'heatmaps'}。"""
    model.eval()

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    if image.dim() == 3:
        image = image.unsqueeze(0)

    outputs = model(image)
    if isinstance(outputs, tuple):
        y_px, heatmaps = outputs[0], outputs[1] if len(outputs) > 1 else None
    else:
        y_px, heatmaps = outputs, None
    if smooth:
        y_px = smooth_y_px(y_px, kernel_size=smooth_kernel, sigma=smooth_sigma)
    return {'y_px': y_px, 'heatmaps': heatmaps}


def visualize_heatmaps(heatmaps, gt_y_px=None, pred_y_px=None, save_path="heatmaps.png"):
    """可视化热力图。"""
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]

    num_channels = heatmaps.shape[0]
    fig, axes = plt.subplots(2, (num_channels + 1) // 2, figsize=(15, 8))
    if num_channels == 1:
        axes = [axes]
    axes = axes.flatten()

    for i in range(num_channels):
        ax = axes[i]
        im = ax.imshow(heatmaps[i], cmap='hot', interpolation='nearest')
        ax.set_title(f'Heatmap {i+1}')
        ax.axis('off')

        if gt_y_px is not None:
            gt_y_px_np = gt_y_px.cpu().numpy() if isinstance(gt_y_px, torch.Tensor) else gt_y_px
            if gt_y_px_np.ndim == 2 and i < gt_y_px_np.shape[0]:
                y_coords = gt_y_px_np[i]
                valid = y_coords >= 0
                if valid.sum() > 0:
                    ax.plot(np.arange(len(y_coords))[valid], y_coords[valid], 'g-', linewidth=2, alpha=0.8, label='GT')

        if pred_y_px is not None:
            pred_y_px_np = pred_y_px.cpu().numpy() if isinstance(pred_y_px, torch.Tensor) else pred_y_px
            if pred_y_px_np.ndim == 2 and i < pred_y_px_np.shape[0]:
                y_coords = pred_y_px_np[i]
                valid = y_coords >= 0
                if valid.sum() > 0:
                    ax.plot(np.arange(len(y_coords))[valid], y_coords[valid], 'r--', linewidth=2, alpha=0.8, label='Pred')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if gt_y_px is not None or pred_y_px is not None:
            ax.legend()

    for i in range(num_channels, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_pred_gt(img_tensor, gt_y_px, pred_y_px, gt_mask=None, pred_seg=None,
                      heatmaps=None, save_path="inference_gt.png"):
    """显示 GT 线条和预测线条在图像上。"""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    if img_tensor.shape[0] == 1:
        img_np = img_tensor.squeeze(0).cpu().numpy()
    else:
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    gt_y_px = gt_y_px.cpu().numpy() if isinstance(gt_y_px, torch.Tensor) else gt_y_px
    pred_y_px = pred_y_px.cpu().numpy() if isinstance(pred_y_px, torch.Tensor) else pred_y_px

    gt_y_px = gt_y_px.squeeze(0)
    pred_y_px = pred_y_px.squeeze(0)

    plt.figure(figsize=(12, 5))
    colors = ['red', 'blue', 'green', 'yellow', 'cyan']

    plt.subplot(1, 2, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title("GT Lines")
    for i in range(gt_y_px.shape[0]):
        y_coords = gt_y_px[i]
        x_coords = np.arange(len(y_coords))
        valid = (y_coords >= 0) & (y_coords < img_np.shape[0])
        if valid.sum() > 0:
            plt.plot(x_coords[valid], y_coords[valid], color=colors[i % len(colors)], linewidth=2, alpha=0.6,
                     label=f'GT Line {i+1}')
    plt.legend()
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_np, cmap='gray')
    plt.title("Predicted Lines")
    for i in range(pred_y_px.shape[0]):
        y_coords = pred_y_px[i]
        x_coords = np.arange(len(y_coords))
        valid = (y_coords >= 0) & (y_coords < img_np.shape[0])
        if valid.sum() > 0:
            plt.plot(x_coords[valid], y_coords[valid], color=colors[i % len(colors)], linewidth=2, alpha=0.6,
                     label=f'Pred Line {i+1}')
    plt.legend()
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
