# -*- coding: utf-8 -*-
"""Dataset utilities for layered line detection."""
import random
import torch


class JointAugment:
    def __init__(self, out_size=(256, 256),
                 p_flip=0.5, p_erase=0.2, p_resize_crop=0.1, p_vertical_distort=0.15):
        self.out_size = out_size
        self.p_flip = p_flip
        self.p_erase = p_erase
        self.p_resize_crop = p_resize_crop
        self.p_vertical_distort = p_vertical_distort

    def __call__(self, img, y_px, valid):
        if random.random() < self.p_flip:
            img = torch.flip(img, dims=[2])
            y_px = torch.flip(y_px, dims=[1])
            valid = torch.flip(valid, dims=[1])

        if random.random() < self.p_erase:
            _, h, w = img.shape
            erase_width = random.randint(0, w // 4)
            side = random.choice(["left", "right"])
            if side == "left":
                img[:, :, :erase_width] = 0
                y_px[:, :erase_width] = y_px[:, erase_width][:, None]
            elif side == "right":
                img[:, :, -erase_width:] = 0
                y_px[:, -erase_width:] = y_px[:, -erase_width - 1][:, None]

        if random.random() < 0.2:
            min_per_line = y_px.min(dim=1).values
            min_all = float(min_per_line.min().item())
            max_shift_allowed = int(min_all) - 3
            max_shift = max(0, max_shift_allowed)

            if max_shift >= 1:
                shift = random.randint(1, max_shift)
                img = torch.roll(img, shifts=-shift, dims=1)
                img[:, -shift:, :] = 0
                y_px = y_px - shift
                y_px = y_px.clamp(min=0)
        else:
            if random.random() < 0.2:
                H = img.shape[1]
                max_y = float(y_px.max().item())
                max_shift_down = int(H - max_y - 1)
                shift_min = 5
                shift_max = 30
                shift_max = min(shift_max, max_shift_down)

                if shift_max >= shift_min:
                    shift = random.randint(shift_min, shift_max)
                    img = torch.roll(img, shifts=-shift, dims=1)
                    img[:, :shift, :] = 0
                    y_px = y_px - shift
                    y_px = y_px.clamp(min=0)

        return img, y_px, valid


def generate_heatmap_targets(y_px, valid, image_height, image_width, sigma=2.0):
    num_lines = len(y_px)
    heatmaps = torch.zeros(num_lines, image_height, image_width)

    y_coords = torch.arange(image_height).float().unsqueeze(1)
    x_coords = torch.arange(image_width).float().unsqueeze(0)

    for line_idx in range(num_lines):
        if valid.dim() == 1:
            if not valid[line_idx].item():
                continue
        else:
            if not valid[line_idx].any().item():
                continue

        for x in range(image_width):
            if valid.dim() == 2 and not valid[line_idx, x].item():
                continue

            if y_px.dim() == 1:
                center_y = y_px[line_idx].item()
            else:
                center_y = y_px[line_idx, x].item()

            gaussian = torch.exp(-((y_coords[:, 0] - center_y) ** 2) / (2 * sigma ** 2))
            heatmaps[line_idx, :, x] = gaussian

    return heatmaps
