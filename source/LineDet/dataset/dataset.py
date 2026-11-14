import os
import random
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.abstract_transforms import Compose

from .dataset_utils import JointAugment, generate_heatmap_targets
from .special_dataset_utils import special_enhancement
from .special_dataset_utils2 import special_augmentation2
from .line_utils import parse_djm_annotation, resize_y_and_valid
from .split_manifest import load_manifest_pairs
from .utils import find_image_annotation_pairs

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 统一尺寸配置（仅在内存中变换，不改原始数据集）：
#   LINEDET_UNIFY_MODE = none | crop | resize | unify    默认 none
#   LINEDET_UNIFY_SIZE = "W,H"                   默认 516,594
#   crop  ：超出目标的部分裁掉（宽度裁右、高度裁下），窄/矮的保持原尺寸
#   resize：整体拉伸到目标尺寸，标注按比例缩放
#   unify ：只按目标宽高比(H/W)裁掉底部多余高度（不做缩放），再统一拉伸到模型 img_size。
#           横边物理尺度相同的前提下，把各尺寸裁到同一物理宽高比；516×594 的图不裁。
#           标注同步（y>=裁高 的列置 invalid）。
UNIFY_MODE = os.environ.get("LINEDET_UNIFY_MODE", "none").strip().lower()
UNIFY_SIZE = tuple(int(x) for x in os.environ.get("LINEDET_UNIFY_SIZE", "516,594").split(","))


def crop_annotation(y_px_list, valid_list, new_w, new_h):
    """裁剪标注：x 超过 new_w 的列去掉；y >= new_h（落在被裁掉的底部）的列置为 invalid。"""
    y_px_arr = np.asarray(y_px_list, dtype=float)
    valid_arr = np.asarray(valid_list, dtype=bool)
    if y_px_arr.ndim == 1:
        y_px_arr = y_px_arr[None, :]
        valid_arr = valid_arr[None, :]
    keep_x = np.arange(min(new_w, y_px_arr.shape[1]))
    y_px_arr = y_px_arr[:, keep_x]
    valid_arr = valid_arr[:, keep_x]
    out_of_h = y_px_arr >= new_h
    y_px_arr = np.where(out_of_h, 0.0, y_px_arr)
    valid_arr = valid_arr & ~out_of_h
    return y_px_arr, valid_arr


def crop_would_keep_target_lines(ann_path, img_size, crop_h):
    """模拟 unify 裁剪后各目标线是否仍有有效列。

    某条目标线在裁剪高度 crop_h 之下（y >= crop_h）完全无标注点时，
    裁剪版将失去该线的监督，此时不应生成裁剪副本。解析失败时返回 True
    （保持原行为，不因额外判断丢数据）。
    """
    try:
        y_px, valid = parse_djm_annotation(ann_path, img_size)
    except Exception:
        return True
    if y_px.ndim == 1:
        y_px = y_px[None, :]
        valid = valid[None, :]
    _, valid_cropped = crop_annotation(y_px, valid, img_size[0], crop_h)
    for r in range(valid_cropped.shape[0]):
        if not valid_cropped[r].any():
            return False
    return True


class LineDataset(Dataset):
    def __init__(self, data_root: str, img_size=(256, 256), mode="train",
                 sigma: float = 2.0, augment: bool = True,
                 split_manifest: str = None):
        self.data_root = data_root
        self.img_size = img_size
        self.sigma = sigma
        self.augment = augment
        self.mode = mode

        if split_manifest:
            pairs = load_manifest_pairs(self.data_root, split_manifest, mode)
        else:
            pairs = find_image_annotation_pairs(self.data_root)
        self.img_paths = [p[0] for p in pairs]
        self.ann_paths = [p[1] for p in pairs]
        print(f"找到 {len(self.img_paths)} 个有效的图像-标注对")

        if not split_manifest:
            split_ratio = 0.9
            total_len = len(self.img_paths)
            train_len = int(total_len * split_ratio)
            if mode == "train":
                pass
            elif mode == 'all':
                pass
            else:
                self.img_paths = self.img_paths[train_len:]
                self.ann_paths = self.ann_paths[train_len:]

        # ---- unify 模式：原图 + 裁剪版两份都参与训练 ----
        # 516×594 不需要裁剪，只保留原图；其余(如 422×728)保留原图 + 追加一个裁剪版。
        # 若裁剪会把某条目标线(如脂肪层下界)整条裁没，则该图不追加裁剪版(只保留原图)。
        if UNIFY_MODE == "unify":
            tgt_w, tgt_h = UNIFY_SIZE
            new_imgs, new_anns, new_flags = [], [], []
            n_crop = 0
            n_crop_skipped = 0
            for ip, ap in zip(self.img_paths, self.ann_paths):
                new_imgs.append(ip); new_anns.append(ap); new_flags.append(False)
                with Image.open(ip) as im:
                    w, h = im.size
                crop_h = int(w * tgt_h / tgt_w)
                if h > crop_h:
                    if crop_would_keep_target_lines(ap, (w, h), crop_h):
                        new_imgs.append(ip); new_anns.append(ap); new_flags.append(True)
                        n_crop += 1
                    else:
                        n_crop_skipped += 1
            self.img_paths = new_imgs
            self.ann_paths = new_anns
            self.crop_flags = new_flags
            print(f'[unify] 裁剪扩充: 追加 {n_crop} 个裁剪版, 因整条目标线被裁没跳过 {n_crop_skipped} 个, 总样本 {len(new_flags)}')
        else:
            self.crop_flags = [False] * len(self.img_paths)

        print(f'Now mode is: {mode}', len(self.img_paths))

        if mode == "train":
            tr_transforms = []
            tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
            tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True,
                                                      p_per_sample=0.2, p_per_channel=0.5))
            tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
            tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
            tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                                p_per_channel=0.5, order_downsample=0, order_upsample=3,
                                                                p_per_sample=0.25, ignore_axes=None))
            tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
            tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))
            self.transform = Compose(tr_transforms)
            self.JointAugment = JointAugment()
        else:
            self.transform = transforms.ToTensor()
            self.JointAugment = None

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]

        image = Image.open(img_path).convert('L')
        original_size = image.size

        y_px_list, valid_list = parse_djm_annotation(ann_path, (original_size[0], original_size[1]))

        # ---- 统一尺寸（内存中变换，不动原始数据集） ----
        if UNIFY_MODE == "crop":
            new_w = min(original_size[0], UNIFY_SIZE[0])
            new_h = min(original_size[1], UNIFY_SIZE[1])
            image = image.crop((0, 0, new_w, new_h))
            unified_size = (new_w, new_h)
            y_px_list, valid_list = crop_annotation(y_px_list, valid_list, new_w, new_h)
        elif UNIFY_MODE == "resize":
            image = image.resize(UNIFY_SIZE, Image.LANCZOS)
            unified_size = UNIFY_SIZE
            y_px_list, valid_list = resize_y_and_valid(y_px_list, valid_list, original_size, UNIFY_SIZE)
        elif UNIFY_MODE == "unify":
            if self.crop_flags[idx]:
                orig_w, orig_h = original_size
                tgt_w, tgt_h = UNIFY_SIZE
                crop_h = int(orig_w * tgt_h / tgt_w)  # 按目标宽高比(H/W)裁高
                crop_h = min(orig_h, crop_h)
                image = image.crop((0, 0, orig_w, crop_h))
                y_px_list, valid_list = crop_annotation(y_px_list, valid_list, orig_w, crop_h)
                unified_size = (orig_w, crop_h)
            else:
                unified_size = original_size
        else:
            unified_size = original_size

        y_px_list, valid_list = resize_y_and_valid(y_px_list, valid_list, unified_size, self.img_size)
        y_px = torch.from_numpy(y_px_list.astype(np.float32))
        valid = torch.from_numpy(valid_list.astype(np.bool))

        image_np = np.array(image.resize(self.img_size, Image.LANCZOS))
        if self.mode == 'train' and random.random() < 0.2:
            image_np = special_enhancement(image_np)

        if self.mode == 'train':
            img = np.array(image_np)[None] / 255.0
            img = self.transform(data=img)['data']
            img = torch.from_numpy(np.array(img, dtype=np.float32))
        else:
            img = self.transform(image_np)

        img_mean = img.mean()
        img_std = img.std()
        img = (img - img_mean) / (img_std + 1e-7)

        if self.JointAugment is not None:
            img, y_px, valid = self.JointAugment(img, y_px, valid)
            if random.random() < 0.3:
                img, y_px = special_augmentation2(img, y_px)

        heatmap_tensor = generate_heatmap_targets(y_px, valid, self.img_size[1], self.img_size[0], self.sigma)
        return {
            'image': img,
            'heatmaps': heatmap_tensor,
            'y_px': y_px,
            'valid': valid,
            'name': img_path,
            'ann_path': ann_path,
        }


def default_collate(batch):
    imgs = torch.stack([b['image'] for b in batch])
    y = torch.stack([b['y_px'] for b in batch])
    v = torch.stack([b['valid'] for b in batch])
    names = [b['name'] for b in batch]

    result = {
        'image': imgs,
        'y_px': y,
        'valid': v,
        'name': names,
    }

    if 'heatmaps' in batch[0]:
        heatmaps = torch.stack([b['heatmaps'] for b in batch])
        result['heatmaps'] = heatmaps

    return result
