import torch
import torch.nn.functional as F
import random

def crop_upper_and_stretch(img, y_px, x=None, rand_ratio=0.125):
    """
    img  : [1, H, W]
    y_px : [N, W]

    x=None 时随机生成：
        x ∈ [5, H * rand_ratio]

    rand_ratio 默认 0.125 (≈ H/8)
    """

    assert img.dim() == 3, "img must be [1, H, W]"
    assert y_px.dim() == 2, "y_px must be [N, W]"

    _, H, W = img.shape
    N, W2 = y_px.shape
    assert W == W2, "Width mismatch"

    # -------------------------
    # 随机 x
    # -------------------------
    if x is None:
        max_x = max(6, int(H * rand_ratio))
        x = random.randint(5, max_x)

    if x <= 0 or x >= H:
        return img, y_px

    # -------------------------
    # 1️⃣ 裁剪图像
    # -------------------------
    cropped = img[:, :H - x, :]          # [1, H-x, W]
    cropped = cropped.unsqueeze(0)       # [1,1,H-x,W]

    # -------------------------
    # 2️⃣ 拉伸回原尺寸
    # -------------------------
    stretched = F.interpolate(
        cropped,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    new_img = stretched.squeeze(0)       # [1,H,W]

    # -------------------------
    # 3️⃣ 同步变换 y 坐标
    # -------------------------
    scale = H / (H - x)
    new_y = y_px * scale
    new_y = torch.clamp(new_y, 0, H - 1)

    return new_img, new_y


def compress_height_with_pad(img, y_px, x=None, rand_ratio=0.25):
    """
    img  : [1, H, W]
    y_px : [N, W]

    高度压缩到 H-x
    顶部填充黑条
    同步更新 y_px

    x=None 时随机:
        x ∈ [5, H*rand_ratio]
    """

    assert img.dim() == 3, "img must be [1, H, W]"
    assert y_px.dim() == 2, "y_px must be [N, W]"

    C, H, W = img.shape
    N, W2 = y_px.shape
    assert W == W2, "Width mismatch"

    # -------------------------
    # 随机 x
    # -------------------------
    if x is None:
        max_x = max(6, int(H * rand_ratio))
        x = random.randint(5, max_x)

    if x <= 0 or x >= H:
        return img, y_px

    # -------------------------
    # 1️⃣ 压缩图像到 H-x
    # -------------------------
    new_h = H - x

    img_4d = img.unsqueeze(0)  # [1,1,H,W]

    resized = F.interpolate(
        img_4d,
        size=(new_h, W),
        mode="bilinear",
        align_corners=False
    )

    resized = resized.squeeze(0)  # [1,new_h,W]

    # -------------------------
    # 2️⃣ 创建黑色画布
    # -------------------------
    padded = torch.zeros_like(img)

    # 放到底部
    padded[:, x:H, :] = resized

    # -------------------------
    # 3️⃣ 同步变换 y_px
    # -------------------------
    scale = (H - x) / H

    new_y = y_px * scale + x
    new_y = torch.clamp(new_y, 0, H - 1)

    return padded, new_y

def crop_lower_and_stretch(img, y_px, x=None, rand_ratio=0.25):
    """
    img  : [1, H, W]
    y_px : [N, W]

    丢弃顶部 x 像素
    拉伸回原高度
    保证不裁掉最上面那条线
    """

    assert img.dim() == 3
    assert y_px.dim() == 2

    C, H, W = img.shape
    N, W2 = y_px.shape
    assert W == W2

    # -------------------------
    # 计算最上面那条线的最小值
    # -------------------------
    top_line_min = torch.min(y_px[0]).item()

    # 最大允许裁剪
    max_safe_x = int(top_line_min)

    if max_safe_x <= 1:
        return img, y_px

    # -------------------------
    # 随机 x
    # -------------------------
    if x is None:
        max_rand = min(int(H * rand_ratio), max_safe_x)
        if max_rand < 1:
            return img, y_px
        x = random.randint(1, max_rand)

    # 强制安全
    x = min(x, max_safe_x)

    if x <= 0 or x >= H:
        return img, y_px

    # -------------------------
    # 1️⃣ 裁剪
    # -------------------------
    cropped = img[:, x:H, :]
    cropped = cropped.unsqueeze(0)  # [1,1,H-x,W]

    # -------------------------
    # 2️⃣ 拉伸
    # -------------------------
    stretched = F.interpolate(
        cropped,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    new_img = stretched.squeeze(0)

    # -------------------------
    # 3️⃣ 同步 y
    # -------------------------
    scale = H / (H - x)

    new_y = (y_px - x) * scale
    new_y = torch.clamp(new_y, 0, H - 1)

    return new_img, new_y

