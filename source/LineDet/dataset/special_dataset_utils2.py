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

def add_diffraction_strips(img, y_px, radius_range=(5,20), num_range=(2,4),
                                        sigma_factor=(1.6, 1.8), y_noise=0.00):
    """
    img  : [1,H,W] z-score float tensor (均值0，方差1)
    y_px : [N,W] float tensor
    
    功能：
        - 在上方1/6区域添加随机衍射条
        - 中心高亮 3px，向上下 radius 扩散平方衰减
        - 强度按 sigma_factor 乘以 std (z-score)
        - y_px 可选微调 ±y_noise
    """
    assert img.dim() == 3, "img must be [1,H,W]"
    assert y_px.dim() == 2, "y_px must be [N,W]"
    
    C, H, W = img.shape
    N, W2 = y_px.shape
    assert W == W2, "Width mismatch"
    
    aug_img = img.clone()
    new_y = y_px.clone()
    
    limit_y = H // 6
    num_strips = random.randint(*num_range)
    radius = random.randint(*radius_range)
    for _ in range(num_strips):
        # 随机中心位置
        cy = random.uniform(10, limit_y-1)
        
        y_min = max(0, int(cy - radius))
        y_max = min(H, int(cy + radius))
        if y_min >= y_max:
            continue
        
        # 垂直方向距离中心
        y_indices = torch.arange(y_min, y_max, device=img.device, dtype=torch.float32)
        dist = torch.abs(y_indices - cy)
        
        # 平方衰减
        intensity = (1.0 - dist / radius)**2
        intensity = torch.clamp(intensity, 0.0, 1.0)
        intensity[dist <= 1.5] = 1.0  # 中心高亮3px
        
        # 随机强度 (z-score 标准化单位)
        sigma_val = random.uniform(*sigma_factor)
        strip_pattern = intensity.view(-1,1) * sigma_val
        strip_pattern = strip_pattern.expand(-1, W).unsqueeze(0)  # [1,H_strip,W]
        
        # 叠加到图像
        aug_img[:, y_min:y_max, :] += strip_pattern
        
        # -----------------------------
        # 微调 y_px
        # mask 仅对条覆盖区域进行微小偏移
        # -----------------------------
        mask = (new_y >= y_min) & (new_y <= y_max)
        noise = (torch.rand_like(new_y) - 0.5) * 2 * y_noise
        new_y[mask] = new_y[mask] + noise[mask]
    
    return aug_img, new_y


def smooth_noise_1d(W: int, num_freqs: int = 4, device='cpu', dtype=torch.float32) -> torch.Tensor:
    """
    生成 [W] 的平滑随机噪声，通过多频正弦叠加模拟 Perlin noise。
    值域约 [-1, 1]（未严格归一化，调用处自行 clamp/scale）。
    """
    noise = torch.zeros(W, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, W, device=device, dtype=dtype)
    for k in range(1, num_freqs + 1):
        amp   = 1.0 / k                            # 高频幅度衰减
        freq  = k * (1 + torch.randint(1, 4, (1,)).item())
        phase = torch.rand(1).item() * 2 * 3.14159
        noise += amp * torch.sin(2 * 3.14159 * freq * xs + phase)
    noise = noise / noise.abs().max().clamp(min=1e-6)  # 归一化到 [-1,1]
    return noise


def smooth_noise_2d(H: int, W: int, num_freqs: int = 4,
                    device='cpu', dtype=torch.float32) -> torch.Tensor:
    """
    生成 [H, W] 的平滑 2D 随机噪声，x/y 独立频率叠加。
    """
    noise_x = smooth_noise_1d(W, num_freqs, device, dtype)   # [W]
    noise_y = smooth_noise_1d(H, num_freqs, device, dtype)   # [H]
    # 外积：每列由 y-noise 调制，每行由 x-noise 调制
    noise_2d = noise_y.unsqueeze(1) * noise_x.unsqueeze(0)   # [H, W]
    noise_2d = noise_2d / noise_2d.abs().max().clamp(min=1e-6)
    return noise_2d


def tissue_nonuniform_deform(
    img: torch.Tensor,
    y_px: torch.Tensor,
    index_top: int,
    index_btn: int,
    scale: float,
    # ── 非均匀噪声参数 ──────────────────────────
    scale_noise_amp: float = 0.15,   # 列间拉伸比例的随机波动幅度（相对 scale-1）
    lateral_amp: float = 0.5,        # 横向（x）位移幅度，单位 px
    fat_ratio: float = 0.3,          # 区域内顶部视为脂肪层的比例（脂肪更软）
    fat_softness: float = 2.0,       # 脂肪层相对肌肉层的额外形变倍率
    num_freqs: int = 4,              # 噪声频率数，越大细节越丰富
    seed: int = None,
):
    """
    在 regional_vertical_deform 的基础上叠加非均匀组织形变：
    
    1. 列间非均匀拉伸：每列 scale 用平滑噪声微扰
    2. 层内深度权重：脂肪（浅层）比肌肉（深层）形变更大
    3. 横向剪切扰动：模拟组织侧向滑移
    
    参数
    ----
    img            : [C, H, W]
    y_px           : [N, W]  原始坐标（会被同步更新）
    index_top/btn  : 区域上下界行索引
    scale          : 基础拉伸比例
    scale_noise_amp: 列间 scale 波动，0 = 均匀
    lateral_amp    : 横向扰动幅度（px），0 = 无横向位移
    fat_ratio      : 区域顶部占比视为脂肪（更容易形变）
    fat_softness   : 脂肪相对肌肉的额外形变倍率
    num_freqs      : 平滑噪声频率层数
    seed           : 随机种子，None 为随机
    """
    assert img.dim() == 3 and y_px.dim() == 2
    C, H, W = img.shape
    N, W2   = y_px.shape
    assert W == W2

    if seed is not None:
        torch.manual_seed(seed)

    device = img.device
    dtype  = torch.float32

    # ════════════════════════════════════════════════
    # 1. 区域边界
    # ════════════════════════════════════════════════
    y_top = y_px[index_top].clone().float()   # [W]
    y_bot = y_px[index_btn].clone().float()   # [W]
    h_old = (y_bot - y_top).clamp(min=1e-6)

    # ════════════════════════════════════════════════
    # 2. 列间非均匀 scale：每列有独立的拉伸比例
    #    scale_col[w] ∈ [scale - amp*(scale-1), scale + amp*(scale-1)]
    #    使 scale=1 时无扰动（形变为0，噪声也为0）
    # ════════════════════════════════════════════════
    noise_col = smooth_noise_1d(W, num_freqs, device, dtype)  # [W] ∈ [-1,1]
    delta_scale = scale_noise_amp * (scale - 1.0)             # 波动幅度随形变量缩放
    scale_col   = scale + noise_col * delta_scale             # [W] 列间变化的局部 scale
    scale_col   = scale_col.clamp(min=0.1)                    # 防止负值

    h_new       = h_old * scale_col                           # [W] 各列新高度
    y_bot_new   = y_top + h_new                               # [W]
    delta_bot   = y_bot_new - y_bot                           # [W] 下方平移量

    # ════════════════════════════════════════════════
    # 3. 深度权重：区域内 [y_top, y_bot] 归一化深度 t∈[0,1]
    #    脂肪在顶部（t < fat_ratio），形变更大
    #    weight(t) = fat_softness  if t < fat_ratio
    #              = 1.0           otherwise
    #    最终 scale_eff(y, w) = 1 + (scale_col[w]-1) * weight(t(y,w))
    # ════════════════════════════════════════════════
    def depth_weight(t: torch.Tensor) -> torch.Tensor:
        """t: 归一化深度 [0,1]，返回形变权重"""
        w = torch.ones_like(t)
        w[t < fat_ratio] = fat_softness
        return w

    # ════════════════════════════════════════════════
    # 4. 横向噪声场 [H, W]：模拟组织侧向滑移
    # ════════════════════════════════════════════════
    noise_lat = smooth_noise_2d(H, W, num_freqs, device, dtype)  # [H,W] ∈ [-1,1]

    # ════════════════════════════════════════════════
    # 5. 构建反向映射场（供 grid_sample）
    #    对输出像素 (ys, xs)，求其在原图的来源位置 (src_y, src_x)
    # ════════════════════════════════════════════════
    ys = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)

    y_top_b     = y_top.view(1, W).expand(H, W)
    y_bot_new_b = y_bot_new.view(1, W).expand(H, W)
    y_bot_b     = y_bot.view(1, W).expand(H, W)
    scale_col_b = scale_col.view(1, W).expand(H, W)
    delta_bot_b = delta_bot.view(1, W).expand(H, W)

    # 归一化深度 t（仅中间区域有意义）
    region_h_new_b = (y_bot_new_b - y_top_b).clamp(min=1e-6)
    t_map = (ys - y_top_b) / region_h_new_b     # [H,W]，区域外无意义但不影响 mask

    # 深度权重
    dw = depth_weight(t_map)                     # [H,W]

    # 有效局部 scale（含深度权重调制）
    # scale_eff = 1 + (scale_col - 1) * dw
    scale_eff = 1.0 + (scale_col_b - 1.0) * dw  # [H,W]
    scale_eff = scale_eff.clamp(min=0.1)

    # ── 反向映射 src_y ──────────────────────────────
    src_y = torch.zeros_like(ys)

    mask_top = ys < y_top_b
    mask_mid = (ys >= y_top_b) & (ys <= y_bot_new_b)
    mask_bot = ys > y_bot_new_b

    # 上方：不变
    src_y[mask_top] = ys[mask_top]

    # 中间：用 scale_eff 做非均匀反向映射
    # 前向：dst = y_top + (src - y_top) * scale_eff  →  反向：src = y_top + (dst - y_top) / scale_eff
    src_y[mask_mid] = (y_top_b[mask_mid]
                       + (ys[mask_mid] - y_top_b[mask_mid]) / scale_eff[mask_mid])

    # 下方：平移
    src_y[mask_bot] = ys[mask_bot] - delta_bot_b[mask_bot]

    # ── 横向扰动 src_x ──────────────────────────────
    # 仅在区域内施加，强度随深度权重调制，并在区域边界平滑衰减
    lateral_mask_strength = torch.zeros(H, W, device=device, dtype=dtype)
    if lateral_amp > 0:
        # 区域内按 sin 窗平滑衰减（边界为0，中央为1）
        t_clamped = t_map.clamp(0, 1)
        sin_window = torch.sin(t_clamped * 3.14159).clamp(0, 1)  # [H,W]
        lateral_mask_strength[mask_mid] = sin_window[mask_mid]
        lateral_mask_strength[mask_bot] = 1.0   # 下方也跟着整体侧移

    src_x = xs + lateral_amp * noise_lat * lateral_mask_strength

    # ════════════════════════════════════════════════
    # 6. grid_sample
    # ════════════════════════════════════════════════
    grid_x = (src_x / (W - 1)) * 2 - 1
    grid_y = (src_y / (H - 1)) * 2 - 1
    grid   = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0)  # [1,H,W,2]

    new_img = F.grid_sample(
        img.unsqueeze(0), grid,
        mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(0)

    # ════════════════════════════════════════════════
    # 7. y_px 前向映射（原始坐标 → 变形后坐标）
    # ════════════════════════════════════════════════
    new_y = y_px.clone().float()
    for i in range(N):
        y_i = y_px[i].float()

        mask_top_i = y_i < y_top
        mask_mid_i = (y_i >= y_top) & (y_i <= y_bot)
        mask_bot_i = y_i > y_bot

        # 上方不变
        new_y[i, mask_top_i] = y_i[mask_top_i]

        # 中间：归一化深度 → 深度权重 → 前向拉伸
        if mask_mid_i.any():
            t_i   = (y_i[mask_mid_i] - y_top[mask_mid_i]) / h_old[mask_mid_i]  # [k] ∈[0,1]
            dw_i  = depth_weight(t_i)                                            # [k]
            sc_i  = 1.0 + (scale_col[mask_mid_i] - 1.0) * dw_i                 # 列间scale + 深度权重
            sc_i  = sc_i.clamp(min=0.1)
            new_y[i, mask_mid_i] = y_top[mask_mid_i] + (y_i[mask_mid_i] - y_top[mask_mid_i]) * sc_i

        # 下方：整体平移
        new_y[i, mask_bot_i] = y_i[mask_bot_i] + delta_bot[mask_bot_i]

    new_y = new_y.clamp(0, H - 1)
    return new_img, new_y

def special_augmentation2(img, y_px):
    """
    随机从多种增强中选择一个执行，返回增强后的 img 和 y_px。
    """
    N = y_px.shape[0]
    funcs = [
        lambda i, y: crop_upper_and_stretch(i, y, rand_ratio=0.125),
        lambda i, y: compress_height_with_pad(i, y, rand_ratio=0.25),
        lambda i, y: crop_lower_and_stretch(i, y, rand_ratio=0.25),
        lambda i, y: add_diffraction_strips(i, y,
                                             radius_range=(5,20),
                                             num_range=(2,4),
                                             sigma_factor=(1.4,2.0),
                                             y_noise=0.00),
    ]

    if N >= 2:
        funcs.append(lambda i, y: tissue_nonuniform_deform(i, y,
                                             index_top=N-2, index_btn=N-1, scale=random.uniform(0.5, 2.0)))
    if N >= 3:
        funcs.append(lambda i, y: tissue_nonuniform_deform(i, y,
                                             index_top=N-3, index_btn=N-2, scale=random.uniform(0.5, 2.0)))
        funcs.append(lambda i, y: tissue_nonuniform_deform(i, y,
                                             index_top=N-3, index_btn=N-1, scale=random.uniform(0.5, 2.0)))

    # 随机选择一个增强函数执行
    func = random.choice(funcs)
    new_img, new_y = func(img, y_px)
    return new_img, new_y