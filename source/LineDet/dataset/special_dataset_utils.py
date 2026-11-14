import cv2
import numpy as np
import random

def local_transparent_blackening(img, black_ratio=0.15, num_regions=3, alpha=0.6):
    h, w = img.shape[:2]
    img_copy = img.copy()
    total_area = h * w
    single_block_area = int(total_area * black_ratio)
    
    for _ in range(num_regions):
        block_w = random.randint(int(np.sqrt(single_block_area)*0.5), int(np.sqrt(single_block_area)*1.5))
        block_h = int(single_block_area / block_w)
        x1 = random.randint(0, max(0, w - block_w - 1))
        y1 = random.randint(0, max(0, h - block_h - 1))
        x2 = x1 + block_w
        y2 = y1 + block_h
        
        img_block = img_copy[y1:y2, x1:x2].copy()
        black_block = np.zeros_like(img_block, dtype=np.uint8)
        
        transparent_block = cv2.addWeighted(img_block, (1 - alpha), black_block, alpha, gamma=0)
        img_copy[y1:y2, x1:x2] = transparent_block
    
    return img_copy

def enhance_contrast(img, enhance_range=(1.5, 2.5), reduce_range=(0.4, 0.8), beta=0):
    img_copy = img.copy()
    if random.random() > 0.5:
        alpha = random.uniform(*enhance_range)
    else:
        alpha = random.uniform(*reduce_range)
    img_contrast = cv2.convertScaleAbs(img_copy, alpha=alpha, beta=beta)
    return img_contrast

def partial_occlusion(img, occlusion_ratio=0.1, shape="random"):
    h, w = img.shape[:2]
    img_copy = img.copy()
    total_area = h * w
    occlusion_area = int(total_area * occlusion_ratio)
    
    if shape == "random":
        shape = random.choice(["rect", "circle"])
    
    if shape == "rect":
        rect_w = random.randint(int(np.sqrt(occlusion_area)*0.5), int(np.sqrt(occlusion_area)*1.5))
        rect_h = int(occlusion_area / rect_w)
        x1 = random.randint(0, max(0, w - rect_w - 1))
        y1 = random.randint(0, max(0, h - rect_h - 1))
        x2 = x1 + rect_w
        y2 = y1 + rect_h
        occlusion_color = (random.randint(30, 80), random.randint(30, 80), random.randint(30, 80))
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), occlusion_color, -1)
    elif shape == "circle":
        radius = int(np.sqrt(occlusion_area / np.pi))
        center_x = random.randint(radius, w - radius)
        center_y = random.randint(radius, h - radius)
        occlusion_color = (random.randint(30, 80), random.randint(30, 80), random.randint(30, 80))
        cv2.circle(img_copy, (center_x, center_y), radius, occlusion_color, -1)
    return img_copy

def local_contrast_jump(img, jump_ratio_range=(1.0, 3.0), num_regions=2):
    h, w = img.shape[:2]
    img_copy = img.copy()
    region_area_ratio = 0.1
    single_region_area = int(h * w * region_area_ratio)
    
    for _ in range(num_regions):
        region_w = random.randint(int(np.sqrt(single_region_area)*0.6), int(np.sqrt(single_region_area)*1.4))
        region_h = int(single_region_area / region_w)
        x1 = random.randint(0, max(0, w - region_w - 1))
        y1 = random.randint(0, max(0, h - region_h - 1))
        x2 = x1 + region_w
        y2 = y1 + region_h
        
        region = img_copy[y1:y2, x1:x2].copy()
        jump_ratio = random.uniform(*jump_ratio_range)
        region_jump = cv2.convertScaleAbs(region, alpha=jump_ratio, beta=0)
        img_copy[y1:y2, x1:x2] = region_jump
    return img_copy


def special_enhancement(img):
    funcs = [
        lambda x: local_transparent_blackening(x, alpha=0.7, num_regions=2),
        enhance_contrast,
        lambda x: partial_occlusion(x, shape="rect"),
        local_contrast_jump,
    ]
    func = random.choice(funcs)
    img = func(img)
    return img

