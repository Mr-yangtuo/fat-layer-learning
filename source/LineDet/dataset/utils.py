import os
from typing import List, Tuple

from .line_utils import parse_annotation_valid

# 需要整体排除的子目录（按目录名匹配，含该名的子路径全部跳过）
# 由环境变量 LINEDET_EXCLUDE_DIRS 传入（竖线 | 分隔多个），在 train减脂.sh 中配置
EXCLUDE_DIRS = set(x.strip() for x in os.environ.get("LINEDET_EXCLUDE_DIRS", "").split("|") if x.strip())

def collect_pairs(img_dir: str, ann_dir: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not (os.path.isdir(img_dir) and os.path.isdir(ann_dir)):
        return pairs

    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith('.png') or f.lower().endswith('.jpg') :
            base = os.path.splitext(f)[0]
            img_path = os.path.join(img_dir, f)
            ann_path = os.path.join(ann_dir, base + '.json')
            if os.path.exists(ann_path) and parse_annotation_valid(ann_path):
                pairs.append((img_path, ann_path))
    return pairs


def find_image_annotation_pairs(data_root: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    for root, dirs, _ in os.walk(data_root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        if 'images' in dirs and 'annotations' in dirs:
            img_dir = os.path.join(root, 'images')
            ann_dir = os.path.join(root, 'annotations')
            pairs.extend(collect_pairs(img_dir, ann_dir))

    return pairs
