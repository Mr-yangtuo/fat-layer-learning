import json
import numpy as np
from typing import List, Tuple, Optional

# 目标线（按从上到下的顺序，作为模型输出的通道顺序）
TARGET_LABELS = ['真皮层下界', '脂肪层下界']

# 标注里出现过的别名/错别字，映射到正确标签
LABEL_ALIASES = {
    '线条 2': '表皮层下界',
}

def _canonical_label(label):
    if label in LABEL_ALIASES:
        return LABEL_ALIASES[label]
    return label

def select_target_lines(data):
    """按标签选取目标线（真皮层下界/脂肪层下界），
    忽略 表皮层上界/表皮层下界 等其他线；同一标签重复时取第一个。
    返回按 TARGET_LABELS 顺序排列的 items 列表。
    """
    selected = {}
    for line in data.get('items', []):
        label = _canonical_label(line.get('label', ''))
        if label in TARGET_LABELS and label not in selected:
            if line.get('data'):
                selected[label] = line
    return [selected[l] for l in TARGET_LABELS if l in selected]

def parse_djm_annotation(
    json_path: str,
    original_size: Tuple[int, int],
    max_lines: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lines = select_target_lines(data)
    
    y_px_list: List[np.ndarray] = []
    valid_list: List[np.ndarray] = []

    orig_w, orig_h = original_size

    for line in lines:
        points = line.get('data', [])
        if not points:
            continue
        per_x_ys = {}
        for p in points:
            x = p.get('x', None)
            y = p.get('y', None)
            if x is None or y is None:
                continue
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            xi = int(round(float(x)))
            if xi < 0 or xi >= orig_w:
                continue
            yi = float(y)
            if yi < 0 or yi > (orig_h - 1):
                continue
            per_x_ys.setdefault(xi, []).append(yi)

        y_coords = np.zeros(orig_w, dtype=float)
        valid_mask = np.zeros(orig_w, dtype=bool)
        for xi, y_list in per_x_ys.items():
            y_coords[xi] = float(np.mean(y_list))
            valid_mask[xi] = True

        y_px_list.append(y_coords)
        valid_list.append(valid_mask)

    return np.stack(y_px_list, axis=0), np.stack(valid_list, axis=0)


def parse_annotation_valid(json_path: str) -> bool:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels = [_canonical_label(i.get('label', '')) for i in data.get('items', [])]
    if all(t in labels for t in TARGET_LABELS):
        return True
    else:
        print('ERROR annotations:', json_path)
        return False


def resize_y_and_valid(y_px_list: np.ndarray, valid_list: np.ndarray, original_size, target_size) -> Tuple[np.ndarray, np.ndarray]:
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]
    
    y_px_arr = np.asarray(y_px_list, dtype=float)
    valid_arr = np.asarray(valid_list, dtype=bool)

    if y_px_arr.ndim == 1:
        y_px_arr = y_px_arr[None, :]
    if valid_arr.ndim == 1:
        valid_arr = valid_arr[None, :]

    new_y_px_list: List[np.ndarray] = []
    new_valid_list: List[np.ndarray] = []

    for line_idx in range(int(y_px_arr.shape[0])):
        y = y_px_arr[line_idx]
        valid = valid_arr[line_idx]

        width_old = int(y.shape[0])
        width_new = int(target_size[0])

        x_old = np.linspace(0, original_size[0] - 1, width_old, dtype=float)
        x_new = np.linspace(0, target_size[0] - 1, width_new, dtype=float)
        x_query = x_new / scale_x

        known = valid & np.isfinite(y)
        if not np.any(known):
            new_y_px_list.append(np.zeros(width_new, dtype=float))
            new_valid_list.append(np.zeros(width_new, dtype=bool))
            continue

        x_known = x_old[known]
        y_known = y[known]
        order = np.argsort(x_known)
        x_known = x_known[order]
        y_known = y_known[order]

        y_new = np.interp(x_query, x_known, y_known, left=y_known[0], right=y_known[-1]) * scale_y

        if x_known.shape[0] <= 1:
            valid_new = np.ones(width_new, dtype=bool)
        else:
            valid_new = (x_query >= x_known[0]) & (x_query <= x_known[-1])

        new_y_px_list.append(y_new)
        new_valid_list.append(valid_new)

    if len(new_y_px_list) == 0:
        width_new = int(target_size[0])
        return np.zeros((0, width_new), dtype=float), np.zeros((0, width_new), dtype=bool)

    return np.stack(new_y_px_list, axis=0), np.stack(new_valid_list, axis=0)

