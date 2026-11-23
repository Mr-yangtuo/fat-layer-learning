"""直接读取原始图像，保留旧裁剪逻辑；不保存磁盘图像缓存。"""
import json
import random
from functools import lru_cache
from PIL import Image
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from legacy_crops import crop_upper_and_stretch, crop_lower_and_stretch, compress_height_with_pad


def augment(image, y, valid, settings=None):
    settings = settings or {}
    _, h, w = image.shape
    if random.random() < 0.5:
        image, y, valid = image.flip(-1), y.flip(-1), valid.flip(-1)
    # Keep all three original crop/resize operations (30% chance to choose one).
    if random.random() < 0.3:
        kind = random.randrange(3)
        if kind == 0:
            amount = random.randint(5, max(6, int(h * 0.125)))
            new_valid = valid & (y < h - amount) & (y * h / (h-amount) <= h-1)
            if new_valid.any(1).all():
                image, y = crop_upper_and_stretch(image, y, x=amount)
                valid = new_valid
        elif kind == 1:
            amount = random.randint(5, max(6, int(h * 0.25)))
            image, y = compress_height_with_pad(image, y, x=amount)
        else:
            maximum = min(int(h * 0.25), int(y[0].min().item()))
            if maximum >= 1:
                amount = random.randint(1, maximum)
                new_valid = valid & (y >= amount) & ((y-amount) * h / (h-amount) <= h-1)
                if new_valid.any(1).all():
                    image, y = crop_lower_and_stretch(image, y, x=amount)
                    valid = new_valid
    if random.random() < 0.2:
        image = image * random.uniform(0.8, 1.2)
    if random.random() < 0.2:
        mean = image.mean()
        image = (image - mean) * random.uniform(0.8, 1.2) + mean
    if random.random() < 0.15:
        image = image.clamp(0, 1).pow(random.uniform(0.8, 1.25))
    if random.random() < settings.get('noise_probability', .15):
        image = image + image * torch.randn_like(image) * random.uniform(0.03, 0.12)
    if random.random() < settings.get('blur_probability', .1):
        image = F.avg_pool2d(image.unsqueeze(0), 3, 1, 1).squeeze(0)
    if random.random() < 0.1:
        small = random.randint(int(h * 0.65), h-1)
        image = F.interpolate(F.interpolate(image[None], size=(small, small), mode='bilinear', align_corners=False),
                              size=(h, w), mode='bilinear', align_corners=False)[0]
    # Low-frequency gain field and a faint false bright ridge, masked away from true boundaries.
    if random.random() < settings.get('ridge_probability', .15):
        row = torch.arange(h, dtype=torch.float32).view(h, 1)
        col = torch.linspace(0, 2 * torch.pi, w)
        center = random.uniform(0.1*h, 0.9*h) + random.uniform(1, 7) * torch.sin(col + random.uniform(0, 6))
        ridge = torch.exp(-0.5 * ((row - center[None]) / random.uniform(1.5, 3))**2)
        distance = (row[None] - y[:, None]).abs().amin(0)
        safe = valid.all(0)[None] & (distance > 8)
        image = image + ridge[None] * safe[None] * random.uniform(0.08, 0.22)
    if random.random() < 0.15:
        gain = torch.linspace(random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), h).view(1, h, 1)
        image = image * gain
    return image.clamp(0, 1), y, valid



SIZE=256
LABELS=["真皮层下界","脂肪层下界"]

@lru_cache(maxsize=256)
def annotation(path, w, h):
    obj = json.loads(path.read_text())
    result, masks = [], []
    for name in LABELS:
        item = next((it for it in obj.get('items', []) if it.get('label') == name and it.get('data')), None)
        if item is None:
            raise ValueError('missing target: ' + name)
        columns = {}
        for point in item['data']:
            x, y = point.get('x'), point.get('y')
            if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                continue
            ix = int(round(float(x)))
            if 0 <= ix < w and 0 <= y <= h - 1:
                columns.setdefault(ix, []).append(float(y))
        if len(columns) < 2:
            raise ValueError('fewer than two valid columns: ' + name)
        xs = np.array(sorted(columns))
        ys = np.array([np.mean(columns[x]) for x in xs])
        # Same original coordinate convention: scale_x = 256 / original_width.
        query = np.arange(SIZE) * w / SIZE
        result.append(np.interp(query, xs, ys))
        masks.append((query >= xs[0]) & (query <= xs[-1]))
    return np.array(result, dtype=np.float32), np.array(masks, dtype=bool)


class BoundaryDataset(Dataset):
    def __init__(self, index_file, split, training, limit=0, settings=None):
        payload=json.loads(Path(index_file).read_text())
        self.data_root=Path(payload['data_root'])
        self.training=training
        self.settings=settings or {}
        self.indices=payload['indices'][split if training else 'evaluation']
        if limit:self.indices=self.indices[:limit]
        self.items=payload['items']
        self.audit=payload['audit']

    def __len__(self):
        return len(self.indices)

    def load_view(self,index):
        item=self.items[index]
        path=self.data_root/item['image']
        ann=path.parent.parent/'annotations'/(path.stem+'.json')
        width,height=item['source_width'],item['source_height']
        y,valid=annotation(ann,width,height)
        crop_h=item['view_height']
        with Image.open(path) as image:
            if image.size!=(width,height):raise ValueError('原图尺寸与固定索引不一致：'+str(path))
            image=image.convert('L')
            if item['crop']:image=image.crop((0,0,width,crop_h))
            pixels=np.array(image.resize((SIZE,SIZE),Image.Resampling.LANCZOS),copy=True)
        coords=np.clip(y*SIZE/crop_h,0,SIZE-1)
        mask=valid & (y<crop_h)
        return pixels,coords,mask

    def __getitem__(self,idx):
        index=self.indices[idx]
        pixels,coords,mask=self.load_view(index)
        image=torch.from_numpy(pixels).float()[None]/255
        y=torch.from_numpy(coords.copy())
        valid=torch.from_numpy(mask.copy())
        if self.training:image,y,valid=augment(image,y,valid,self.settings)
        return dict(image=image,y=y,valid=valid,index=index,
                    original_scale=self.items[index]['view_height']/256)
