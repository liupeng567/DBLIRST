"""IRSTD-1k 数据集（M2-L1 文献复现层：MSHNet 原基准对拍）。

口径 = 官方 MSHNet utils/data.py IRSTD_Dataset 逐行移植：
  - train：随机水平翻转 p=0.5 → 短边随机缩放 [0.5,2.0]×256 → 短边不足 256 补零 →
    随机 256×256 裁剪 → 高斯模糊 p=0.5（radius∈[0,1]）；
  - val（testval）：整图 bilinear resize 256×256（掩码 nearest）；
  - 图像 RGB + ImageNet 归一化；掩码 ToTensor [0,1]；
  - split：trainval.txt 800 / test.txt 201（与官方一致）。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Irstd1k(Dataset):
    def __init__(self, root: str, mode: str = "train", base_size: int = 256,
                 crop_size: int = 256):
        root = Path(root)
        list_file = "trainval.txt" if mode == "train" else "test.txt"
        self.names = (root / list_file).read_text(encoding="utf-8").split()
        self.img_dir = root / "IRSTD1k_Img"
        self.label_dir = root / "IRSTD1k_Label"
        self.mode = mode
        self.base_size = base_size
        self.crop_size = crop_size
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.names)

    def _img_to_tensor(self, img: Image.Image) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1)  # HWC RGB → CHW
        return (x - self.mean) / self.std

    def __getitem__(self, i: int):
        name = self.names[i]
        img = Image.open(self.img_dir / f"{name}.png").convert("RGB")
        mask = Image.open(self.label_dir / f"{name}.png")

        if self.mode == "train":
            img, mask = self._sync_transform(img, mask)
        else:
            img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
            mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)

        return {
            "name": name,
            "windows": self._img_to_tensor(img),  # [3,256,256] ImageNet 归一化
            "target": torch.from_numpy(
                np.asarray(mask, dtype=np.float32) / 255.0
            ).unsqueeze(0),  # [1,256,256]
        }

    def _sync_transform(self, img: Image.Image, mask: Image.Image):
        rng = random.random()
        if rng < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop = self.crop_size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh, ow = long_size, int(1.0 * w * long_size / h + 0.5)
        else:
            ow, oh = long_size, int(1.0 * h * long_size / w + 0.5)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if min(ow, oh) < crop:
            padw = crop - ow if ow < crop else 0
            padh = crop - oh if oh < crop else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - crop)
        y1 = random.randint(0, h - crop)
        img = img.crop((x1, y1, x1 + crop, y1 + crop))
        mask = mask.crop((x1, y1, x1 + crop, y1 + crop))
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img, mask
