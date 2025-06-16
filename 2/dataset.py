
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class HyperspectralDataset(Dataset):
    def __init__(self, hsi_dir, rgb_dir, label_dir, list_file, image_size=256, transform=None, train=True):
        self.hsi_dir = hsi_dir
        self.rgb_dir = rgb_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.train = train
        with open(list_file, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # 加载高光谱数据 (.mat)
        hsi_path = os.path.join(self.hsi_dir, f"{filename}.mat")
        hsi_data = sio.loadmat(hsi_path)
        hsi = hsi_data['data']  # (H, W, C)

        hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min() + 1e-6)
        hsi = torch.tensor(hsi, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)

        rgb_path = os.path.join(self.rgb_dir, f"{filename}.png")
        rgb = Image.open(rgb_path).convert("RGB")

        label_path = os.path.join(self.label_dir, f"{filename}.png")
        label = Image.open(label_path).convert("L")
        label = np.array(label)
        if label.max() > 1:
            label = label / 255.0
        label = (label > 0.5).astype(np.int64)
        label = torch.tensor(label, dtype=torch.long)

        if self.train and self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            if torch.rand(1) > 0.5:
                rgb = TF.hflip(rgb)
                label = TF.hflip(label)
            if torch.rand(1) > 0.5:
                rgb = TF.vflip(rgb)
                label = TF.vflip(label)
            angle = torch.randint(-15, 16, (1,)).item()
            rgb = TF.rotate(rgb, angle)
            label = TF.rotate(label, angle)

            i, j, h, w = transforms.RandomResizedCrop.get_params(
                label, scale=(0.8, 1.0), ratio=(0.75, 1.33)
            )
            rgb = TF.crop(rgb, i, j, h, w)
            label = TF.crop(label, i, j, h, w)

        # Resize
        hsi = TF.resize(hsi, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.BILINEAR)
        rgb = TF.resize(rgb, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.BILINEAR)
        label = TF.resize(label, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)

        rgb = TF.to_tensor(rgb)
        rgb = TF.normalize(rgb, mean=[0.5] * 3, std=[0.5] * 3)

        if idx == 0:
            print(f"HSI range: [{hsi.min().item():.4f}, {hsi.max().item():.4f}]")
            print(f"RGB range: [{rgb.min().item():.4f}, {rgb.max().item():.4f}]")
            print(f"Label unique values: {torch.unique(label)}")

        return hsi, rgb, label


def get_transform(image_size, train=True):
    if train:
        return None
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
