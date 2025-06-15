import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from utils.image_transform import extract_fft, extract_lbp, extract_srm_residual

class GenImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df['file_name'].iloc[idx])
        label = int(self.df['label'].iloc[idx])
        # print(f"Processing image: {img_path}, label: {label}")
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {img_path}")
        # 中心裁剪224x224，若图片小于224则补全为224x224
        h, w = img.shape[:2]
        top = max((h - 224) // 2, 0)
        left = max((w - 224) // 2, 0)
        img_cropped = img[top:top+224, left:left+224]
        pad_h = max(224 - img_cropped.shape[0], 0)
        pad_w = max(224 - img_cropped.shape[1], 0)
        if pad_h > 0 or pad_w > 0:
            img_cropped = cv2.copyMakeBorder(
                img_cropped,
                top=0, bottom=pad_h,
                left=0, right=pad_w,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
        img = img_cropped

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fft = extract_fft(gray)
        srm = extract_srm_residual(gray)
        lbp = extract_lbp(gray)

        rgb = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = (rgb - mean) / std
        multi_channel = np.concatenate([rgb, fft[None, ...], srm[None, ...], lbp[None, ...]], axis=0)
        multi_channel_tensor = torch.tensor(multi_channel, dtype=torch.float32)

        if self.transform:
            multi_channel_tensor = self.transform(multi_channel_tensor)
        return multi_channel_tensor, torch.tensor(label)

def make_dataframe(root_dir):
    data = []
    ai_dir = os.path.join(root_dir, 'ai')
    nature_dir = os.path.join(root_dir, 'nature')
    for fname in os.listdir(ai_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            data.append({'file_name': f'ai/{fname}', 'label': 0})
    for fname in os.listdir(nature_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            data.append({'file_name': f'nature/{fname}', 'label': 1})
    df = pd.DataFrame(data)
    return df

def get_dataloaders(batch_size=256, val_ratio=0.2, shuffle=True):
    dataset_root = './dataset/imagenet_ai_0419_sdv4/val'
    df = make_dataframe(dataset_root)
    # print(df)
    train_df, val_df = np.split(df.sample(frac=1, random_state=42), [int((1 - val_ratio) * len(df))])
    train_dataset = GenImageDataset(train_df, dataset_root)
    val_dataset = GenImageDataset(val_df, dataset_root)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, val_loader

# 用法示例
# train_loader, val_loader = get_dataloaders(batch_size=256)