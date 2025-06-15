import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

import cv2
from utils.image_transform import extract_fft,extract_lbp,extract_srm_residual
import h5py

class MultimodalFusionImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_test=False):
        """
        :param image_paths: List[str] - 图像文件路径
        :param labels: List[int] - 标签，0为真实图，1为伪造图
        :param transform: torchvision.transforms 对图像增强（可选）
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test  # Flag to indicate if this is the test dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.is_test:
            # Use the first column (assumed to be 'id' or 'file_name')
            img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])  
            h5_path = os.path.join(self.root_dir, 'preprocessed',
                            os.path.splitext(self.df.iloc[idx, 0])[0] + '_compressed.h5')
        else:
            # Use column names instead of hardcoded index
            img_path = os.path.join(self.root_dir, self.df['file_name'].iloc[idx])  
            h5_path = os.path.join(self.root_dir, 'preprocessed',
                            os.path.splitext(self.df['file_name'].iloc[idx])[0] + '_compressed.h5')
            label = int(self.df['label'].iloc[idx])  

        
        # 从 h5 文件加载数据
        with h5py.File(h5_path, 'r') as f:
            array = f['data'][:]

        # 转为 tensor
        tensor = torch.tensor(array, dtype=torch.float32)

        if self.transform:
            tensor = self.transform(tensor)
        if self.is_test:
            return tensor, -1 , self.df.iloc[idx, 0]
        else:
            return tensor, torch.tensor(label)
        
# Training Transform (with data augmentation)
train_transform = None # transforms.Compose([
#     transforms.ToPILImage(),
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomRotation(10),      # Augmentation: Rotate images slightly
#     # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#     # transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# Test/Validation Transform (NO augmentation)
test_transform = None # transforms.Compose([
#     transforms.ToPILImage(),
#     # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#     # transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

def get_dataloaders(batch_size=256):
    # Define the root directory of the dataset
    dataset_root = './dataset/ai-vs-human-generated-dataset/'
    # Load the CSV file
    train_df = pd.read_csv(os.path.join(dataset_root, 'train.csv'), index_col=0)
    test_df = pd.read_csv(os.path.join(dataset_root, 'test.csv'))

    # Split into training and validation (80% train, 20% validation)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])
    # Create datasets
    train_dataset = MultimodalFusionImageDataset(train_df, dataset_root, transform=train_transform, is_test=False)
    val_dataset = MultimodalFusionImageDataset(val_df, dataset_root, transform=test_transform, is_test=False)
    test_dataset = MultimodalFusionImageDataset(test_df, dataset_root, transform=test_transform, is_test=True)
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader