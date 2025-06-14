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
        else:
            # Use column names instead of hardcoded index
            img_path = os.path.join(self.root_dir, self.df['file_name'].iloc[idx])  
            label = int(self.df['label'].iloc[idx])  

        # 读取图像
        img = cv2.imread(img_path)

        # 缩放为224x224
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 通道 4: FFT频域
        fft = extract_fft(gray)

        # 通道 5: SRM滤波
        srm = extract_srm_residual(gray)

        # 通道 6: LBP纹理
        lbp = extract_lbp(gray)

        # 合并6通道，转Tensor
        rgb = img.astype(np.float32) / 255.0  # HWC, [0,1]
        # 标准化RGB通道
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        rgb = np.transpose(rgb, (2, 0, 1))    # CHW, 3x224x224
        rgb = (rgb - mean) / std
        multi_channel = np.concatenate([rgb, 
                        fft[None, ...], 
                        srm[None, ...], 
                        lbp[None, ...]], axis=0)  # 6x224x224
        
        multi_channel_tensor = torch.tensor(multi_channel, dtype=torch.float32)

        if self.transform:
            multi_channel_tensor = self.transform(multi_channel_tensor)
        if self.is_test:
            return multi_channel_tensor, -1 , self.df.iloc[idx, 0]
        else:
            return multi_channel_tensor, torch.tensor(label)
        
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader