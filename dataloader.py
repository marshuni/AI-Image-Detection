import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_test=False):
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

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, -1 
        else:
            return image, label
        

# Training Transform (with data augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Augmentation: Randomly flip images
    transforms.RandomRotation(10),      # Augmentation: Rotate images slightly
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test/Validation Transform (NO augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Keep it consistent
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the root directory of the dataset
dataset_root = './dataset/ai-vs-human-generated-dataset/'
# Load the CSV file
train_df = pd.read_csv(os.path.join(dataset_root, 'train.csv'), index_col=0)
test_df = pd.read_csv(os.path.join(dataset_root, 'test.csv'))

# Split into training and validation (80% train, 20% validation)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

# Create datasets
train_dataset = ImageDataset(train_df, dataset_root, transform=train_transform, is_test=False)
val_dataset = ImageDataset(val_df, dataset_root, transform=test_transform, is_test=False)
test_dataset = ImageDataset(test_df, dataset_root, transform=test_transform, is_test=True)

def get_dataloaders(batch_size=256):
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader





