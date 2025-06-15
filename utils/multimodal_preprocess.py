import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import h5py

from image_transform import extract_fft, extract_lbp, extract_srm_residual

def preprocess_and_save(df, root_dir, preprocessed_root, is_test=False):
    os.makedirs(preprocessed_root, exist_ok=True)
    for idx in tqdm(range(len(df))):
        if is_test:
            img_name = df.iloc[idx, 0]
        else:
            img_name = df['file_name'].iloc[idx]
        img_path = os.path.join(root_dir, img_name)
        save_path = os.path.join(preprocessed_root, os.path.splitext(img_name)[0] + '_compressed.h5')

        # 如果当前文件已存在，则跳过
        if os.path.exists(save_path):
            continue
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            # print(f"Warning: {img_path} not found or cannot be read.")
            continue

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

        # 缩放为224x224
        # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 通道 4: FFT频域
        fft = extract_fft(gray)
        # 通道 5: SRM滤波
        srm = extract_srm_residual(gray)
        # 通道 6: LBP纹理
        lbp = extract_lbp(gray)

        # 合并6通道，转Tensor
        rgb = img.astype(np.float32) / 255.0  # HWC, [0,1]
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        rgb = np.transpose(rgb, (2, 0, 1))    # CHW, 3x224x224
        rgb = (rgb - mean) / std
        multi_channel = np.concatenate([rgb, 
                                        fft[None, ...], 
                                        srm[None, ...], 
                                        lbp[None, ...]], axis=0)  # 6x224x224

        # 转换为 float16
        multi_channel = multi_channel.astype(np.float16)

        # 使用 HDF5 存储并启用 gzip 压缩
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('data', data=multi_channel, compression='gzip', compression_opts=4)

if __name__ == "__main__":
    # 配置参数
    dataset_root = './dataset/ai-vs-human-generated-dataset/'
    preprocessed_root = './dataset/ai-vs-human-generated-dataset/preprocessed/'
    os.makedirs(preprocessed_root, exist_ok=True)

    # 处理train.csv
    train_df = pd.read_csv(os.path.join(dataset_root, 'train.csv'), index_col=0)
    preprocess_and_save(train_df, dataset_root, os.path.join(preprocessed_root), is_test=False)

    # 处理test.csv
    test_df = pd.read_csv(os.path.join(dataset_root, 'test.csv'))
    preprocess_and_save(test_df, dataset_root, os.path.join(preprocessed_root), is_test=True)