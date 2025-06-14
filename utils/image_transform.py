import cv2
import numpy as np

from skimage.feature import local_binary_pattern
import os

def extract_fft(img: np.ndarray) -> np.ndarray:
    """
    提取图像的频域幅度谱，输出归一化图像
    输入: img - RGB图像 (H, W, 3) 或 灰度图像 (H, W)
    返回: 幅度谱图像 (H, W) [float32, 范围0~1]
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fft = 20 * np.log(np.abs(fshift) + 1e-8)  # 避免log(0)

    # 归一化到0-1之间
    fft = (fft - fft.min()) / (fft.max()-fft.min())

    return fft.astype(np.float32)

def extract_srm_residual(img: np.ndarray) -> np.ndarray:
    """
    提取SRM噪声残差图像（此处用高通边缘增强近似SRM滤波器）
    输入: RGB图像或灰度图像
    返回: 噪声图 (H, W) [float32, 范围0~1]
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 简单高通滤波器（Laplacian近似）
    kernel = np.array([[0, -1, 0],
                       [-1,  4, -1],
                       [0, -1, 0]])

    residual = cv2.filter2D(img.astype(np.float32), -1, kernel)

    # 归一化并增强对比度
    residual = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
    # 增强对比度（例如使用gamma校正或线性拉伸）
    gamma = 0.5  # gamma < 1 提高暗部对比度
    residual = np.power(residual, gamma)

    return residual.astype(np.float32)

def extract_lbp(img: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """
    提取LBP纹理图像
    输入: RGB图像或灰度图像
    返回: LBP图 (H, W) [float32, 归一化到0~1]
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(img, P=n_points, R=radius, method='uniform')

    # 归一化
    lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())

    return lbp.astype(np.float32)

def test_feature_extraction():
    """
    测试特征提取函数，展示样例图片的FFT、SRM残差和LBP特征
    """
    import matplotlib.pyplot as plt

    # 选择一张样例图片
    img_path = "dataset/ai-vs-human-generated-dataset/train_data/"
    # 获取目录下第一张图片
    img_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not img_files:
        raise FileNotFoundError("未在指定目录下找到图片文件")
    sample_img_path = os.path.join(img_path, img_files[457])
    print(f"使用样例图片: {sample_img_path}")

    # 读取图片
    img = cv2.imread(sample_img_path)

    # 提取特征
    fft_img = extract_fft(img)
    srm_img = extract_srm_residual(img)
    lbp_img = extract_lbp(img)

    # 展示与保存（保持原始像素大小）
    h, w = img.shape[:2]
    fig, axes = plt.subplots(1, 4, figsize=(w * 4 / 100, h / 100), dpi=100)
    titles = ["Original", "FFT", "SRM Residual", "LBP"]
    images = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        fft_img,
        srm_img,
        lbp_img
    ]
    cmaps = [None, 'gray', 'gray', 'gray']

    for ax, im, title, cmap in zip(axes, images, titles, cmaps):
        if cmap:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0)
    plt.savefig("save/feature_visualization.png", bbox_inches='tight', pad_inches=0.5)
    plt.close()
