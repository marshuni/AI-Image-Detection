import torch
import tqdm

from config import *

# 测试函数
def test_model(model, test_loader):
    model.eval()
    predicted_all = []
    labels_all = []

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            predicted = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy().flatten()
            predicted_all.extend(predicted)
            labels_all.extend(labels.cpu().numpy().flatten())

    return labels_all, predicted_all


def test_model_for_kaggle_submission(model, test_loader):
    paths = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for inputs, labels, img_path in tqdm.tqdm(test_loader):
            paths.extend(img_path)

            inputs = inputs.to(device)
            outputs = model(inputs)

            predicted = (torch.sigmoid(outputs) > 0.5).long()
            predictions.extend(predicted.cpu().numpy().flatten())

    # 将预测结果与文件id一一对应，保存到文件
    import pandas as pd
    submission_df = pd.DataFrame({
        'id': paths,
        'label': predictions
    })
    submission_df.to_csv('submission.csv', index=False)
    print(submission_df.head())

import matplotlib.pyplot as plt
from sklearn.metrics import *
import numpy as np
import seaborn as sns

def calculate_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1
    
# 绘制ROC曲线
def calculate_roc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('ROC - Graph Anomaly Detection')
    plt.legend(loc="lower right")
    plt.savefig(f"./save/ROC_Curve.png",dpi=300)
    plt.close()

    return roc_auc