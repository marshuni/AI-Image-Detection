import torch
import tqdm

from config import *

# 测试函数
def test_model(model, test_loader):
    model.eval()
    predicted_all = []
    labels_all = []

    with torch.no_grad():
        for inputs, labels, _ in tqdm.tqdm(test_loader):
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