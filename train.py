import torch
import tqdm

from config import *

import pylab
import matplotlib.pyplot as plt
import os

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    losses = []
    best_loss = float('inf')
    patience = 5  # 早停容忍度
    trigger_times = 0

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Epochs"):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            # Adjust outputs to match labels shape for BCEWithLogitsLoss
            outputs = outputs.squeeze(dim=1) if outputs.dim() > 1 else outputs
            loss = criterion(outputs, labels.float())
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        epoch_acc = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        # 学习率调度
        scheduler.step(epoch_loss)

        # 早停机制
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    plt.figure()
    pylab.xlim(0, num_epochs)
    plt.plot(range(1, num_epochs+1), losses, label='loss',color='blue')
    plt.legend()
    plt.savefig(os.path.join("./save/train", f"loss_{model.__class__.__name__}.png"))
    plt.close()