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
        # Plot the curve of the loss values
    plt.figure()
    pylab.xlim(0, num_epochs)
    plt.plot(range(1, num_epochs+1), losses, label='loss',color='blue')
    plt.legend()
    plt.savefig(os.path.join("./save/train", f"loss_{model.__class__.__name__}.png"))
    plt.close()