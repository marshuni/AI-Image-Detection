import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

# 设置随机种子保证可重复性
torch.manual_seed(42)

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集示例（这里使用CIFAR10作为示例）
train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义ResNet50模型
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=10, weights=ResNet50_Weights.DEFAULT):
        super(ResNet50Classifier, self).__init__()
        # 加载预训练的ResNet50模型
        self.resnet50 = torchvision.models.resnet50(weights=weights)
        
        # 冻结所有层（如果只想训练分类层）
        # for param in self.resnet50.parameters():
        #     param.requires_grad = False
            
        # 替换最后的全连接层以适应我们的分类任务
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet50(x)

# 初始化模型
num_classes = 10  # CIFAR10有10个类别
model = ResNet50Classifier(num_classes=num_classes, weights=ResNet50_Weights.DEFAULT)

# 将模型移动到GPU（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# 测试函数
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# 训练和测试模型
train_model(model, train_loader, criterion, optimizer, num_epochs=5)
test_model(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'resnet50_classifier.pth')

# 加载保存的模型
# loaded_model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
# loaded_model.load_state_dict(torch.load('resnet50_classifier.pth'))
# loaded_model = loaded_model.to(device)
