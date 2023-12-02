import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


# 卷积网络
class ConvNet(nn.Module):
    def __init__(self, out_dim):
        super(ConvNet, self).__init__()
        # 卷积+池化
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接
        self.linear1 = nn.Sequential(nn.Linear(7*7*32, 256), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.linear3 = nn.Linear(128, out_dim)


    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.pool1(x)
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = F.softmax(x, dim=1)
        return x


# 初始化五个模型
models = [ConvNet(10) for _ in range(6)]

# 加载每个模型的预训练权重
for i, model in enumerate(models):
    model.load_state_dict(torch.load(f'lastest_{i + 1}.pt'))
    model.eval()  # 设置为评估模式

# 加载测试数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
test_dataset = datasets.MNIST('MNIST', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

# 进行模型投票集成的测试
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        votes = torch.zeros((inputs.size(0), 10))  # 用于存储每个样本每个类别的得票数

        # 对每个模型进行预测，并进行投票
        for model in models:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(inputs.size(0)):
                votes[i, predicted[i]] += 1  # 直接使用 predicted 作为索引，无需 .item()

        # 选择得票最多的类别作为最终预测结果
        _, ensemble_prediction = torch.max(votes, 1)

        total += labels.size(0)
        correct += (ensemble_prediction == labels).sum().item()

accuracy = correct / total
print(f'Ensemble Test Accuracy (Voting): {accuracy * 100:.3f}%')
