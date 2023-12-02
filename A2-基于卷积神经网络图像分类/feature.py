import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 卷积网络
class ConvNet(nn.Module):
    def __init__(self, out_dim):
        super(ConvNet, self).__init__()
        # 卷积+池化+激活函数
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3),
                                    nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=9, out_channels=16, kernel_size=3),
                                    nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU())
        # 全连接
        self.layer3 = nn.Sequential(nn.Linear(5*5*16, 128), nn.ReLU())
        self.layer4 = nn.Linear(128, out_dim)
        # Dropout
        self.drop = nn.Dropout(0.5)
        # Normalization

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, 1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.softmax(x, dim=1)
        return x


model = ConvNet(10)
model.load_state_dict(torch.load('lastest.pt', map_location=torch.device('cpu')))
model.eval()

for z in range(10):
    # 读取并处理输入图像
    image_path = f'{z}.png'  # 替换为你的图像路径
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    input_image = transform(image).unsqueeze(0)  # 添加批次维度

    # 进行前向传播
    with torch.no_grad():
        output = model.layer1[0](input_image)

    # 获取所有通道的特征图
    feature_maps = output.squeeze().numpy()

    # 显示所有通道的特征图在一条轴上
    plt.figure(figsize=(15, 5))
    for i in range(feature_maps.shape[0]):
        plt.subplot(1, feature_maps.shape[0], i + 1)
        plt.imshow(feature_maps[i], cmap='gray')
        plt.title(f'Channel {i + 1}')
        plt.axis('off')

    # 保存图像
    plt.savefig(f'i{z}.png')

