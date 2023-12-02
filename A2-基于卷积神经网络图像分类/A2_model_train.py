import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import imageio.v3 as io

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

train_batch_size = 100
test_batch_size = 1024
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])

train_dataset = DataLoader(torchvision.datasets.MNIST('MNIST', train=True, download=True, transform=transform), batch_size=train_batch_size, shuffle=True)
test_dataset = DataLoader(torchvision.datasets.MNIST('MNIST', train=False, download=True, transform=transform), batch_size=test_batch_size, shuffle=False)


# 卷积网络
class ConvNet(nn.Module):
    def __init__(self, out_dim):
        super(ConvNet, self).__init__()
        # 卷积+池化+激活函数
        # self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3),
        #                             nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU())
        # self.layer2 = nn.Sequential(nn.Conv2d(in_channels=9, out_channels=16, kernel_size=3),
        #                             nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU())
        # self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3),
        #                             nn.BatchNorm2d(9), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer2 = nn.Sequential(nn.Conv2d(in_channels=9, out_channels=16, kernel_size=3),
        #                             nn.BatchNorm(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3),
        #                             nn.LayerNorm((9, 26, 26)), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer2 = nn.Sequential(nn.Conv2d(in_channels=9, out_channels=16, kernel_size=3),
        #                             nn.LayerNorm((16, 11, 11)), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3),
        #                             nn.GroupNorm(num_groups=3, num_channels=9), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer2 = nn.Sequential(nn.Conv2d(in_channels=9, out_channels=16, kernel_size=3),
        #                             nn.GroupNorm(num_groups=4, num_channels=16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3),
        #                             nn.InstanceNorm2d(9), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer2 = nn.Sequential(nn.Conv2d(in_channels=9, out_channels=16, kernel_size=3),
        #                             nn.InstanceNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))

        # 全连接
        self.layer3 = nn.Sequential(nn.Linear(7*7*32, 256), nn.ReLU())
        self.layer4 = nn.Linear(256, out_dim)
        # Dropout
        # self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, 1)
        x = self.layer3(x)
        # x = self.drop(x)
        x = self.layer4(x)
        x = F.softmax(x, dim=1)
        return x


# L1正则化
def l1_regularization(model, l1_alpha):
    l1_loss = 0
    for name, parameters in model.named_parameters():
        if name in ["layer1.0.weight", "layer2.0.weight", "layer3.weight"]:
            l1_loss += torch.abs(parameters).sum()
    return l1_alpha * l1_loss


# L2正则化
def l2_regularization(model, l2_alpha):
    l2_loss = 0
    for name, parameters in model.named_parameters():
        if name in ["layer1.0.weight", "layer2.0.weight", "layer3.weight"]:
            l2_loss += (parameters ** 2).sum() / 2.0
    return l2_alpha * l2_loss


epochs = 20
lr = 1e-2
momentum = 0.9
model = ConvNet(10).to('cuda')
optimizer = optim.SGD(lr=lr, momentum=momentum, params=model.parameters())


# 模型训练
train_loss_value = []
eval_loss_value = []
eval_acc_value = []
for epoch in range(epochs):
    train_loss = 0
    for img, label in train_dataset:
        img = img.clone().detach().to('cuda')
        label = label.clone().detach().to('cuda')
        optimizer.zero_grad()
        output = model(img)
        _, pred = output.max(1)
        # 交叉熵损失
        loss = F.cross_entropy(output, label)
        # L1正则化
        # loss += l1_regularization(model, 1e-4)
        # L2正则化
        # loss += l2_regularization(model, 1e-4)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
    print('epoch: {}, Train Loss: {:.6f}'.format(epoch + 1, train_loss / len(train_dataset)))
    # all_weights = np.zeros([0, ])
    # for name, parameters in model.named_parameters():
    #     if name in ["layer1.0.weight", "layer2.0.weight", "layer3.weight"]:
    #         w_flatten = np.reshape(parameters.clone().detach(), [-1])
    #         all_weights = np.concatenate([all_weights, w_flatten], axis=0)
    # plt.hist(all_weights, bins=100, color="b", range=[-1, 1])
    # plt.title("epoch=" + str(epoch) + " loss=%.2f" % (train_loss / len(train_dataset)))
    # plt.savefig("mnist_model_weights_hist_%d.png" % (epoch))
    # plt.clf()


    # 模型验证
    eval_acc = 0
    eval_loss = 0
    test_loss = 0
    with torch.no_grad():
        for img, label in train_dataset:
            img = img.clone().detach().to('cuda')
            label = label.clone().detach().to('cuda')
            output = model(img)
            _, pred = output.max(1)
            loss = F.cross_entropy(output, label)
            test_loss += loss.data

        for img, label in test_dataset:
            img = img.clone().detach().to('cuda')
            label = label.clone().detach().to('cuda')
            output = model(img)
            _, pred = output.max(1)
            correct = (pred == label).sum().item()
            acc = correct / img.shape[0]
            eval_acc += acc
            loss = F.cross_entropy(output, label)
            eval_loss += loss.data

        train_loss_value.append(test_loss.item() / len(train_dataset))
        eval_loss_value.append(eval_loss.item() / len(test_dataset))
        eval_acc_value.append(eval_acc / len(test_dataset))
        print('Test Loss: {:.6f}, Test Acc: {:.6f}'.format(eval_loss / len(test_dataset), eval_acc / len(test_dataset)))


# gif_images = []
# for i in range(epochs):
#     gif_images.append(io.imread("mnist_model_weights_hist_"+str(i)+".png"))   # 读取多张图片
# io.imwrite("mnist_model_weights.gif", gif_images, fps=epochs/10)


with open('eval-acc.txt', 'w') as file:
    file.write(', '.join(map(str, eval_acc_value)))
    file.close()

with open('eval-loss.txt', 'w') as file:
    file.write(', '.join(map(str, eval_loss_value)))
    file.close()

with open('train-loss.txt', 'w') as file:
    file.write(', '.join(map(str, train_loss_value)))
    file.close()

torch.save(model.state_dict(), 'lastest.pt')
