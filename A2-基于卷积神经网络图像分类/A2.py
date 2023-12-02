import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold


torch.manual_seed(1)

train_batch_size = 100
test_batch_size = 1024
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
train_dataset = torchvision.datasets.MNIST('MNIST', train=True, download=True, transform=transform)
test_dataset = DataLoader(torchvision.datasets.MNIST('MNIST', train=False, download=True, transform=transform), batch_size=test_batch_size, shuffle=False)


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


epochs = 10
lr = 5e-2
momentum = 0.9

total_train_loss_value = []
total_eval_loss_value = []
total_eval_acc_value = []

# 交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(train_dataset)), train_dataset.targets)):
    print(f'Fold {fold + 1}')
    model = ConvNet(10).to('cuda')
    optimizer = optim.SGD(lr=lr, momentum=momentum, params=model.parameters())

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=val_sampler)

    train_loss_value = []
    eval_loss_value = []
    eval_acc_value = []
    for epoch in range(epochs):
        # 模型训练
        train_loss = 0
        train_acc = 0
        model.train()
        for img, label in train_loader:
            img = img.clone().detach().to('cuda')
            label = label.clone().detach().to('cuda')
            optimizer.zero_grad()
            output = model(img)
            _, pred = output.max(1)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.data
        print('epoch: {}, Train Loss: {:.6f}'.format(epoch + 1, train_loss / len(train_loader)))
        train_loss_value.append(train_loss.item() / len(train_loader))

        # 模型验证
        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for img, label in val_loader:
                img = img.clone().detach().to('cuda')
                label = label.clone().detach().to('cuda')
                output = model(img)
                _, pred = output.max(1)
                correct = (pred == label).sum().item()
                acc = correct / img.shape[0]
                val_acc += acc
                loss = F.cross_entropy(output, label)
                val_loss += loss.data
            eval_loss_value.append(val_loss.item() / len(val_loader))
            eval_acc_value.append(val_acc / len(val_loader))
            print('epoch: {}, Val Loss: {:.6f}, Val Acc: {:.6f}'.format(epoch + 1, val_loss / len(val_loader), val_acc / len(val_loader)))

    total_train_loss_value.append(train_loss_value)
    total_eval_acc_value.append(eval_acc_value)
    total_eval_loss_value.append(eval_loss_value)


# 平均精度计算
avr_train_loss_value = []
avr_eval_acc_value = []
avr_eval_loss_value = []
train_loss = 0
eval_acc = 0
eval_loss = 0
for i in range(epochs):
    for j in range(5):
        train_loss += total_train_loss_value[j][i]
        eval_acc += total_eval_acc_value[j][i]
        eval_loss += total_eval_loss_value[j][i]
    avr_train_loss_value.append(train_loss / 5)
    avr_eval_acc_value.append(eval_acc / 5)
    avr_eval_loss_value.append(eval_loss / 5)
    train_loss = 0
    eval_acc = 0
    eval_loss = 0


# 模型测试
test_loss = 0
test_acc = 0
test_loss_value = []
test_acc_value = []
model.eval()
with torch.no_grad():
    for img, label in test_dataset:
        img = img.clone().detach().to('cuda')
        label = label.clone().detach().to('cuda')
        output = model(img)
        _, pred = output.max(1)
        correct = (pred == label).sum().item()
        acc = correct / img.shape[0]
        test_acc += acc
        loss = F.cross_entropy(output, label)
        test_loss += loss.data
    test_loss_value.append(test_loss.item() / len(test_dataset))
    test_acc_value.append(test_acc / len(test_dataset))
    print('Test Loss: {:.6f}, Test Acc: {:.6f}'.format(test_loss / len(test_dataset), test_acc / len(test_dataset)))


with open('eval-acc.txt', 'w') as file:
    file.write(', '.join(map(str, avr_eval_acc_value)))
    file.close()

with open('eval-loss.txt', 'w') as file:
    file.write(', '.join(map(str, avr_eval_loss_value)))
    file.close()

with open('train-loss.txt', 'w') as file:
    file.write(', '.join(map(str, avr_train_loss_value)))
    file.close()

with open('test-acc.txt', 'w') as file:
    file.write(', '.join(map(str, test_acc_value)))
    file.close()

with open('test-loss.txt', 'w') as file:
    file.write(', '.join(map(str, test_loss_value)))
    file.close()
