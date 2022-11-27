import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchsummary import summary

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Resnet(torch.nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], in_channels=3, num_classes=10):
        super(Resnet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2_x = self._make_layer(block, layers[0], 64)
        self.conv3_x = self._make_layer(block, layers[1], 128, 2)
        self.conv4_x = self._make_layer(block, layers[2], 256, 2)
        self.conv5_x = self._make_layer(block, layers[3], 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block=BasicBlock, blocks=2, in_channels=3, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != in_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, in_channels * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(in_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, in_channels, downsample, stride)
        )

        self.inplanes = in_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, in_channels)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(224),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64

    # CIFAR10 데이터셋 다운로드
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # gpu 동작 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model
    model = Resnet(Bottleneck, [3, 4, 6, 3], 3, 10).cuda()
    # summary(model, input_size=(3, 224, 224), device='cuda')

    lr = 1e-5
    epochs = 1  # epochs

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training 시작
    for epoch in range(epochs):
        print("\nEpoch ", epoch)

        # 훈련
        print("\nTrain:")
        model.train()
        train_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            prediction = model(images)
            loss = F.cross_entropy(prediction, targets)

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i + 1) % (len(train_loader) // 10) == 1) or i + 1 == len(train_loader):
                print('[%3d/%3d] | Loss: %.5f' % (i + 1, len(train_loader), train_loss / (i + 1)))

        # 검증
        print("\nValidation")
        model.eval()
        val_loss = 0
        correct = 0
        for i, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)

            out = model(images)
            loss = F.cross_entropy(out, targets)
            val_loss += loss.item()
            prediction = torch.argmax(out, 1)

            correct += (prediction == targets).sum().item()

            if ((i + 1) % (len(test_loader) // 3) == 1) or i + 1 == len(test_loader):
                print('[%3d/%3d] | Loss: %.5f' % (i + 1, len(test_loader), val_loss / (i + 1)))
        print(f"Accuracy of the network : {100 * correct // (len(test_loader) * batch_size)}%")


