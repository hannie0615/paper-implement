import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary


class VGGNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes, layer_config="A"):
        super().__init__()

        layers = layer_configs[layer_config]

        self.features = make_layer(in_channels, layers)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(7 * 7 * 512, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def make_layer(in_channels, layers):
    modules = []

    in_channels = in_channels
    for layer_type in layers:
        if layer_type == "M":
            modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif layer_type == "LRN":
            modules.append(nn.LocalResponseNorm(2))
        else:
            k, layer = layer_type
            modules.append(nn.Conv2d(in_channels, layer, kernel_size=k, padding=1))
            modules.append(nn.ReLU(inplace=True))
            in_channels = layer

    return nn.Sequential(*modules)

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

    # 타입별로 모델 선언할 수 있음
    layer_configs = {
        "A": [(3, 64), 'M', (3, 128), 'M', (3, 256), (3, 256), 'M', (3, 512), (3, 512), 'M', (3, 512), (3, 512), 'M'],
        "A-LRN": [(3, 64), 'LRN', 'M', (3, 128), 'M', (3, 256), (3, 256), 'M', (3, 512), (3, 512), 'M', (3, 512),
                  (3, 512), 'M'],
        "B": [(3, 64), (3, 64), 'M', (3, 128), (3, 128), 'M', (3, 256), (3, 256), 'M', (3, 512), (3, 512), 'M',
              (3, 512), (3, 512), 'M'],
        "C": [(3, 64), (3, 64), 'M', (3, 128), (3, 128), 'M', (3, 256), (3, 256), (1, 256), 'M', (3, 512), (3, 512),
              (1, 512), 'M', (3, 512), (3, 512), (1, 512), 'M'],
        "D": [(3, 64), (3, 64), 'M', (3, 128), (3, 128), 'M', (3, 256), (3, 256), (3, 256), 'M', (3, 512), (3, 512),
              (3, 512), 'M', (3, 512), (3, 512), (3, 512), 'M'],
        "E": [(3, 64), (3, 64), 'M', (3, 128), (3, 128), 'M', (3, 256), (3, 256), (3, 256), (3, 256), 'M', (3, 512),
              (3, 512), (3, 512), (3, 512), 'M', (3, 512), (3, 512), (3, 512), (3, 512), 'M'],
    }

    # A 타입
    model = VGGNet(3, 10, "A").to(device)
    #summary(model, input_size=(3, 224, 224), device='cuda')

    lr = 1e-5
    epochs = 50

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print("\nEpoch ", epoch)
        # 훈련
        print("\nTrain:")
        model.train()
        train_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            prediction = model(images)  # 예측값
            loss = F.cross_entropy(prediction, targets)  # 손실 계산
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 1 train 당 출력은 10개만.
            if ((i + 1) % (len(train_loader) // 10) == 1) or i + 1 == len(train_loader):
                print('[%3d/%3d] | Loss: %.5f' % (i + 1, len(train_loader), train_loss / (i + 1)))
        # 검증
        print("\nValidation")
        model.eval()
        val_loss = 0
        for i, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            loss = F.cross_entropy(preds, targets)
            val_loss += loss.item()

            # 1 validation 당 출력은 3개만
            if ((i + 1) % (len(test_loader) // 3) == 1) or i + 1 == len(test_loader):
                print('[%3d/%3d] | Loss: %.5f' % (i + 1, len(test_loader), val_loss / (i + 1)))
