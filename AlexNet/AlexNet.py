import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from torchsummary import summary

def custom_imshow(img):
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

# 구현
class Alexnet(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 96, kernel_size=11, stride=4),
            torch.nn.ReLU(inplace=True),
            torch.nn.LocalResponseNorm(2),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.LocalResponseNorm(2),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True)
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(6 * 6 * 256, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(227),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

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

    sample_num = 694

    custom_imshow(trainset[sample_num][0])
    print(classes[trainset[sample_num][1]])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Alexnet(3, 10).to(device)
    summary(model, input_size=(3, 227, 227), device='cuda')

    lr = 1e-4
    epochs = 50

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print("\nEpoch ", epoch)
        # train
        print("\nTrain:")
        model.train()
        train_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            pred = model(images)
            loss = F.cross_entropy(pred, targets)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i + 1 == len(train_loader):
                print('[%3d/%3d] | Loss: %.5f' % (i + 1, len(train_loader), train_loss / (i + 1)))
        # val
        print("\nValidation")
        model.eval()
        val_loss = 0
        for i, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            val_loss += loss.item()

            if i + 1 == len(test_loader):
                print('[%3d/%3d] | Loss: %.5f' % (i + 1, len(test_loader), val_loss / (i + 1)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    train()



