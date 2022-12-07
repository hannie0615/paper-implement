## 1. module import ##
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


## 2. gpu check
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

batch_size = 16
epochs = 50

## 3. data download
train_dataset = datasets.MNIST(root='C:/Users/KETI/Downloads/',
                               train=True,    # 학습용 데이터임을 지정(false는 검증용)
                               download=True, # 다운로드 할 것인지 지정
                               transform=transforms.ToTensor())   # 이미지 전처리(0~1로 자동 정규화)

test_dataset = datasets.MNIST(root='C:/Users/KETI/Downloads/',
                              train=False,
                              transform=transforms.ToTensor())
# Mini-batch 단위로 데이터 할당
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # shuffle: 순서 섞기
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


## 4. data check
for(x_train, y_train) in train_loader:
    print("=========================================")
    print('X_train: ', x_train.size(), 'type: ', x_train.type()) # X_train:  torch.Size([16, 1, 28, 28])  => 배치 사이즈*rgb*가로*세로
    print('Y_train: ', y_train.size(), 'type: ', y_train.type()) # Y_train:  torch.Size([16]) => 배치 사이즈
    print("=========================================")
    break

## 5. mlp
class Net(nn.Module):   # nn.Module에 딥러닝에 필요한 함수들이 내제.
    def __init__(self):
        super(Net, self).__init__()     # nn.Module 내의 메서드를 상속받는다.
        self.fc1 = nn.Linear(28*28*1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)   # 10: 클래스의 수

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

## 6. optimizer, objective Function
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()   # output과 원-핫 인코딩 값과의 loss

## 7. train 과정
def train(model, train_loader, optimizer, log_interval): # 67? 87 페이지
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{}({:.0f}%)]"
                  .format(epoch, batch_idx*len(image), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():   # 평가 단계에서는 gradient 억제
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.*correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for epoch in range(1, epochs+1):
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[Epoch: {}], \t test loss : {:.4f}, \t Test Accuracy: {:.2f} %\n".format(epoch, test_loss, test_accuracy))





