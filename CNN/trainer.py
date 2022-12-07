import numpy as np
import torch


# class Trainer:
#
#     def __init__(self, model, criterion, metrics, optimizer, config, device,
#                  train_loader, valid_data_loader=None, lr_scheduler=None, epochs):
#         self.config = config
#         self.model = model
#         self.criterion = criterion
#         self.metrics = metrics
#         self.optimizer = optimizer
#         self.device = device
#         self.train_loader = train_loader
#         self.epochs = epochs
#

def train(model, train_loader, device, optimizer, criterion, Epoch):

    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print("Train Epoch: {} [{}/{}({:.0f}%)]\t Train Loss: {:.6f}".format(
                Epoch, batch_idx * len(image), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))


def evaluation(model, test_loader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            test_loss += criterion(output, label).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.*correct / len(test_loader.dataset)

    return test_loss, test_accuracy




