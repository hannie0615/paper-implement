import argparse
import collections
import torch
import torch.nn as nn
from data_loaders import Cifar10_DataLoader
from model import CNNModel
import trainer

def main(config):
    batch_size = 32
    epochs = 100

    train_loader, test_loader = Cifar10_DataLoader(BATCH_SIZE=batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()


    for Epoch in range(1, epochs+1):
        trainer.train(model, train_loader, device, optimizer, criterion, Epoch)
        test_loss, test_accuracy = trainer.evaluation(model, test_loader, device, criterion)
        print("\n[Epoch: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f}%\n".format(
            Epoch, test_loss, test_accuracy)
        )


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = args.parse_args()

    print(config)
    main(config)


