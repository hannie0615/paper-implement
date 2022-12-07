from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#
# class Cifar10_DataLoader(DataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#
#         transform = transforms.Compose([
#             transforms.ToTensor()
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=transform)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def Cifar10_DataLoader(BATCH_SIZE):
    data_dir = './data'
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(data_dir,
                                     train=True,
                                     download=True,
                                     transform=transform)

    test_dataset = datasets.CIFAR10(data_dir,
                                     train=False,
                                     transform=transform)

    train_loader = DataLoader(dataset=train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    return train_loader, test_loader