import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_MNIST(batch_size, root='data'):

    root = 'data'

    # get train data
    train_data = datasets.MNIST(root=root, 
                                train=True, 
                                download=True)

    # calculate mean and std on train data
    mean = train_data.data.float().mean() / 255
    std = train_data.data.float().std() / 255

    # define transforms
    data_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[mean], 
                                             std=[std])
                        ])

    # get train data w/ transforms
    train_data = datasets.MNIST(root=root, 
                                train=True, 
                                download=True,
                                transform=data_transforms)

    test_data = datasets.MNIST(root=root, 
                               train=False, 
                               download=True,
                               transform=data_transforms)

    train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_iterator = data.DataLoader(test_data, batch_size=batch_size)

    return train_iterator, test_iterator

def get_fashion_MNIST(batch_size, root='data'):

    root = 'data'

    # get train data
    train_data = datasets.FashionMNIST(root=root, 
                                       train=True, 
                                       download=True)

    # calculate mean and std on train data
    mean = train_data.data.float().mean() / 255
    std = train_data.data.float().std() / 255

    # define transforms
    data_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[mean], 
                                             std=[std])
                        ])

    # get train data w/ transforms
    train_data = datasets.FashionMNIST(root=root, 
                                       train=True, 
                                       download=True,
                                       transform=data_transforms)

    # get test data w/ transforms
    test_data = datasets.FashionMNIST(root=root, 
                                      train=False, 
                                      download=True,
                                      transform=data_transforms)

    train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_iterator = data.DataLoader(test_data, batch_size=batch_size)

    return train_iterator, test_iterator

def get_KMNIST(batch_size, root='data'):

    root = 'data'

    # get train data
    train_data = datasets.KMNIST(root=root, 
                                 train=True, 
                                 download=True)

    # calculate mean and std on train data
    mean = train_data.data.float().mean() / 255
    std = train_data.data.float().std() / 255

    # define transforms
    data_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[mean], 
                                             std=[std])
                        ])

    # get train data w/ transforms
    train_data = datasets.KMNIST(root=root, 
                                 train=True, 
                                 download=True,
                                 transform=data_transforms)

    # get test data w/ transforms
    test_data = datasets.KMNIST(root=root, 
                                train=False, 
                                download=True,
                                transform=data_transforms)

    train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_iterator = data.DataLoader(test_data, batch_size=batch_size)

    return train_iterator, test_iterator