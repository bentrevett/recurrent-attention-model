import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random

def get_data(which, batch_size, root='data'):

    # get train data
    train_data = getattr(datasets, which)(root=root,
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
    train_data = getattr(datasets, which)(root=root,
                                          train=True,
                                          download=True,
                                          transform=data_transforms)

    # get test data w/ transforms
    test_data = getattr(datasets, which)(root=root,
                                         train=False,
                                         download=True,
                                         transform=data_transforms)

    # load train and test iterators
    train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_iterator = data.DataLoader(test_data, batch_size=batch_size)

    return train_iterator, test_iterator

def get_translated_data(which, translated_size, batch_size, root='data'):

    # hard coded for MNIST style datasets
    image_size = 28

    # define transform that does nothing
    data_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        ])

    # get train data
    train_data = getattr(datasets, which)(root=root,
                                          train=True,
                                          download=True,
                                          transform=data_transforms)

    # create collator
    collator = TranslatedCollator(image_size, translated_size)

    # create iterator
    train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator.collate)

    # stack all images in giant tensor
    images = [image for image, label in train_iterator]
    images = torch.cat(images, dim=0)

    #get mean and std
    mean = images.mean()
    std = images.std()

    # define transforms with normalization
    data_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[mean],
                                             std=[std])
                        ])

    # get train data w/ transforms
    train_data = getattr(datasets, which)(root=root,
                                          train=True,
                                          download=True,
                                          transform=data_transforms)

    # get test data w/ transforms
    test_data = getattr(datasets, which)(root=root,
                                         train=False,
                                         download=True,
                                         transform=data_transforms)

    # load collator
    collator = TranslatedCollator(image_size, translated_size)

    # load train and test iterators
    train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator.collate)
    test_iterator = data.DataLoader(test_data, batch_size=batch_size, collate_fn=collator.collate)

    return train_iterator, test_iterator

def get_cluttered_data(which, translated_size, n_clutter, clutter_size, batch_size, root='data'):

    # hard coded for MNIST style datasets
    image_size = 28

    # define transform that does nothing
    data_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        ])

    # get train data w/ transforms
    train_data = getattr(datasets, which)(root=root,
                                          train=True,
                                          download=True,
                                          transform=data_transforms)

    # load collator
    collator = ClutteredCollator(image_size, translated_size, n_clutter, clutter_size)

    # create iterator
    train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator.collate)

    # stack all images in giant tensor
    images = [image for image, label in train_iterator]
    images = torch.cat(images, dim=0)

    #get mean and std
    mean = images.mean()
    std = images.std()

    # define transforms with normalization
    data_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[mean],
                                             std=[std])
                        ])

    # get train data w/ transforms
    train_data = getattr(datasets, which)(root=root,
                                          train=True,
                                          download=True,
                                          transform=data_transforms)

    # get test data w/ transforms
    test_data = getattr(datasets, which)(root=root,
                                         train=False,
                                         download=True,
                                         transform=data_transforms)

    # load collator
    collator = ClutteredCollator(image_size, translated_size, n_clutter, clutter_size)

    # load train and test iterators
    train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator.collate)
    test_iterator = data.DataLoader(test_data, batch_size=batch_size, collate_fn=collator.collate)

    return train_iterator, test_iterator

class TranslatedCollator:
    def __init__(self, image_size, translated_size):
        self.image_size = image_size
        self.translated_size = translated_size

    def collate(self, batch):
        images, labels = zip(*batch)
        batch_size = len(images)
        background = torch.zeros(batch_size, 1, self.translated_size, self.translated_size)
        image_pos = torch.randint(0, self.translated_size-self.image_size, (batch_size, 2))
        for i, image in enumerate(images):
            background[i,:,image_pos[i][1]:image_pos[i][1]+self.image_size,image_pos[i][0]:image_pos[i][0]+self.image_size] = image
        labels = torch.LongTensor(labels)
        return background, labels

class ClutteredCollator:
    def __init__(self, image_size, translated_size, n_clutter, clutter_size):
        self.image_size = image_size
        self.translated_size = translated_size
        self.n_clutter = n_clutter
        self.clutter_size = clutter_size

    def collate(self, batch):
        images, labels = zip(*batch)
        batch_size = len(images)
        background = torch.zeros(batch_size, 1, self.translated_size, self.translated_size)
        clutter_slice = torch.randint(0, self.image_size-self.clutter_size, (batch_size, self.n_clutter, 2))
        clutter_pos = torch.randint(0, self.translated_size-self.clutter_size, (batch_size, self.n_clutter, 2))
        image_pos = torch.randint(0, self.translated_size-self.image_size, (batch_size, 2))
        for i, image in enumerate(images):
            for j in range(self.n_clutter):
                clutter_full = random.choice(images)
                clutter = clutter_full[:,clutter_slice[i][j][1]:clutter_slice[i][j][1]+self.clutter_size,clutter_slice[i][j][0]:clutter_slice[i][j][0]+self.clutter_size]
                background[i,:,clutter_pos[i][j][1]:clutter_pos[i][j][1]+self.clutter_size,clutter_pos[i][j][0]:clutter_pos[i][j][0]+self.clutter_size] = clutter
            background[i,:,image_pos[i][1]:image_pos[i][1]+self.image_size,image_pos[i][0]:image_pos[i][0]+self.image_size] = image
        labels = torch.LongTensor(labels)
        return background, labels
