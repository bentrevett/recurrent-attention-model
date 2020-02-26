import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os

import model

root = 'data'

# get data
data = datasets.MNIST(root=root, 
                      train=True, 
                      download=True)

# calculate mean and std
mean = data.data.float().mean() / 255
std = data.data.float().std() / 255

# define transforms
transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[mean], 
                                                std=[std])
                       ])

# get data w/ transforms
data = datasets.MNIST(root=root, 
                      train=True, 
                      download=True,
                      transform=transforms)

# get data loader
batch_size = 25
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)

# get batch 
data_iter = iter(data_loader)
images, labels = data_iter.next()

# create glimpse sensor module
n_glimpses = 6
n_channels = 1
patch_size = 8
n_patches = 1
scale = 1
glimpse_hid_dim = 128
locations_hid_dim = 128
recurrent_hid_dim = 256
std = 0.1
output_dim = 10
ram = model.RecurrentAttentionModel(n_glimpses, n_channels, patch_size, 
                                    n_patches, scale, glimpse_hid_dim, 
                                    locations_hid_dim, recurrent_hid_dim, 
                                    std, output_dim)

#_glimpses, n_channels, patch_size, n_patches, scale, glimpse_hid_dim, locations_hid_dim, recurrent_hid_dim, std, output_dim)

# coords, [0,0] implies centre of the image
locations = torch.zeros(batch_size, 2)

# put images through glimpse network
log_classifier_actions, log_locations_actions, baselines, locations = ram(images)

print(log_classifier_actions.shape)
print(log_locations_actions.shape)
print(baselines.shape)
print(locations.shape)