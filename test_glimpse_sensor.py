import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import modules
import os

def save_images(images, labels, name):

    assert len(images) == len(labels) == 25

    # get rid of channel dimension of 1, if exists
    images = images.squeeze()

    # Create figure with sub-plots.
    fig, axes = plt.subplots(5, 5)
    fig.tight_layout(pad=0.1)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i], cmap="Greys_r")

        xlabel = f'{labels[i]}'
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    #save image
    fig.savefig(name)

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

# plot raw images
os.makedirs('images', exist_ok=True)
X = images.numpy()
X = np.transpose(X, [0, 2, 3, 1])
save_images(X, labels, 'images/raw')

# create glimpse sensor module
patch_size = 8
n_patches = 3
scale = 2
glimpse_sensor = modules.GlimpseSensor(patch_size, n_patches, scale)

# coords, [0,0] implies centre of the image
coords = torch.zeros(batch_size, 2)

# put images through glimpse sensor, making it return list of pytorch tensor images
x = glimpse_sensor.get_patches(images, coords, return_images=True)

# plot all patches for each of the batch of images
for i, _x in enumerate(x):
    X = _x.cpu().numpy()
    X = np.transpose(X, [0,2,3,1])
    save_images(X, labels, f'images/patch_{i}')