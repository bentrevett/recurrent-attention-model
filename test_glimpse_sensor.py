import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import data_loader
from modules.glimpse_sensor import GlimpseSensor

def save_images(images, labels, name):

    assert len(images) == len(labels) == 5

    # get rid of channel dimension of 1, if exists
    images = images.squeeze()

    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 5)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0.1)


    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i], cmap="Greys_r")

        #xlabel = f'{labels[i]}'
        #ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    # save image
    fig.savefig(name, bbox_inches = 'tight')

data = 'MNIST'
batch_size = 5
train_data, test_data = data_loader.get_data(data, batch_size)

# get batch 
test_iter = iter(test_data)
images, labels = test_iter.next()

# plot raw images
os.makedirs('images', exist_ok=True)
X = images.numpy()
X = np.transpose(X, [0, 2, 3, 1])
save_images(X, labels, 'images/raw')

# create glimpse sensor module
patch_size = 8
n_patches = 3
scale = 2
glimpse_sensor = GlimpseSensor(patch_size, n_patches, scale)

# location, [0,0] implies centre of the image
location = torch.zeros(batch_size, 2)

# put images through glimpse sensor, making it return list of pytorch tensor images
x = glimpse_sensor.get_patches(images, location, return_images=True)

# plot all patches for each of the batch of images
s = 1
for i, _x in enumerate(x):
    X = _x.cpu().numpy()
    X = np.transpose(X, [0,2,3,1])
    save_images(X, labels, f'images/patch_{i+1}_scale_{s}')
    s *= scale

# test shape of output
x = glimpse_sensor.get_patches(images, location)

n_channels = 1

assert x.shape == (batch_size, n_patches*n_channels*patch_size*patch_size)