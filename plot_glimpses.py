import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='mnist')
parser.add_argument('--n_to_plot', type=int, default=10)
args = parser.parse_args()

images = torch.load(f'checkpoints/{args.data}-images.pt')
predictions = torch.load(f'checkpoints/{args.data}-predictions.pt')
locations = torch.load(f'checkpoints/{args.data}-locations.pt')
params = torch.load(f'checkpoints/{args.data}-params.pt')
patch_size = params['patch_size']
n_patches = params['n_patches']
scale = params['scale']

# images = [batch, n channels, height, width]
# locations = [batch, n glimpses, 2]

batch_size, n_channels, height, _ = images.shape
_, n_glimpses, _ = locations.shape

n_to_plot = min(args.n_to_plot, batch_size)

images = images[:n_to_plot]
predictions = predictions[:n_to_plot]
locations = locations[:n_to_plot]

# convert locations from [-1, +1] to [0, height]
locations = (0.5 * ((locations + 1.0) * height))

fig, axes = plt.subplots(nrows=1, ncols=n_to_plot)

images = images.cpu().numpy()
images = np.transpose(images, [0, 2, 3, 1])
images = images.squeeze()

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap="Greys_r")
    xlabel = f'{predictions[i]}'
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_yticks([])

def bounding_box(x, y, size, color='w'):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect

def update_image(i):
        color = 'r'
        location = locations[:,i]
        for j, ax in enumerate(axes.flat):
            for p in ax.patches:
                p.remove()
            loc = location[j]
            rect = bounding_box(
                loc[0], loc[1], patch_size, color
            )
            ax.add_patch(rect)

anim = animation.FuncAnimation(
    fig, update_image, frames=n_glimpses, interval=500, repeat=True)

name = f'images/{args.data}.mp4'
anim.save(name, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])