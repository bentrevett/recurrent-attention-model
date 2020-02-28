import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--n_to_plot', type=int, default=25)
parser.add_argument('--n_rows', type=int, default=5)
args = parser.parse_args()

assert args.n_to_plot % args.n_rows == 0 

images = torch.load(f'checkpoints/{args.name}/images.pt')
predictions = torch.load(f'checkpoints/{args.name}/predictions.pt')
locations = torch.load(f'checkpoints/{args.name}/locations.pt')
params = torch.load(f'checkpoints/{args.name}/params.pt')
patch_size = params['patch_size']
n_patches = params['n_patches']
scale = params['scale']
std = params['std']

# images = [batch, n channels, height, width]
# locations = [batch, n glimpses, 2]

batch_size, n_channels, height, _ = images.shape
_, n_glimpses, _ = locations.shape

assert batch_size >= args.n_to_plot 

images = images[:args.n_to_plot]
predictions = predictions[:args.n_to_plot]
locations = locations[:args.n_to_plot]

# convert locations from [-1, +1] to [0, height]
locations = (0.5 * ((locations + 1.0) * height))

fig, axes = plt.subplots(nrows=args.n_rows, ncols=args.n_to_plot//args.n_rows)
fig.tight_layout(pad=0.1)

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
            while len(ax.patches) > 0:
                for p in ax.patches:
                    p.remove()
            loc = location[j]
            s = 1
            for k in range(n_patches):
                rect = bounding_box(
                    loc[0], loc[1], patch_size*s, color
                )
                ax.add_patch(rect)
                s = s * scale

anim = animation.FuncAnimation(
    fig, update_image, frames=n_glimpses, interval=1000, repeat=True)

name = f'images/{args.name}.gif'
anim.save(name, writer='imagemagick')