import torch
import torch.nn.functional as F

class GlimpseSensor:
    def __init__(self, patch_size, n_patches, scale):

        self.patch_size = patch_size # size of patch
        self.n_patches = n_patches # number of patches
        self.scale = scale # size of subsequent patches

    def get_patches(self, image, location, return_images=False):

        # image = [batch size, n channels, height, width]
        # location = [batch size, 2]
        # if return_images == True, returns list of downsampled patches

        _, _, height, width = image.shape

        assert height == width, f'only works on square images, got [{height},{width}]'
        assert torch.max(location).item() <= 1.0, 'location x and y must be between [-1,+1]'
        assert torch.min(location).item() >= -1, 'location x and y must be between [-1,+1]'

        patches = []
        size = self.patch_size

        # extract `n_patches` that get bigger by `scale` each time
        for i in range(self.n_patches):
            patch = self.get_patch(image, location, size)
            patches.append(patch)
            size = int(size * self.scale)

        # resize patches by scaling down to `patch_size`
        for i in range(1, self.n_patches):
            downscale = patches[i].shape[-1] // self.patch_size
            patches[i] = F.avg_pool2d(patches[i], downscale)

        if return_images:
            return patches

        # concat and flatten
        patches = torch.cat(patches, 1)

        patches = patches.view(patches.shape[0], -1)

        # patches = [batch, n_patches*n_channels*patch_size*patch_size]

        return patches

    def get_patch(self, image, location, size):

        batch_size, _, height, _ = image.shape

        # convert `location` from [-1, 1] to [0, height]
        location = (0.5 * ((location + 1.0) * height)).long()

        # how much padding for the (left, right, top, bottom)
        pad_dims = (
                    size//2, size//2,
                    size//2, size//2,
                    )
        
        # pad images
        image = F.pad(image, pad_dims, 'replicate')

        # patch x, y location
        from_x, from_y = location[:, 0], location[:, 1]
        to_x, to_y = from_x + size, from_y + size

        patches = []
        
        # get patches from padded images
        for i in range(batch_size):
            patches.append(image[i, :, from_y[i]:to_y[i], from_x[i]:to_x[i]].unsqueeze(0))

        patches = torch.cat(patches)

        return patches