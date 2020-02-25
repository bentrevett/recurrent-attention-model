import torch
import torch.nn as nn
import torch.nn.functional as F

class GlimpseSensor:
    def __init__(self, patch_size, n_patches, scale):

        self.patch_size = patch_size # size of patch
        self.n_patches = n_patches # number of patches
        self.scale = scale # size of subsequent patches

    def get_patches(self, images, locations, return_images=False):

        # images = [batch size, n channels, height, width]
        # locations = [batch size, 2]
        # if return_images == True, returns list of downsampled patches

        _, _, height, width = images.shape

        assert height == width, f'only works on square images, got [{height},{width}]'
        assert torch.max(locations).item() <= 1.0, 'locations must be between [-1,+1]'
        assert torch.min(locations).item() >= -1, 'locations must be between [-1,+1]'

        patches = []
        size = self.patch_size

        # extract `n_patches` that get bigger by `scale` each time
        for i in range(self.n_patches):
            patch = self.get_patch(images, locations, size)
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

    def get_patch(self, images, locations, size):

        batch_size, _, height, _ = images.shape

        # convert `locations` from [-1, 1] to [0, height]
        locations = (0.5 * ((locations + 1.0) * height)).long()

        # how much padding for the (left, right, top, bottom)
        pad_dims = (
                    size//2, size//2,
                    size//2, size//2,
                    )
        
        # pad images
        images = F.pad(images, pad_dims, 'replicate')

        # patch x, y locations
        from_x, from_y = locations[:, 0], locations[:, 1]
        to_x, to_y = from_x + size, from_y + size

        patches = []
        
        # get patches from padded images
        for i in range(batch_size):
            patches.append(images[i, :, from_y[i]:to_y[i], from_x[i]:to_x[i]].unsqueeze(0))

        patches = torch.cat(patches)

        return patches

class GlimpseNetwork(nn.Module):
    def __init__(self, n_channels, patch_size, n_patches, scale, glimpse_hid_dim, locations_hid_dim):
        super().__init__()

        self.glimpse_sensor = GlimpseSensor(patch_size, n_patches, scale)
        self.fc_glimpse = nn.Linear(n_patches*n_channels*patch_size*patch_size, glimpse_hid_dim)
        self.fc_glimpse_out = nn.Linear(locations_hid_dim, glimpse_hid_dim+locations_hid_dim)
        self.fc_locations = nn.Linear(2, locations_hid_dim)
        self.fc_locations_out = nn.Linear(locations_hid_dim, glimpse_hid_dim+locations_hid_dim)

    def forward(self, images, locations):

        # images = [batch size, n channels, height, width]
        # locations = [batch size, 2]

        glimpse = self.glimpse_sensor.get_patches(images, locations)

        # glimpse = [batch size, n_patches*n_channels*patch_size*patch_size]

        glimpse = F.relu(self.fc_glimpse(glimpse))

        # glimpse = [batch size, glimpse hid dim]

        glimpse = self.fc_glimpse_out(glimpse)

        # glimpse = [batch size, glimpse hid dim + locations hid dim]

        locations = F.relu(self.fc_locations(locations))

        # locations = [batch size, locations hid dim]

        locations = self.fc_locations_out(locations)

        # locations = [batch size, glimpse hid dim + locations hid dim]

        out = F.relu(glimpse + locations)

        # out = [batch size, glimpse hid dim + locations hid dim]

        return out