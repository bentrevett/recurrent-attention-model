import torch
import torch.nn as nn
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

class GlimpseNetwork(nn.Module):
    def __init__(self, n_channels, patch_size, n_patches, scale, glimpse_hid_dim, location_hid_dim):
        super().__init__()

        self.glimpse_sensor = GlimpseSensor(patch_size, n_patches, scale)
        self.fc_glimpse = nn.Linear(n_patches*n_channels*patch_size*patch_size, glimpse_hid_dim)
        self.fc_glimpse_out = nn.Linear(location_hid_dim, glimpse_hid_dim+location_hid_dim)
        self.fc_location = nn.Linear(2, location_hid_dim)
        self.fc_location_out = nn.Linear(location_hid_dim, glimpse_hid_dim+location_hid_dim)

    def forward(self, image, location):

        # image = [batch size, n channels, height, width]
        # location = [batch size, 2]

        glimpse = self.glimpse_sensor.get_patches(image, location)

        # glimpse = [batch size, n_patches*n_channels*patch_size*patch_size]

        glimpse = F.relu(self.fc_glimpse(glimpse))

        # glimpse = [batch size, glimpse hid dim]

        glimpse = self.fc_glimpse_out(glimpse)

        # glimpse = [batch size, glimpse hid dim + location hid dim]

        location = F.relu(self.fc_location(location))

        # location = [batch size, location hid dim]

        location = self.fc_location_out(location)

        # location = [batch size, glimpse hid dim + location hid dim]

        glimpse_hidden = F.relu(glimpse + location)

        # glimpse_hidden = [batch size, glimpse hid dim + location hid dim]

        return glimpse_hidden

class CoreNetwork(nn.Module):
    def __init__(self, glimpse_hid_dim, location_hid_dim, recurrent_hid_dim):
        super().__init__()

        self.i2h = nn.Linear(glimpse_hid_dim+location_hid_dim, recurrent_hid_dim)
        self.h2h = nn.Linear(recurrent_hid_dim, recurrent_hid_dim) 

    def forward(self, glimpse_hidden, recurrent_hidden):

        # glimpse_hidden = [batch size, glimpse_hid_dim+location_hid_dim]
        # recurrent_hidden = [batch size, recurrent_hid_dim]

        glimpse_hidden = self.i2h(glimpse_hidden)
        recurrent_hidden = self.h2h(recurrent_hidden)
        recurrent_hidden = F.relu(glimpse_hidden + recurrent_hidden)

        # recurrent_hidden = [batch size, recurrent_hid_dim]

        return recurrent_hidden

class LocationNetwork(nn.Module):
    def __init__(self, recurrent_hidden_dim, std):
        super().__init__()

        self.std = std
        self.fc = nn.Linear(recurrent_hidden_dim, 2)

    def forward(self, recurrent_hidden):

        # hidden = [batch size, hidden dim]

        # potentially this should be torch.clamp(self.fc(recurrent_hidden), -1, +1).detach()
        # https://github.com/kevinzakka/recurrent-visual-attention/issues/12
        # https://github.com/hehefan/Recurrent-Attention-Model/blob/master/model.py#L75
        location_mu = F.tanh(self.fc(recurrent_hidden))

        # mu = [batch size, hidden dim]

        noise = torch.zeros_like(location_mu)
        noise.data.normal_(std=self.std)

        # potentially this should be torch.clamp(mu + noise, -1, +1).detach()
        # https://github.com/kevinzakka/recurrent-visual-attention/issues/12
        # https://github.com/hehefan/Recurrent-Attention-Model/blob/master/model.py#L82
        location_noise = F.tanh(location_mu + noise).detach()

        # location_noise = [batch size, 2]
        # location_mu = [batch size, 2]

        return location_mu, location_noise

class ActionNetwork(nn.Module):
    def __init__(self, recurrent_hid_dim, output_dim):
        super().__init__()

        self.fc = nn.Linear(recurrent_hid_dim, output_dim)

    def forward(self, recurrent_hidden):

        # recurrent_hidden = [batch size, recurrent hid dim]

        action_logits = self.fc(recurrent_hidden)

        # action_logits = [batch size, output dim]

        return action_logits

class BaselineNetwork(nn.Module):
    def __init__(self, recurrent_hid_dim):
        super().__init__()

        self.fc = nn.Linear(recurrent_hid_dim, 1)

    def forward(self, recurrent_hidden):

        # recurrent_hidden = [batch size, recurrent hid dim]

        baseline = self.fc(recurrent_hidden)

        # baseline = [batch size, 1]

        return baseline
