import torch.nn as nn
import torch.nn.functional as F

from .glimpse_sensor import GlimpseSensor

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