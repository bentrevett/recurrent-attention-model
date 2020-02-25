import torch.nn as nn
import torch.distributions.normal.Normal as Normal

from modules import GlimpseNetwork
from modules import CoreNetwork
from modules import LocationNetwork
from modules import ActionNetwork
from modules import CriticNetwork

class RecurrentAttentionModule(nn.Module):
    def __init__(self, n_glimpses, n_channels, patch_size, n_patches, scale, glimpse_hid_dim, locations_hid_dim, recurrent_hid_dim, std, output_dim):
        super().__init__()
        
        self.n_glimpses = n_glimpses
        self.std = std

        self.glimpse_network = GlimpseNetwork(n_channels, patch_size, n_patches, scale, glimpse_hid_dim, locations_hid_dim)
        self.core_network = CoreNetwork(glimpse_hid_dim, locations_hid_dim, recurrent_hid_dim)
        self.location_network = LocationNetwork(recurrent_hid_dim, std)
        self.action_network = ActionNetwork(recurrent_hid_dim, output_dim)
        self.critic_network = CriticNetwork(recurrent_hid_dim)

        self.recurrent_hidden = nn.Parameter(torch.zeros(1, recurrent_hid_dim))

    def step(self, images, locations, recurrent_hidden):

        # images = [batch size, n channels, height, width]
        # locations = [batch size, 2]
        # recurrent_hidden = [batch size, recurrent hid dim]

        glimpse_hidden = self.glimpse_network(images, locations)

        # glimpse_hidden = [batch size, glimpse hid dim + locations hid dim]

        recurrent_hidden = self.core_network(glimpse_hidden, recurrent_hidden)

        # recurrent_hidden = [batch size, recurrent hid dim]

        locations_mu, locations_noise = self.location_network(recurrent_hidden)

        # locations_mu = [batch size, 2]
        # locations_noise = [batch size, 2]

        baseline = self.critic_network(recurrent_hidden)

        # baseline = [batch size, 1]

        log_actions = Normal(locations_mu, self.std).log_pob(locations_noise)

        log_actions = log_actions.sum(dim=1)

        # log_actions = [batch size]

        return recurrent_hidden, locations_mu, locations_noise, baseline, log_actions

    def forward(self, images, locations=None):

        if locations is None:
            locations = torch.FloatTensor(batch_size, 2).uniform_(-1, +1)

        # images = [batch size, n channels, height, width]
        # locations = [batch size, 2]

        batch_size = images.shape[0]

        recurrent_hidden = self.recurrent_hidden.repeat(batch_size, 1)

        # https://github.com/ipod825/recurrent-visual-attention/blob/master/modules.py

        for t in range(self.n_glimpses):
            recurrent_hidden, locations_mu, locations_noise, baseline, log_actions = self.step(images, locations, recurrent_hidden)