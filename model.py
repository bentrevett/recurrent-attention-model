import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from modules.glimpse_network import GlimpseNetwork
from modules.core_network import CoreNetwork
from modules.location_network import LocationNetwork
from modules.action_network import ActionNetwork
from modules.baseline_network import BaselineNetwork

class RecurrentAttentionModel(nn.Module):
    def __init__(self, n_glimpses, n_channels, patch_size, n_patches, scale, glimpse_hid_dim, location_hid_dim, recurrent_hid_dim, std, output_dim):
        super().__init__()

        assert n_glimpses >= 1
        assert n_channels in [1, 3]
        assert patch_size >= 1
        assert n_patches >= 1
        assert scale >= 1
        assert glimpse_hid_dim >= 1
        assert location_hid_dim >= 1
        assert glimpse_hid_dim + location_hid_dim == recurrent_hid_dim
        assert std >= 0
        assert output_dim >= 2

        self.n_glimpses = n_glimpses
        self.std = std

        self.glimpse_network = GlimpseNetwork(n_channels, patch_size, n_patches, scale, glimpse_hid_dim, location_hid_dim)
        self.core_network = CoreNetwork(glimpse_hid_dim, location_hid_dim, recurrent_hid_dim)
        self.location_network = LocationNetwork(recurrent_hid_dim)
        self.action_network = ActionNetwork(recurrent_hid_dim, output_dim)
        self.baseline_network = BaselineNetwork(recurrent_hid_dim)

        self.recurrent_hidden = nn.Parameter(torch.zeros(1, recurrent_hid_dim))

    def step(self, image, location, recurrent_hidden, std):

        # image = [batch size, n channels, height, width]
        # location = [batch size, 2]
        # recurrent_hidden = [batch size, recurrent hid dim]

        glimpse_hidden = self.glimpse_network(image, location)

        # glimpse_hidden = [batch size, glimpse hid dim + location hid dim]

        recurrent_hidden = self.core_network(glimpse_hidden, recurrent_hidden)

        # recurrent_hidden = [batch size, recurrent hid dim]

        location_mu, location_noise = self.location_network(recurrent_hidden, std)

        # location_mu = [batch size, 2]
        # location_noise = [batch size, 2]

        baseline = self.baseline_network(recurrent_hidden)

        # baseline = [batch size, 1]

        log_location_action = Normal(location_mu, std).log_prob(location_noise)

        log_location_action = log_location_action.sum(dim=1)

        # log_location_action = [batch size]

        return recurrent_hidden, location_noise, baseline, log_location_action

    def forward(self, image, location=None, std=None):

        assert len(image.shape) == 4

        batch_size = image.shape[0]

        if location is None:
            location = torch.FloatTensor(batch_size, 2).uniform_(-1, +1)
        else:
            assert len(location.shape) == 2 and location.shape[-1] == 2
            assert torch.max(location).item() <= 1.0, 'location x and y must be between [-1,+1]'
            assert torch.min(location).item() >= -1, 'location x and y must be between [-1,+1]'

        if std is None:
            std = self.std
        else:
            assert std >= 0

        # images = [batch size, n channels, height, width]
        # location = [batch size, 2]

        recurrent_hidden = self.recurrent_hidden.repeat(batch_size, 1)

        locations = []
        baselines = []
        log_location_actions = []

        for t in range(self.n_glimpses):
            recurrent_hidden, location, baseline, log_location_action = self.step(image, location, recurrent_hidden, std)
            locations.append(location)
            baselines.append(baseline)
            log_location_actions.append(log_location_action)

        log_classifier_actions = F.log_softmax(self.action_network(recurrent_hidden), dim=1)
        locations = torch.stack(locations, dim=1)
        baselines = torch.stack(baselines, dim=1)
        log_location_actions = torch.stack(log_location_actions, dim=1)

        return log_classifier_actions, log_location_actions, baselines, locations