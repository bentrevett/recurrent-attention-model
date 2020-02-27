import torch
import torch.nn as nn
import torch.nn.functional as F

class LocationNetwork(nn.Module):
    def __init__(self, recurrent_hidden_dim, std):
        super().__init__()

        self.fc = nn.Linear(recurrent_hidden_dim, 2)
        self.std = std

    def forward(self, recurrent_hidden):

        # hidden = [batch size, hidden dim]

        location_mu = torch.clamp(self.fc(recurrent_hidden.detach()), -1, +1)

        # location_mu = [batch size, hidden dim]

        noise = torch.zeros_like(location_mu)
        noise.data.normal_(mean=0, std=self.std)

        location = torch.clamp(location_mu + noise, -1, +1).detach()

        # location_noise = [batch size, 2]
        # location_mu = [batch size, 2]

        return location, location_mu