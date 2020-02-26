import torch
import torch.nn as nn
import torch.nn.functional as F

class LocationNetwork(nn.Module):
    def __init__(self, recurrent_hidden_dim):
        super().__init__()

        self.fc = nn.Linear(recurrent_hidden_dim, 2)

    def forward(self, recurrent_hidden, std):

        # hidden = [batch size, hidden dim]

        # potentially this should be torch.clamp(self.fc(recurrent_hidden), -1, +1).detach()
        # https://github.com/kevinzakka/recurrent-visual-attention/issues/12
        # https://github.com/hehefan/Recurrent-Attention-Model/blob/master/model.py#L75
        location_mu = torch.tanh(self.fc(recurrent_hidden))

        # mu = [batch size, hidden dim]

        noise = torch.zeros_like(location_mu)
        noise.data.normal_(std=std)

        # potentially this should be torch.clamp(mu + noise, -1, +1).detach()
        # https://github.com/kevinzakka/recurrent-visual-attention/issues/12
        # https://github.com/hehefan/Recurrent-Attention-Model/blob/master/model.py#L82
        location_noise = torch.tanh(location_mu + noise).detach()

        # location_noise = [batch size, 2]
        # location_mu = [batch size, 2]

        return location_mu, location_noise