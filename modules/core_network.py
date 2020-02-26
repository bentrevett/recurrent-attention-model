import torch.nn as nn
import torch.nn.functional as F

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