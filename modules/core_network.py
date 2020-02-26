import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNCoreNetwork(nn.Module):
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

class LSTMCoreNetwork(nn.Module):
    def __init__(self, glimpse_hid_dim, location_hid_dim, recurrent_hid_dim):
        super().__init__()

        self.x2f = nn.Linear(glimpse_hid_dim+location_hid_dim, recurrent_hid_dim)
        self.h2f = nn.Linear(recurrent_hid_dim, recurrent_hid_dim) 

        self.x2i = nn.Linear(glimpse_hid_dim+location_hid_dim, recurrent_hid_dim)
        self.h2i = nn.Linear(recurrent_hid_dim, recurrent_hid_dim)

        self.x2o = nn.Linear(glimpse_hid_dim+location_hid_dim, recurrent_hid_dim)
        self.h2o = nn.Linear(recurrent_hid_dim, recurrent_hid_dim)

        self.x2c = nn.Linear(glimpse_hid_dim+location_hid_dim, recurrent_hid_dim)
        self.h2c = nn.Linear(recurrent_hid_dim, recurrent_hid_dim)

    def forward(self, glimpse_hidden, recurrent_hidden, recurrent_cell):

        # glimpse_hidden = [batch size, glimpse_hid_dim+location_hid_dim]
        # recurrent_hidden = [batch size, recurrent_hid_dim]
        # recurrent_hidden = [batch size, recurrent_hid_dim]

        f = torch.sigmoid(self.x2f(glimpse_hidden) + self.h2f(recurrent_hidden))
        i = torch.sigmoid(self.x2i(glimpse_hidden) + self.h2i(recurrent_hidden))
        o = torch.sigmoid(self.x2o(glimpse_hidden) + self.h2o(recurrent_hidden))

        recurrent_cell = f * recurrent_cell + i * torch.tanh(self.x2c(glimpse_hidden) + self.h2c(recurrent_hidden))

        recurrent_hidden = o * recurrent_cell

        # recurrent_hidden = [batch size, recurrent_hid_dim]

        return recurrent_hidden, recurrent_cell