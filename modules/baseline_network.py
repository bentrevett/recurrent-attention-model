import torch.nn as nn
import torch.nn.functional as F

class BaselineNetwork(nn.Module):
    def __init__(self, recurrent_hid_dim):
        super().__init__()

        self.fc = nn.Linear(recurrent_hid_dim, 1)

    def forward(self, recurrent_hidden):

        # recurrent_hidden = [batch size, recurrent hid dim]

        baseline = F.relu(self.fc(recurrent_hidden))

        # baseline = [batch size, 1]

        return baseline