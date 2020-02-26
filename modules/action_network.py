import torch.nn as nn

class ActionNetwork(nn.Module):
    def __init__(self, recurrent_hid_dim, output_dim):
        super().__init__()

        self.fc = nn.Linear(recurrent_hid_dim, output_dim)

    def forward(self, recurrent_hidden):

        # recurrent_hidden = [batch size, recurrent hid dim]

        action_logits = self.fc(recurrent_hidden)

        # action_logits = [batch size, output dim]

        return action_logits