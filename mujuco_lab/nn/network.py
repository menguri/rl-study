import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic network (Q-network)
        self.fc1 = nn.Linear(state_dim, 128)  # State + action as input
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Output V(s) as a single value

    def forward(self, state, action):
        # Concatenate state and action for V(s) computation
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v_value = self.fc3(x)
        return v_value
