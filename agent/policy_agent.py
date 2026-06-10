
#why is agent moving in a straight line??
#balance exploitation and exploration

import torch
import torch.nn as nn
import numpy as np


class PolicyNetwork(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        return self.net(x)
    
class PolicyAgent:

    def __init__(self, input_dim):

        self.policy = PolicyNetwork(input_dim)

    def act(self, state):

        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():
            theta = self.policy(state)

        theta = np.pi * torch.tanh(theta)

        return theta.item()

