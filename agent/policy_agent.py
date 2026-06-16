
# make an update function for backprop

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

            nn.Linear(64, 2)
        )

    def forward(self, x):

        return self.net(x)
    
#i also want to test this one

class PolicyNetwork2(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(64, 1)
        self.log_std_head = nn.Linear(64, 1)

    def forward(self, x):

        h = self.shared(x)

        mean = self.mean_head(h)
        log_std = self.log_std_head(h)

        return mean, log_std
    
class PolicyAgent:

    def __init__(self, input_dim):

        self.policy = PolicyNetwork(input_dim)

    def act(self, state):

        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        mean_theta, log_sigma = self.policy(state)
        sigma = torch.exp(log_sigma)
        theta = torch.normal(mean_theta, sigma)

        theta = np.pi * torch.tanh(theta)

        return theta.item()

