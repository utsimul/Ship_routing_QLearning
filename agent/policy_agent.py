
# make an update function for backprop

from shapely import distance
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from Helpers import *



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


    
class PolicyAgent:

    def __init__(self, input_dim):

        self.policy = PolicyNetwork(input_dim)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=1e-4
        )

    def act(self, state, goal_direction, distance):

        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        mean_theta, log_sigma = self.policy(state)

        sigma = torch.exp(log_sigma)

        # less exploration near goal
        #sigma *= min(distance / 200.0, 1.0)

        dist = Normal(mean_theta, sigma)

        raw_theta = dist.sample()

        log_prob = dist.log_prob(raw_theta)

        theta = np.pi * torch.tanh(raw_theta)
        theta += goal_direction

        print(
            "mean=", mean_theta.item(),
            "sigma=", sigma.item()
        )

        return theta.item(), log_prob
    
    def update(self, rewards, log_probs, gamma=0.99):

        returns = []

        G = 0

        for r in reversed(rewards):

            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        # optional normalization
        returns = (
            returns - returns.mean()
        ) / (returns.std() + 1e-8)

        loss = 0

        for log_prob, G in zip(log_probs, returns):

            loss += -log_prob * G

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        print("Policy loss:", loss.item())
    
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

