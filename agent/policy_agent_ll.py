import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


# ACTOR NETWORK

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

        output = self.net(x)

        mean = output[..., 0]
        log_std = output[..., 1]

        return mean, log_std

# CRITIC NETWORK

class ValueNetwork(nn.Module):

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

        return self.net(x).squeeze(-1)


# PPO AGENT

class PolicyAgent:

    def __init__(
        self,
        input_dim,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=10,
        minibatch_size=256
    ):

        self.policy = PolicyNetwork(input_dim) #actor
        self.value_network = ValueNetwork(input_dim) #critic

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) +
            list(self.value_network.parameters()),
            lr=learning_rate
        )



        self.gamma = gamma

        self.gae_lambda = gae_lambda

        self.clip_epsilon = clip_epsilon

        self.value_coef = value_coef

        self.entropy_coef = entropy_coef

        self.ppo_epochs = ppo_epochs

        self.minibatch_size = minibatch_size

    def get_value(self, state):

        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():

            value = self.value_network(state)

        return value.item()

    def act(
        self,
        state,
        goal_direction,
        distance
    ):

        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        # Actor
        mean, log_std = self.policy(state)


        log_std = torch.clamp(
            log_std,
            -5.0,
            2.0
        )

        std = torch.exp(log_std)


        dist = Normal(mean, std)

        raw_action = dist.sample()


        log_prob = dist.log_prob(
            raw_action
        )


        value = self.value_network(state)


        theta = (
            np.pi *
            torch.tanh(raw_action)
        )

        theta = 0.5*theta + goal_direction

        theta = (
            theta + np.pi
        ) % (
            2 * np.pi
        ) - np.pi


        return (
            theta.item(),
            log_prob.item(),
            value.item(),
            raw_action.item()
        )


    def update(
        self,
        states,
        raw_actions,
        old_log_probs,
        returns,
        advantages
    ):


        states = torch.tensor(
            states,
            dtype=torch.float32
        )

        raw_actions = torch.tensor(
            raw_actions,
            dtype=torch.float32
        )

        old_log_probs = torch.tensor(
            old_log_probs,
            dtype=torch.float32
        )

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        advantages = torch.tensor(
            advantages,
            dtype=torch.float32
        )


        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )


        num_samples = len(states)


        for epoch in range(self.ppo_epochs):


            indices = torch.randperm(
                num_samples
            )


            for start in range(
                0,
                num_samples,
                self.minibatch_size
            ):

                end = min(
                    start + self.minibatch_size,
                    num_samples
                )

                batch_indices = indices[start:end]


                batch_states = states[
                    batch_indices
                ]

                batch_actions = raw_actions[
                    batch_indices
                ]

                batch_old_log_probs = old_log_probs[
                    batch_indices
                ]

                batch_returns = returns[
                    batch_indices
                ]

                batch_advantages = advantages[
                    batch_indices
                ]

                mean, log_std = self.policy(
                    batch_states
                )


                log_std = torch.clamp(
                    log_std,
                    -5.0,
                    2.0
                )

                std = torch.exp(log_std)


                dist = Normal(
                    mean,
                    std
                )


                new_log_probs = dist.log_prob(
                    batch_actions
                )

                entropy = dist.entropy().mean()



                ratio = torch.exp(
                    new_log_probs -
                    batch_old_log_probs
                )


                surr1 = (
                    ratio *
                    batch_advantages
                )

                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon
                    )
                    *
                    batch_advantages
                )


                policy_loss = -torch.min(
                    surr1,
                    surr2
                ).mean()

                values = self.value_network(
                    batch_states
                )


                value_loss = (
                    batch_returns -
                    values
                ).pow(2).mean()


                total_loss = (
                    policy_loss
                    +
                    self.value_coef *
                    value_loss
                    -
                    self.entropy_coef *
                    entropy
                )


                self.optimizer.zero_grad()

                total_loss.backward()


                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) +
                    list(self.value_network.parameters()),
                    max_norm=0.5
                )


                self.optimizer.step()


        print(
            f"PPO update | "
            f"Policy Loss: {policy_loss.item():.4f} | "
            f"Value Loss: {value_loss.item():.4f} | "
            f"Entropy: {entropy.item():.4f}"
        )