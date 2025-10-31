import math

import torch
import torch.nn as nn
from bbrl.agents import Agent
from bbrl_utils.nn import build_mlp


class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, device):
        super().__init__()
        self.is_q_function = True
        self.device = device
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.Tanh()
        )

    def forward(self, t):
        obs = self.get(("env/env_obs", t)).to(self.device)
        action = self.get(("action", t)).to(self.device)
        obs_act = torch.cat((obs, action), dim=1)
        q_value = self.model(obs_act).squeeze(-1)
        self.set((f"{self.prefix}q_value", t), q_value)


class ContinuousDeterministicActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, device):
        super().__init__()
        self.device = device
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(layers, activation=nn.Tanh(), output_activation=nn.Tanh())

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t)).to(self.device)
        action = self.model(obs)
        self.set(("action", t), action)


class AddOUNoise(Agent):
    def __init__(self, std_dev, device, theta=0.15, dt=1e-2):
        super().__init__()
        self.theta = theta
        self.std_dev = std_dev
        self.dt = dt
        self.device = device
        self.x_prev = None

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        if act is None:
            return
        act = act.to(self.device)
        if self.x_prev is None:
            self.x_prev = torch.zeros_like(act, device=self.device)
        x = (
            self.x_prev
            + self.theta * (act - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape, device=self.device)
        )
        self.x_prev = x
        self.set(("action", t), x)
