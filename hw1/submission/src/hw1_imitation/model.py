"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.network = nn.Sequential()
        self.network.add_module("input", nn.Linear(state_dim, hidden_dims[0]))
        self.network.add_module("relu_in", nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            self.network.add_module(f"hidden_{i}", nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.network.add_module(f"relu_{i}", nn.ReLU())
        self.network.add_module("output", nn.Linear(hidden_dims[-1], action_dim * chunk_size))
        print(self.network)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # print("state shape: ", state.shape) --> (B, 5)
        # print("action_chunk shape: ", action_chunk.shape) --> (B, 8, 2)
        pred = self.network(state).view(-1, self.chunk_size, self.action_dim)
        # print(pred.shape) --> (B, 8, 2)
        loss = nn.MSELoss()(pred, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            pred = self.network(state).view(-1, self.chunk_size, self.action_dim)
        return pred


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.network = nn.Sequential()
        self.network.add_module("input", nn.Linear(state_dim + action_dim * chunk_size + 1, hidden_dims[0]))
        self.network.add_module("relu_in", nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            self.network.add_module(f"hidden_{i}", nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.network.add_module(f"relu_{i}", nn.ReLU())
        self.network.add_module("output", nn.Linear(hidden_dims[-1], action_dim * chunk_size))
        print(self.network)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.size(0)
        noise = torch.randn_like(action_chunk)
        # print("noise shape: ", noise.shape)
        # print("action_chunk shape: ", action_chunk.shape)
        tau = torch.rand(B, 1, 1, device=state.device)
        # print("tau shape: ", tau.shape)
        interpolate = (action_chunk * tau + noise * (1 - tau)).reshape(B, -1)
        tau = tau.reshape(B, 1)
        # print("interpolate shape: ", interpolate.shape)
        
        # print("state shape: ", state.shape)
        input = torch.cat([state, interpolate, tau], dim=-1)
        # print("input shape: ", input.shape)
        vels = self.network(input).view(-1, self.chunk_size, self.action_dim)
        # print("vels shape: ", vels.shape)
        diffs = action_chunk - noise
        # print("diffs shape: ", diffs.shape)
        loss = nn.MSELoss()(vels, diffs)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.size(0)
        dt = 1.0 / num_steps
        noise = torch.randn(B, self.chunk_size * self.action_dim, device=state.device)
        # print("noise shape: ", noise.shape)
        for step in range(num_steps):
            tau = torch.full((B, 1), (step + 1) * dt, device=state.device)
            vels = self.network(torch.cat([state, noise, tau], dim=-1))
            # print("vels shape: ", vels.shape)
            noise = noise + vels * dt
            # print("tau shape: ", tau.shape)
        res = noise.view(-1, self.chunk_size, self.action_dim)
        # print("res shape: ", res.shape)
        return res

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
