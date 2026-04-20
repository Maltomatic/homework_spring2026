from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class SACBCAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_beta,
        make_beta_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
    ):
        super().__init__()

        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.beta = make_beta()

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.beta_optimizer = make_beta_optimizer(self.beta.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha

        self.target_entropy = -action_dim / 2  # Heuristic value (|A| / 2) from the SAC paper.

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # Get the mode action from a tanh transformed distribution.
        action = self.actor(observation).base_dist.base_dist.mode.tanh()
        return ptu.to_numpy(action[0])

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(V)(student): Compute the Q loss
        q = self.critic(observations, actions)
        with torch.no_grad():
            next_act = self.actor(next_observations).rsample()
            next_q = self.target_critic(next_observations, next_act)
            y = rewards + self.discount * (1 - dones) * next_q.mean(dim = 0)
            
        y = y.unsqueeze(0).expand_as(q)
        loss = nn.MSELoss(reduction='none')(q, y).mean() * 2

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor
        """
        # TODO(V)(student): Compute the actor loss
        act_dist = self.actor(observations)
        q_loss = self.critic(observations, act_dist.rsample()).mean() * -0.5

        mses = nn.MSELoss(reduction="none")(act_dist.base_dist.base_dist.mean, actions)
        bc_loss = mses.mean() * self.alpha

        entropy_loss = self.beta() * act_dist.log_prob(act_dist.rsample()).mean()

        loss = q_loss + bc_loss + entropy_loss

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "total_loss": loss,
            "q_loss": q_loss,
            "bc_loss": bc_loss,
            "entropy_loss": entropy_loss,
            "mse": mses.mean(),
        }

    @torch.compile
    def update_beta(
        self,
        observations: torch.Tensor,
    ):
        """
        Update the beta parameter using dual gradient descent.
        """
        actor_dists = self.actor(observations)
        actor_actions = actor_dists.rsample()
        log_probs = actor_dists.log_prob(actor_actions)

        loss = self.beta() * (-log_probs - self.target_entropy).detach().mean()

        self.beta_optimizer.zero_grad()
        loss.backward()
        self.beta_optimizer.step()

        return {
            "beta_loss": loss,
            "beta": self.beta(),
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics_beta = self.update_beta(observations)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
            **{f"beta/{k}": v.item() for k, v in metrics_beta.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(V)(student): Update target_critic using Polyak averaging with self.target_update_rate
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data)
