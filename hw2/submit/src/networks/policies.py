import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu

debug = False

class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO(?V): implement get_action
        obs = ptu.from_numpy(obs).unsqueeze(0)
        action = self.forward(obs).sample().cpu().numpy()
        # NTS: if pure exploitation action = action.mean
        if(0):
            print(f"get_action: obs shape: {obs.shape} \n\tobs: {obs} \n\taction shape: {action.shape} \n\taction: {action}")

        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        ret = None
        if self.discrete:
            # TODO(?V): define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)
            # x_softmax = F.softmax(logits, dim=-1)
            # ret = D.Categorical(probs=x_softmax)
            ret = D.Categorical(logits=logits)
        else:
            # TODO(?V): define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs)
            logstd = self.logstd.expand_as(mean)
            ret = D.Normal(mean, torch.exp(logstd))
        return ret

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions).squeeze(-2)
        advantages = ptu.from_numpy(advantages)

        if self.discrete:
            actions = actions.long().squeeze(-1)

        # TODO(?V): compute the policy gradient actor loss
        if(debug):
            print(f"Shape of obs: {obs.shape}")
            print(f"Shape of actions: {actions.shape}")
        if(self.discrete):
            log_probs = self.forward(obs).log_prob(actions)
        else:
            log_probs = self.forward(obs).log_prob(actions).sum(dim=-1)
        loss = -log_probs * advantages
        if(debug):
            print(f"Shape of log probabilities: {log_probs.shape}")
            print(f"Shape of advantages: {advantages.shape}")
            print(f"Shape of loss: {loss.shape} \n\tloss: {loss}")
        loss = loss.mean()

        # TODO(V): perform an optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
        }
