from typing import Optional, Sequence
import numpy as np
import torch

from networks.critics import ValueCritic
from networks.policies import MLPPolicyPG
from infrastructure import pytorch_util as ptu
from torch import nn

debug = False

class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO(V): flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        q_values = np.concatenate(q_values, axis=0)

        if(debug):
            print(f"Agent update: Rewards datatype: {type(rewards)}")
            print(f"Agent update: Rewards[0] datatype: {type(rewards[0])}")
            print(f"Agent update: Rewards shape: {rewards.shape}")
            print(f"Agent update: Q values datatype: {type(q_values)}")
            print(f"Agent update: Q values shape: {q_values.shape}")

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO(V): update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO(V): perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info = None
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, q_values)
                
            info.update(critic_info)

        return info

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        rwd = 0
        # if(debug):
        #     print(f"Length of rewards: {len(rewards)}")
        #     print(f"Rewards datatype: {type(rewards[0])}")
        for reward in rewards[:0:-1]:
            rwd += reward
            rwd *= self.gamma
        rwd += rewards[0]
        disc_return = [rwd] * len(rewards)
        return disc_return

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        rwd = 0
        rtg = []
        for t in range(len(rewards)):
            rwd += rewards[-1 - t]
            rtg.append(rwd)
            rwd *= self.gamma
        rtg = rtg[::-1]
        return rtg

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""
        if(debug):
            print(f"Datatype of rewards: {type(rewards)}")
            print(f"Datatype of rewards[0]: {type(rewards[0])}")
            print(f"Length of rewards: {len(rewards)}")
            print(f"Shape of rewards: {rewards[0].shape}")
        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO(V): use the helper function self._discounted_return to calculate the Q-values
            q_values = []
            for reward in rewards:
                q_values.append(np.array(self._discounted_return(reward)))
            if(debug):
                print(f"Datatype Q values: {type(q_values)}")
                print(f"Datatype Q values[0]: {type(q_values[0])}")
                print(f"Length of Q values: {len(q_values)}")
                print(f"Shape of Q values: {q_values[0].shape}")
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO(V): use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = []
            for reward in rewards:
                q_values.append(np.array(self._discounted_reward_to_go(reward)))
            if(debug):
                print(f"Datatype Q values: {type(q_values)}")
                print(f"Shape of Q values: {q_values[0].shape}")
        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO(V): if no baseline, then what are the advantages?
            advantages = q_values
        else:
            # TODO(V): run the critic and use it as a baseline
            ob_tensor = ptu.from_numpy(obs)
            values = self.critic(ob_tensor).detach().cpu().numpy()
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO(?V): if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0] ####

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    if terminals[i] == 1:
                        advantages[i] = q_values[i] - values[i]
                    else:
                        delta = rewards[i] + self.gamma * values[i + 1] - values[i]
                        advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i + 1]

                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            advantages = np.array(advantages)
            mean = np.mean(advantages)
            std = np.std(advantages)
            advantages = (advantages - mean) / (std + 1e-8)

        return advantages