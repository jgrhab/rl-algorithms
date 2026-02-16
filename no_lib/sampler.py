from dataclasses import dataclass

import torch
from gymnasium.vector import VectorEnv
from torch import Tensor, nn
from torch.distributions.categorical import Categorical


@dataclass
class Sample:
    """A sample from a vectorized environment."""

    states: Tensor
    """States (observations) `s_t` returned by the vectorized environment at time-step `t`.
    `shape = [steps, num_envs, *]`
    `dtype = float32`"""

    actions: Tensor
    """Actions `a_t` taken at time-step `t` in state `s_t`.
    `shape = [steps, num_envs]`
    `dtype = int64`"""

    rewards: Tensor
    """Rewards `r_t` obtained from taking action `a_t` in state `s_t` at time-step `t`.
    `shape = [steps, num_envs]`
    `dtype = float32`"""

    terminal: Tensor
    """Boolean mask indicating terminal states.
    `shape = [steps, num_envs]`
    `dtype = bool`

    Only terminal states are marked here; for states where either termination or truncation
    has occured, see `final`."""

    final: Tensor
    """Boolean mask indicating time-steps `t` where the environment has terminated or truncated.
    `shape = [steps, num_envs]`
    `dtype = bool`

    This is necessary to reset the computation of accumulated quantities for single episodes,
    such as episode returns (rewards-to-go) or GAE. The mask allows for resetting the accumulator
    resets to zero on each episode end.

    See https://farama.org/Vector-Autoreset-Mode for a visualization how environments reset.

    Information about termination/truncation returned by `env.step()` refers to the
    next state (from the point of view of calling the function). That is, if we run
    ```s_0, _ = env.reset()
    s_1, r_0, term, trunc, _ = env.step(a_0)```
    then the values of `term` and `trunc` refer to the state `s_1`."""

    next_state: Tensor
    """State after the last step sampled.
    `shape = [num_envs, *]`
    `dtype = float32`

    This is necessary for computing the GAE, where the advantage at time-step `t`
    depends on the value at time-step `t+1`. """

    @torch.no_grad()
    def __init__(
        self,
        env: VectorEnv,
        policy_net: nn.Module,
        steps: int = 512,
        device: torch.device | str = "cpu",
    ):
        """Samples `steps` time-steps from the vectorized environment following the policy."""

        obs_shape: tuple[int] = env.observation_space.shape  # ty:ignore[invalid-assignment]

        states = torch.empty((steps, *obs_shape), dtype=torch.float32, device=device)
        actions = torch.empty((steps, env.num_envs), dtype=torch.int64)
        rewards = torch.empty((steps, env.num_envs), dtype=torch.float32)
        terminal = torch.empty((steps, env.num_envs), dtype=torch.bool)
        final = torch.empty((steps, env.num_envs), dtype=torch.bool)

        # get initial state and make terminated/truncated markers for it
        state, _ = env.reset()
        terminated = torch.zeros((env.num_envs,), dtype=torch.bool)  # False
        truncated = torch.zeros((env.num_envs,), dtype=torch.bool)  # False

        for t in range(steps):
            states[t] = state.to(device)
            terminal[t] = terminated
            final[t] = torch.logical_or(terminated, truncated)  # if s_t is final

            policy = Categorical(logits=policy_net(states[t]))  # policy pi(.|s_t)
            action = policy.sample().to("cpu")  # sample action a_t from policy
            state, reward, terminated, truncated, _ = env.step(action)

            actions[t] = action
            rewards[t] = reward.to(torch.float32)

        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.terminal = terminal
        self.final = final
        self.next_state = state.to(device)

    @property
    def returns(self) -> Tensor:
        """Compute the returns (rewards-to-go) per episode."""

        # create returns tensor with correct values at the last time-step
        returns = self.rewards * ~self.final

        # Accumulate rewards backwards, resetting the accumulation on episode end.
        # Recall that final[t] = 1 iff states[t] is terminal. Accumulation resets
        # on terminal states to not include rewards from the later episode.
        for t in reversed(range(len(returns) - 1)):
            returns[t] = self.rewards[t] + returns[t + 1] * ~self.final[t]
            # TODO more efficient computation? (without loop)

        return returns

    @property
    def average_reward(self) -> float:
        """Computes the average reward per episode in the sample."""

        return (self.rewards.sum() / self.final.sum()).item()
