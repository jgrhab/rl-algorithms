# Proximal Policy Optimization (PPO)
#
# https://spinningup.openai.com/en/latest/algorithms/ppo.html

import gymnasium as gym
import torch
from gymnasium.vector import VectorEnv
from sampler import Sample
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.optim import Optimizer


def compute_GAE(
    sample: Sample, critic_net: nn.Module, discount: float = 0.99, weight: float = 0.95
) -> Tensor:
    """Computes the Generalized Advantage Estimate of the sample."""

    # compute the values V(s_t) for 0 <= t <= T and for the "next state" at time T+1
    values = critic_net(sample.states).squeeze().to("cpu")
    next_value = critic_net(sample.next_state).to("cpu").view([1, -1])

    # create tensor with next_values[t] = values[t+1] and set next_values[t] = 0 for t terminal
    next_values = torch.where(
        ~sample.terminal, torch.cat([values[1:], next_value]), 0.0
    )

    deltas = sample.rewards + discount * next_values - values

    # initialize advantages with correct value for the last time-step
    advs = deltas.clone()

    # compute advantages by accumulating backwards from the last time-step
    # reset accumulation on episode end
    for t in reversed(range(len(advs) - 1)):
        advs[t] = deltas[t] + (discount * weight) * advs[t + 1] * ~sample.final[t]

    return (advs - advs.mean()) / advs.std()


def compute_ppo_clip_objective(
    sample: Sample, actor_net: nn.Module, advantages: Tensor, clip_param: float = 0.2
) -> Tensor:
    """Computes the surrogate objective function.

    Let theta_k denotes the current policy weights and theta denotes the "new policy" weights,
    and denote their ratio by r(a|s) := pi_theta(a|s) / pi_{theta_k}(a|s).

    The clipped objective is defined as
    L(s, a, theta_k, theta) := min{ r(a|s) * A(a,s), clip(r(a|s)) * A(a,s) },
    and the update step is given by
    theta_{k+1} <- argmax_theta E[L(s, a, theta_k, theta)].

    In practice, there is only one policy pi_{theta_k} to sample from. We have to see theta_k
    as fixed in the computations and take gradients with respect to theta (the "new policy") only.
    """

    with torch.no_grad():
        old_policy = Categorical(logits=actor_net(sample.states).to("cpu"))
        old_log_probs = old_policy.log_prob(sample.actions)

    new_policy = Categorical(logits=actor_net(sample.states).to("cpu"))
    new_log_probs = new_policy.log_prob(sample.actions)

    # compute the ratio as the difference of logs for stability
    ratio = torch.exp(new_log_probs - old_log_probs)
    clip_ratio = torch.clamp(ratio, min=1 - clip_param, max=1 + clip_param)

    return torch.minimum(ratio * advantages, clip_ratio * advantages).mean()


def compute_critic_loss(sample: Sample, values: Tensor) -> Tensor:
    return nn.functional.mse_loss(values, sample.returns)


def train_step(
    env: VectorEnv,
    actor_net: nn.Module,
    critic_net: nn.Module,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    epochs: int = 5,
    sample_steps: int = 512,
    device: str = "cpu",
) -> float:
    """Sample one `Sample` and use it to train the actor and critic for multiple epochs."""

    sample = Sample(env, actor_net, sample_steps, device)

    for _ in range(epochs):
        # Compute the advantages (GAE) without gradient as it is used to update the actor.
        # Note that this is not strictly necessary since the advantages depend only on the
        # critic weights, which are not updated by the actor optimizer; it is however faster.
        with torch.no_grad():
            advantages = compute_GAE(sample, critic_net)

        actor_obj = compute_ppo_clip_objective(sample, actor_net, advantages)
        actor_obj.backward()
        actor_optimizer.step()
        actor_optimizer.zero_grad()

        values = critic_net(sample.states).squeeze().to("cpu")

        critic_loss = compute_critic_loss(sample, values)
        critic_loss.backward()
        critic_optimizer.step()
        critic_optimizer.zero_grad()

    # return the average reward of the sample (computed before any optimization step)
    return sample.average_reward


# --- TESTING --- #


def test():
    # Use GPU when available for forward pass through neural networks.
    # The model weights and model inputs need to be on the GPU; the model outputs are on the GPU.
    # All computations other than those of the neural networks are done on the CPU.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Using a vectorized environment produces batched data with dimensions `(num_envs, *)`.
    env = gym.make_vec("CartPole-v1", num_envs=8, vectorization_mode="sync")
    env = gym.wrappers.vector.NumpyToTorch(env)

    obs_len = env.single_observation_space.shape[0]
    num_actions = env.single_action_space.n

    actor_net = nn.Sequential(
        nn.Linear(obs_len, 32),
        nn.Tanh(),
        nn.Linear(32, num_actions),
    ).to(device)

    critic_net = nn.Sequential(
        nn.Linear(obs_len, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    ).to(device)

    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=5e-3, maximize=True)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=5e-3)

    for step in range(10):
        avg_reward = train_step(
            env, actor_net, critic_net, actor_optimizer, critic_optimizer, device=device
        )

        print(f"(step {step}) avg. reward: {avg_reward:.1f}")


if __name__ == "__main__":
    test()
