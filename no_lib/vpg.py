# Vanilla Policy Gradient with Vectorized Environment
#
# https://spinningup.openai.com/en/latest/algorithms/vpg.html
# https://gymnasium.farama.org/api/vector/

import gymnasium as gym
import torch
from gymnasium.vector import VectorEnv
from sampler import Sample
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.optim import Optimizer


def compute_actor_loss(sample: Sample, actor_net: nn.Module, values: Tensor) -> Tensor:
    # forward pass (with gradient) to get the policy and then the log-probs
    policy = Categorical(logits=actor_net(sample.states).to("cpu"))  # policy on CPU
    log_probs = policy.log_prob(sample.actions)

    # use the simplest form of advantages A_t = R_t - V(s_t)
    return -1 * (log_probs * (sample.returns - values)).sum()


def compute_critic_loss(sample: Sample, values: Tensor) -> Tensor:
    return nn.functional.mse_loss(values, sample.returns)


def train_step(
    env: VectorEnv,
    actor_net: nn.Module,
    critic_net: nn.Module,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    sample_steps: int = 512,
    epochs: int = 5,
    device: str = "cpu",
) -> float:
    """Sample one `Sample` and use it to train the actor and critic."""

    sample = Sample(env, actor_net, sample_steps, device=device)

    for _ in range(epochs):
        values = critic_net(sample.states).squeeze().to("cpu")

        # Detach values to compute actor loss to optimize only the policy.
        # This is not strictly necessary here since the actor optimizer only updates the
        # actor weights, but would be in case of a shared optimizer for actor and critic.
        actor_loss = compute_actor_loss(sample, actor_net, values.detach())
        actor_loss.backward()
        actor_optimizer.step()
        actor_optimizer.zero_grad()

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

    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=5e-3)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=5e-3)

    for step in range(10):
        avg_reward = train_step(
            env, actor_net, critic_net, actor_optimizer, critic_optimizer, device=device
        )

        print(f"(step {step}) avg. reward: {avg_reward:.1f}")


if __name__ == "__main__":
    test()
