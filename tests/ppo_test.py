# %%
import numpy as np
import pytest
import torch

from luau.iaa_env import get_env
from luau.ppo import PPO


# %%
@pytest.mark.parametrize("num_envs", [1, 2, 4])
@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["mps"] if torch.backends.mps.is_available() else ["cpu"])
def test_preprocess(num_envs: int, device: str) -> None:
    """Test the preprocess method of the PPO class."""
    # Dummy PPO object
    env = get_env(seed=47, num_envs=num_envs)
    ppo = PPO(
        env=env,
        lr_actor=0.0003,
        gamma=0.99,
        k_epochs=3,
        eps_clip=0.2,
        minibatch_size=4,
        horizon=5,
        gae_lambda=0.95,
    )

    # Create dummy NumPy input data
    sample_obs, _ = env.reset()

    # Test preprocess method
    preprocessed = ppo.preprocess(sample_obs)

    # Check types
    assert isinstance(preprocessed["image"], torch.Tensor), "Image should be a tensor."
    assert isinstance(preprocessed["direction"], torch.Tensor), "Direction should be a tensor."

    # Check device
    assert preprocessed["image"].device.type == device, f"Image tensor should be on {device}."
    assert preprocessed["direction"].device.type == device, f"Direction tensor should be on {device}."

    # Check shape
    assert preprocessed["image"].shape == (num_envs, 3, 7, 7), "Image tensor should have shape (num_envs, 3, 64, 64)."
    assert preprocessed["direction"].shape == (num_envs, 1), "Direction tensor should have shape (num_envs, 1)."


# %%
@pytest.mark.parametrize("num_envs", [1])
@pytest.mark.parametrize(("gamma", "gae_lambda"), [(0.99, 0.8)])
def test_calculate_gae(num_envs: int, gamma: float, gae_lambda: float) -> None:
    """Test the calculation of the Generalized Advantage Estimation."""
    env = get_env(seed=47, num_envs=num_envs)
    ppo = PPO(
        env=env,
        lr_actor=0.0003,
        gamma=gamma,
        k_epochs=3,
        eps_clip=0.2,
        minibatch_size=4,
        horizon=5,
        gae_lambda=gae_lambda,
    )

    # Create dummy tensors for next_obs and next_done
    next_obs, _ = env.reset()

    # Preprocess next observation
    next_obs = ppo.preprocess(next_obs)
    next_dones = np.zeros(num_envs, dtype=bool)

    step = 1
    done = next_dones
    ppo.buffer.images[step] = next_obs["image"]
    ppo.buffer.directions[step] = next_obs["direction"]
    ppo.buffer.is_terminals[step] = torch.from_numpy(done)

    # Select actions and store them in the PPO agent's buffer
    with torch.no_grad():
        actions, action_logprobs, state_vals = ppo.policy(next_obs)
        ppo.buffer.state_values[step] = state_vals
    ppo.buffer.actions[step] = actions
    ppo.buffer.logprobs[step] = action_logprobs

    # Step the environment and store the rewards
    next_obs, rewards, next_dones, truncated, info = env.step(actions.tolist())
    next_dones = np.logical_or(next_dones, truncated)
    ppo.buffer.rewards[step] = torch.from_numpy(rewards)

    # Test GAE calculation
    rewards, advantages = ppo._calculate_gae(next_obs, next_dones)  # noqa: SLF001

    # Check that rewards and advantages have the expected shape
    assert rewards.shape == (ppo.horizon, ppo.num_envs), "Rewards tensor should have the correct shape."
    assert advantages.shape == (ppo.horizon, ppo.num_envs), "Advantages tensor should have the correct shape."
