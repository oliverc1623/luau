# %%

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from luau.iaa_env import IntrospectiveEnv, get_env
from luau.ppo import PPO, SingleEnvPPO


# %%
@pytest.mark.parametrize("num_envs", [1, 2, 4])
@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["cpu"])
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
    assert preprocessed["direction"].shape == (1, num_envs), "Direction tensor should have shape (1, num_envs)."


# %%
@pytest.mark.parametrize("num_envs", [1, 2, 4])
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


# %%
@pytest.mark.parametrize(
    ("num_envs", "max_timesteps", "horizon", "minibatch_size", "k_epochs"),
    [
        (1, 20, 10, 5, 3),
        (1, 10, 6, 3, 3),
    ],
)
def test_update(num_envs: int, max_timesteps: int, horizon: int, minibatch_size: int, k_epochs: int) -> None:
    """Test the update method of the PPO class."""
    seed = 47
    rng = np.random.default_rng(seed)
    env = IntrospectiveEnv(rng=rng, size=9, locked=False, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    ppo = SingleEnvPPO(
        env=env,
        lr_actor=0.0003,
        gamma=0.99,
        k_epochs=k_epochs,
        eps_clip=0.2,
        minibatch_size=minibatch_size,
        horizon=horizon,
        gae_lambda=0.95,
    )

    # Create dummy tensors for next_obs and next_done
    next_obs, _ = env.reset()
    next_dones = np.zeros(num_envs, dtype=bool)

    num_updates = max_timesteps // (horizon * num_envs)  # TODO: make 10 and 5 a parameter
    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * 0.0003  # TODO: make 0.0003 a parameter
        ppo.optimizer.param_groups[0]["lr"] = lrnow
        for step in range(5):
            obs = ppo.preprocess(next_obs)
            done = next_dones
            ppo.buffer.images[step] = obs["image"]
            ppo.buffer.directions[step] = obs["direction"]
            ppo.buffer.is_terminals[step] = torch.from_numpy(np.array(done))

            # Select actions and store them in the PPO agent's buffer
            with torch.no_grad():
                actions, action_logprobs, state_vals = ppo.policy(obs)
                ppo.buffer.state_values[step] = state_vals
            ppo.buffer.actions[step] = actions
            ppo.buffer.logprobs[step] = action_logprobs

        # Step the environment and store the rewards
        next_obs, rewards, next_dones, truncated, info = env.step(actions.item())
        next_dones = np.logical_or(next_dones, truncated)
        ppo.buffer.rewards[step] = torch.from_numpy(np.array(rewards))

        # Capture the initial weights
        initial_weights = [param.clone() for param in ppo.policy.parameters()]

        # PPO update at the end of the horizon
        writer = SummaryWriter()
        ppo.update(next_obs, np.array(next_dones), writer, step)

        # Capture the weights after the update
        updated_weights = [param.clone() for param in ppo.policy.parameters()]

        # Check if any weights have changed
        weights_changed = any(not torch.equal(initial, updated) for initial, updated in zip(initial_weights, updated_weights, strict=False))

        assert weights_changed, "Policy weights did not change after the update."

        # After update, we can check the buffer is cleared
        assert torch.all(ppo.buffer.images == 0), "Buffer images should be cleared."
        assert torch.all(ppo.buffer.directions == 0), "Buffer directions should be cleared."
        assert torch.all(ppo.buffer.logprobs == 0), "Buffer logprobs should be cleared."
        assert torch.all(ppo.buffer.state_values == 0), "Buffer state_values should be cleared."
        assert torch.all(ppo.buffer.actions == 0), "Buffer actions should be cleared."
        assert torch.all(ppo.buffer.rewards == 0), "Buffer rewards should be cleared."
        assert torch.all(ppo.buffer.is_terminals == 0), "Buffer is_terminals should be cleared."


# %%


@pytest.mark.parametrize("num_envs", [2, 4])
def test_vec_env_update(num_envs: int) -> None:
    """Test the update method of the PPO class."""
    seed = 47
    env = get_env(seed=seed, num_envs=num_envs)
    ppo = PPO(
        env=env,
        lr_actor=0.0003,
        gamma=0.99,
        k_epochs=3,
        eps_clip=0.2,
        minibatch_size=3,
        horizon=6,
        gae_lambda=0.95,
    )
    # Create dummy tensors for next_obs and next_done
    next_obs, _ = env.reset()
    next_dones = np.zeros(num_envs, dtype=bool)

    num_updates = 10 // (5 * num_envs)  # TODO: make 10 and 5 a parameter
    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * 0.0003  # TODO: make 0.0003 a parameter
        ppo.optimizer.param_groups[0]["lr"] = lrnow
        for step in range(5):
            obs = ppo.preprocess(next_obs)
            done = next_dones
            ppo.buffer.images[step] = obs["image"]
            ppo.buffer.directions[step] = obs["direction"]
            ppo.buffer.is_terminals[step] = torch.from_numpy(np.array(done))

            # Select actions and store them in the PPO agent's buffer
            with torch.no_grad():
                actions, action_logprobs, state_vals = ppo.policy(obs)
                ppo.buffer.state_values[step] = state_vals
            ppo.buffer.actions[step] = actions
            ppo.buffer.logprobs[step] = action_logprobs

        # Step the environment and store the rewards
        next_obs, rewards, next_dones, truncated, info = env.step(actions.tolist())
        next_dones = np.logical_or(next_dones, truncated)
        ppo.buffer.rewards[step] = torch.from_numpy(np.array(rewards))

        # Capture the initial weights
        initial_weights = [param.clone() for param in ppo.policy.parameters()]

        # PPO update at the end of the horizon
        writer = SummaryWriter()
        ppo.update(next_obs, np.array(next_dones), writer, step)

        # Capture the weights after the update
        updated_weights = [param.clone() for param in ppo.policy.parameters()]

        # Check if any weights have changed
        weights_changed = any(not torch.equal(initial, updated) for initial, updated in zip(initial_weights, updated_weights, strict=False))

        assert weights_changed, "Policy weights did not change after the update."

        # After update, we can check the buffer is cleared
        assert torch.all(ppo.buffer.images == 0), "Buffer images should be cleared."
        assert torch.all(ppo.buffer.directions == 0), "Buffer directions should be cleared."
        assert torch.all(ppo.buffer.logprobs == 0), "Buffer logprobs should be cleared."
        assert torch.all(ppo.buffer.state_values == 0), "Buffer state_values should be cleared."
        assert torch.all(ppo.buffer.actions == 0), "Buffer actions should be cleared."
        assert torch.all(ppo.buffer.rewards == 0), "Buffer rewards should be cleared."
        assert torch.all(ppo.buffer.is_terminals == 0), "Buffer is_terminals should be cleared."
