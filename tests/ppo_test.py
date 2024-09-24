from pathlib import Path

import numpy as np
import pytest
import torch

from luau.ppo import PPO


# Mock device to CPU for testing
device = torch.device("cpu")


@pytest.fixture()
def ppo_agent() -> None:
    """Create a PPO agent for testing."""
    state_dim = torch.tensor(3)  # Example state dimension
    action_dim = 2  # Example action dimension
    lr_actor = 0.001
    gamma = 0.99
    k_epochs = 4
    eps_clip = 0.2
    return PPO(state_dim, action_dim, lr_actor, gamma, k_epochs, eps_clip)


def test_save_load(ppo_agent: PPO) -> None:
    """Test saving and loading the PPO agent."""
    checkpoint_path = Path("tests/test_models/checkpoint.pth")
    ppo_agent.save(checkpoint_path)
    assert checkpoint_path.exists(), "Checkpoint file was not saved."

    # Modify the model to ensure load is working
    for param in ppo_agent.policy.parameters():
        param.data.fill_(0)
    for param in ppo_agent.policy_old.parameters():
        param.data.fill_(0)

    ppo_agent.load(checkpoint_path)

    for param, old_param in zip(ppo_agent.policy.parameters(), ppo_agent.policy_old.parameters(), strict=False):
        assert torch.equal(param, old_param), "Model parameters were not loaded correctly."


def test_preprocess_image_observation(ppo_agent: PPO) -> None:
    """Test preprocessing image observations."""
    rng = np.random.default_rng()
    x = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    processed_x = ppo_agent.preprocess(x, image_observation=True)
    assert processed_x.shape == (1, 3, 64, 64), "Image observation preprocessing failed."
    assert processed_x.max() <= 1.0
    assert processed_x.min() >= 0.0, "Image normalization failed."


def test_preprocess_non_image_observation(ppo_agent: PPO) -> None:
    """Test preprocessing non-image observations."""
    rng = np.random.default_rng()
    x = rng.random((10, 64, 64, 3), dtype=np.float32)
    processed_x = ppo_agent.preprocess(x, image_observation=False)
    assert processed_x.shape == (10, 3, 64, 64), "Non-image observation preprocessing failed."


def test_preprocess_invert_image(ppo_agent: PPO) -> None:
    """Test preprocessing inverted image observations."""
    rng = np.random.default_rng()
    x = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    processed_x = ppo_agent.preprocess(x, image_observation=True, invert=True)
    assert processed_x.shape == (1, 3, 64, 64), "Inverted image observation preprocessing failed."
    assert processed_x.max() <= 1.0
    assert processed_x.min() >= 0.0, "Inverted image normalization failed."
