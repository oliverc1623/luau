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


def test_weights_update_after_one_update() -> None:
    """Test that the neural network weights update after one update."""
    # Define dummy parameters
    state_dim = 3  # Example state dimension (e.g., number of channels in an image)
    action_dim = 2  # Example action dimension (number of possible actions)
    lr_actor = 0.001
    gamma = 0.99
    k_epochs = 4
    eps_clip = 0.2

    # Instantiate the PPO agent
    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        gamma=gamma,
        k_epochs=k_epochs,
        eps_clip=eps_clip,
    )

    # Record initial weights
    initial_weights = {}
    for name, param in ppo_agent.policy.named_parameters():
        initial_weights[name] = param.clone().detach()

    # Simulate interactions to fill the buffer
    num_steps = 5  # Number of steps to simulate
    for _ in range(num_steps):
        # Create a dummy state
        rng = np.random.default_rng()
        image = rng.random((7, 7, state_dim), dtype=np.float32)  # Example image size
        direction = np.array(rng.random(), dtype=np.float32)  # Example scalar direction

        state = {
            "image": image,
            "direction": direction,
        }

        # Select an action
        _ = ppo_agent.select_action(state)

        # Append dummy reward and is_terminal to the buffer
        ppo_agent.buffer.rewards.append(rng.random())
        ppo_agent.buffer.is_terminals.append(False)

    # Perform an update
    ppo_agent.update()

    # Record weights after update
    updated_weights = {}
    for name, param in ppo_agent.policy.named_parameters():
        updated_weights[name] = param.clone().detach()

    # Check if any weights have changed
    weights_changed = False
    for name in initial_weights:
        if not torch.equal(initial_weights[name], updated_weights[name]):
            weights_changed = True
            break

    assert weights_changed, "Weights did not change after update."


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
