# %%
import pytest
import torch

from luau.iaa_env import get_env
from luau.ppo import PPO


# %%
@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["mps"] if torch.backends.mps.is_available() else ["cpu"])
def test_preprocess(device: str) -> None:
    """Test the preprocess method of the PPO class."""
    # Dummy PPO object
    env = get_env(seed=47, num_envs=1)
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
    assert preprocessed["image"].shape == (1, 3, 7, 7), "Image tensor should have shape (1, 3, 64, 64)."
    assert preprocessed["direction"].shape == (1, 1), "Direction tensor should have shape (1, 1)."
