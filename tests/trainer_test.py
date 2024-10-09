# %%
import shutil
from pathlib import Path

import numpy as np
import pytest

from luau.ppo import IAAPPO, PPO
from luau.train import Trainer


root_path = Path(__file__).resolve().parent.parent
TEST_CONFIG = Path(f"{root_path}/tests/test_configs/test_config.yaml")
ALGORITHM_CLASSES = {
    "PPO": PPO,
    "IAAPPO": IAAPPO,
}

# %%


@pytest.fixture()
def trainer() -> Trainer:
    """Set fixture for creating a Trainer object."""
    log_dir = "tests/test_logs"
    model_dir = "tests/test_models"
    random_seed = 47
    run_id = 1
    return Trainer(TEST_CONFIG, log_dir, model_dir, random_seed, run_id)


def test_setup_directories(trainer: Trainer) -> None:
    """Test that the directories are created correctly."""
    log_dir, model_dir = trainer.setup_directories()

    assert log_dir.exists(), f"Log directory {log_dir} does not exist"
    assert model_dir.exists(), f"Model directory {model_dir} does not exist"

    # Clean up created directories
    shutil.rmtree(log_dir.parent.parent.parent)
    shutil.rmtree(model_dir.parent.parent.parent)


def test_get_vector_env_seeding(trainer: Trainer) -> None:
    """Test that each environment in the vectorized environment has a different seed."""
    seed = 47
    vector_env = trainer.get_vector_env(seed)
    num_envs = trainer.num_envs

    obs, _ = vector_env.reset()
    assert obs["image"].shape[0] == num_envs, f"Expected {num_envs} environments, got {obs['image'].shape[0]}"
    for i in range(num_envs):
        for j in range(i + 1, num_envs):
            assert not np.array_equal(obs["image"][i], obs["image"][j]), f"Expected different images for environments {i} and {j}"


def test_ppo_agent(trainer: Trainer) -> None:
    """Test that the PPO agent is created correctly."""
    env = trainer.get_vector_env(47)
    ppo_agent = trainer.get_ppo_agent(env)
    assert isinstance(ppo_agent, ALGORITHM_CLASSES[trainer.algorithm]), "ppo_agent is not an instance of PPO"

    # Test ppo agent is saved correctly
    log_dir, model_dir = trainer.setup_directories()
    checkpoint_path = f"{model_dir}/{trainer.algorithm}_{trainer.env_name}_run_{trainer.run_id}_seed_{trainer.random_seed}.pth"
    ppo_agent.save(checkpoint_path)
    assert Path(checkpoint_path).exists(), f"Checkpoint file {checkpoint_path} does not exist"
