# %%
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from luau.ppo import IAAPPO, PPO
from luau.train import Trainer


root_path = Path(__file__).resolve().parent.parent
TEST_CONFIG = Path(f"{root_path}/tests/test_configs/test_iaappo_config.yaml")
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
    """Test that the IAAPPO agent is created correctly."""
    env = trainer.get_vector_env(47)
    ppo_agent = trainer.get_ppo_agent(env)
    assert isinstance(ppo_agent, ALGORITHM_CLASSES[trainer.algorithm]), "ppo_agent is not an instance of PPO or IAAPPO"

    # Test ppo agent is saved correctly
    log_dir, model_dir = trainer.setup_directories()
    checkpoint_path = f"{model_dir}/{trainer.algorithm}_{trainer.env_name}_run_{trainer.run_id}_seed_{trainer.random_seed}.pth"
    ppo_agent.save(checkpoint_path)
    assert Path(checkpoint_path).exists(), f"Checkpoint file {checkpoint_path} does not exist"

    ppo_policy_weights = ppo_agent.policy.state_dict()
    teacher_source_weights = ppo_agent.teacher_ppo_agent.policy.state_dict()
    teacher_target_weights = ppo_agent.teacher_target.policy.state_dict()

    for key in ppo_policy_weights:
        assert not np.array_equal(
            ppo_policy_weights[key].cpu().numpy(),
            teacher_source_weights[key].cpu().numpy(),
        ), f"Expected different weights for key {key} in policy models"

    # Before training teacher source and target weights should be the same
    for key in teacher_source_weights:
        assert np.array_equal(
            teacher_source_weights[key].cpu().numpy(),
            teacher_target_weights[key].cpu().numpy(),
        ), f"Expected same weights for key {key} in policy models"

    # TODO: make sure teacher target and source weights are updated after training - should be different
    t = 0
    next_obs, _ = env.reset()
    next_dones = np.zeros(ppo_agent.num_envs, dtype=bool)
    obs = ppo_agent.preprocess(next_obs)
    assert obs["image"].shape[0] == ppo_agent.num_envs, f"Expected {ppo_agent.num_envs} environments, got {obs['image'].shape[0]}"
    assert obs["direction"].shape[0] == ppo_agent.num_envs, f"Expected {ppo_agent.num_envs} environments, got {obs['direction'].shape[0]}"
    done = next_dones
    ppo_agent.buffer.images[t] = obs["image"]
    ppo_agent.buffer.directions[t] = obs["direction"]
    ppo_agent.buffer.is_terminals[t] = torch.from_numpy(done)

    # # Select actions and store them in the PPO agent's buffer
    with torch.no_grad():
        # introspect is false
        h = ppo_agent.introspect(obs, t)
        assert h.shape == torch.Size([ppo_agent.num_envs]), f"Expected shape {torch.Size([ppo_agent.num_envs])}, got {h.shape}"

        # introspect is true
        t = 1
        h = ppo_agent.introspect(obs, t)
        assert h.shape == torch.Size([ppo_agent.num_envs]), f"Expected shape {torch.Size([ppo_agent.num_envs])}, got {h.shape}"
        ppo_agent.buffer.indicators[t] = h

    actions, action_logprobs, state_vals = ppo_agent.select_action(obs, t)
    assert actions.shape == torch.Size([ppo_agent.num_envs]), f"Expected shape {torch.Size([ppo_agent.num_envs])}, got {actions.shape}"
    assert action_logprobs.shape == torch.Size(
        [ppo_agent.num_envs],
    ), f"Expected shape {torch.Size([ppo_agent.num_envs])}, got {action_logprobs.shape}"
    assert state_vals.shape == torch.Size([ppo_agent.num_envs]), f"Expected shape {torch.Size([ppo_agent.num_envs])}, got {state_vals.shape}"

    teacher_correction, student_correction = ppo_agent.correct()
    assert teacher_correction.shape == torch.Size(
        [
            ppo_agent.horizon,
            ppo_agent.num_envs,
        ],
    ), f"Expected shape {torch.Size([ppo_agent.horizon, ppo_agent.num_envs])}, got {teacher_correction.shape}"
    assert student_correction.shape == torch.Size(
        [
            ppo_agent.horizon,
            ppo_agent.num_envs,
        ],
    ), f"Expected shape {torch.Size([ppo_agent.horizon, ppo_agent.num_envs])}, got {student_correction.shape}"
