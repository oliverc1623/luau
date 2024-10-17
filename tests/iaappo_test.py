# %%

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from luau.iaa_env import get_env
from luau.ppo import IAAPPO, PPO


# %%


@pytest.mark.parametrize(
    ("num_envs", "max_timesteps", "horizon", "minibatch_size", "k_epochs"),
    [
        (2, 12, 6, 3, 3),
    ],
)
def test_iaappo_init(num_envs: int, max_timesteps: int, horizon: int, minibatch_size: int, k_epochs: int) -> None:
    """Test the initialization of the IAAPPO class."""
    seed = 47
    env = get_env(seed=seed, num_envs=num_envs)
    teacher_ppo = PPO(
        env=env,
        lr_actor=0.0003,
        gamma=0.99,
        k_epochs=k_epochs,
        eps_clip=0.2,
        minibatch_size=minibatch_size,
        horizon=horizon,
        gae_lambda=0.8,
    )
    teacher_model_path = Path("models/PPO/IntrospectiveEnvUnlocked/run_1_seed_1623/PPO_IntrospectiveEnvUnlocked_run_1_seed_1623.pth")
    assert teacher_model_path.exists(), f"Teacher model path {teacher_model_path} does not exist."
    teacher_ppo.load(teacher_model_path)
    original_teacher = [param.clone() for param in teacher_ppo.policy.parameters()]

    student_ppo = IAAPPO(
        env=env,
        lr_actor=0.0003,
        gamma=0.99,
        k_epochs=k_epochs,
        eps_clip=0.2,
        minibatch_size=minibatch_size,
        horizon=horizon,
        gae_lambda=0.8,
        teacher_source=teacher_ppo,
        introspection_decay=0.99999,
        burn_in=0,
        introspection_threshold=0.9,
    )

    assert student_ppo.teacher_source is not None, "Teacher model should not be None."

    # Check if the teacher model is loaded correctly
    for teacher_param, student_param in zip(teacher_ppo.policy.parameters(), student_ppo.policy.parameters(), strict=False):
        assert not torch.equal(teacher_param, student_param), "Teacher model and student model should have different weights."

    for teacher_source, teacher_target in zip(
        student_ppo.teacher_source.policy.parameters(),
        student_ppo.teacher_target.policy.parameters(),
        strict=False,
    ):
        assert torch.equal(teacher_source, teacher_target), "Teacher source model and teacher target model should have same weights."

    # Reset all environments
    next_obs, _ = env.reset()
    next_dones = np.zeros(num_envs, dtype=bool)
    time_step = 0
    writer = SummaryWriter()

    # Training loop
    num_updates = max_timesteps // (horizon * num_envs)
    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * student_ppo.lr_actor
        student_ppo.optimizer.param_groups[0]["lr"] = lrnow
        student_ppo.teacher_target.optimizer.param_groups[0]["lr"] = lrnow
        for step in range(horizon):
            # Preprocess the next observation and store relevant data in the PPO agent's buffer
            obs = student_ppo.preprocess(next_obs)
            done = next_dones
            student_ppo.buffer.images[step] = obs["image"]
            student_ppo.buffer.directions[step] = obs["direction"]
            student_ppo.buffer.is_terminals[step] = torch.from_numpy(done)

            # Select actions and store them in the PPO agent's buffer
            with torch.no_grad():
                if isinstance(student_ppo, IAAPPO):
                    actions, action_logprobs, state_vals = student_ppo.select_action(obs, step)
                else:
                    actions, action_logprobs, state_vals = student_ppo.select_action(obs)
                student_ppo.buffer.state_values[step] = state_vals
            student_ppo.buffer.actions[step] = actions
            student_ppo.buffer.logprobs[step] = action_logprobs

            # Step the environment and store the rewards
            next_obs, rewards, next_dones, truncated, info = env.step(actions.tolist())
            next_dones = np.logical_or(next_dones, truncated)
            student_ppo.buffer.rewards[step] = torch.from_numpy(rewards)

        initial_weights = [param.clone() for param in student_ppo.policy.parameters()]
        initial_teacher_source_weights = [param.clone() for param in student_ppo.teacher_source.policy.parameters()]
        initial_teacher_target_weights = [param.clone() for param in student_ppo.teacher_target.policy.parameters()]

        # PPO update at the end of the horizon
        student_ppo.update(next_obs, next_dones, writer, time_step)

        # Capture the weights after the update
        updated_weights = [param.clone() for param in student_ppo.policy.parameters()]

        # Check if any weights have changed
        weights_changed = any(not torch.equal(initial, updated) for initial, updated in zip(initial_weights, updated_weights, strict=False))
        assert weights_changed, "Policy weights did not change after the update."

        # Check if the teacher source model has not changed
        for initial, updated in zip(initial_teacher_source_weights, student_ppo.teacher_source.policy.parameters(), strict=False):
            assert torch.equal(initial, updated), "Teacher source model weights should not change."

        # Check if the original teacher source model has not changed
        for initial, updated in zip(original_teacher, student_ppo.teacher_source.policy.parameters(), strict=False):
            assert torch.equal(initial, updated), "Teacher source model weights should not change."

        # Check if the teacher target model have changed
        for initial, updated in zip(initial_teacher_target_weights, student_ppo.teacher_target.policy.parameters(), strict=False):
            assert not torch.equal(initial, updated), "Teacher target model weights should change."


# %%
