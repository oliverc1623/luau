# %%

from pathlib import Path

import torch

from luau.iaa_env import get_env
from luau.ppo import IAAPPO, PPO


# %%


def test_iaappo_init() -> None:
    """Test the initialization of the IAAPPO class."""
    seed = 47
    num_envs = 2
    env = get_env(seed=seed, num_envs=num_envs)
    teacher_ppo = PPO(
        env=env,
        lr_actor=0.0003,
        gamma=0.99,
        k_epochs=3,
        eps_clip=0.2,
        minibatch_size=3,
        horizon=6,
        gae_lambda=0.8,
    )
    teacher_model_path = Path("models/PPO/IntrospectiveEnvUnlocked/run_1_seed_1623/PPO_IntrospectiveEnvUnlocked_run_1_seed_1623.pth")
    assert teacher_model_path.exists(), f"Teacher model path {teacher_model_path} does not exist."
    teacher_ppo.load(teacher_model_path)

    student_ppo = IAAPPO(
        env=env,
        lr_actor=0.0003,
        gamma=0.99,
        k_epochs=3,
        eps_clip=0.2,
        minibatch_size=3,
        horizon=6,
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


# %%
