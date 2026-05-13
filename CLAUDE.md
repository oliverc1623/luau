# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate the micromamba environment
micromamba activate luau

# Lint and format
ruff check --fix
ruff format

# Run all checks (lint + format + tests + coverage)
inv checks

# Run tests (excluding CI-skipped tests)
pytest -m "not skipci"

# Run a single test file
pytest tests/ppo_test.py

# Run training scripts (from inside luau/ or luau/driving-envs/)
python sac_diaa.py --env-id LunarLander-v3 --seed 1 --pretrained_run_id "luau/<run_id>"
```

## Architecture

LUAU is a research framework for teacher-student transfer in RL. A pretrained "teacher" agent guides a "student" learning a harder variant of the same task, with the student deciding per-state whether the teacher's knowledge is still relevant.

### Algorithm files (`luau/`)

| File | Algorithm | Env type |
|---|---|---|
| `sac_torchcompile.py` | Baseline SAC | Continuous |
| `sac_finetune.py` | Fine-tune from teacher weights | Continuous |
| `sac_iaa.py` | IAA — uses teacher action when teacher Q-value confidence exceeds threshold | Continuous |
| `sac_diaa.py` | DIAA — uses frozen vs. trainable teacher Q divergence as the introspection signal | Continuous |
| `sac_tgrl.py` / `sac_tgrl_iaa.py` | TGRL baselines | Continuous |
| `ppo.py` | Baseline PPO | Discrete (MiniGrid) |
| `iaa.py` | IAA for PPO | Discrete (MiniGrid) |
| `inference.py` | Run a trained policy for one episode | - |
| `evaluate.py` | Compare teacher vs. student action distributions | - |

The `luau/driving-envs/` subtree mirrors this structure for MetaDrive environments (`CyCurveMerge-v0`, `MergeTurn-v0`, `TIntersection-v0`).

### Key design patterns

**CLI arguments**: SAC variants use `tyro` with a `@dataclass Args`; PPO/IAA use `argparse`. Common args: `--seed`, `--env-id`, `--pretrained_run_id`, `--num-envs`, `--total-timesteps`, `--compile`, `--cudagraphs`.

**Teacher loading**: All student algorithms load teacher checkpoints from WandB artifacts via `pretrained_run_id` (format: `"luau/<run_id>"`). The teacher actor and Q-network are fetched, frozen, and used during training.

**Performance**: SAC variants support `torch.compile` + CUDA graphs (`--compile --cudagraphs`) and `--gradient_steps -1` (UTD ratio = number of envs). Use `num_envs=8` for vectorized training.

**Experiment tracking**: All runs log to Weights & Biases (project `luau`). Model checkpoints (`actor.pt`, `qnet.pt`) are saved as WandB artifacts.

**MetaDrive**: `CustomMetaDriveEnv` wraps `MetaDriveEnv` with top-down rendering. The driving-envs scripts accept additional args: `--traffic_density`, `--accident_prob`, `--map`, `--use_lateral_reward`.

**Transfer Efficacy (TE)**: The primary evaluation metric: `(AUC_method - AUC_baseline) / AUC_baseline`. Also tracked: jumpstart (initial reward advantage) and asymptotic reward.

### Environments

- **Discrete**: MiniGrid-Empty-5x5, FourRoomDoorKey (PPO/IAA)
- **Continuous**: BipedalWalker-v3 (hardcore), LunarLander-v3 (wind, gravity=-4) (SAC variants)
- **Driving**: CyCurveMerge-v0, MergeTurn-v0, TIntersection-v0 via MetaDrive (driving-envs SAC variants)

### Testing

Tests live in `tests/`. The `checks.py` task auto-creates skeleton test files for any module missing a `_test.py`. Tests that require GPU or long runtime should be marked `@pytest.mark.skipci(reason="...")` — these are excluded from CI but can run locally.
