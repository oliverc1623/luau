# LUAU Demo

A short walkthrough of a full teacher → student transfer run on **BipedalWalker**:
train a teacher on the normal task, transfer it to the hardcore variant with DIAA,
compare against a from-scratch baseline, and watch the result.

## 1. Setup

```bash
micromamba activate luau
wandb login            # runs and model checkpoints are logged to the `luau` project
```

## 2. Train a teacher (source task)

Train baseline SAC on regular BipedalWalker. The run's actor and Q-network are
saved to WandB as artifacts; note the run ID (e.g. `luau/abc123xy`).

```bash
python luau/sac_torchcompile.py \
  --env-id BipedalWalker-v3 \
  --exp_name teacher \
  --num-envs 8 \
  --total-timesteps 1_000_000
```

## 3. Transfer to the target task with DIAA

Point `--pretrained_run_id` at the teacher run from step 2. The target task is the
hardcore variant (stumps, pits, stairs).

```bash
python luau/sac_diaa.py \
  --env-id BipedalWalker-v3 \
  --env_kwargs hardcore True \
  --exp_name diaa_demo \
  --num-envs 8 \
  --total-timesteps 1_000_000 \
  --pretrained_run_id "luau/<teacher-run-id>" \
  --burn_in 1000 \
  --introspection_threshold 0.5 \
  --introspection_decay 0.99999
```

Swap in `luau/sac_iaa.py` or `luau/sac_finetune.py` with the same arguments to
compare the other transfer methods. Add `--compile --cudagraphs` on a CUDA machine
for a large speedup.

What to watch in WandB:

| Metric | Meaning |
|---|---|
| `episode_return` | Student's episodic return |
| `advice` | Average number of envs following the teacher's action per step; decays toward 0 over training |
| `abs_diff` | Disagreement between the student's twin Q-heads, the signal DIAA thresholds on |
| `introspection_threshold` | DIAA's adaptive threshold; rises while teacher-guided episodes outperform the student, falls otherwise |

## 4. Train the from-scratch baseline

```bash
python luau/sac_torchcompile.py \
  --env-id BipedalWalker-v3 \
  --env_kwargs hardcore True \
  --exp_name student \
  --num-envs 8 \
  --total-timesteps 1_000_000
```

## 5. Compare

```bash
python luau/analysis/learning_curves.py   # learning curves, method vs. baseline
python luau/analysis/te_table.py          # Transfer Efficacy table
```

Transfer Efficacy is `(AUC_method - AUC_baseline) / AUC_baseline`; a positive value
means the transfer method learned faster than training from scratch.

## 6. Watch the trained policy

`luau/inference.py` is a cell-style script. Set `run_id` and `env_id` near the top to
the run you want, then run it to record one episode to `videos/inference/`:

```bash
python luau/inference.py
```

## Other domains

- **MiniGrid (discrete):** `luau/ppo.py` (baseline) and `luau/iaa.py --teacher-model <path.pth>`
- **MetaDrive (driving):** the same SAC scripts live in `luau/driving-envs/`, with launchers
  in `luau/driving-envs/run*.sh`
