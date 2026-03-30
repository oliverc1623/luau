#!/bin/bash
set -e

BWHM_SEEDS=(11 21 31 41)
LLWE_SEEDS=(61 72 82 92)

# ── Burn-in ablation (decay fixed at default 0.99999) ────────────────────────
for burn_in in 100 1000 5000; do
  for seed in "${BWHM_SEEDS[@]}"; do
    python luau/sac_diaa.py \
      --seed=$seed \
      --env-id "BipedalWalker-v3" \
      --exp_name "burnin_ablation" \
      --num-envs 8 \
      --gradient_steps -1 \
      --cudagraphs \
      --compile \
      --total-timesteps 1_000_000 \
      --pretrained_run_id "luau/wxi10qyt" \
      --env_kwargs hardcore True \
      --burn_in $burn_in \
      --introspection_decay 0.99999
  done

  for seed in "${LLWE_SEEDS[@]}"; do
    python luau/sac_diaa.py \
      --seed=$seed \
      --env-id "LunarLander-v3" \
      --exp_name "burnin_ablation" \
      --num-envs 8 \
      --gradient_steps -1 \
      --cudagraphs \
      --compile \
      --total-timesteps 1_000_000 \
      --pretrained_run_id "luau/vmfqjyly" \
      --env_kwargs continuous True enable_wind True wind_power 20.0 turbulence_power 2.0 gravity -4 \
      --burn_in $burn_in \
      --introspection_decay 0.99999
  done
done

# ── Decay ablation (burn-in fixed at default 1000) ───────────────────────────
for decay in 0.9999 0.99999 0.999999; do
  for seed in "${BWHM_SEEDS[@]}"; do
    python luau/sac_diaa.py \
      --seed=$seed \
      --env-id "BipedalWalker-v3" \
      --exp_name "decay_ablation" \
      --num-envs 8 \
      --gradient_steps -1 \
      --cudagraphs \
      --compile \
      --total-timesteps 1_000_000 \
      --pretrained_run_id "luau/wxi10qyt" \
      --env_kwargs hardcore True \
      --burn_in 1000 \
      --introspection_decay $decay
  done

  for seed in "${LLWE_SEEDS[@]}"; do
    python luau/sac_diaa.py \
      --seed=$seed \
      --env-id "LunarLander-v3" \
      --exp_name "decay_ablation" \
      --num-envs 8 \
      --gradient_steps -1 \
      --cudagraphs \
      --compile \
      --total-timesteps 1_000_000 \
      --pretrained_run_id "luau/vmfqjyly" \
      --env_kwargs continuous True enable_wind True wind_power 20.0 turbulence_power 2.0 gravity -4 \
      --burn_in 1000 \
      --introspection_decay $decay
  done
done
