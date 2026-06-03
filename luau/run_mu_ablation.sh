#!/bin/bash
# Sensitivity analysis of mu (--threshold_lr, the dynamic-threshold step size)
# on BipedalWalker-v3 hardcore for DIAA. Sweeps mu over an order of magnitude
# around the default (0.01) with 4 seeds each = 12 runs total.
# Teacher: luau/wxi10qyt. Mirrors run_bipedal.sh otherwise so curves are
# directly comparable to the existing DIAA results.

set -e

for mu in 0.001 0.01 0.1; do
    for seed in 11 21 31 41; do
        python sac_diaa.py --seed=$seed \
            --env-id "BipedalWalker-v3" \
            --exp_name "diaa_mu_${mu}" \
            --num-envs 8 \
            --gradient_steps -1 \
            --cudagraphs \
            --compile \
            --total-timesteps 1_000_000 \
            --pretrained_run_id "luau/wxi10qyt" \
            --threshold_lr $mu \
            --env_kwargs hardcore True
    done
done
