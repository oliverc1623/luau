#!/bin/bash
# TGRL baseline runs for DIAA rebuttal (LunarLander + BipedalWalker, 4 seeds each).
# Mirrors hyperparameters in run.sh / run_bipedal.sh so curves are directly
# comparable to sac_diaa.py / sac_iaa.py results.

set -e

for seed in 11 21 31 41; do
    python sac_tgrl.py --seed=$seed \
        --env-id "LunarLander-v3" \
        --exp_name "tgrl_baseline" \
        --num-envs 8 \
        --gradient_steps -1 \
        --cudagraphs \
        --compile \
        --total-timesteps 1_000_000 \
        --pretrained_run_id "luau/vmfqjyly" \
        --env_kwargs continuous True enable_wind True wind_power 20.0 turbulence_power 2.0 gravity -4
done

for seed in 11 21 31 41; do
    python sac_tgrl.py --seed=$seed \
        --env-id "BipedalWalker-v3" \
        --exp_name "tgrl_baseline" \
        --num-envs 8 \
        --gradient_steps -1 \
        --cudagraphs \
        --compile \
        --total-timesteps 1_000_000 \
        --pretrained_run_id "luau/wxi10qyt" \
        --env_kwargs hardcore True
done
