#!/bin/bash
# SAC teacher on HumanoidStandup (baseline)

set -e

for seed in 11 21 31 41; do
    python sac_torchcompile.py --seed=$seed \
        --env-id "HumanoidStandup-v5" \
        --exp_name "teacher_sac" \
        --num-envs 8 \
        --gradient_steps -1 \
        --cudagraphs \
        --compile \
        --total-timesteps 2_000_000
done
