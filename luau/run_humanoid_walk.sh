#!/bin/bash
# SAC DIAA Student on HumanoidWalk

for seed in 11 21 31 41; do
    python sac_diaa.py \
    --env-id "Humanoid-v5" \
    --exp_name "diaa_student" \
    --pretrained_run_id "luau/x4m9jjp8" \
    --total-timesteps 2_000_000 \
    --num-envs 8 \
    --seed 11 \
    --compile \
    --cudagraphs
done
