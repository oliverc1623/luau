#!/bin/bash
# Execute TGRL across the three MetaDrive envs with multiple seeds.
scripts=(
    sac_tgrl.py
)

# env-id, map pairs
envs=(
    "TIntersection-v0 T"
    "CyCurveMerge-v0 Cy"
    "MergeTurn-v0 yT"
)

for script in "${scripts[@]}"; do
    for env_pair in "${envs[@]}"; do
        env_id="${env_pair% *}"
        map="${env_pair#* }"
        for seed in 11 21 31 42; do
            python $script --seed=$seed \
                --env-id "$env_id" \
                --exp_name "student" \
                --num-envs 8 \
                --gradient_steps -1 \
                --cudagraphs \
                --compile \
                --total-timesteps 1_000_000 \
                --traffic_density 0.2 \
                --accident_prob 1.0 \
                --map "$map" \
                --use_lateral_reward \
                --pretrained_run_id "luau/kg8xhrne"
        done
    done
done
