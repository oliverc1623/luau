#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    sac_finetune.py
    sac_iaa.py
    sac_diaa.py
)
for script in "${scripts[@]}"; do
    for seed in 11 21 31 42; do
        if [[ $script == *.py ]]; then
            python $script --seed=$seed \
                --env-id "MergeTurn-v0" \
                --exp_name "student" \
                --num-envs 8 \
                --gradient_steps -1 \
                --cudagraphs \
                --compile \
                --total-timesteps 1_000_000 \
                --traffic_density 0.2 \
                --accident_prob 1.0 \
                --map "yT" \
                --use_lateral_reward \
                --pretrained_run_id "luau/kg8xhrne"
        else
            python $script --seed=$seed
        fi
    done
done
