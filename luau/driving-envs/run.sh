#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    sac_finetune.py
    sac_iaa.py
    sac_diaa.py
)
for script in "${scripts[@]}"; do
    for seed in 61 72 82 92; do
        if [[ $script == *.py ]]; then
            python $script --seed=$seed \
                --env-id "CurveRoadDense-v0" \
                --exp_name "student" \
                --num-envs 8 \
                --gradient_steps -1 \
                --cudagraphs \
                --compile \
                --total-timesteps 1_000_000 \
                --traffic_density 0.3 \
                --map "C" \
                --pretrained_run_id "luau/kg8xhrne"
        else
            python $script --seed=$seed
        fi
    done
done
