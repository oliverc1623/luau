#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    sac_torchcompile.py
)
for script in "${scripts[@]}"; do
    for seed in 11 21 31 42; do
        if [[ $script == *.py ]]; then
            python $script --seed=$seed \
                --env-id "StraightRoad-v0" \
                --exp_name "teacher" \
                --num-envs 8 \
                --gradient_steps -1 \
                --cudagraphs \
                --compile \
                --total-timesteps 1_000_000
        else
            python $script --seed=$seed
        fi
    done
done
