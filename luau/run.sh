#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    sac_torchcompile.py
)
for script in "${scripts[@]}"; do
    for seed in 11 21 31 41; do
        if [[ $script == *_torchcompile.py ]]; then
            python $script --seed=$seed \
                --env-id "HumanoidStandup-v5" \
                --exp_name "teacher" \
                --num-envs 8 \
                --gradient_steps -1 \
                --cudagraphs \
                --compile \
                --total-timesteps 2_000_000
        else
            python $script --seed=$seed
        fi
    done
done
