#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    sac_torchcompile.py
)
for script in "${scripts[@]}"; do
    for seed in 21 31 41; do
        if [[ $script == *.py ]]; then
            python $script --seed=$seed \
                --env-id "BipedalWalker-v3" \
                --exp_name "student" \
                --num-envs 8 \
                --gradient_steps -1 \
                --cudagraphs \
                --compile \
                --total-timesteps 1_000_000 \
                --env_kwargs hardcore True
        else
            python $script --seed=$seed
        fi
    done
done
