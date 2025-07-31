#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    sac_iaa.py
    sac_diaa.py
)
for script in "${scripts[@]}"; do
    for seed in 11 21 31 42; do
        for threshold in 0.25 0.75 0.9; do
            if [[ $script == *.py ]]; then
                python $script --seed=$seed \
                    --env-id "BipedalWalker-v3" \
                    --exp_name "student" \
                    --num-envs 8 \
                    --gradient_steps -1 \
                    --cudagraphs \
                    --compile \
                    --total-timesteps 1_000_000 \
                    --pretrained_run_id "luau/wxi10qyt" \
                    --env_kwargs hardcore True \
                    --introspection_threshold $threshold \
                    --burn_in 0
            else
                python $script --seed=$seed
            fi
        done
    done
done
