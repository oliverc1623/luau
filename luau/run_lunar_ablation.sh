#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    # sac_finetune.py
    sac_iaa.py
    sac_diaa.py
)
for script in "${scripts[@]}"; do
    for seed in 61 72 82 92; do
        for threshold in 0.25 0.75 0.9; do
            if [[ $script == *.py ]]; then
                python $script --seed=$seed \
                    --env-id "LunarLander-v3" \
                    --exp_name "student_ablation" \
                    --num-envs 8 \
                    --gradient_steps -1 \
                    --cudagraphs \
                    --compile \
                    --total-timesteps 1_000_000 \
                    --pretrained_run_id "luau/vmfqjyly" \
                    --introspection_threshold $threshold \
                    --env_kwargs continuous True enable_wind True wind_power 20.0 turbulence_power 2.0 gravity -4
            else
                python $script --seed=$seed
            fi
        done
    done
done
