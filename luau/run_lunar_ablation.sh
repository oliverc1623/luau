#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    sac_finetune.py
    sac_iaa.py
    sac_diaa.py
)
for script in "${scripts[@]}"; do
    for seed in 92; do
        if [[ $script == *.py ]]; then
            for gravity in -10 -6 -2 0; do
                python $script --seed=$seed \
                    --env-id "LunarLander-v3" \
                    --exp_name "student" \
                    --num-envs 8 \
                    --gradient_steps -1 \
                    --cudagraphs \
                    --compile \
                    --total-timesteps 1_000_000 \
                    --pretrained_run_id "luau/vmfqjyly" \
                    --env_kwargs continuous True enable_wind True wind_power 20.0 turbulence_power 2.0 gravity $gravity
            fi
        fi
    done
done
