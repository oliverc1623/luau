#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    evaluate.py
)
for script in "${scripts[@]}"; do
    for seed in 61; do
        if [[ $script == *.py ]]; then
            python $script --seed=$seed \
                --env-id "BipedalWalker-v3" \
                --student_run_id "kswwn4xa" \
                --teacher_run_id "wxi10qyt" \
                --env_kwargs hardcore True render_mode rgb_array
        else
            python $script --seed=$seed
        fi
    done
done
