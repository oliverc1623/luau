#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    evaluate.py
)
for script in "${scripts[@]}"; do
    for seed in 61; do
        if [[ $script == *.py ]]; then
            python $script --seed=$seed \
                --env-id "CyCurveMerge-v0" \
                --student_run_id "wjbish1a" \
                --teacher_run_id "kg8xhrne" \
                --traffic_density 0.2 \
                --accident_prob 1.0 \
                --map "Cy"
        else
            python $script --seed=$seed
        fi
    done
done
