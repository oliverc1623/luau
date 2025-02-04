#!/bin/bash

# Define the number of runs
num_runs=5

# Define the parameters for each run
declare -a seeds=("11" "47" "97" "457" "809")  # Example seeds

# Loop through the runs and execute the Python script with different arguments
for ((run=1; run<=num_runs; run++)); do
  seed=${seeds[$((run-1))]}  # Get the seed for this run
  echo "Running experiment $run with seed $seed"

  python luau/dqn.py \
    --exp-name "DQN_Teacher_Source_Run${run}" \
    --gym-id "MiniGrid-Empty-5x5-v0" \
    --total-timesteps 500_000 \
    --num-envs 10 \
    --target-network-frequency 100 \
    --tau 0.005 \
    --save-model-freq 100_000 \
    --seed "$seed" \

  echo "Experiment $run completed"
  echo "----------------------------------------"
done
