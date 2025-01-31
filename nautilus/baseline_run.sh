#!/bin/bash

# Define the number of runs
num_runs=5

# Define the parameters for each run
declare -a seeds=("11" "47" "97" "457" "809")  # Example seeds

# Loop through the runs and execute the Python script with different arguments
for ((run=1; run<=num_runs; run++)); do
  seed=${seeds[$((run-1))]}  # Get the seed for this run
  echo "Running experiment $run with seed $seed"

  python luau/ppo.py \
    --exp-name "PPO_Teacher_Source_Run${run}" \
    --gym-id "TrafficLight5x5-v0" \
    --learning-rate 0.0005 \
    --total-timesteps 500_000 \
    --num-envs 10 \
    --num-minibatches 8 \
    --vf-coef 10.0 \
    --target-kl 0.01 \
    --kl-loss True \
    --seed "$seed"

  echo "Experiment $run completed"
  echo "----------------------------------------"
done
