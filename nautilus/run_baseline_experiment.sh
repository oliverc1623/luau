#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba create -n poetry python=3.11 -c conda-forge -y
micromamba run -n poetry pipx ensurepath
export PATH="/root/.local/bin:$PATH"
micromamba run -n poetry pipx install poetry

eval "$(micromamba shell hook --shell=bash)"
micromamba create -n luau python=3.11 -c conda-forge -y
export PATH="/root/.local/bin:$PATH"
micromamba run -n luau poetry install
micromamba run -n luau poetry run inv setup
micromamba run -n luau pip install gymnasium[mujoco]
micromamba run -n luau tensorboard --logdir=~/../pvcvolume/runs/ --samples_per_plugin scalars=5000 &
micromamba run -n luau python luau/sac_continuous.py --env-id "Humanoid-v5" --num-envs 8 --seed 1 --exp-name "SAC-Student-Baseline-Run1"
micromamba run -n luau python luau/sac_continuous.py --env-id "Humanoid-v5" --num-envs 8 --seed 17 --exp-name "SAC-Student-Baseline-Run2"
micromamba run -n luau python luau/sac_continuous.py --env-id "Humanoid-v5" --num-envs 8 --seed 26 --exp-name "SAC-Student-Baseline-Run3"
micromamba run -n luau python luau/sac_continuous.py --env-id "Humanoid-v5" --num-envs 8 --seed 45 --exp-name "SAC-Student-Baseline-Run4"
micromamba run -n luau python luau/sac_continuous.py --env-id "Humanoid-v5" --num-envs 8 --seed 72 --exp-name "SAC-Student-Baseline-Run5"
