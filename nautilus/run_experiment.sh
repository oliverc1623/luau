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
micromamba run -n luau tensorboard --logdir=~/../pvcvolume/PPO_logs &
micromamba run -n luau python luau/train.py --config_path ./hyperparams/ppo-unlocked-env.yaml --log_dir ~/../pvcvolume --model_dir ~/../pvcvolume --num_experiments 1
