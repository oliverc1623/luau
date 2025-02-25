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
micromamba run -n luau apt-get update -y 
micromamba run -n luau apt-get install ffmpeg libsm6 libxext6  -y
micromamba run -n luau apt-get install gcc -y
micromamba run -n luau pip install tensordict
micromamba run -n luau pip install jax-jumpy==1.0.0
micromamba run -n luau pip install torchrl
micromamba run -n luau pip install highway-env
