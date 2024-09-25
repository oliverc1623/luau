#!/bin/bash

micromamba create -n poetry python=3.11 -c conda-forge -y
micromamba run -n poetry pipx ensurepath
micromamba run -n poetry pipx install poetry

micromamba create -n luau python=3.11 -c conda-forge -y
micromamba run -n luau poetry install
micromamba run -n luau poetry run inv setup
micromamba run -n luau python luau/train.py --log_dir ~/../pvcvolume --model_dir ~/../pvcvolume --num_experiments 4
