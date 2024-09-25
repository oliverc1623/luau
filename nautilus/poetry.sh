#!/bin/bash

micromamba create -n poetry python=3.11 -c conda-forge -y
micromamba run -n poetry pipx ensurepath
micromamba run -n poetry pipx install poetry

micromamba create -n luau python=3.11 -c conda-forge -y
micromamba activate luau
git clone https://github.com/oliverc1623/luau.git
cd luau
poetry install
poetry run inv setup
python luau/train.py --log_dir ~/../pvcvolume --model_dir ~/../pvcvolume --num_experiments 4
