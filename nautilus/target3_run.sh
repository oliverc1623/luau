#!/bin/bash

# eval "$(micromamba shell hook --shell=bash)"
# micromamba create -n poetry python=3.11 -c conda-forge -y
# micromamba run -n poetry pipx ensurepath
# export PATH="/root/.local/bin:$PATH"
# micromamba run -n poetry pipx install poetry

# eval "$(micromamba shell hook --shell=bash)"
# micromamba create -n luau python=3.11 -c conda-forge -y
# export PATH="/root/.local/bin:$PATH"
# micromamba run -n luau poetry install
# micromamba run -n luau poetry run inv setup
# micromamba run -n luau pip install gymnasium[mujoco]
# micromamba run -n luau apt-get update -y 
# micromamba run -n luau apt-get install ffmpeg libsm6 libxext6  -y
# micromamba run -n luau apt-get install gcc -y
# micromamba run -n luau pip install tensordict
# micromamba run -n luau pip install jax-jumpy==1.0.0
# micromamba run -n luau pip install torchrl
# micromamba run -n luau pip install highway-env

micromamba run -n luau python luau/diaa_sac_continuous.py --exp-name "student-diaa-target3" --seed 1 --compile --cudagraphs --learning-starts 100 --burn-in 0 --introspection-decay 0.9999 --teacher-actor ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_actor.pt --teacher-qnet ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_qnet.pt --total-timesteps 100_000
micromamba run -n luau python luau/diaa_sac_continuous.py --exp-name "student-diaa-target3" --seed 17 --compile --cudagraphs --learning-starts 100 --burn-in 0 --introspection-decay 0.9999 --teacher-actor ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_actor.pt --teacher-qnet ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_qnet.pt --total-timesteps 100_000
micromamba run -n luau python luau/diaa_sac_continuous.py --exp-name "student-diaa-target3" --seed 22 --compile --cudagraphs --learning-starts 100 --burn-in 0 --introspection-decay 0.9999 --teacher-actor ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_actor.pt --teacher-qnet ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_qnet.pt --total-timesteps 100_000
micromamba run -n luau python luau/diaa_sac_continuous.py --exp-name "student-diaa-target3" --seed 50 --compile --cudagraphs --learning-starts 100 --burn-in 0 --introspection-decay 0.9999 --teacher-actor ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_actor.pt --teacher-qnet ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_qnet.pt --total-timesteps 100_000
micromamba run -n luau python luau/diaa_sac_continuous.py --exp-name "student-diaa-target3" --seed 99 --compile --cudagraphs --learning-starts 100 --burn-in 0 --introspection-decay 0.9999 --teacher-actor ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_actor.pt --teacher-qnet ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_qnet.pt --total-timesteps

micromamba run -n luau python luau/iaa_sac_continuous.py --exp-name "student-iaa-target3" --seed 1 --compile --cudagraphs --learning-starts 100 --burn-in 0 --introspection-decay 0.9999 --teacher-actor ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_actor.pt --teacher-qnet ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_qnet.pt --total-timesteps 100_000
micromamba run -n luau python luau/iaa_sac_continuous.py --exp-name "student-iaa-target3" --seed 17 --compile --cudagraphs --learning-starts 100 --burn-in 0 --introspection-decay 0.9999 --teacher-actor ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_actor.pt --teacher-qnet ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_qnet.pt --total-timesteps 100_000
micromamba run -n luau python luau/iaa_sac_continuous.py --exp-name "student-iaa-target3" --seed 22 --compile --cudagraphs --learning-starts 100 --burn-in 0 --introspection-decay 0.9999 --teacher-actor ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_actor.pt --teacher-qnet ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_qnet.pt --total-timesteps 100_000
micromamba run -n luau python luau/iaa_sac_continuous.py --exp-name "student-iaa-target3" --seed 50 --compile --cudagraphs --learning-starts 100 --burn-in 0 --introspection-decay 0.9999 --teacher-actor ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_actor.pt --teacher-qnet ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_qnet.pt --total-timesteps 100_000
micromamba run -n luau python luau/iaa_sac_continuous.py --exp-name "student-iaa-target3" --seed 99 --compile --cudagraphs --learning-starts 100 --burn-in 0 --introspection-decay 0.9999 --teacher-actor ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_actor.pt --teacher-qnet ../../pvcvolume/models/highway-fast-v0__sac_torch_compile_pixels__1__True__True_qnet.pt --total-timesteps

micromamba run -n luau python luau/sac_torch_compile_pixels.py --exp-name "student-baseline-target3" --seed 1 --compile --cudagraphs --learning-starts 100 --total-timesteps 100_000
micromamba run -n luau python luau/sac_torch_compile_pixels.py --exp-name "student-baseline-target3" --seed 17 --compile --cudagraphs --learning-starts 100 --total-timesteps 100_000
micromamba run -n luau python luau/sac_torch_compile_pixels.py --exp-name "student-baseline-target3" --seed 22 --compile --cudagraphs --learning-starts 100 --total-timesteps 100_000
micromamba run -n luau python luau/sac_torch_compile_pixels.py --exp-name "student-baseline-target3" --seed 50 --compile --cudagraphs --learning-starts 100 --total-timesteps 100_000
micromamba run -n luau python luau/sac_torch_compile_pixels.py --exp-name "student-baseline-target3" --seed 99 --compile --cudagraphs --learning-starts 100 --total-timesteps 100_000
