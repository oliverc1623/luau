#!/bin/bash

micromamba create -n poetry python=3.11 -c conda-forge
micromamba activate poetry
pipx ensurepath
pipx install poetry
micromamba deactivate
