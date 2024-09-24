#!/bin/bash

micromamba create -n poetry python=3.11 -c conda-forge
micromamba run -n poetry pipx ensurepath
micromamba run -n poetry pipx install poetry
