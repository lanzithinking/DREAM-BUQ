#!/usr/bin/env bash -l

# load FEniCS environment
# source ${HOME}/FEniCS/fenics.sh
source ${HOME}/miniconda3/bin/activate fenics-2019

# run python script to get ESS of samples stored in h5 format
python -u get_ESS_dolfin.py