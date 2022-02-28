#!/bin/bash

set -ev

# Install Miniconda3
CONDA_PREFIX="${CONDA_PREFIX:-$HOME/.local/miniconda3}"
pushd /tmp
curl -O "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
bash "Miniconda3-latest-Linux-x86_64.sh" -b -p "$CONDA_PREFIX"
rm "Miniconda3-latest-Linux-x86_64.sh"
popd

# Initialize conda
"$HOME/.local/miniconda3/condabin/conda init"
source "$HOME/.local/miniconda3/etc/profile.d/conda.sh"

# Create python environment
conda create -y -n simple python=3.8
conda activate simple

# Install dependencies
conda install -y -c pytorch pytorch==1.10.1 cudatoolkit==11.3.1
pip install -r requirements.txt
