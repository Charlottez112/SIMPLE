#!/bin/bash

set -ev

# Download the Phys101 dataset
mkdir -p ./data/phys101
pushd ./data/phys101
curl -O http://phys101.csail.mit.edu/data/phys101_v1.0.zip
echo Unzipping dataset...
unzip -q phys101_v1.0.zip
rm phys101_v1.0.zip
popd

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
conda create -y -n noether python=3.8
conda activate noether

# Install dependencies
conda install -y -c pytorch pytorch==1.10.1 torchvision==0.11.2 cudatoolkit==11.3.1
pip install -r requirements.txt
