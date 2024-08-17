#!/bin/bash
apt-get update
apt-get install -y vim nano

if [ ! -d "/workspace/miniconda3" ];
then
    echo "Conda not found, installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3
    source /workspace/miniconda3/etc/profile.d/conda.sh
    conda init
fi

export PATH="/workspace/miniconda3/bin:$PATH"
source "/workspace/miniconda3/etc/profile.d/conda.sh"
