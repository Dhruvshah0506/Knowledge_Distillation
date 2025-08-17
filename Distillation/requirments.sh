#!/bin/bash

# Create virtual environment
python3 -m venv venv

pip install transformers datasets accelerate -q
pip install -U datasets
pip install -q pyarrow
pip install nltk
pip uninstall torch -y
pip install torchvision torchaudio -q
pip install optuna
pip install tensorboard
pip install torch


pip install \
    peft  \
    bitsandbytes  \
    pandas \
    tqdm \
    dataclasses

echo 'Virtual environment created and dependencies installed successfully.'