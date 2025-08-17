#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
#pip install -r requirment.txt
pip install pandas sentence-transformers tqdm hdbscan transformers
pip install accelerate

# Print success message
echo 'Virtual environment created and dependencies installed successfully.'
