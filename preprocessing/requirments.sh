#!/bin/bash

# Create virtual environment
#python3 -m venv venv

# Activate virtual environment
#source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install \
    accelerate  \
    hdbscan \
    huggingface-hub  \
    pandas \
    safetensors \
    sentence-transformers \
    sentencepiece  \
    torch  \
    tqdm  \
    transformers  \
    config.yaml \
    python-dotenv
    
pip install language_tool_python
pip install tiktoken
pip install einops transformers_stream_generator


# Print success message
echo 'Virtual environment created and dependencies installed successfully.'
