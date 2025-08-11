#!/bin/bash

echo "Setting up LLM Testbench Generation Project..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check for Icarus Verilog
if ! command -v iverilog &> /dev/null; then
    echo "Warning: Icarus Verilog not found. Please install it:"
    echo "  Ubuntu/Debian: sudo apt-get install iverilog"
    echo "  macOS: brew install icarus-verilog"
    echo "  Windows: Download from http://iverilog.icarus.com/"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# Weights & Biases API key (optional)
WANDB_API_KEY=

# HuggingFace token (for gated models)
HF_TOKEN=

# Project settings
PROJECT_NAME=llm-testbench-gen
CUDA_VISIBLE_DEVICES=0
EOL
fi

echo "Setup complete! Activate the environment with: source venv/bin/activate"