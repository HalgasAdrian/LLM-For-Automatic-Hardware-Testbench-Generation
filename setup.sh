#!/bin/bash

echo "Setting up LLM Testbench Generation Project (macOS)..."

# Check if Python 3.11 is available
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "Warning: Python 3.11 is recommended. Your version:"
    $PYTHON_CMD --version
else
    echo "Error: Python 3 not found!"
    exit 1
fi

# Create virtual environment
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core requirements first
echo "Installing core requirements..."
pip install requests tqdm huggingface-hub pyyaml click numpy

# Try to install ML packages
echo "Installing ML packages..."
pip install torch transformers datasets accelerate peft || echo "Warning: Some ML packages failed to install"

# Install optional packages
echo "Installing optional packages..."
pip install pyverilog jsonlines pandas scikit-learn nltk rouge-score || echo "Warning: Some optional packages failed to install"

# Check for Icarus Verilog
if ! command -v iverilog &> /dev/null; then
    echo ""
    echo "Icarus Verilog not found. Installing with Homebrew..."
    if command -v brew &> /dev/null; then
        brew install icarus-verilog
    else
        echo "Homebrew not found. Please install Icarus Verilog manually:"
        echo "  1. Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "  2. Then run: brew install icarus-verilog"
    fi
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

echo ""
echo "Checking installed packages..."
python -c "import requests; print('✓ requests installed')" 2>/dev/null || echo "✗ requests not installed"
python -c "import tqdm; print('✓ tqdm installed')" 2>/dev/null || echo "✗ tqdm not installed"
python -c "import huggingface_hub; print('✓ huggingface_hub installed')" 2>/dev/null || echo "✗ huggingface_hub not installed"
python -c "import yaml; print('✓ pyyaml installed')" 2>/dev/null || echo "✗ pyyaml not installed"

echo ""
echo "Setup complete! Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run: python scripts/download_data.py"