# LLM-based Hardware Testbench Generation

This project implements an automated Verilog testbench generation system using Large Language Models (LLMs). It fine-tunes open-source LLMs to generate functional testbenches for hardware designs described in Verilog HDL.

## Features

- 🔧 Automatic testbench generation from Verilog DUT code
- 🚀 Fine-tuning pipeline using LoRA/QLoRA for efficiency
- 📊 Comprehensive evaluation metrics (compilation, simulation, coverage)
- 🛠️ Support for multiple datasets (AutoBench, MG-Verilog, HDLBits)
- 📈 Experiment tracking with Weights & Biases

## Project Structure

```
llm-testbench-gen/
├── configs/           # Configuration files
├── data/             # Dataset directory
│   ├── raw/          # Raw datasets
│   ├── processed/    # Processed train/val/test splits
│   └── test_results/ # Evaluation results
├── models/           # Model checkpoints
├── scripts/          # Main scripts
├── utils/            # Utility modules
├── notebooks/        # Analysis notebooks
└── tests/           # Unit tests
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Icarus Verilog (for simulation)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-testbench-gen.git
cd llm-testbench-gen
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

4. (Optional) Configure environment variables:
```bash
# Edit .env file with your API keys
nano .env
```

## Quick Start

### 1. Data Pipeline

```bash
python scripts/data_pipeline.py
```

### 2. Zip files to compute on Colab

```bash
python scripts/prepare_for_colab.py
```

### 3. Upload zip to colab and Train Model

```bash
python scripts/train.py
```

### 4. Evaluate on Colab Notebook

```bash
python scripts/evaluate.py
```

## Datasets

The project uses three main datasets:

1. **AutoBench**: Verilog designs with corresponding testbenches
2. **MG-Verilog**: Hardware descriptions paired with implementations
3. **HDLBits**: Beginner-friendly Verilog problems (manual collection required)

## Model Architecture

- **Base Model**: TinyLlama-1.1B
- **Fine-tuning**: LoRA with rank 16
- **Quantization**: 4-bit quantization using BitsAndBytes

## Evaluation Metrics

- **Compilation Success Rate**: Percentage of testbenches that compile without errors
- **Simulation Pass Rate**: Percentage of testbenches that simulate successfully
- **Coverage Score**: Code coverage achieved by generated tests
- **Generation Time**: Time comparison vs manual writing

## Configuration

Edit `configs/config.yaml` to customize:
- Model selection
- Training hyperparameters
- Evaluation settings
- Generation parameters

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black scripts/ utils/
flake8 scripts/ utils/
```

## Results

Results will be saved in:
- `models/checkpoints/`: Trained model weights
- `data/test_results/`: Evaluation metrics and generated testbenches
- `wandb/`: Training logs (if configured)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{llm_testbench_gen,
  title = {LLM-based Hardware Testbench Generation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/llm-testbench-gen}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AutoBench dataset creators
- MG-Verilog dataset from GaTech-EIC
- HDLBits for Verilog problems
- Hugging Face for transformer models
- Claude for coding help
