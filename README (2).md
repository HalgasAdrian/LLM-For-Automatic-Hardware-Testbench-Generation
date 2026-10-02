# LLM-Based Hardware Testbench Generation

A proof of concept that fine-tunes a small open-source language model to automatically write test code for hardware designs.

Built as the final project for CS6120 Natural Language Processing at Northeastern University.

## What problem this solves

Hardware engineers describe chips and circuits in a language called Verilog. Before a design can be trusted, someone has to write a **testbench**: a separate piece of Verilog that feeds inputs into the design and checks that the outputs are correct. Writing testbenches by hand is slow and repetitive.

This project asks a simple question: can a language model read a Verilog design and write a working testbench for it? It fine-tunes TinyLlama-1.1B on examples of designs paired with testbenches, then automatically grades what the model produces.

## How it works

1. **Prepare the data.** Pull Verilog designs and testbenches from three public datasets, clean them, and split them into training, validation, and test sets.
2. **Fine-tune the model.** Train TinyLlama-1.1B using LoRA and 4-bit quantization, which keeps memory use low enough to train on a Google Colab GPU.
3. **Generate testbenches.** Give the fine-tuned model designs it hasn't seen before and have it write a testbench for each one.
4. **Grade the results.** Run every generated testbench through Icarus Verilog, an open-source Verilog simulator, and score it automatically (see Evaluation below).

Training runs are tracked in Weights & Biases.

## Results

This is a proof of concept. The full pipeline works end to end: data preparation, fine-tuning, generation, and automatic grading.

Fine-tuning did not produce large improvements over the base model. The main limit is the size of the training data. A model this small needs many more examples of designs paired with good testbenches to learn the task well.

Evaluation outputs and generated testbenches are saved in `data/test_results/`.

## Next steps

1. **Collect more training data.** This is the biggest bottleneck and the most likely way to improve results.
2. **Try a larger base model** once more training data is available.
3. **Measure generation time** against writing the same testbenches by hand.

## Evaluation

Every generated testbench is scored automatically:

| Metric | What it measures |
| --- | --- |
| Compilation success rate | The share of generated testbenches that compile without errors |
| Simulation pass rate | The share that run successfully in simulation |
| Coverage score | How much of the design the generated tests actually exercise |

## Datasets

| Dataset | What it contains |
| --- | --- |
| AutoBench | Verilog designs with matching testbenches |
| MG-Verilog | Hardware descriptions paired with Verilog implementations |
| HDLBits | Beginner-friendly Verilog problems (must be collected manually) |

## Model setup

| Setting | Value |
| --- | --- |
| Base model | TinyLlama-1.1B |
| Fine-tuning | LoRA, rank 16 |
| Quantization | 4-bit, using BitsAndBytes |

## Tech used

| Purpose | Technology |
| --- | --- |
| Language | Python |
| Model training | PyTorch, Hugging Face, BitsAndBytes |
| Verilog simulation | Icarus Verilog |
| Experiment tracking | Weights & Biases |
| GPU compute | Google Colab |

## What's in the repo

```
LLM-For-Automatic-Hardware-Testbench-Generation/
├── configs/                  # Model, training, and evaluation settings
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/            # Cleaned train/validation/test splits
│   └── test_results/         # Evaluation results and generated testbenches
├── models/                   # Model checkpoints
├── notebooks/                # Analysis notebooks
├── scripts/                  # Data pipeline, training, and evaluation scripts
├── utils/                    # Shared helper code
├── llm_testbench_colab.zip   # Project packaged for running on Colab
├── trained_model.zip         # Trained model weights
├── requirements.txt
└── setup.sh
```

## Getting it running

### Prerequisites

- Python 3.8+
- A CUDA-capable GPU (or Google Colab)
- Icarus Verilog, for simulating the generated testbenches

### Setup

1. Clone the repository:

```bash
git clone https://github.com/HalgasAdrian/LLM-For-Automatic-Hardware-Testbench-Generation.git
cd LLM-For-Automatic-Hardware-Testbench-Generation
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

4. (Optional) Add any API keys you use, such as a Weights & Biases key for experiment tracking, to a `.env` file.

### Running the pipeline

Steps 1 and 2 run on your own machine. Steps 3 and 4 run on Google Colab.

1. Prepare the data:

```bash
python scripts/data_pipeline.py
```

2. Package the project for Colab:

```bash
python scripts/prepare_for_colab.py
```

3. Upload the zip to Colab and train the model:

```bash
python scripts/train.py
```

4. Evaluate the trained model in the Colab notebook:

```bash
python scripts/evaluate.py
```

To change the model, training settings, or evaluation settings, edit `configs/config.yaml`.

### Code formatting

```bash
black scripts/ utils/
flake8 scripts/ utils/
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- The AutoBench dataset creators
- MG-Verilog dataset from GaTech-EIC
- HDLBits for Verilog problems
- Hugging Face for transformer models
- Claude for coding help
