#!/usr/bin/env python3
"""Train the LLM for Verilog testbench generation."""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import wandb
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VerilogDataset:
    """Custom dataset for Verilog testbench generation."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as conversation
        text = f"{item['instruction']}\n\n### Response:\n{item['response']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Set up labels for language modeling
        encoding["labels"] = encoding["input_ids"].clone()
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["labels"].squeeze()
        }


def setup_model_and_tokenizer(config: dict):
    """Set up the model and tokenizer with quantization and LoRA."""
    model_name = config['model']['base_model']
    
    logger.info(f"Loading model: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Quantization config for 4-bit
    bnb_config = None
    if config['model']['quantization']['load_in_4bit']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=config['model']['quantization']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=config['model']['quantization']['bnb_4bit_quant_type']
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Prepare model for k-bit training
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=config['model']['lora']['r'],
        lora_alpha=config['model']['lora']['lora_alpha'],
        target_modules=config['model']['lora']['target_modules'],
        lora_dropout=config['model']['lora']['lora_dropout'],
        bias=config['model']['lora']['bias'],
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_datasets(config: dict, tokenizer):
    """Load and prepare datasets for training."""
    # Load data
    train_data = load_dataset(Path(config['data']['processed_data_path']) / "train" / "train.jsonl")
    val_data = load_dataset(Path(config['data']['processed_data_path']) / "val" / "val.jsonl")
    
    logger.info(f"Loaded {len(train_data)} training examples")
    logger.info(f"Loaded {len(val_data)} validation examples")
    
    # Create datasets
    train_dataset = VerilogDataset(train_data, tokenizer, config['data']['max_testbench_length'])
    val_dataset = VerilogDataset(val_data, tokenizer, config['data']['max_testbench_length'])
    
    return train_dataset, val_dataset


def setup_training_args(config: dict):
    """Set up training arguments."""
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        eval_strategy="steps",
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        fp16=config['training']['fp16'],
        optim=config['training']['optim'],
        max_grad_norm=config['training']['max_grad_norm'],
        warmup_ratio=config['training']['warmup_ratio'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name=f"verilog-tb-gen-{config['model']['base_model'].split('/')[-1]}",
        push_to_hub=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    
    return training_args


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    # For language modeling, we typically use perplexity
    # But for now, we'll just return loss
    return {}


def main():
    """Main training function."""
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb if API key is available
    if os.getenv("WANDB_API_KEY"):
        wandb.init(
            project=config['project']['name'],
            config=config,
            name=f"train-{config['model']['base_model'].split('/')[-1]}"
        )
    
    # Set up model and tokenizer
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset, val_dataset = prepare_datasets(config, tokenizer)
    
    # Set up training arguments
    training_args = setup_training_args(config)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Check if we can do a quick test
    if len(train_dataset) < 5:
        logger.warning("Very small dataset - running in test mode")
        training_args.num_train_epochs = 1
        training_args.eval_steps = 2
        training_args.save_steps = 10
        training_args.logging_steps = 1
    
    # Start training
    logger.info("Starting training...")
    logger.info(f"Total training steps: {trainer.state.max_steps}")
    
    # Train
    train_result = trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Save training results
    with open(Path(training_args.output_dir) / "training_results.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(f"Model saved to: {training_args.output_dir}")
    
    # Final metrics
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Final loss: {train_result.metrics.get('train_loss', 'N/A')}")
    print(f"Total training time: {train_result.metrics.get('train_runtime', 'N/A')} seconds")
    print("\nNext step: Run 'python scripts/evaluate.py' to evaluate the model")


if __name__ == "__main__":
    # Check for GPU
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available, training will be slow!")
    
    main()