#!/usr/bin/env python3
"""Simple evaluation script that just tests generation without compilation."""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
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


def generate_testbench(model, tokenizer, dut_code: str, max_new_tokens: int = 1024) -> str:
    """Generate a testbench for given DUT code."""
    # Format prompt
    prompt = f"""Generate a Verilog testbench for the following design under test (DUT). The testbench should include proper initialization, stimulus generation, and output verification.

```verilog
{dut_code.strip()}
```

### Response:"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    logger.info("Generating testbench...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    if "### Response:" in generated_text:
        generated_text = generated_text.split("### Response:")[-1].strip()
    
    # Clean markdown if present
    if "```verilog" in generated_text:
        generated_text = generated_text.split("```verilog")[1].split("```")[0]
    elif "```" in generated_text:
        generated_text = generated_text.split("```")[1].split("```")[0]
    
    return generated_text.strip()


def main():
    """Main evaluation function."""
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    test_data_path = Path(config['data']['processed_data_path']) / "test" / "test.jsonl"
    test_data = load_dataset(test_data_path)
    
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Load model
    logger.info("Loading trained model...")
    model_path = Path(config['training']['output_dir'])
    base_model_name = config['model']['base_model']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model
    logger.info(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Load LoRA weights
    logger.info(f"Loading LoRA weights from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    
    # Set to evaluation mode
    model.eval()
    
    # Create output directory
    output_dir = Path(config['data']['test_results_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate testbenches
    results = []
    for i, item in enumerate(tqdm(test_data, desc="Generating testbenches")):
        dut_code = item['dut_code']
        reference_tb = item.get('testbench_code', '')
        
        try:
            generated_tb = generate_testbench(model, tokenizer, dut_code)
            
            # Save generated testbench
            tb_file = output_dir / f"generated_tb_{i}.v"
            with open(tb_file, "w") as f:
                f.write(generated_tb)
            
            # Save comparison
            comparison_file = output_dir / f"comparison_{i}.txt"
            with open(comparison_file, "w") as f:
                f.write("=== DUT CODE ===\n")
                f.write(dut_code)
                f.write("\n\n=== GENERATED TESTBENCH ===\n")
                f.write(generated_tb)
                f.write("\n\n=== REFERENCE TESTBENCH ===\n")
                f.write(reference_tb)
            
            results.append({
                'id': i,
                'success': True,
                'generated_length': len(generated_tb),
                'reference_length': len(reference_tb)
            })
            
        except Exception as e:
            logger.error(f"Error generating testbench {i}: {e}")
            results.append({
                'id': i,
                'success': False,
                'error': str(e)
            })
    
    # Save results
    with open(output_dir / "generation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = sum(1 for r in results if r.get('success', False))
    print("\n" + "="*60)
    print("GENERATION RESULTS")
    print("="*60)
    print(f"Total examples: {len(test_data)}")
    print(f"Successful generations: {successful}")
    print(f"Failed generations: {len(test_data) - successful}")
    print(f"\nGenerated testbenches saved to: {output_dir}")
    print("\nYou can manually inspect the generated testbenches to assess quality.")


if __name__ == "__main__":
    main()