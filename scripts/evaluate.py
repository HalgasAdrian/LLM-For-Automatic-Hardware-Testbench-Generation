#!/usr/bin/env python3
"""Evaluate the trained testbench generation model."""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import subprocess
import tempfile
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_dataset
from utils.verilog_utils import VerilogProcessor, clean_verilog_code

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestbenchEvaluator:
    """Evaluate generated testbenches."""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.verilog_processor = VerilogProcessor(
            timeout=config['evaluation']['iverilog_timeout']
        )
        
    def generate_testbench(self, dut_code: str, max_new_tokens: int = None) -> str:
        """Generate a testbench for given DUT code with simplified prompting."""
        
        # Use config value if not specified, default to 4096
        if max_new_tokens is None:
            max_new_tokens = self.config['generation'].get('max_new_tokens', 4096)
        
        # Extract module information from DUT
        module_match = re.search(r'module\s+(\w+)\s*\((.*?)\);', dut_code, re.DOTALL)
        if module_match:
            module_name = module_match.group(1)
            ports_section = module_match.group(2)
            
            # Extract ports
            inputs = re.findall(r'input\s+(?:\[\d+:\d+\]\s+)?(\w+)', ports_section)
            outputs = re.findall(r'output\s+(?:reg\s+)?(?:\[\d+:\d+\]\s+)?(\w+)', ports_section)
        else:
            module_name = "top_module"
            inputs = []
            outputs = []
        
        # SIMPLIFIED PROMPT - Much shorter to save tokens
        prompt = f"""Generate a complete Verilog testbench for this DUT:

```verilog
{dut_code.strip()}
```

Requirements:
- Start with `timescale 1ns / 1ps
- Module name: {module_name}_tb;
- Inputs ({', '.join(inputs)}) as reg
- Outputs ({', '.join(outputs)}) as wire
- Instantiate DUT as 'uut'
- Add initial block with test cases
- Include $display statements
- End with $finish; and endmodule

### Response:
`timescale 1ns / 1ps

module {module_name}_tb;"""
        
        # Tokenize with SMALLER max_length for prompt to leave room for generation
        inputs_tok = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024  # Reduced from 2048 to leave room for generation
        )
        inputs_tok = {k: v.to(self.model.device) for k, v in inputs_tok.items()}
        
        # Generate with MORE tokens
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs_tok,
                max_new_tokens=max_new_tokens,  # Now uses 4096 from config
                temperature=self.config['generation']['temperature'],
                top_p=self.config['generation']['top_p'],
                do_sample=self.config['generation']['do_sample'],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "### Response:" in generated_text:
            generated_text = generated_text.split("### Response:")[-1].strip()
        
        # Clean the generated code
        generated_text = clean_verilog_code(generated_text)
        
        # Additional post-processing to fix common issues
        generated_text = self.post_process_testbench(generated_text)
        
        # Ensure it starts with timescale if missing
        if not generated_text.startswith('`timescale'):
            generated_text = '`timescale 1ns / 1ps\n\n' + generated_text
        
        return generated_text
    
    def post_process_testbench(self, tb_code: str) -> str:
        """Post-process generated testbench to fix common syntax errors."""
        # Remove any markdown artifacts
        tb_code = re.sub(r'```verilog\s*', '', tb_code)
        tb_code = re.sub(r'```\s*', '', tb_code)
        
        # Fix common syntax errors
        tb_code = tb_code.replace('initial {', 'initial begin')
        
        # Be more careful with brace replacement - only replace standalone braces
        lines = tb_code.split('\n')
        fixed_lines = []
        for line in lines:
            # Don't replace } in things like 4'b{1,0,1,0}
            if line.strip() == '}':
                fixed_lines.append(line.replace('}', 'end'))
            elif line.strip().startswith('}') and '//' in line:
                # Handle "} // comment" case
                fixed_lines.append(line.replace('}', 'end', 1))
            else:
                fixed_lines.append(line)
        tb_code = '\n'.join(fixed_lines)
        
        # Fix incorrect bit width syntax (e.g., 10'b101 should be 3'b101)
        def fix_bit_width(match):
            width = match.group(1)
            binary = match.group(2)
            actual_width = len(binary)
            return f"{actual_width}'b{binary}"
        
        tb_code = re.sub(r"(\d+)'b([01]+)", fix_bit_width, tb_code)
        
        # Ensure proper module ending
        if 'endmodule' not in tb_code:
            # Check if we need to close any open blocks first
            begin_count = tb_code.count('begin')
            end_count = tb_code.count('end')
            
            if begin_count > end_count:
                # Add missing ends
                for _ in range(begin_count - end_count):
                    tb_code += '\nend'
            
            tb_code = tb_code.rstrip() + '\nendmodule'
        
        # Fix missing semicolons after $finish
        tb_code = re.sub(r'\$finish\s*(?!;)', '$finish;', tb_code)
        
        # Remove duplicate initial blocks if they exist
        initial_blocks = tb_code.split('initial begin')
        if len(initial_blocks) > 5:  # Too many initial blocks
            # Keep first block (module declaration) and next 2-3 test blocks
            reconstructed = initial_blocks[0]
            for i in range(1, min(4, len(initial_blocks))):
                if 'endmodule' not in initial_blocks[i]:
                    reconstructed += 'initial begin' + initial_blocks[i]
            if 'endmodule' not in reconstructed:
                reconstructed += '\nendmodule'
            tb_code = reconstructed
        
        return tb_code.strip()
    
    def evaluate_single(self, dut_code: str, reference_tb: str = None) -> Dict:
        """Evaluate a single DUT-testbench pair."""
        results = {
            'generation_success': False,
            'compilation_success': False,
            'simulation_success': False,
            'syntax_valid': False,
            'has_required_components': {},
            'error_message': None,
            'generated_testbench': None,
            'generation_length': 0
        }
        
        try:
            # Generate testbench with increased token limit
            logger.info("Generating testbench...")
            generated_tb = self.generate_testbench(dut_code, max_new_tokens=4096)
            results['generated_testbench'] = generated_tb
            results['generation_success'] = True
            results['generation_length'] = len(generated_tb)
            
            # Log generation details for debugging
            logger.debug(f"Generated testbench length: {len(generated_tb)} characters")
            logger.debug(f"Ends with endmodule: {'endmodule' in generated_tb}")
            
            # Validate syntax and structure
            validations = self.verilog_processor.validate_testbench_structure(generated_tb)
            results['has_required_components'] = validations
            results['syntax_valid'] = all(validations.values())
            
            # Try to compile
            logger.info("Compiling testbench...")
            compile_success, compile_error = self.verilog_processor.compile_verilog(
                dut_code, generated_tb
            )
            results['compilation_success'] = compile_success
            
            if compile_success:
                # Try to simulate
                logger.info("Running simulation...")
                sim_success, sim_output, sim_error = self.verilog_processor.run_simulation()
                results['simulation_success'] = sim_success
                
                if not sim_success:
                    results['error_message'] = f"Simulation error: {sim_error}"
            else:
                results['error_message'] = f"Compilation error: {compile_error}"
                
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            results['error_message'] = str(e)
        
        return results
    
    def evaluate_dataset(self, test_data: List[Dict]) -> Dict:
        """Evaluate the model on a test dataset."""
        all_results = []
        
        for item in tqdm(test_data, desc="Evaluating"):
            dut_code = item['dut_code']
            reference_tb = item.get('testbench_code', None)
            
            results = self.evaluate_single(dut_code, reference_tb)
            results['item_id'] = item.get('id', len(all_results))
            all_results.append(results)
        
        # Compute aggregate metrics
        metrics = self.compute_metrics(all_results)
        
        return {
            'metrics': metrics,
            'detailed_results': all_results
        }
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics from evaluation results."""
        total = len(results)
        if total == 0:
            return {}
        
        metrics = {
            'total_examples': total,
            'generation_success_rate': sum(r['generation_success'] for r in results) / total,
            'compilation_success_rate': sum(r['compilation_success'] for r in results) / total,
            'simulation_success_rate': sum(r['simulation_success'] for r in results) / total,
            'syntax_valid_rate': sum(r['syntax_valid'] for r in results) / total,
        }
        
        # Component presence rates
        component_stats = {}
        for component in ['has_timescale', 'has_module', 'has_initial', 'has_finish', 'has_display']:
            component_stats[f'{component}_rate'] = sum(
                r['has_required_components'].get(component, False) for r in results
            ) / total
        
        metrics.update(component_stats)
        
        # Add generation length statistics
        gen_lengths = [r['generation_length'] for r in results if r['generation_success']]
        if gen_lengths:
            metrics['avg_generation_length'] = sum(gen_lengths) / len(gen_lengths)
            metrics['min_generation_length'] = min(gen_lengths)
            metrics['max_generation_length'] = max(gen_lengths)
        
        return metrics


def load_model(config: dict):
    """Load the trained model."""
    model_path = Path(config['training']['output_dir'])
    base_model_name = config['model']['base_model']
    
    logger.info(f"Loading base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model with better memory management
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        # Use float16 for GPU
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        # Use float32 for CPU
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    # Load LoRA weights
    logger.info(f"Loading LoRA weights from: {model_path}")
    try:
        model = PeftModel.from_pretrained(
            model, 
            model_path,
            device_map="auto" if device == "cuda" else None,
            offload_folder=None
        )
    except Exception as e:
        logger.warning(f"Error loading with PeftModel: {e}")
        logger.info("Trying alternative loading method...")
        # Alternative: manually load the adapter
        from peft import LoraConfig, get_peft_model
        import json
        
        # Load adapter config
        with open(model_path / "adapter_config.json", "r") as f:
            adapter_config = json.load(f)
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=adapter_config["r"],
            lora_alpha=adapter_config["lora_alpha"],
            target_modules=adapter_config["target_modules"],
            lora_dropout=adapter_config["lora_dropout"],
            bias=adapter_config["bias"],
            task_type=adapter_config["task_type"]
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Load the weights manually
        adapter_weights = torch.load(
            model_path / "adapter_model.safetensors",
            map_location=device
        )
        model.load_state_dict(adapter_weights, strict=False)
    
    # Move to appropriate device
    if device == "cuda":
        model = model.cuda()
    elif torch.backends.mps.is_available():
        # For Apple Silicon
        model = model.to("mps")
    
    # Set to evaluation mode
    model.eval()
    
    return model, tokenizer


def test_generation_completeness(model, tokenizer, config):
    """Quick test to verify generation completes properly."""
    print("\n" + "="*60)
    print("TESTING GENERATION COMPLETENESS")
    print("="*60)
    
    # Simple test DUT
    test_dut = """module counter(
    input clk,
    input reset,
    output reg [3:0] count
);
    always @(posedge clk or posedge reset) begin
        if (reset)
            count <= 4'b0000;
        else
            count <= count + 1;
    end
endmodule"""
    
    evaluator = TestbenchEvaluator(model, tokenizer, config)
    
    # Test with different token limits
    for max_tokens in [2048, 4096, 6000]:
        print(f"\nTesting with max_new_tokens={max_tokens}")
        result = evaluator.generate_testbench(test_dut, max_new_tokens=max_tokens)
        
        print(f"  Generated length: {len(result)} characters")
        print(f"  Has endmodule: {'endmodule' in result}")
        print(f"  Has $finish: {'$finish' in result}")
        
        if len(result) > 200:
            print(f"  Last 200 chars: ...{result[-200:]}")
        
        # Check if it's complete
        if 'endmodule' in result and '$finish' in result:
            print(f"  ✅ Complete testbench generated!")
            break
        else:
            print(f"  ⚠️ Incomplete generation, trying with more tokens...")
    
    return result


def main():
    """Main evaluation function."""
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update generation config to use more tokens
    config['generation']['max_new_tokens'] = 4096
    
    # Load test data
    test_data_path = Path(config['data']['processed_data_path']) / "test" / "test.jsonl"
    test_data = load_dataset(test_data_path)
    
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Limit test set size if configured
    if config['evaluation']['test_set_size'] and len(test_data) > config['evaluation']['test_set_size']:
        test_data = test_data[:config['evaluation']['test_set_size']]
        logger.info(f"Limited to {len(test_data)} examples for evaluation")
    
    # Load model
    logger.info("Loading trained model...")
    model, tokenizer = load_model(config)
    
    # Optional: Run quick test first
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_generation_completeness(model, tokenizer, config)
        return
    
    # Create evaluator
    evaluator = TestbenchEvaluator(model, tokenizer, config)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    logger.info(f"Using max_new_tokens: {config['generation']['max_new_tokens']}")
    results = evaluator.evaluate_dataset(test_data)
    
    # Save results
    output_dir = Path(config['data']['test_results_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save generated testbenches
    for i, result in enumerate(results['detailed_results']):
        if result['generated_testbench']:
            tb_file = output_dir / f"generated_tb_{i}.v"
            with open(tb_file, "w") as f:
                f.write(result['generated_testbench'])
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    metrics = results['metrics']
    print(f"Total examples evaluated: {metrics['total_examples']}")
    print(f"\nSuccess Rates:")
    print(f"  Generation: {metrics['generation_success_rate']:.1%}")
    print(f"  Compilation: {metrics['compilation_success_rate']:.1%}")
    print(f"  Simulation: {metrics['simulation_success_rate']:.1%}")
    print(f"  Valid Syntax: {metrics['syntax_valid_rate']:.1%}")
    
    print(f"\nComponent Presence:")
    for key, value in metrics.items():
        if '_rate' in key and 'has_' in key:
            print(f"  {key.replace('_rate', '').replace('has_', '')}: {value:.1%}")
    
    if 'avg_generation_length' in metrics:
        print(f"\nGeneration Statistics:")
        print(f"  Average length: {metrics['avg_generation_length']:.0f} chars")
        print(f"  Min length: {metrics['min_generation_length']} chars")
        print(f"  Max length: {metrics['max_generation_length']} chars")
    
    print(f"\nDetailed results saved to: {output_dir}")
    print("\nNext step: Run 'python scripts/generate.py --dut_file <your_design.v>' to generate testbenches for new designs")


if __name__ == "__main__":
    # Check for required tools
    if subprocess.run(["which", "iverilog"], capture_output=True).returncode != 0:
        logger.warning("Icarus Verilog not found! Install with: brew install icarus-verilog")
        logger.warning("Continuing without compilation/simulation testing...")
    
    main()