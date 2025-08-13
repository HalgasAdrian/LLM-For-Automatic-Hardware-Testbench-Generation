#!/usr/bin/env python3
"""
Filter training data to keep only valid, compilable examples.
This will help train a model that generates correct Verilog.
"""

import json
import re
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Tuple, Dict

def check_verilog_syntax(dut_code: str, tb_code: str) -> Tuple[bool, str]:
    """Check if DUT and testbench compile successfully."""
    
    # Quick validation first
    if not validate_structure(tb_code):
        return False, "Structure validation failed"
    
    # Try to compile with iverilog
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            dut_file = os.path.join(tmpdir, "dut.v")
            tb_file = os.path.join(tmpdir, "tb.v")
            
            with open(dut_file, 'w') as f:
                f.write(dut_code)
            with open(tb_file, 'w') as f:
                f.write(tb_code)
            
            # Try compilation
            result = subprocess.run(
                ['iverilog', '-o', os.path.join(tmpdir, 'sim'), tb_file, dut_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return True, "Compilation successful"
            else:
                return False, result.stderr[:200]  # First 200 chars of error
                
    except subprocess.TimeoutExpired:
        return False, "Compilation timeout"
    except FileNotFoundError:
        # iverilog not installed, use basic validation only
        return validate_structure(tb_code), "iverilog not available, basic validation only"
    except Exception as e:
        return False, str(e)

def validate_structure(tb_code: str) -> bool:
    """Basic structural validation of testbench."""
    
    checks = {
        'has_timescale': '`timescale' in tb_code,
        'has_module': bool(re.search(r'module\s+\w+\s*;', tb_code)),
        'has_endmodule': tb_code.strip().endswith('endmodule'),
        'has_initial': 'initial begin' in tb_code,
        'has_finish': '$finish' in tb_code,
        'balanced_begin_end': tb_code.count('begin') == (tb_code.count('end') - tb_code.count('endmodule')),
        'has_uut': bool(re.search(r'\w+\s+(uut|UUT|dut|DUT)\s*\(', tb_code)),
        'no_weird_repetition': tb_code.count('endmodule') == 1,
    }
    
    # All checks must pass
    return all(checks.values())

def fix_common_issues(tb_code: str) -> str:
    """Fix common issues in testbench code."""
    
    # Remove multiple endmodules
    while tb_code.count('endmodule') > 1:
        # Keep the first one
        parts = tb_code.split('endmodule', 1)
        tb_code = parts[0] + 'endmodule'
    
    # Fix missing $finish
    if '$finish' not in tb_code and 'initial begin' in tb_code:
        lines = tb_code.split('\n')
        # Find the last end before endmodule
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == 'end' and i > 0:
                # Check if this is part of initial block
                found_initial = False
                for j in range(i-1, -1, -1):
                    if 'initial begin' in lines[j]:
                        found_initial = True
                        break
                if found_initial:
                    lines.insert(i, '        $finish;')
                    tb_code = '\n'.join(lines)
                    break
    
    # Ensure single endmodule at end
    if not tb_code.strip().endswith('endmodule'):
        tb_code = tb_code.rstrip() + '\nendmodule'
    
    return tb_code

def filter_training_data(input_file: str, output_file: str, synthetic_only: bool = False):
    """Filter training data to keep only valid examples."""
    
    print(f"Processing: {input_file}")
    
    all_examples = []
    valid_examples = []
    invalid_examples = []
    fixed_examples = []
    
    # Load data
    with open(input_file, 'r') as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line)
                all_examples.append(item)
                
                dut_code = item.get('dut_code', '')
                tb_code = item.get('response', '')
                
                # Check if valid
                is_valid, error = check_verilog_syntax(dut_code, tb_code)
                
                if is_valid:
                    valid_examples.append(item)
                else:
                    # Try to fix
                    fixed_tb = fix_common_issues(tb_code)
                    is_fixed_valid, _ = check_verilog_syntax(dut_code, fixed_tb)
                    
                    if is_fixed_valid:
                        item['response'] = fixed_tb
                        item['testbench_code'] = fixed_tb
                        fixed_examples.append(item)
                        valid_examples.append(item)
                    else:
                        invalid_examples.append((idx, error))
                        
            except Exception as e:
                print(f"Error processing line {idx}: {e}")
    
    print(f"\nResults:")
    print(f"Total examples: {len(all_examples)}")
    print(f"Originally valid: {len(valid_examples) - len(fixed_examples)}")
    print(f"Fixed: {len(fixed_examples)}")
    print(f"Still invalid: {len(invalid_examples)}")
    print(f"Final valid: {len(valid_examples)}")
    
    # Show some invalid examples
    if invalid_examples:
        print(f"\nFirst 3 invalid examples:")
        for idx, error in invalid_examples[:3]:
            print(f"  Example {idx}: {error[:100]}")
    
    # Save valid examples
    if valid_examples:
        with open(output_file, 'w') as f:
            for item in valid_examples:
                f.write(json.dumps(item) + '\n')
        print(f"\nSaved {len(valid_examples)} valid examples to: {output_file}")
    else:
        print("\n⚠️ No valid examples found!")
    
    return valid_examples

def create_clean_training_set():
    """Create a clean training dataset."""
    
    print("="*60)
    print("CREATING CLEAN TRAINING DATASET")
    print("="*60)
    
    # Check synthetic examples first (these should be good)
    synthetic_file = Path("data/processed/new_examples.jsonl")
    if synthetic_file.exists():
        print("\n1. Checking synthetic examples...")
        synthetic_valid = filter_training_data(
            str(synthetic_file),
            "data/processed/train/synthetic_valid.jsonl"
        )
        print(f"   Synthetic valid: {len(synthetic_valid)}")
    else:
        synthetic_valid = []
    
    # Check original training data
    train_file = Path("data/processed/train/train.jsonl")
    if train_file.exists():
        print("\n2. Checking original training data...")
        train_valid = filter_training_data(
            str(train_file),
            "data/processed/train/original_valid.jsonl"
        )
        
        # Remove synthetic examples from train_valid to avoid duplicates
        train_valid_filtered = []
        synthetic_instructions = {item['instruction'] for item in synthetic_valid}
        for item in train_valid:
            if item['instruction'] not in synthetic_instructions:
                train_valid_filtered.append(item)
        
        print(f"   Original valid (non-synthetic): {len(train_valid_filtered)}")
    else:
        train_valid_filtered = []
    
    # Combine all valid examples
    all_valid = synthetic_valid + train_valid_filtered
    
    print(f"\n3. Combined valid examples: {len(all_valid)}")
    
    # Save combined clean dataset
    clean_file = Path("data/processed/train/train_clean.jsonl")
    with open(clean_file, 'w') as f:
        for item in all_valid:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Clean training data saved to: {clean_file}")
    print(f"   Total examples: {len(all_valid)}")
    
    if len(all_valid) < 20:
        print("\n⚠️ WARNING: Very few valid examples!")
        print("   Consider:")
        print("   1. Using relaxed validation (structure only, no compilation)")
        print("   2. Manually creating more examples")
        print("   3. Fixing the invalid examples manually")
    else:
        print("\n✅ Ready for retraining!")
        print("\nNext steps:")
        print("1. Backup current model:")
        print("   mv models/checkpoints models/checkpoints_backup")
        print("\n2. Use clean training data:")
        print("   cp data/processed/train/train.jsonl data/processed/train/train_original.jsonl")
        print("   cp data/processed/train/train_clean.jsonl data/processed/train/train.jsonl")
        print("\n3. Retrain the model:")
        print("   python scripts/train.py")

if __name__ == "__main__":
    create_clean_training_set()