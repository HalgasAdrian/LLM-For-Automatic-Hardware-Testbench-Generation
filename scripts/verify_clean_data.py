#!/usr/bin/env python3
"""Verify training data has no markdown in responses."""

import json
from pathlib import Path

def check_data_quality(file_path):
    """Check if data file has markdown or other issues."""
    print(f"\nChecking: {file_path}")
    
    if not Path(file_path).exists():
        print(f"  File not found!")
        return
    
    total = 0
    has_markdown = 0
    has_timescale = 0
    has_module = 0
    has_endmodule = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            response = item['response']
            total += 1
            
            # Check for markdown
            if '```' in response:
                has_markdown += 1
            
            # Check for proper Verilog structure
            if '`timescale' in response:
                has_timescale += 1
            if 'module' in response:
                has_module += 1
            if 'endmodule' in response:
                has_endmodule += 1
    
    print(f"  Total examples: {total}")
    print(f"  Has markdown (```): {has_markdown} ({has_markdown/total*100:.1f}%)")
    print(f"  Has timescale: {has_timescale} ({has_timescale/total*100:.1f}%)")
    print(f"  Has module: {has_module} ({has_module/total*100:.1f}%)")
    print(f"  Has endmodule: {has_endmodule} ({has_endmodule/total*100:.1f}%)")
    
    # Show a sample
    if total > 0:
        with open(file_path, 'r') as f:
            first_line = f.readline()
            first_item = json.loads(first_line)
            print(f"\n  First response preview (first 300 chars):")
            print(f"  {first_item['response'][:300]}...")

# Check both files
print("="*60)
print("TRAINING DATA VERIFICATION")
print("="*60)

check_data_quality("data/processed/train/train.jsonl")
check_data_quality("data/processed/train/train_augmented.jsonl")

print("\n" + "="*60)
print("RECOMMENDATION:")
if Path("data/processed/train/train_augmented.jsonl").exists():
    print("Use train_augmented.jsonl for training (rename to train.jsonl)")
else:
    print("Current train.jsonl should be clean")
print("="*60)
