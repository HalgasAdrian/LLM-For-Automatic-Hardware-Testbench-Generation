#!/usr/bin/env python3
"""Fix markdown in training data responses."""

import json
from pathlib import Path
import shutil

def clean_response(response):
    """Remove markdown code blocks from response."""
    # Check if response contains markdown
    if "```verilog" in response:
        response = response.split("```verilog")[1]
        if "```" in response:
            response = response.split("```")[0]
    elif "```" in response:
        parts = response.split("```")
        if len(parts) >= 3:
            response = parts[1]
    
    return response.strip()

def main():
    """Fix all processed data files."""
    processed_path = Path("data/processed")
    
    # Track statistics
    total_fixed = 0
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_file = processed_path / split / f"{split}.jsonl"
        
        if not split_file.exists():
            print(f"Skipping {split} - file not found")
            continue
        
        print(f"\nProcessing {split} data...")
        
        # Read and fix data
        fixed_data = []
        markdown_count = 0
        
        with open(split_file, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    item = json.loads(line)
                    
                    # Check and clean the response
                    original_response = item['response']
                    cleaned_response = clean_response(original_response)
                    
                    # Check if it had markdown
                    if "```" in original_response:
                        markdown_count += 1
                    
                    # Update the item
                    item['response'] = cleaned_response
                    item['testbench_code'] = cleaned_response
                    
                    fixed_data.append(item)
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
        
        # Backup original file
        backup_file = split_file.parent / f"{split}_with_markdown.jsonl"
        shutil.copy(split_file, backup_file)
        print(f"Backed up original to: {backup_file}")
        
        # Save fixed data
        with open(split_file, 'w') as f:
            for item in fixed_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Fixed {split} data:")
        print(f"  - Total examples: {len(fixed_data)}")
        print(f"  - Examples with markdown: {markdown_count}")
        print(f"  - Percentage with markdown: {markdown_count/len(fixed_data)*100:.1f}%")
        
        total_fixed += len(fixed_data)
    
    print(f"\n{'='*50}")
    print(f"Total examples fixed: {total_fixed}")
    print("All data files have been cleaned!")
    print("\nNext steps:")
    print("1. Run augment_data.py to add more examples")
    print("2. Run prepare_for_colab.py to create new training package")
    print("3. Upload to Colab and retrain")

if __name__ == "__main__":
    main()