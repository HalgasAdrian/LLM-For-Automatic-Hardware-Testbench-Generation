#!/usr/bin/env python3
"""Analyze the structure of downloaded datasets."""

import os
from pathlib import Path
import json
from collections import defaultdict

def analyze_autobench():
    """Analyze AutoBench dataset structure in detail."""
    print("\n=== AutoBench Dataset Analysis ===")
    autobench_path = Path("data/raw/autobench")
    
    if not autobench_path.exists():
        print(f"AutoBench path not found: {autobench_path}")
        return
    
    # Find all Verilog files
    verilog_files = list(autobench_path.rglob("*.v"))
    print(f"Total Verilog files: {len(verilog_files)}")
    
    # Analyze directory structure
    dir_structure = defaultdict(list)
    for vfile in verilog_files:
        rel_path = vfile.relative_to(autobench_path)
        dir_structure[str(rel_path.parent)].append(vfile.name)
    
    print("\nDirectory structure:")
    for dir_path, files in sorted(dir_structure.items())[:10]:  # Show first 10
        print(f"\n{dir_path}:")
        for f in sorted(files):
            print(f"  - {f}")
    
    # Look for testbench patterns
    tb_files = [f for f in verilog_files if '_tb.v' in f.name or 'testbench' in f.name.lower()]
    dut_files = [f for f in verilog_files if '_tb.v' not in f.name and 'testbench' not in f.name.lower()]
    
    print(f"\nTestbench files found: {len(tb_files)}")
    print(f"Potential DUT files found: {len(dut_files)}")
    
    # Find pairs in the same directory
    pairs_found = 0
    print("\nSample DUT-TB pairs found:")
    for tb_file in tb_files[:5]:  # Show first 5 pairs
        tb_dir = tb_file.parent
        # Look for corresponding DUT
        base_name = tb_file.stem.replace('_tb', '')
        potential_duts = list(tb_dir.glob(f"{base_name}.v"))
        
        if potential_duts:
            print(f"\nPair {pairs_found + 1}:")
            print(f"  TB: {tb_file.relative_to(autobench_path)}")
            print(f"  DUT: {potential_duts[0].relative_to(autobench_path)}")
            pairs_found += 1
    
    print(f"\nTotal potential pairs: {pairs_found}")
    
    # Check for README or documentation
    readme_files = list(autobench_path.rglob("README*"))
    if readme_files:
        print("\nREADME files found:")
        for readme in readme_files:
            print(f"  - {readme.relative_to(autobench_path)}")


def analyze_mg_verilog():
    """Analyze MG-Verilog dataset structure."""
    print("\n=== MG-Verilog Dataset Analysis ===")
    mg_path = Path("data/raw/mg_verilog")
    
    if not mg_path.exists():
        print(f"MG-Verilog path not found: {mg_path}")
        return
    
    # List all files
    all_files = list(mg_path.rglob("*"))
    print(f"Total files: {len(all_files)}")
    
    # Categorize by extension
    extensions = defaultdict(int)
    for f in all_files:
        if f.is_file():
            extensions[f.suffix] += 1
    
    print("\nFile types:")
    for ext, count in sorted(extensions.items()):
        print(f"  {ext}: {count}")
    
    # Check JSON files
    json_files = list(mg_path.rglob("*.json"))
    if json_files:
        print(f"\nAnalyzing {len(json_files)} JSON files:")
        for json_file in json_files[:3]:  # Sample first 3
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                print(f"\n{json_file.name}:")
                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())[:10]}")  # First 10 keys
                elif isinstance(data, list) and data:
                    print(f"  List with {len(data)} items")
                    if isinstance(data[0], dict):
                        print(f"  First item keys: {list(data[0].keys())}")
            except Exception as e:
                print(f"  Error reading: {e}")
    
    # Check for Parquet files (common HuggingFace format)
    parquet_files = list(mg_path.rglob("*.parquet"))
    if parquet_files:
        print(f"\nFound {len(parquet_files)} Parquet files")
        print("Consider using pandas to read these files")


def analyze_hdlbits():
    """Check HDLBits directory."""
    print("\n=== HDLBits Dataset Analysis ===")
    hdl_path = Path("data/raw/hdlbits")
    
    if not hdl_path.exists():
        print(f"HDLBits path not found: {hdl_path}")
        return
    
    files = list(hdl_path.rglob("*"))
    print(f"Total files: {len(files)}")
    
    for f in files[:10]:
        if f.is_file():
            print(f"  - {f.relative_to(hdl_path)}")


def main():
    """Run all analyses."""
    print("Analyzing dataset structure...")
    
    analyze_autobench()
    analyze_mg_verilog()
    analyze_hdlbits()
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print("="*50)


if __name__ == "__main__":
    main()