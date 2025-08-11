#!/usr/bin/env python3
"""Process raw Verilog data into training format."""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import random
import re
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import DataProcessor
from utils.verilog_utils import VerilogProcessor, clean_verilog_code

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_autobench_structure():
    """Analyze the structure of AutoBench dataset."""
    autobench_path = Path("data/raw/autobench")
    
    if not autobench_path.exists():
        logger.error(f"AutoBench path not found: {autobench_path}")
        return
    
    logger.info("Analyzing AutoBench structure...")
    
    # Common patterns for DUT and testbench files
    dut_patterns = ['*design*.v', '*dut*.v', '*.v']
    tb_patterns = ['*tb*.v', '*test*.v', '*testbench*.v']
    
    verilog_files = list(autobench_path.rglob("*.v"))
    logger.info(f"Found {len(verilog_files)} Verilog files")
    
    # Try to identify DUT-TB pairs
    pairs_found = 0
    for vfile in verilog_files[:10]:  # Sample first 10 files
        logger.info(f"  - {vfile.relative_to(autobench_path)}")
    
    return verilog_files


def process_autobench_enhanced(data_processor: DataProcessor) -> List[Dict[str, str]]:
    """Enhanced AutoBench processing with better pattern matching."""
    data_pairs = []
    autobench_path = Path("data/raw/autobench")
    
    if not autobench_path.exists():
        logger.warning(f"AutoBench path not found: {autobench_path}")
        return data_pairs
    
    # Find all testbench files
    tb_files = list(autobench_path.rglob("*_tb.v"))
    logger.info(f"Found {len(tb_files)} testbench files")
    
    # Process each testbench file
    for tb_file in tb_files:
        try:
            # Get the base name by removing _tb suffix
            base_name = tb_file.stem.replace('_tb', '')
            
            # Look for corresponding DUT file in the same directory
            dut_file = tb_file.parent / f"{base_name}.v"
            
            if dut_file.exists():
                tb_content = tb_file.read_text()
                dut_content = dut_file.read_text()
                
                # Create the instruction pair
                pair = data_processor.create_instruction_pair(dut_content, tb_content)
                data_pairs.append(pair)
                
                logger.debug(f"Found pair: {dut_file.name} <-> {tb_file.name}")
            else:
                logger.debug(f"No matching DUT for {tb_file.name}")
                
        except Exception as e:
            logger.error(f"Error processing {tb_file}: {e}")
    
    # Also look for files with 'testbench' in the name
    testbench_files = [f for f in autobench_path.rglob("*testbench*.v") 
                       if f not in tb_files]
    
    for tb_file in testbench_files:
        try:
            # Try to find corresponding design file
            tb_dir = tb_file.parent
            all_v_files = list(tb_dir.glob("*.v"))
            
            # Find non-testbench file in same directory
            for v_file in all_v_files:
                if 'testbench' not in v_file.name.lower() and '_tb' not in v_file.name:
                    tb_content = tb_file.read_text()
                    dut_content = v_file.read_text()
                    
                    pair = data_processor.create_instruction_pair(dut_content, tb_content)
                    data_pairs.append(pair)
                    logger.debug(f"Found pair: {v_file.name} <-> {tb_file.name}")
                    break
                    
        except Exception as e:
            logger.error(f"Error processing {tb_file}: {e}")
    
    logger.info(f"Processed {len(data_pairs)} AutoBench pairs")
    return data_pairs


def process_mg_verilog_enhanced(data_processor: DataProcessor) -> List[Dict[str, str]]:
    """Process MG-Verilog dataset with enhanced parsing."""
    data_pairs = []
    mg_path = Path("data/raw/mg_verilog")
    
    if not mg_path.exists():
        logger.warning(f"MG-Verilog path not found: {mg_path}")
        return data_pairs
    
    # Try to load with datasets library (HuggingFace format)
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(str(mg_path))
        
        logger.info(f"Loaded MG-Verilog dataset with {len(dataset)} examples")
        
        # Process dataset entries
        for item in dataset:
            # Check common field names
            if 'code' in item and 'testbench' in item:
                pair = data_processor.create_instruction_pair(
                    item['code'], 
                    item['testbench']
                )
                data_pairs.append(pair)
            elif 'design' in item and 'testbench' in item:
                pair = data_processor.create_instruction_pair(
                    item['design'], 
                    item['testbench']
                )
                data_pairs.append(pair)
            elif 'module_code' in item and 'testbench_code' in item:
                pair = data_processor.create_instruction_pair(
                    item['module_code'], 
                    item['testbench_code']
                )
                data_pairs.append(pair)
                
    except Exception as e:
        logger.info(f"Could not load as HuggingFace dataset: {e}")
        
        # Fall back to JSON processing
        json_files = list(mg_path.rglob("*.json"))
        jsonl_files = list(mg_path.rglob("*.jsonl"))
        
        logger.info(f"Found {len(json_files)} JSON files and {len(jsonl_files)} JSONL files")
        
        # Process JSON files
        for json_file in json_files:
            if 'dataset_info' in json_file.name or 'state' in json_file.name:
                continue  # Skip metadata files
                
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Handle different possible structures
                if isinstance(data, list):
                    for item in data:
                        if 'design' in item and 'testbench' in item:
                            pair = data_processor.create_instruction_pair(
                                item['design'], 
                                item['testbench']
                            )
                            data_pairs.append(pair)
                elif isinstance(data, dict):
                    if 'design' in data and 'testbench' in data:
                        pair = data_processor.create_instruction_pair(
                            data['design'], 
                            data['testbench']
                        )
                        data_pairs.append(pair)
                        
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
    
    logger.info(f"Processed {len(data_pairs)} MG-Verilog pairs")
    return data_pairs


def create_sample_data(data_processor: DataProcessor) -> List[Dict[str, str]]:
    """Create sample DUT-testbench pairs for testing."""
    sample_pairs = []
    
    # Sample 1: Simple AND gate
    and_gate_dut = """module and_gate(
    input a,
    input b,
    output y
);
    assign y = a & b;
endmodule"""
    
    and_gate_tb = """`timescale 1ns / 1ps

module and_gate_tb;
    reg a, b;
    wire y;
    
    and_gate uut (
        .a(a),
        .b(b),
        .y(y)
    );
    
    initial begin
        $display("Testing AND gate");
        a = 0; b = 0; #10;
        $display("a=%b, b=%b, y=%b", a, b, y);
        
        a = 0; b = 1; #10;
        $display("a=%b, b=%b, y=%b", a, b, y);
        
        a = 1; b = 0; #10;
        $display("a=%b, b=%b, y=%b", a, b, y);
        
        a = 1; b = 1; #10;
        $display("a=%b, b=%b, y=%b", a, b, y);
        
        $finish;
    end
endmodule"""
    
    sample_pairs.append(data_processor.create_instruction_pair(and_gate_dut, and_gate_tb))
    
    # Sample 2: 2-bit counter
    counter_dut = """module counter_2bit(
    input clk,
    input reset,
    output reg [1:0] count
);
    always @(posedge clk or posedge reset) begin
        if (reset)
            count <= 2'b00;
        else
            count <= count + 1;
    end
endmodule"""
    
    counter_tb = """`timescale 1ns / 1ps

module counter_2bit_tb;
    reg clk, reset;
    wire [1:0] count;
    
    counter_2bit uut (
        .clk(clk),
        .reset(reset),
        .count(count)
    );
    
    // Clock generation
    always #5 clk = ~clk;
    
    initial begin
        $display("Testing 2-bit counter");
        clk = 0;
        reset = 1;
        #20 reset = 0;
        
        repeat(8) begin
            #10 $display("Time=%0t, count=%b", $time, count);
        end
        
        $finish;
    end
endmodule"""
    
    sample_pairs.append(data_processor.create_instruction_pair(counter_dut, counter_tb))
    
    logger.info(f"Created {len(sample_pairs)} sample pairs")
    return sample_pairs


def validate_data_pairs(data_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Validate and filter data pairs."""
    verilog_processor = VerilogProcessor()
    valid_pairs = []
    
    logger.info(f"Validating {len(data_pairs)} data pairs...")
    
    for pair in tqdm(data_pairs, desc="Validating pairs"):
        try:
            dut_code = pair['dut_code']
            tb_code = pair['testbench_code']
            
            # Basic validation
            if len(dut_code.strip()) < 50 or len(tb_code.strip()) < 50:
                continue
            
            # Check if it's actually Verilog code
            if 'module' not in dut_code or 'module' not in tb_code:
                continue
            
            # Clean the code
            pair['dut_code'] = clean_verilog_code(dut_code)
            pair['testbench_code'] = clean_verilog_code(tb_code)
            
            valid_pairs.append(pair)
            
        except Exception as e:
            logger.debug(f"Validation error: {e}")
            continue
    
    logger.info(f"Valid pairs after filtering: {len(valid_pairs)}")
    return valid_pairs


def main():
    """Main processing function."""
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data processor
    data_processor = DataProcessor(config)
    
    logger.info("Starting data processing...")
    
    # Analyze dataset structure first
    logger.info("\n=== Analyzing dataset structure ===")
    analyze_autobench_structure()
    
    # Collect all data pairs
    all_data = []
    
    # Process AutoBench with enhanced method
    logger.info("\n=== Processing AutoBench ===")
    autobench_pairs = process_autobench_enhanced(data_processor)
    all_data.extend(autobench_pairs)
    
    # Process MG-Verilog
    logger.info("\n=== Processing MG-Verilog ===")
    mg_pairs = process_mg_verilog_enhanced(data_processor)
    all_data.extend(mg_pairs)
    
    # If we don't have enough data, add sample pairs
    if len(all_data) < 10:
        logger.warning("Not enough data found, adding sample pairs for testing")
        sample_pairs = create_sample_data(data_processor)
        all_data.extend(sample_pairs)
    
    # Validate and clean data
    logger.info("\n=== Validating data ===")
    all_data = validate_data_pairs(all_data)
    
    if not all_data:
        logger.error("No valid data pairs found! Check your dataset structure.")
        logger.info("Creating sample data for testing purposes...")
        all_data = create_sample_data(data_processor)
    
    logger.info(f"\nTotal valid data pairs: {len(all_data)}")
    
    # Split data
    random.seed(config['project']['seed'])
    random.shuffle(all_data)
    
    train_data, val_data, test_data = data_processor.split_data(
        all_data,
        config['data']['train_split'],
        config['data']['val_split']
    )
    
    # Save processed data
    logger.info("\n=== Saving processed data ===")
    data_processor.save_processed_data(train_data, val_data, test_data)
    
    # Print statistics
    print("\n" + "="*50)
    print("Data Processing Complete!")
    print("="*50)
    print(f"Total examples: {len(all_data)}")
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Test examples: {len(test_data)}")
    print("\nNext step: Run 'python scripts/train.py' to start training")


if __name__ == "__main__":
    main()