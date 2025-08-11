"""Data processing utilities."""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import re
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process Verilog DUT and testbench pairs."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_path = Path(config['data']['raw_data_path'])
        self.processed_path = Path(config['data']['processed_data_path'])
        
    def create_instruction_pair(self, dut_code: str, testbench_code: str) -> Dict[str, str]:
        """Create instruction-response pair for training."""
        instruction = (
            "Generate a Verilog testbench for the following design under test (DUT). "
            "The testbench should include proper initialization, stimulus generation, "
            "and output verification.\n\n"
            f"```verilog\n{dut_code.strip()}\n```"
        )
        
        response = f"```verilog\n{testbench_code.strip()}\n```"
        
        return {
            "instruction": instruction,
            "response": response,
            "dut_code": dut_code,
            "testbench_code": testbench_code
        }
    
    def process_autobench_data(self) -> List[Dict[str, str]]:
        """Process AutoBench dataset."""
        data_pairs = []
        autobench_path = self.data_path / "autobench"
        
        if not autobench_path.exists():
            logger.warning(f"AutoBench path not found: {autobench_path}")
            return data_pairs
        
        # Look for DUT and testbench pairs
        for dut_file in autobench_path.glob("**/design*.v"):
            # Find corresponding testbench
            tb_name = dut_file.name.replace("design", "testbench")
            tb_file = dut_file.parent / tb_name
            
            if tb_file.exists():
                try:
                    dut_code = dut_file.read_text()
                    tb_code = tb_file.read_text()
                    
                    pair = self.create_instruction_pair(dut_code, tb_code)
                    data_pairs.append(pair)
                except Exception as e:
                    logger.error(f"Error processing {dut_file}: {e}")
        
        logger.info(f"Processed {len(data_pairs)} AutoBench pairs")
        return data_pairs
    
    def process_hdlbits_data(self) -> List[Dict[str, str]]:
        """Process HDLBits data if available."""
        data_pairs = []
        hdlbits_path = self.data_path / "hdlbits"
        
        if not hdlbits_path.exists():
            logger.warning(f"HDLBits path not found: {hdlbits_path}")
            return data_pairs
        
        # Process HDLBits format
        # This is a placeholder - actual implementation depends on data format
        
        return data_pairs
    
    def process_mg_verilog_data(self) -> List[Dict[str, str]]:
        """Process MG-Verilog dataset from HuggingFace."""
        data_pairs = []
        mg_path = self.data_path / "mg_verilog"
        
        if not mg_path.exists():
            logger.warning(f"MG-Verilog path not found: {mg_path}")
            return data_pairs
        
        # Load JSON files from MG-Verilog
        for json_file in mg_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract DUT and testbench if available
                if 'design_code' in data and 'testbench_code' in data:
                    pair = self.create_instruction_pair(
                        data['design_code'], 
                        data['testbench_code']
                    )
                    data_pairs.append(pair)
                    
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        logger.info(f"Processed {len(data_pairs)} MG-Verilog pairs")
        return data_pairs
    
    def split_data(self, data: List[Dict], 
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train, validation, and test sets."""
        random.shuffle(data)
        
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def save_processed_data(self, train_data: List[Dict], 
                           val_data: List[Dict], 
                           test_data: List[Dict]):
        """Save processed data to files."""
        # Save as JSONL files
        datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, data in datasets.items():
            output_file = self.processed_path / split_name / f"{split_name}.jsonl"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            
            logger.info(f"Saved {len(data)} examples to {output_file}")
    
    def process_all_data(self):
        """Process all available datasets."""
        all_data = []
        
        # Process each data source
        all_data.extend(self.process_autobench_data())
        all_data.extend(self.process_hdlbits_data())
        all_data.extend(self.process_mg_verilog_data())
        
        if not all_data:
            logger.error("No data found to process!")
            return
        
        logger.info(f"Total data pairs collected: {len(all_data)}")
        
        # Split data
        train_data, val_data, test_data = self.split_data(
            all_data,
            self.config['data']['train_split'],
            self.config['data']['val_split']
        )
        
        # Save processed data
        self.save_processed_data(train_data, val_data, test_data)
        
        # Save statistics
        stats = {
            'total_examples': len(all_data),
            'train_examples': len(train_data),
            'val_examples': len(val_data),
            'test_examples': len(test_data)
        }
        
        stats_file = self.processed_path / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset statistics saved to {stats_file}")


def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data