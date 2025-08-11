#!/usr/bin/env python3
"""Download and prepare datasets for training."""

import os
import sys
import subprocess
from pathlib import Path
import requests
import zipfile
import tarfile
from tqdm import tqdm
import logging
from huggingface_hub import snapshot_download

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, dest_path: str, desc: str = "Downloading"):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_autobench():
    """Download AutoBench dataset."""
    logger.info("Downloading AutoBench dataset...")
    
    data_dir = Path("data/raw/autobench")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone AutoBench repository
    if not (data_dir / ".git").exists():
        cmd = ["git", "clone", "https://github.com/AutoBench/AutoBench.git", str(data_dir)]
        subprocess.run(cmd, check=True)
        logger.info("AutoBench downloaded successfully")
    else:
        logger.info("AutoBench already exists, pulling latest changes...")
        cmd = ["git", "-C", str(data_dir), "pull"]
        subprocess.run(cmd, check=True)


def download_mg_verilog():
    """Download MG-Verilog dataset from HuggingFace."""
    logger.info("Downloading MG-Verilog dataset...")
    
    data_dir = Path("data/raw/mg_verilog")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download from HuggingFace
        snapshot_download(
            repo_id="GaTech-EIC/MG-Verilog",
            repo_type="dataset",
            local_dir=str(data_dir),
            ignore_patterns=["*.md", ".gitattributes"]
        )
        logger.info("MG-Verilog downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading MG-Verilog: {e}")
        logger.info("Please download manually from https://huggingface.co/datasets/GaTech-EIC/MG-Verilog")


def download_hdlbits_examples():
    """Download example Verilog files (HDLBits solutions if available)."""
    logger.info("Setting up HDLBits examples directory...")
    
    data_dir = Path("data/raw/hdlbits")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a README for manual data collection
    readme_content = """# HDLBits Data

HDLBits problems need to be collected manually from https://hdlbits.01xz.net/

## How to collect:
1. Visit HDLBits website
2. Copy problem descriptions and solutions
3. Create pairs of DUT and testbench files
4. Save as: problem_name_dut.v and problem_name_tb.v

## Example structure:
- counter_dut.v (the counter design)
- counter_tb.v (the testbench for counter)
"""
    
    with open(data_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    logger.info("HDLBits directory created. Please add data manually.")


def verify_downloads():
    """Verify that datasets are downloaded correctly."""
    logger.info("\nVerifying downloads...")
    
    checks = {
        "AutoBench": Path("data/raw/autobench"),
        "MG-Verilog": Path("data/raw/mg_verilog"),
        "HDLBits": Path("data/raw/hdlbits")
    }
    
    for name, path in checks.items():
        if path.exists():
            # Count Verilog files
            verilog_files = list(path.rglob("*.v")) + list(path.rglob("*.sv"))
            logger.info(f"✓ {name}: Found {len(verilog_files)} Verilog files")
        else:
            logger.warning(f"✗ {name}: Directory not found")


def main():
    """Main download function."""
    logger.info("Starting dataset download process...")
    
    # Download each dataset
    try:
        download_autobench()
    except Exception as e:
        logger.error(f"Failed to download AutoBench: {e}")
    
    try:
        download_mg_verilog()
    except Exception as e:
        logger.error(f"Failed to download MG-Verilog: {e}")
    
    download_hdlbits_examples()
    
    # Verify downloads
    verify_downloads()
    
    logger.info("\nDownload process completed!")
    logger.info("Next step: Run 'python scripts/process_data.py' to process the data")


if __name__ == "__main__":
    main()