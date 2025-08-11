#!/usr/bin/env python3
"""Prepare project files for Google Colab."""

import os
import shutil
import zipfile
from pathlib import Path

def create_colab_package():
    """Create a zip file with all necessary files for Colab."""
    
    # Files and directories to include
    include_items = [
        'configs',
        'utils',
        'scripts',
        'notebooks',  # Include notebooks
        'data/processed',  # Only processed data
        'requirements.txt',
        '.env'  # If you have API keys
    ]
    
    # Create a temporary directory
    temp_dir = Path('colab_package')
    temp_dir.mkdir(exist_ok=True)
    
    # Copy files
    for item in include_items:
        src = Path(item)
        if src.exists():
            dst = temp_dir / item
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    
    # Create a Colab-specific requirements file
    colab_requirements = """torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
accelerate>=0.24.0
peft>=0.7.0
bitsandbytes>=0.41.0
wandb>=0.15.0
pyyaml>=6.0
tqdm>=4.65.0
jsonlines>=3.1.0
"""
    
    with open(temp_dir / 'requirements_colab.txt', 'w') as f:
        f.write(colab_requirements)
    
    # Create zip file
    zip_name = 'llm_testbench_colab.zip'
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    print(f"Created {zip_name}")
    print(f"Size: {os.path.getsize(zip_name) / 1024 / 1024:.2f} MB")
    print("\nUpload this file to Google Drive or Colab directly")

if __name__ == "__main__":
    create_colab_package()