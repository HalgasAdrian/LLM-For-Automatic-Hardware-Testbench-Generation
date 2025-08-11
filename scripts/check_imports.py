#!/usr/bin/env python3
"""Check if required packages are installed."""

import sys
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def main():
    """Check all required packages."""
    print("Checking required packages...\n")
    
    required_packages = [
        ("requests", "requests"),
        ("tqdm", "tqdm"),
        ("huggingface-hub", "huggingface_hub"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("pyyaml", "yaml"),
        ("pyverilog", "pyverilog"),
    ]
    
    missing_packages = []
    
    for package, import_name in required_packages:
        if not check_package(package, import_name):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing_packages)}")
        return 1
    else:
        print("\nAll required packages are installed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())