"""Verilog processing utilities."""

import subprocess
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class VerilogProcessor:
    """Handle Verilog compilation and simulation."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.temp_dir = tempfile.mkdtemp()
        
    def __del__(self):
        """Clean up temporary directory."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def extract_module_name(self, verilog_code: str) -> Optional[str]:
        """Extract module name from Verilog code."""
        pattern = r'module\s+(\w+)\s*[\(#]'
        match = re.search(pattern, verilog_code)
        return match.group(1) if match else None
    
    def compile_verilog(self, dut_code: str, testbench_code: str) -> Tuple[bool, str]:
        """
        Compile Verilog DUT and testbench.
        
        Returns:
            (success, error_message)
        """
        try:
            # Save files
            dut_file = os.path.join(self.temp_dir, "dut.v")
            tb_file = os.path.join(self.temp_dir, "testbench.v")
            
            with open(dut_file, 'w') as f:
                f.write(dut_code)
            
            with open(tb_file, 'w') as f:
                f.write(testbench_code)
            
            # Compile with iverilog
            cmd = ['iverilog', '-o', os.path.join(self.temp_dir, 'sim'), tb_file, dut_file]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Compilation timeout"
        except Exception as e:
            return False, str(e)
    
    def run_simulation(self) -> Tuple[bool, str, str]:
        """
        Run the compiled simulation.
        
        Returns:
            (success, output, error_message)
        """
        try:
            sim_file = os.path.join(self.temp_dir, 'sim')
            if not os.path.exists(sim_file):
                return False, "", "Simulation file not found"
            
            cmd = ['vvp', sim_file]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                return True, result.stdout, ""
            else:
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "", "Simulation timeout"
        except Exception as e:
            return False, "", str(e)
    
    def validate_testbench_structure(self, testbench_code: str) -> Dict[str, bool]:
        """Validate testbench has required components."""
        validations = {
            'has_timescale': bool(re.search(r'`timescale', testbench_code)),
            'has_module': bool(re.search(r'module\s+\w+\s*[;(]', testbench_code)),
            'has_endmodule': bool(re.search(r'endmodule', testbench_code)),
            'has_initial': bool(re.search(r'initial\s+begin', testbench_code)),
            'has_finish': bool(re.search(r'\$finish', testbench_code)),
            'has_display': bool(re.search(r'\$display', testbench_code)),
            'has_clock': bool(re.search(r'always\s+#\d+\s+\w+\s*=\s*~\w+', testbench_code)),
        }
        return validations
    
    def extract_signals(self, verilog_code: str) -> Dict[str, List[str]]:
        """Extract input/output signals from module."""
        signals = {'inputs': [], 'outputs': [], 'inouts': []}
        
        # Find module declaration
        module_match = re.search(
            r'module\s+\w+\s*\((.*?)\);', 
            verilog_code, 
            re.DOTALL
        )
        
        if module_match:
            port_section = module_match.group(1)
            
            # Extract inputs
            inputs = re.findall(r'input\s+(?:\[\d+:\d+\]\s+)?(\w+)', port_section)
            signals['inputs'].extend(inputs)
            
            # Extract outputs
            outputs = re.findall(r'output\s+(?:reg\s+)?(?:\[\d+:\d+\]\s+)?(\w+)', port_section)
            signals['outputs'].extend(outputs)
            
            # Extract inouts
            inouts = re.findall(r'inout\s+(?:\[\d+:\d+\]\s+)?(\w+)', port_section)
            signals['inouts'].extend(inouts)
        
        return signals


def clean_verilog_code(code: str) -> str:
    """Clean and format Verilog code."""
    # Remove markdown code blocks if present
    code = re.sub(r'```verilog\s*\n?', '', code)
    code = re.sub(r'```\s*\n?', '', code)
    
    # Remove leading/trailing whitespace
    code = code.strip()
    
    # Ensure proper spacing
    code = re.sub(r'\s+', ' ', code)
    code = re.sub(r'\s*;\s*', ';\n', code)
    code = re.sub(r'\s*begin\s*', ' begin\n', code)
    code = re.sub(r'\s*end\s*', '\nend', code)
    
    return code


def compare_testbenches(generated: str, reference: str) -> float:
    """Compare generated testbench with reference."""
    # Simple similarity based on common elements
    gen_elements = set(re.findall(r'\b\w+\b', generated.lower()))
    ref_elements = set(re.findall(r'\b\w+\b', reference.lower()))
    
    if not ref_elements:
        return 0.0
    
    common = gen_elements.intersection(ref_elements)
    return len(common) / len(ref_elements)