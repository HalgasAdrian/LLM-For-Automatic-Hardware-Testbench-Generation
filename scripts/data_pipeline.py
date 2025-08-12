#!/usr/bin/env python3
"""Streamlined data pipeline that orchestrates existing utilities.

This script coordinates the data pipeline using the existing
DataProcessor and VerilogProcessor utilities, eliminating
redundant code from individual scripts.

Usage:
    python data_pipeline.py --help           # Show all options
    python data_pipeline.py --all           # Run complete pipeline
    python data_pipeline.py --download      # Download datasets
    python data_pipeline.py --process       # Process raw data
    python data_pipeline.py --augment       # Add synthetic examples
    python data_pipeline.py --validate      # Validate and fix data
    python data_pipeline.py --analyze       # Analyze datasets
"""

import os
import sys
import json
import yaml
import shutil
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing utilities
from utils.data_utils import DataProcessor, load_dataset
from utils.verilog_utils import VerilogProcessor, clean_verilog_code

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Orchestrates the data pipeline using existing utilities."""
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        """Initialize pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_processor = DataProcessor(self.config)
        self.verilog_processor = VerilogProcessor()
        
    def download_datasets(self):
        """Download all required datasets."""
        logger.info("Downloading datasets...")
        
        # AutoBench
        autobench_dir = Path("data/raw/autobench")
        autobench_dir.mkdir(parents=True, exist_ok=True)
        
        if not (autobench_dir / ".git").exists():
            logger.info("Cloning AutoBench repository...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/AutoBench/AutoBench.git", 
                str(autobench_dir)
            ], check=True)
        else:
            logger.info("Updating AutoBench repository...")
            subprocess.run(["git", "-C", str(autobench_dir), "pull"], check=True)
        
        # MG-Verilog
        mg_dir = Path("data/raw/mg_verilog")
        mg_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from huggingface_hub import snapshot_download
            logger.info("Downloading MG-Verilog from HuggingFace...")
            snapshot_download(
                repo_id="GaTech-EIC/MG-Verilog",
                repo_type="dataset",
                local_dir=str(mg_dir),
                ignore_patterns=["*.md", ".gitattributes"]
            )
        except Exception as e:
            logger.error(f"Could not download MG-Verilog: {e}")
        
        # HDLBits placeholder
        hdl_dir = Path("data/raw/hdlbits")
        hdl_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Dataset download complete!")
    
    def analyze_datasets(self):
        """Analyze the structure and content of datasets."""
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        
        for dataset_name in ['autobench', 'mg_verilog', 'hdlbits']:
            dataset_path = Path("data/raw") / dataset_name
            
            if not dataset_path.exists():
                print(f"\n{dataset_name}: Not found")
                continue
            
            print(f"\n=== {dataset_name.upper()} ===")
            
            # Count files by type
            v_files = list(dataset_path.rglob("*.v"))
            sv_files = list(dataset_path.rglob("*.sv"))
            json_files = list(dataset_path.rglob("*.json"))
            
            print(f"Verilog files (.v): {len(v_files)}")
            print(f"SystemVerilog files (.sv): {len(sv_files)}")
            print(f"JSON files: {len(json_files)}")
            
            # Identify testbenches
            tb_files = [f for f in v_files if '_tb' in f.name or 'testbench' in f.name.lower()]
            print(f"Potential testbenches: {len(tb_files)}")
            
            # Check for pairs
            pairs = 0
            for tb_file in tb_files:
                base_name = tb_file.stem.replace('_tb', '').replace('testbench', '')
                for potential_dut in tb_file.parent.glob(f"{base_name}*.v"):
                    if potential_dut != tb_file:
                        pairs += 1
                        break
            
            print(f"Potential DUT-TB pairs: {pairs}")
    
    def process_datasets(self):
        """Process all datasets using DataProcessor."""
        logger.info("Processing datasets...")
        
        # Use the existing DataProcessor method
        self.data_processor.process_all_data()
        
        # Load and display statistics
        stats_file = Path(self.config['data']['processed_data_path']) / 'dataset_stats.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print("\nProcessing Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    def augment_data(self):
        """Add synthetic training examples - FULL augmentation from original script."""
        logger.info("Augmenting training data with comprehensive synthetic examples...")
        
        # Load existing training data
        train_path = Path(self.config['data']['processed_data_path']) / "train" / "train.jsonl"
        existing_data = load_dataset(train_path) if train_path.exists() else []
        
        logger.info(f"Existing training examples: {len(existing_data)}")
        
        # Create ALL synthetic examples
        basic_examples = self._create_basic_modules()
        parameterized_examples = self._create_parameterized_variations(basic_examples)
        
        # Convert to instruction format
        all_synthetic = []
        for dut, tb in basic_examples + parameterized_examples:
            pair = self.data_processor.create_instruction_pair(dut, tb)
            all_synthetic.append(pair)
        
        logger.info(f"Created {len(all_synthetic)} synthetic examples")
        logger.info(f"  - Basic modules: {len(basic_examples)}")
        logger.info(f"  - Parameterized variations: {len(parameterized_examples)}")
        
        # Combine data
        all_data = existing_data + all_synthetic
        
        # Save augmented data
        augmented_path = train_path.parent / "train_augmented.jsonl"
        with open(augmented_path, 'w') as f:
            for item in all_data:
                f.write(json.dumps(item) + '\n')
        
        # Also save just the new examples for review
        new_examples_file = Path(self.config['data']['processed_data_path']) / "new_examples.jsonl"
        with open(new_examples_file, 'w') as f:
            for item in all_synthetic:
                f.write(json.dumps(item) + '\n')
        
        print(f"\nAugmentation complete!")
        print(f"  Original: {len(existing_data)} examples")
        print(f"  Added: {len(all_synthetic)} synthetic examples")
        print(f"  Total: {len(all_data)} examples")
        print(f"\nFiles created:")
        print(f"  - {augmented_path} (full dataset)")
        print(f"  - {new_examples_file} (new examples only)")
        print("\nTo use augmented data: rename train_augmented.jsonl to train.jsonl")
    
    def _create_basic_modules(self) -> List[Tuple[str, str]]:
        """Create comprehensive set of basic module templates - FROM ORIGINAL augment_data.py"""
        examples = []
        
        # 1. Simple Multiplexer
        mux_dut = """module mux4to1(
    input [3:0] data_in,
    input [1:0] sel,
    output reg data_out
);
    always @(*) begin
        case (sel)
            2'b00: data_out = data_in[0];
            2'b01: data_out = data_in[1];
            2'b10: data_out = data_in[2];
            2'b11: data_out = data_in[3];
        endcase
    end
endmodule"""
        
        mux_tb = """`timescale 1ns / 1ps

module mux4to1_tb;
    reg [3:0] data_in;
    reg [1:0] sel;
    wire data_out;
    
    mux4to1 uut (
        .data_in(data_in),
        .sel(sel),
        .data_out(data_out)
    );
    
    initial begin
        $display("Testing 4-to-1 Multiplexer");
        
        data_in = 4'b1010;
        
        sel = 2'b00; #10;
        $display("sel=%b, data_out=%b (expected: %b)", sel, data_out, data_in[0]);
        
        sel = 2'b01; #10;
        $display("sel=%b, data_out=%b (expected: %b)", sel, data_out, data_in[1]);
        
        sel = 2'b10; #10;
        $display("sel=%b, data_out=%b (expected: %b)", sel, data_out, data_in[2]);
        
        sel = 2'b11; #10;
        $display("sel=%b, data_out=%b (expected: %b)", sel, data_out, data_in[3]);
        
        $finish;
    end
endmodule"""
        
        examples.append((mux_dut, mux_tb))
        
        # 2. D Flip-Flop with Reset
        dff_dut = """module d_flip_flop(
    input clk,
    input rst,
    input d,
    output reg q
);
    always @(posedge clk or posedge rst) begin
        if (rst)
            q <= 1'b0;
        else
            q <= d;
    end
endmodule"""
        
        dff_tb = """`timescale 1ns / 1ps

module d_flip_flop_tb;
    reg clk, rst, d;
    wire q;
    
    d_flip_flop uut (
        .clk(clk),
        .rst(rst),
        .d(d),
        .q(q)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $display("Testing D Flip-Flop");
        
        rst = 1; d = 0;
        #10 rst = 0;
        
        d = 1; #10;
        $display("Time=%0t, d=%b, q=%b", $time, d, q);
        
        d = 0; #10;
        $display("Time=%0t, d=%b, q=%b", $time, d, q);
        
        d = 1; #10;
        $display("Time=%0t, d=%b, q=%b", $time, d, q);
        
        rst = 1; #10;
        $display("After reset: q=%b (should be 0)", q);
        
        #10 $finish;
    end
endmodule"""
        
        examples.append((dff_dut, dff_tb))
        
        # 3. 3-bit Ripple Counter
        counter_dut = """module ripple_counter(
    input clk,
    input reset,
    output [2:0] count
);
    wire q0, q1, q2;
    
    assign count = {q2, q1, q0};
    
    t_flip_flop tff0(.clk(clk), .reset(reset), .t(1'b1), .q(q0));
    t_flip_flop tff1(.clk(~q0), .reset(reset), .t(1'b1), .q(q1));
    t_flip_flop tff2(.clk(~q1), .reset(reset), .t(1'b1), .q(q2));
endmodule

module t_flip_flop(
    input clk,
    input reset,
    input t,
    output reg q
);
    always @(posedge clk or posedge reset) begin
        if (reset)
            q <= 1'b0;
        else if (t)
            q <= ~q;
    end
endmodule"""
        
        counter_tb = """`timescale 1ns / 1ps

module ripple_counter_tb;
    reg clk, reset;
    wire [2:0] count;
    
    ripple_counter uut (
        .clk(clk),
        .reset(reset),
        .count(count)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $display("Testing 3-bit Ripple Counter");
        $monitor("Time=%0t, count=%b", $time, count);
        
        reset = 1;
        #15 reset = 0;
        
        // Let it count for 10 clock cycles
        #100;
        
        $display("Final count: %d", count);
        $finish;
    end
endmodule"""
        
        examples.append((counter_dut, counter_tb))
        
        # 4. Simple ALU
        alu_dut = """module simple_alu(
    input [3:0] a,
    input [3:0] b,
    input [1:0] op,
    output reg [3:0] result,
    output zero
);
    assign zero = (result == 4'b0000);
    
    always @(*) begin
        case (op)
            2'b00: result = a + b;    // ADD
            2'b01: result = a - b;    // SUB
            2'b10: result = a & b;    // AND
            2'b11: result = a | b;    // OR
        endcase
    end
endmodule"""
        
        alu_tb = """`timescale 1ns / 1ps

module simple_alu_tb;
    reg [3:0] a, b;
    reg [1:0] op;
    wire [3:0] result;
    wire zero;
    
    simple_alu uut (
        .a(a),
        .b(b),
        .op(op),
        .result(result),
        .zero(zero)
    );
    
    initial begin
        $display("Testing Simple ALU");
        
        // Test ADD
        a = 4'b0011; b = 4'b0101; op = 2'b00; #10;
        $display("ADD: %d + %d = %d, zero=%b", a, b, result, zero);
        
        // Test SUB
        a = 4'b1000; b = 4'b0011; op = 2'b01; #10;
        $display("SUB: %d - %d = %d, zero=%b", a, b, result, zero);
        
        // Test AND
        a = 4'b1100; b = 4'b1010; op = 2'b10; #10;
        $display("AND: %b & %b = %b, zero=%b", a, b, result, zero);
        
        // Test OR
        a = 4'b1100; b = 4'b0011; op = 2'b11; #10;
        $display("OR: %b | %b = %b, zero=%b", a, b, result, zero);
        
        // Test zero flag
        a = 4'b0101; b = 4'b0101; op = 2'b01; #10;
        $display("SUB (zero test): %d - %d = %d, zero=%b", a, b, result, zero);
        
        $finish;
    end
endmodule"""
        
        examples.append((alu_dut, alu_tb))
        
        # 5. Shift Register
        shift_dut = """module shift_register(
    input clk,
    input reset,
    input serial_in,
    input shift_enable,
    output reg [3:0] parallel_out
);
    always @(posedge clk or posedge reset) begin
        if (reset)
            parallel_out <= 4'b0000;
        else if (shift_enable)
            parallel_out <= {parallel_out[2:0], serial_in};
    end
endmodule"""
        
        shift_tb = """`timescale 1ns / 1ps

module shift_register_tb;
    reg clk, reset, serial_in, shift_enable;
    wire [3:0] parallel_out;
    
    shift_register uut (
        .clk(clk),
        .reset(reset),
        .serial_in(serial_in),
        .shift_enable(shift_enable),
        .parallel_out(parallel_out)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $display("Testing Shift Register");
        $monitor("Time=%0t, parallel_out=%b", $time, parallel_out);
        
        reset = 1; shift_enable = 0; serial_in = 0;
        #10 reset = 0;
        
        // Shift in pattern 1011
        shift_enable = 1;
        serial_in = 1; #10;
        serial_in = 1; #10;
        serial_in = 0; #10;
        serial_in = 1; #10;
        
        shift_enable = 0; #10;
        $display("Final output: %b (expected: 1011)", parallel_out);
        
        $finish;
    end
endmodule"""
        
        examples.append((shift_dut, shift_tb))
        
        # 6. Priority Encoder
        priority_dut = """module priority_encoder(
    input [3:0] in,
    output reg [1:0] out,
    output reg valid
);
    always @(*) begin
        valid = 1'b1;
        if (in[3])
            out = 2'b11;
        else if (in[2])
            out = 2'b10;
        else if (in[1])
            out = 2'b01;
        else if (in[0])
            out = 2'b00;
        else begin
            out = 2'b00;
            valid = 1'b0;
        end
    end
endmodule"""
        
        priority_tb = """`timescale 1ns / 1ps

module priority_encoder_tb;
    reg [3:0] in;
    wire [1:0] out;
    wire valid;
    
    priority_encoder uut (
        .in(in),
        .out(out),
        .valid(valid)
    );
    
    initial begin
        $display("Testing Priority Encoder");
        $monitor("Time=%0t, in=%b, out=%b, valid=%b", $time, in, out, valid);
        
        in = 4'b0000; #10;
        in = 4'b0001; #10;
        in = 4'b0010; #10;
        in = 4'b0100; #10;
        in = 4'b1000; #10;
        in = 4'b1010; #10;
        in = 4'b1111; #10;
        
        $finish;
    end
endmodule"""
        
        examples.append((priority_dut, priority_tb))
        
        # 7. Gray Code Counter
        gray_dut = """module gray_counter(
    input clk,
    input reset,
    output reg [3:0] gray_out
);
    reg [3:0] binary_count;
    
    always @(posedge clk or posedge reset) begin
        if (reset)
            binary_count <= 4'b0000;
        else
            binary_count <= binary_count + 1;
    end
    
    always @(*) begin
        gray_out = binary_count ^ (binary_count >> 1);
    end
endmodule"""
        
        gray_tb = """`timescale 1ns / 1ps

module gray_counter_tb;
    reg clk, reset;
    wire [3:0] gray_out;
    
    gray_counter uut (
        .clk(clk),
        .reset(reset),
        .gray_out(gray_out)
    );
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $display("Testing Gray Code Counter");
        $monitor("Time=%0t, gray_out=%b", $time, gray_out);
        
        reset = 1;
        #15 reset = 0;
        
        // Count for 16 cycles to see full sequence
        #160;
        
        $finish;
    end
endmodule"""
        
        examples.append((gray_dut, gray_tb))
        
        # 8. Decoder 2-to-4
        decoder_dut = """module decoder_2to4(
    input [1:0] in,
    input enable,
    output reg [3:0] out
);
    always @(*) begin
        if (enable) begin
            case (in)
                2'b00: out = 4'b0001;
                2'b01: out = 4'b0010;
                2'b10: out = 4'b0100;
                2'b11: out = 4'b1000;
            endcase
        end else
            out = 4'b0000;
    end
endmodule"""
        
        decoder_tb = """`timescale 1ns / 1ps

module decoder_2to4_tb;
    reg [1:0] in;
    reg enable;
    wire [3:0] out;
    
    decoder_2to4 uut (
        .in(in),
        .enable(enable),
        .out(out)
    );
    
    initial begin
        $display("Testing 2-to-4 Decoder");
        
        // Test with enable = 0
        enable = 0;
        in = 2'b00; #10;
        $display("enable=%b, in=%b, out=%b", enable, in, out);
        
        // Test with enable = 1
        enable = 1;
        in = 2'b00; #10;
        $display("enable=%b, in=%b, out=%b", enable, in, out);
        
        in = 2'b01; #10;
        $display("enable=%b, in=%b, out=%b", enable, in, out);
        
        in = 2'b10; #10;
        $display("enable=%b, in=%b, out=%b", enable, in, out);
        
        in = 2'b11; #10;
        $display("enable=%b, in=%b, out=%b", enable, in, out);
        
        $finish;
    end
endmodule"""
        
        examples.append((decoder_dut, decoder_tb))
        
        # 9. Simple state machine (2-bit sequence detector)
        fsm_dut = """module sequence_detector(
    input clk,
    input reset,
    input in,
    output reg detected
);
    reg [1:0] state;
    parameter S0 = 2'b00, S1 = 2'b01, S2 = 2'b10;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= S0;
            detected <= 1'b0;
        end else begin
            case (state)
                S0: begin
                    if (in) state <= S1;
                    detected <= 1'b0;
                end
                S1: begin
                    if (in) state <= S2;
                    else state <= S0;
                    detected <= 1'b0;
                end
                S2: begin
                    if (in) begin
                        state <= S2;
                        detected <= 1'b1;
                    end else begin
                        state <= S0;
                        detected <= 1'b0;
                    end
                end
                default: state <= S0;
            endcase
        end
    end
endmodule"""
        
        fsm_tb = """`timescale 1ns / 1ps

module sequence_detector_tb;
    reg clk, reset, in;
    wire detected;
    
    sequence_detector uut (
        .clk(clk),
        .reset(reset),
        .in(in),
        .detected(detected)
    );
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $display("Testing Sequence Detector (111)");
        
        reset = 1; in = 0;
        #10 reset = 0;
        
        // Test sequence: 0110111
        in = 0; #10;
        $display("Time=%0t, in=%b, detected=%b", $time, in, detected);
        
        in = 1; #10;
        $display("Time=%0t, in=%b, detected=%b", $time, in, detected);
        
        in = 1; #10;
        $display("Time=%0t, in=%b, detected=%b", $time, in, detected);
        
        in = 0; #10;
        $display("Time=%0t, in=%b, detected=%b", $time, in, detected);
        
        in = 1; #10;
        $display("Time=%0t, in=%b, detected=%b", $time, in, detected);
        
        in = 1; #10;
        $display("Time=%0t, in=%b, detected=%b", $time, in, detected);
        
        in = 1; #10;
        $display("Time=%0t, in=%b, detected=%b (should be 1)", $time, in, detected);
        
        #10 $finish;
    end
endmodule"""
        
        examples.append((fsm_dut, fsm_tb))
        
        # 10. Comparator
        comp_dut = """module comparator_4bit(
    input [3:0] a,
    input [3:0] b,
    output a_greater,
    output a_equal,
    output a_less
);
    assign a_greater = (a > b);
    assign a_equal = (a == b);
    assign a_less = (a < b);
endmodule"""
        
        comp_tb = """`timescale 1ns / 1ps

module comparator_4bit_tb;
    reg [3:0] a, b;
    wire a_greater, a_equal, a_less;
    
    comparator_4bit uut (
        .a(a),
        .b(b),
        .a_greater(a_greater),
        .a_equal(a_equal),
        .a_less(a_less)
    );
    
    initial begin
        $display("Testing 4-bit Comparator");
        
        a = 4'd5; b = 4'd3; #10;
        $display("a=%d, b=%d: greater=%b, equal=%b, less=%b", a, b, a_greater, a_equal, a_less);
        
        a = 4'd7; b = 4'd7; #10;
        $display("a=%d, b=%d: greater=%b, equal=%b, less=%b", a, b, a_greater, a_equal, a_less);
        
        a = 4'd2; b = 4'd9; #10;
        $display("a=%d, b=%d: greater=%b, equal=%b, less=%b", a, b, a_greater, a_equal, a_less);
        
        a = 4'd15; b = 4'd0; #10;
        $display("a=%d, b=%d: greater=%b, equal=%b, less=%b", a, b, a_greater, a_equal, a_less);
        
        $finish;
    end
endmodule"""
        
        examples.append((comp_dut, comp_tb))
        
        return examples
    
    def _create_parameterized_variations(self, base_examples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Create parameterized variations of examples - FROM ORIGINAL augment_data.py"""
        variations = []
        
        # Counter with different widths
        for width in [4, 5, 6, 8]:
            counter_dut = f"""module counter_{width}bit(
    input clk,
    input reset,
    input enable,
    output reg [{width-1}:0] count
);
    always @(posedge clk or posedge reset) begin
        if (reset)
            count <= {width}'b0;
        else if (enable)
            count <= count + 1;
    end
endmodule"""
            
            counter_tb = f"""`timescale 1ns / 1ps

module counter_{width}bit_tb;
    reg clk, reset, enable;
    wire [{width-1}:0] count;
    
    counter_{width}bit uut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .count(count)
    );
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $display("Testing {width}-bit Counter");
        reset = 1; enable = 0;
        #10 reset = 0; enable = 1;
        
        repeat(10) begin
            #10 $display("Count = %d", count);
        end
        
        $finish;
    end
endmodule"""
            
            variations.append((counter_dut, counter_tb))
        
        # Shift register with different widths
        for width in [8, 16]:
            shift_dut = f"""module shift_register_{width}bit(
    input clk,
    input reset,
    input serial_in,
    input shift_enable,
    output reg [{width-1}:0] parallel_out
);
    always @(posedge clk or posedge reset) begin
        if (reset)
            parallel_out <= {width}'b0;
        else if (shift_enable)
            parallel_out <= {{parallel_out[{width-2}:0], serial_in}};
    end
endmodule"""
            
            shift_tb = f"""`timescale 1ns / 1ps

module shift_register_{width}bit_tb;
    reg clk, reset, serial_in, shift_enable;
    wire [{width-1}:0] parallel_out;
    
    shift_register_{width}bit uut (
        .clk(clk),
        .reset(reset),
        .serial_in(serial_in),
        .shift_enable(shift_enable),
        .parallel_out(parallel_out)
    );
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $display("Testing {width}-bit Shift Register");
        $monitor("Time=%0t, parallel_out=%h", $time, parallel_out);
        
        reset = 1; shift_enable = 0; serial_in = 0;
        #10 reset = 0;
        
        shift_enable = 1;
        repeat({width}) begin
            serial_in = $random;
            #10;
        end
        
        shift_enable = 0;
        #10 $finish;
    end
endmodule"""
            
            variations.append((shift_dut, shift_tb))
        
        return variations
    
    def validate_and_clean(self):
        """Validate and clean processed data."""
        logger.info("Validating and cleaning data...")
        
        processed_path = Path(self.config['data']['processed_data_path'])
        
        for split in ['train', 'val', 'test']:
            split_file = processed_path / split / f"{split}.jsonl"
            
            if not split_file.exists():
                logger.warning(f"Skipping {split} - file not found")
                continue
            
            print(f"\n=== {split.upper()} ===")
            
            # Analyze data
            data = load_dataset(split_file)
            total = len(data)
            
            # Check for issues
            has_markdown = 0
            has_timescale = 0
            has_module = 0
            has_endmodule = 0
            needs_cleaning = False
            
            cleaned_data = []
            for item in data:
                response = item['response']
                
                # Check for markdown
                if '```' in response:
                    has_markdown += 1
                    needs_cleaning = True
                    # Clean it
                    response = self._clean_markdown(response)
                    item['response'] = response
                    item['testbench_code'] = response
                
                # Check structure
                if '`timescale' in response:
                    has_timescale += 1
                if 'module' in response:
                    has_module += 1
                if 'endmodule' in response:
                    has_endmodule += 1
                
                cleaned_data.append(item)
            
            # Display statistics
            print(f"Total examples: {total}")
            print(f"Has markdown: {has_markdown} ({has_markdown/total*100:.1f}%)")
            print(f"Has timescale: {has_timescale} ({has_timescale/total*100:.1f}%)")
            print(f"Has module: {has_module} ({has_module/total*100:.1f}%)")
            print(f"Has endmodule: {has_endmodule} ({has_endmodule/total*100:.1f}%)")
            
            # Save cleaned data if needed
            if needs_cleaning:
                # Backup original
                backup_file = split_file.parent / f"{split}_backup.jsonl"
                shutil.copy(split_file, backup_file)
                logger.info(f"Backed up to: {backup_file}")
                
                # Save cleaned
                with open(split_file, 'w') as f:
                    for item in cleaned_data:
                        f.write(json.dumps(item) + '\n')
                
                logger.info(f"Cleaned {has_markdown} items with markdown")
    
    def _clean_markdown(self, text: str) -> str:
        """Remove markdown code blocks from text."""
        if "```verilog" in text:
            text = text.split("```verilog")[1]
            if "```" in text:
                text = text.split("```")[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
        return text.strip()
    
    def run_pipeline(self, steps: List[str]):
        """Run specified pipeline steps."""
        step_methods = {
            'download': self.download_datasets,
            'analyze': self.analyze_datasets,
            'process': self.process_datasets,
            'augment': self.augment_data,
            'validate': self.validate_and_clean,
        }
        
        for step in steps:
            if step in step_methods:
                print(f"\n{'='*60}")
                print(f"RUNNING: {step.upper()}")
                print('='*60)
                step_methods[step]()
            else:
                logger.warning(f"Unknown step: {step}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Streamlined data pipeline for Verilog testbench generation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--all', action='store_true', 
                       help='Run complete pipeline')
    parser.add_argument('--download', action='store_true',
                       help='Download datasets')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze dataset structure')
    parser.add_argument('--process', action='store_true',
                       help='Process raw data into training format')
    parser.add_argument('--augment', action='store_true',
                       help='Add synthetic training examples (10 basic + variations)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate and clean processed data')
    
    args = parser.parse_args()
    
    # Determine which steps to run
    if args.all:
        steps = ['download', 'analyze', 'process', 'augment', 'validate']
    else:
        steps = []
        if args.download: steps.append('download')
        if args.analyze: steps.append('analyze')
        if args.process: steps.append('process')
        if args.augment: steps.append('augment')
        if args.validate: steps.append('validate')
    
    if not steps:
        parser.print_help()
        print("\nExample usage:")
        print("  python data_pipeline.py --all        # Run everything")
        print("  python data_pipeline.py --process    # Just process data")
        print("  python data_pipeline.py --augment    # Add synthetic examples")
        print("  python data_pipeline.py --download --process --augment  # Full data prep")
        return
    
    # Initialize and run pipeline
    pipeline = DataPipeline()
    pipeline.run_pipeline(steps)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run 'python scripts/train.py' to train the model")
    print("2. Run 'python scripts/evaluate.py' to evaluate performance")


if __name__ == "__main__":
    main()