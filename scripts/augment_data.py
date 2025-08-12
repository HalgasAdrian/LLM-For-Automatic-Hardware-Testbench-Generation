#!/usr/bin/env python3
"""Augment training data with synthetic examples and variations."""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import DataProcessor

def create_basic_modules() -> List[Tuple[str, str]]:
    """Create basic module templates with proper testbenches."""
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


def create_parameterized_variations(base_examples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Create variations of examples with different parameters."""
    variations = []
    
    # Counter with different widths
    for width in [4, 5, 6]:
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
    
    return variations


def main():
    """Generate additional training data."""
    import yaml
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_processor = DataProcessor(config)
    
    # Get basic examples
    basic_examples = create_basic_modules()
    print(f"Created {len(basic_examples)} basic examples")
    
    # Get variations
    variations = create_parameterized_variations(basic_examples)
    print(f"Created {len(variations)} parameterized variations")
    
    # Combine all
    all_new_examples = basic_examples + variations
    
    # Convert to instruction format
    new_data = []
    for dut, tb in all_new_examples:
        pair = data_processor.create_instruction_pair(dut, tb)
        new_data.append(pair)
    
    # Load existing data
    existing_train = []
    train_file = Path(config['data']['processed_data_path']) / "train" / "train.jsonl"
    if train_file.exists():
        with open(train_file, 'r') as f:
            for line in f:
                existing_train.append(json.loads(line))
    
    print(f"Existing training examples: {len(existing_train)}")
    
    # Combine and save
    all_train = existing_train + new_data
    print(f"Total training examples after augmentation: {len(all_train)}")
    
    # Save augmented data
    augmented_file = Path(config['data']['processed_data_path']) / "train" / "train_augmented.jsonl"
    with open(augmented_file, 'w') as f:
        for item in all_train:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved augmented training data to: {augmented_file}")
    print("\nTo use this data, rename it to train.jsonl or update your training script")
    
    # Also save just the new examples for review
    new_examples_file = Path(config['data']['processed_data_path']) / "new_examples.jsonl"
    with open(new_examples_file, 'w') as f:
        for item in new_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"New examples saved to: {new_examples_file}")


if __name__ == "__main__":
    main()