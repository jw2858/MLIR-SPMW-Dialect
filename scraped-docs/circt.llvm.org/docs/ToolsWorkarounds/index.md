EDA Tool Workarounds - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

EDA Tool Workarounds
====================

This documents various bugs found in EDA tools and their workarounds in circt.
Each but will have a brief description, example code, and the mitigation added
(with links to the commit when possible).

Automatic Variables Cause Latch Warnings
========================================

Verilator issues a latch warning for fully-initialized, automatic variables. This precludes using locally scoped variables.
<https://github.com/verilator/verilator/issues/4022>

Example [¶](#example)
---------------------

```
module ALU(
  input         clock,
  input  [4:0]  operation,
  input  [63:0] inputs_1,
                inputs_0,
                inputs_2,
  input  [16:0] immediate,
  output [63:0] output_0
);
  reg  [63:0]  casez_tmp_1;
  always_comb begin
    automatic logic [63:0] lowHigh;
    casez (operation)
      5'b00011:
        casez_tmp_1 = inputs_0 & inputs_1;
      5'b00100:
        casez_tmp_1 = inputs_0 | inputs_1;
      5'b00101:
        casez_tmp_1 = inputs_0 ^ inputs_1;
      5'b01001: begin
        automatic logic [16:0] _aluOutput_T_22 =
          immediate >> {14'h0, inputs_2, inputs_1[0], inputs_0[0]};
        casez_tmp_1 = {63'h0, _aluOutput_T_22[0]};
      end
      default:
        casez_tmp_1 = inputs_0;
    endcase
  end
endmodule
```

Gives:

```
$ verilator --version
Verilator 5.008 2023-03-04 rev v5.008
$ verilator --lint-only ALU.sv
%Warning-LATCH: ALU.sv:11:3: Latch inferred for signal 'ALU.unnamedblk1.unnamedblk2._aluOutput_T_22' (not all control paths of combinational always assign a value)
                           : ... Suggest use of always_latch for intentional latches
   11 |   always_comb begin
      |   ^~~~~~~~~~~
                ... For warning description see https://verilator.org/warn/LATCH?v=4.218
                ... Use "/* verilator lint_off LATCH */" and lint_on around source to disable this message.
%Error: Exiting due to 1 warning(s)
```

Workaround [¶](#workaround)
---------------------------

Flag added to promote all storage to the top level of a module.
<https://github.com/llvm/circt/commit/3c8b4b47b600ea6bcc6da56fe9b81d6fe4022e4c>

Inline Array calculations can cause synthesis failures
======================================================

Some tools have bugs (version dependent) in const prop in this case.

Example [¶](#example-1)
-----------------------

```
module Foo (input clock, input in, output [2:0] out);
  reg [2:0] state;
  wire [7:0][2:0] array = 24'h4 << 6;
  wire [2:0] a = array[state];
  wire [2:0] b = array[state + 3'h1 + 3'h1];
  // works:      array[state + (3'h1 + 3'h1)]
  // works:      array[state + 3'h2]
  always @(posedge clock) state <= in ? a : b;
  assign out = b;
endmodule
```

Workaround [¶](#workaround-1)
-----------------------------

Flag added to export verilog to force array index calculations to not be inline.

<https://github.com/llvm/circt/commit/15a1f95f2d59767f20b459a12ac42338de22bc97>

Memory semantics changed by synthesis
=====================================

Read/Write forwarding behavior is dependent on memory size, since the synthesis
tool changes it’s mapping based on that. The “optimized” mapping does not
preserve the behavior of the verilog. This is a known issue reported on various
forums by multiple people. There are some version dependencies on when this
manifests.

Example [¶](#example-2)
-----------------------

```
Qux:
  module Qux:
    input clock: Clock
    input addr: UInt<1>
    input r: {en: UInt<1>, flip data: {a: UInt<32>, b: UInt<32>}, addr: UInt<1>}
    input w: {en: UInt<1>, data: {a: UInt<32>, b: UInt<32>}, addr: UInt<1>, mask: {a: UInt<1>, b: UInt<1>}}

    mem m :
      data-type => {a: UInt<32>, b: UInt<32>}
      depth => 1
      reader => r
      writer => w
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    m.r.clk <= clock
    m.r.en <= r.en
    m.r.addr <= r.addr
    r.data <= m.r.data

    m.w.clk <= clock
    m.w.en <= w.en
    m.w.addr <= w.addr
    m.w.data <= w.data
    m.w.mask <= w.mask
```

Compile with either firtool -repl-seq-mem -repl-seq-mem-file=mem.conf Foo.fir and firrtl -i Foo.fir.

Workaround [¶](#workaround-2)
-----------------------------

FIRRTL memory lowering has a flag to generate attributes on memory
implementations that preserve the behavior described in the verilog. This is
not a general solution, this bug could impact anyone making memory-looking
things. It was decided not to try to reverse engineer the conditions which
cause the bug to manifest (since they are version dependent), thus there isn’t
a universal fix that can be applied in the generated verilog.

<https://github.com/llvm/circt/commit/e9f443be475e0ef796c0c6af1ce09d6e783fcd5a>

Clock Gates and Enables Not Recognized For Registers
====================================================

Clock gates in some rtl-based power estimation tools are unable to recognize
clock gates and enables if they are not generated as if statements in always
blocks. This is a very narrow pattern match with significant implications for
the tools lint results and quality of analysis results.

Example [¶](#example-3)
-----------------------

```
  %count = seq.firreg %2 clock %clock sym @count : i2
￼  %1 = comb.mux %cond, %value, %count : i2		￼  %1 = comb.mux bin %cond, %value, %count : i2
￼  %2 = comb.mux %reset, %c0_i2, %1 : i2		￼  %2 = comb.mux bin %reset, %c0_i2, %1 : i2￼
```

The mux on `cond` must become an `if` in the output since it forms a self-loop
on the register `count`.

Workaround [¶](#workaround-3)
-----------------------------

Effort at several points in lowering make effort to find self-loops through
register read and write ports and muxes. These are generated as `if` statements
in the always block that updates the register.

<https://github.com/llvm/circt/pull/3815>

 [Prev - Verif Dialect](https://circt.llvm.org/docs/Dialects/Verif/ "Verif Dialect")
[Next - Formal Verification Tooling](https://circt.llvm.org/docs/FormalVerification/ "Formal Verification Tooling") 

Powered by [Hugo](https://gohugo.io). Theme by [TechDoc](https://themes.gohugo.io/hugo-theme-techdoc/). Designed by [Thingsym](https://github.com/thingsym/hugo-theme-techdoc).

* [Home](https://circt.llvm.org/)
* [Talks and Related Publications](https://circt.llvm.org/talks/)
* [Getting Started](https://circt.llvm.org/getting_started/)
* [Code Documentation-](https://circt.llvm.org/docs/)
  + [Tools+](https://circt.llvm.org/docs/Tools/)
    - [circt-synth](https://circt.llvm.org/docs/Tools/circt-synth/)
    - [circt-verilog](https://circt.llvm.org/docs/Tools/circt-verilog/)
    - [handshake-runner](https://circt.llvm.org/docs/Tools/handshake-runner/)
  + [CIRCT Charter](https://circt.llvm.org/docs/Charter/)
  + [Dialects+](https://circt.llvm.org/docs/Dialects/)

    - ['arc' Dialect](https://circt.llvm.org/docs/Dialects/Arc/)
    - ['calyx' Dialect](https://circt.llvm.org/docs/Dialects/Calyx/)
    - ['chirrtl' Dialect](https://circt.llvm.org/docs/Dialects/CHIRRTL/)
    - ['comb' Dialect+](https://circt.llvm.org/docs/Dialects/Comb/)
      * [`comb` Dialect Rationale](https://circt.llvm.org/docs/Dialects/Comb/RationaleComb/)
    - ['datapath' Dialect+](https://circt.llvm.org/docs/Dialects/Datapath/)
      * ['datapath' Dialect Rationale](https://circt.llvm.org/docs/Dialects/Datapath/RationaleDatapath/)
    - ['dc' Dialect+](https://circt.llvm.org/docs/Dialects/DC/)
      * [DC Dialect Rationale](https://circt.llvm.org/docs/Dialects/DC/RationaleDC/)
    - ['emit' Dialect+](https://circt.llvm.org/docs/Dialects/Emit/)
      * [Emission (Emit) Dialect Rationale](https://circt.llvm.org/docs/Dialects/Emit/RationaleEmit/)
    - ['esi' Dialect+](https://circt.llvm.org/docs/Dialects/ESI/)
      * [ESI data types and communication types](https://circt.llvm.org/docs/Dialects/ESI/types/)
      * [ESI Global Services](https://circt.llvm.org/docs/Dialects/ESI/services/)
      * [ESI Software APIs](https://circt.llvm.org/docs/Dialects/ESI/software_api/)
      * [Miscellaneous Notes](https://circt.llvm.org/docs/Dialects/ESI/notes/)
      * [The Elastic Silicon Interconnect dialect](https://circt.llvm.org/docs/Dialects/ESI/RationaleESI/)
    - ['firrtl' Dialect+](https://circt.llvm.org/docs/Dialects/FIRRTL/)
      * [FIRRTL Annotations](https://circt.llvm.org/docs/Dialects/FIRRTL/FIRRTLAnnotations/)
      * [FIRRTL Dialect Rationale](https://circt.llvm.org/docs/Dialects/FIRRTL/RationaleFIRRTL/)
      * [Intrinsics](https://circt.llvm.org/docs/Dialects/FIRRTL/FIRRTLIntrinsics/)
    - ['fsm' Dialect+](https://circt.llvm.org/docs/Dialects/FSM/)
      * [FSM Dialect Rationale](https://circt.llvm.org/docs/Dialects/FSM/RationaleFSM/)
    - ['handshake' Dialect+](https://circt.llvm.org/docs/Dialects/Handshake/)
      * [Handshake Dialect Rationale](https://circt.llvm.org/docs/Dialects/Handshake/RationaleHandshake/)
    - ['hw' Dialect+](https://circt.llvm.org/docs/Dialects/HW/)
      * [HW Dialect Rationale](https://circt.llvm.org/docs/Dialects/HW/RationaleHW/)
    - ['hwarith' Dialect+](https://circt.llvm.org/docs/Dialects/HWArith/)
      * [HW Arith Dialect Rationale](https://circt.llvm.org/docs/Dialects/HWArith/RationaleHWArith/)
    - ['kanagawa' Dialect+](https://circt.llvm.org/docs/Dialects/Kanagawa/)
      * [`kanagawa` Dialect Rationale](https://circt.llvm.org/docs/Dialects/Kanagawa/RationaleKanagawa/)
    - ['loopschedule' Dialect+](https://circt.llvm.org/docs/Dialects/LoopSchedule/)
      * [LoopSchedule Dialect Rationale](https://circt.llvm.org/docs/Dialects/LoopSchedule/LoopSchedule/)
    - ['msft' Dialect](https://circt.llvm.org/docs/Dialects/MSFT/)
    - ['om' Dialect+](https://circt.llvm.org/docs/Dialects/OM/)
      * [Object Model Dialect Rationale](https://circt.llvm.org/docs/Dialects/OM/RationaleOM/)
    - ['pipeline' Dialect+](https://circt.llvm.org/docs/Dialects/Pipeline/)
      * [Pipeline Dialect Rationale](https://circt.llvm.org/docs/Dialects/Pipeline/RationalePipeline/)
    - ['rtgtest' Dialect](https://circt.llvm.org/docs/Dialects/RTGTest/)
    - ['seq' Dialect+](https://circt.llvm.org/docs/Dialects/Seq/)
      * [Seq(uential) Dialect Rationale](https://circt.llvm.org/docs/Dialects/Seq/RationaleSeq/)
    - ['ssp' Dialect+](https://circt.llvm.org/docs/Dialects/SSP/)
      * [SSP Dialect Rationale](https://circt.llvm.org/docs/Dialects/SSP/RationaleSSP/)
    - ['sv' Dialect+](https://circt.llvm.org/docs/Dialects/SV/)
      * [SV Dialect Rationale](https://circt.llvm.org/docs/Dialects/SV/RationaleSV/)
    - ['synth' Dialect+](https://circt.llvm.org/docs/Dialects/Synth/)
      * ['synth' Dialect](https://circt.llvm.org/docs/Dialects/Synth/RationaleSynth/)
      * [Synth Longest Path Analysis](https://circt.llvm.org/docs/Dialects/Synth/LongestPathAnalysis/)
    - ['systemc' Dialect+](https://circt.llvm.org/docs/Dialects/SystemC/)
      * [SystemC Dialect Rationale](https://circt.llvm.org/docs/Dialects/SystemC/RationaleSystemC/)
    - [Debug Dialect](https://circt.llvm.org/docs/Dialects/Debug/)
    - [Interop Dialect+](https://circt.llvm.org/docs/Dialects/Interop/)
      * [Interoperability Dialect Rationale](https://circt.llvm.org/docs/Dialects/Interop/RationaleInterop/)
    - [LLHD Dialect](https://circt.llvm.org/docs/Dialects/LLHD/)
    - [LTL Dialect](https://circt.llvm.org/docs/Dialects/LTL/)
    - [Moore Dialect](https://circt.llvm.org/docs/Dialects/Moore/)
    - [Random Test Generation (RTG) Rationale](https://circt.llvm.org/docs/Dialects/RTG/)
    - [Simulation Dialect](https://circt.llvm.org/docs/Dialects/Sim/)
    - [SMT Dialect](https://circt.llvm.org/docs/Dialects/SMT/)
    - [Verif Dialect](https://circt.llvm.org/docs/Dialects/Verif/)
  + [EDA Tool Workarounds](https://circt.llvm.org/docs/ToolsWorkarounds/)
  + [Formal Verification Tooling](https://circt.llvm.org/docs/FormalVerification/)
  + [Getting Started with the CIRCT Project](https://circt.llvm.org/docs/GettingStarted/)
  + [HLS in CIRCT](https://circt.llvm.org/docs/HLS/)
  + [Passes](https://circt.llvm.org/docs/Passes/)
  + [Python CIRCT Design Entry (PyCDE)+](https://circt.llvm.org/docs/PyCDE/)
    - [Compiling CIRCT and PyCDE](https://circt.llvm.org/docs/PyCDE/compiling/)
    - [PyCDE Basics](https://circt.llvm.org/docs/PyCDE/basics/)
  + [Static scheduling infrastructure](https://circt.llvm.org/docs/Scheduling/)
  + [Symbol and Inner Symbol Rationale](https://circt.llvm.org/docs/RationaleSymbols/)
  + [Using the Python Bindings](https://circt.llvm.org/docs/PythonBindings/)
  + [Verilog and SystemVerilog Generation](https://circt.llvm.org/docs/VerilogGeneration/)