LoopSchedule Dialect Rationale - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

LoopSchedule Dialect Rationale
==============================

This document describes various design points of the `loopschedule` dialect, why it is
the way it is, and current status. This follows in the spirit of other
[MLIR
Rationale docs](https://mlir.llvm.org/docs/Rationale/).

Introduction [¶](#introduction)
-------------------------------

The `loopschedule` dialect provides a collection of ops to represent software-like loops
after scheduling. There are currently two main kinds of loops that can be represented:
pipelined and sequential. Pipelined loops allow multiple iterations of the loop to be
in-flight at a time and have an associated initiation interval (`II`) to specify the number
of cycles between the start of successive loop iterations. In contrast, sequential loops
are guaranteed to only have one iteration in-flight at any given time.

A primary goal of the `loopschedule` dialect, as opposed to many other High-Level Synthesis
(HLS) representations, is to maintain the structure of loops after scheduling. As such, the
`loopschedule` ops are inspired by the `scf` and `affine` dialect ops.

Pipelined Loops [¶](#pipelined-loops)
-------------------------------------

Pipelined loops are represented with the `loopschedule.pipeline` op. A `pipeline`
loop resembles a `while` loop in the `scf` dialect, but the body must contain only
`loopschedule.pipeline.stage` and `loopschedule.terminator` ops. To have a better
understanding of how `loopschedule.pipeline` works, we will look at the following
example:

```
func.func @test1(%arg0: memref<10xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = loopschedule.pipeline II = 1 iter_args(%arg1 = %c0, %arg2 = %c0_i32) : (index, i32) -> i32 {
    %1 = arith.cmpi ult, %arg1, %c10 : index
    loopschedule.register %1 : i1
  } do {
    %1:2 = loopschedule.pipeline.stage start = 0 {
      %3 = arith.addi %arg1, %c1 : index
      %4 = memref.load %arg0[%arg1] : memref<10xi32>
      loopschedule.register %3, %4 : index, i32
    } : index, i32
    %2 = loopschedule.pipeline.stage start = 1 {
      %3 = arith.addi %1#1, %arg2 : i32
      pipeline.register %3 : i32
    } : i32
    loopschedule.terminator iter_args(%1#0, %2), results(%2) : (index, i32) -> i32
  }
  return %0 : i32
}
```

A `pipeline` op first defines the initial values for the `iter_args`. `iter_args` are values that will
be passed back to the first stage after the last stage of the pipeline. The pipeline also defines a
specific, static `II`. Each pipeline stage in the `do` block represents a series of ops run in parallel.

Values are registered at the end of a stage and passed out as results for future pipeline stages to
use. Each pipeline stage must have a defined start time, which is the number of cycles between the
start of the pipeline and when the first valid data will be available as input to that stage.

Finally, the terminator is called with the `iter_args` for the next iteration and the result values
that will be returned when the pipeline completes. Even though the terminator is located at the
end of the loop body, its values are passed back to a previous stage whenever needed. We do not
need to wait for an entire iteration to finish before `iter_args` become valid for the next iteration.

Multi-cycle and pipelined ops can also be supported in `pipeline` loops. In the following example,
assume the multiply op is bound to a 3-stage pipelined multiplier:

```
func.func @test1(%arg0: memref<10xi32>, %arg1: memref<10xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  loopschedule.pipeline II = 1 iter_args(%arg2 = %c0) : (index, i32) -> () {
    %1 = arith.cmpi ult, %arg1, %c10 : index
    loopschedule.register %1 : i1
  } do {
    %1:2 = loopschedule.pipeline.stage start = 0 {
      %3 = arith.addi %arg1, %c1 : index
      %4 = memref.load %arg0[%arg2] : memref<10xi32>
      loopschedule.register %3, %4 : index, i32
    } : index, i32
    %2:2 = loopschedule.pipeline.stage start = 1 {
      %3 = arith.muli %1#1, %c1_i32 : i32
      pipeline.register %3, %1#0 : i32
    } : i32
    loopschedule.pipeline.stage start = 4 {
      memref.store %2#0, %arg0[%2#1] : i32
      pipeline.register
    } : i32
    loopschedule.terminator iter_args(%1#0), results() : (index, i32) -> ()
  }
  return
}
```

Here, the `II` is still 1 because new values can be introduced to the multiplier every cycle. The last
stage is delayed by 3 cycles because of the 3 cycle latency of the multiplier. The `pipeline` op is
currently tightly coupled to the lowering implementation used, as the latency of operators is not
represented in the IR, but rather an implicit assumption made when lowering later. The scheduling
problem is constructed with these implicit operator latencies in mind. This coupling can be addressed
in the future with a proper operator library to maintain explicit operator latencies in the IR.

Status [¶](#status)
-------------------

Added pipeline loop representation, more documentation and rationale to come as ops are added.

 [Prev - 'loopschedule' Dialect](https://circt.llvm.org/docs/Dialects/LoopSchedule/ "'loopschedule' Dialect")
[Next - 'msft' Dialect](https://circt.llvm.org/docs/Dialects/MSFT/ "'msft' Dialect") 

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
  + [Dialects-](https://circt.llvm.org/docs/Dialects/)

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
    - ['loopschedule' Dialect-](https://circt.llvm.org/docs/Dialects/LoopSchedule/)
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