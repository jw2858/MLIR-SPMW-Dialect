SystemC Dialect Rationale - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

SystemC Dialect Rationale
=========================

This document describes various design points of the SystemC dialect, why they
are the way they are, and current status. This follows in the spirit of other
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

* [SystemC Dialect Rationale](#systemc-dialect-rationale)
  + [Introduction](#introduction)
  + [Lowering](#lowering)
  + [Q&A](#qa)

Introduction [¶](#introduction)
-------------------------------

[SystemC](https://en.wikipedia.org/wiki/SystemC) is a library written in C++
to allow functional modeling of systems. The included event-driven simulation
kernel can then be used to simulate a system modeled entirely in SystemC.
Additionally, SystemC is a standard (IEEE Std 1666-2011) supported by several
tools (e.g., Verilator) and can thus be used as an interface to such tools as
well as between multiple systems that are internally using custom
implementations.

Enabling CIRCT to emit SystemC code provides another way (next to Verilog
emission) to interface with the outside-world and at the same time
provides another way to simulate systems compiled with CIRCT.

Lowering [¶](#lowering)
-----------------------

In a first step, lowering from
[HW](https://circt.llvm.org/docs/Dialects/HW/)
to the SystemC dialect will be implemented. A tool called ExportSystemC,
which is analogous to ExportVerilog, will then take these SystemC and
[Comb](https://circt.llvm.org/docs/Dialects/Comb/) operations to emit proper
SystemC-C++ code that can be compiled using clang, GCC, or any other
C++-compiler to produce the simulator binary. In the long run support for more
dialects can be added, such as LLHD and SV.

As a simple example we take a look at the following HW module which just adds
two numbers together:

```
hw.module @adder (%a: i32, %b: i32) -> (c: i32) {
    %sum = comb.add %a, %b : i32
    hw.output %sum : i32
}
```

It will then be lowered to the following SystemC IR to make code emission
easier for ExportSystemC:

```
systemc.module @adder(%a: i32, %b: i32) -> (%c: i32) {
    systemc.ctor {
        systemc.method @add
    }
    systemc.func @add() {
        // c = a + b
        %res = comb.add %a, %b : i32
        systemc.con %c, %res : i32
    }
}
```

ExportSystemC will then emit the following C++ code to be compiled by clang or
another C++-compiler:

```
#ifndef ADDER_H
#define ADDER_H

#include <systemc.h>

SC_MODULE(adder) {
    sc_in<sc_uint<32>> a;
    sc_in<sc_uint<32>> b;
    sc_out<sc_uint<32>> c;

    SC_CTOR(adder) {
        SC_METHOD(add);
    }

    void add() {
        c = a + b;
    }
};

#endif // ADDER_H
```

Q&A [¶](#qa)
------------

**Q: Why implement a custom module operation rather than using `hw.module`?**

In SystemC we want to model module outputs as arguments such that the SSA value
is already defined from the beginning which we can then assign to and reference.

**Q: Why implement a custom func operation rather than using `func.func`?**

An important difference compared to the `func.func` operation is that it
represents a member function (method) of a SystemC module, i.e., a C++ struct.
This leads to some implementation differences:

* Not isolated from above: we need to be able to access module fields such as
  the modules inputs, outputs, and signals
* Verified to have no arguments and void return type: this is a restriction
  from SystemC for the function to be passed to SC\_METHOD, etc. This could
  also be achieved with `func.func`, but would require us to write the verifier
  in `systemc.module` instead.
* Region with only a single basic block (structured control flow) and no
  terminator: using structured control-flow leads to easier code emission

**Q: How much of C++ does the SystemC dialect aim to model?**

As much as necessary, as little as possible. Completely capturing C++ in a
dialect would be a huge undertaking and way too much to ‘just’ achieve SystemC
emission. At the same time, it is not possible to not model any C++ at all,
because when only modeling SystemC specific constructs, the gap for
ExportSystemC to bridge would be too big (we want the printer to be as simple
as possible).

**Q: Why does `systemc.module` have a graph region rather than a SSACFG region?**

It contains a single graph region to allow flexible positioning of the fields,
constructor and methods to support different ordering styles (fields at top
or bottom, methods to be registered with SC\_METHOD positioned after the
constructor, etc.) without requiring any logic in ExportSystemC. Program code
to change the emission style can thus be written as part of the lowering from
HW, as a pre-emission transformation, or anywhere else.

 [Prev - 'systemc' Dialect](https://circt.llvm.org/docs/Dialects/SystemC/ "'systemc' Dialect")
[Next - Debug Dialect](https://circt.llvm.org/docs/Dialects/Debug/ "Debug Dialect") 

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
    - ['systemc' Dialect-](https://circt.llvm.org/docs/Dialects/SystemC/)
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