HLS in CIRCT - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

HLS in CIRCT
============

// write a compelling introduction here

* [`hlstool`](#hlstool)
* [Flows](#flows)
* [Dynamically scheduled HLS (DHLS)](#dynamically-scheduled-hls-dhls)
* [Calyx](#calyx)

`hlstool` [¶](#hlstool)
-----------------------

[`hlstool`](https://github.com/llvm/circt/blob/main/tools/hlstool/hlstool.cpp) is a tool for driving various CIRCT-based HLS flows. The tool works by defining
[pass pipelines](https://mlir.llvm.org/docs/PassManagement/) that string together various MLIR and CIRCT passes to realize an HLS flow.  
For new users to MLIR, it is important to recognize the different between such compiler driver tools, and the MLIR `opt` tools (optimization drivers), such as `mlir-opt` and `circt-opt`.  
For instance, in `hlstool` we may specify a pass pipeline such as:

```
  // Memref legalization.
  pm.addPass(circt::createFlattenMemRefPass());
  pm.nest<func::FuncOp>().addPass(
      circt::handshake::createHandshakeLegalizeMemrefsPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
```

For the compiler developer, however, we often *don’t* want to run the entire HLS pass pipeline on our IR, but instead work on a specific level of IR and run a specific pass. Thus, every pass that is nested within `hlstool` can also be driven by an `opt` tool, e.g. `circt::createFlattenMemRefPass` is registered in `circt-opt` as `circt-opt --flatten-memref`.

Flows [¶](#flows)
-----------------

CIRCT homes multiple actively developed HLS flows, each with their own approach to HLS. This section aims to provide a brief overview of the different flows, and how to use them.

**Note:** we do mean ‘active development’**(!)** - most of these flows are the combined efforts of various student, phd and research projects, and should **not** be considered production-ready. You **will** encounter bugs, but you will also have a group of people are happy to help you fix them, and improve the flows. Feel free to reach out to the
[codeowner](https://github.com/llvm/circt/blob/main/CODEOWNERS) of the dialect/pass that you’re working on, or ask a question in the
[LLVM discord](https://discord.gg/Jsktb5PR) which has a dedicated `circt` channel.

How flow documentation is structured:

* **Description**: A brief description/rationale of the flow, what it is and why it is how it is.
* **Relevant dialects**: A listing of the dialects used in the flow (preferably in order), as well as links to the dialect rationales. We encourage the user to read the dialect rationales to understand the design decisions behind the dialects.
* **Example usage/relevant tests**: A vast majority of CIRCTs flow-specific knowledge is (unfortunately) institutional. However, users are often referred to the CIRCT tests to understand what different passes do, and how to drive the various CIRCT tools to achieve their goals. This section thus lists some relevant tests that may serve as important references of how to use the flow.

Dynamically scheduled HLS (DHLS) [¶](#dynamically-scheduled-hls-dhls)
---------------------------------------------------------------------

#### Relevant dialects: [¶](#relevant-dialects)

* [cf](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/)
* [Handshake](https://circt.llvm.org/docs/Dialects/Handshake/RationaleHandshake/)

#### Example usage/Relevant tests [¶](#example-usagerelevant-tests)

* [handshake integration tests](https://github.com/llvm/circt/tree/main/integration_test/Dialect/Handshake) is a collection of cocotb-based tests that use `hlstool` to convert `cf`-level code to verilog.
* [handshake-runner](https://github.com/llvm/circt/tree/main/integration_test/handshake-runner) tests show how to use the `handshake-runner` to interpret `handshake`-level IR

#### Resources [¶](#resources)

The following is a list of work that has used or contributed to the DHLS flow in CIRCT:

* (10/2022)
  [Multi-Level Rewriting for Stream Processing to RTL compilation (M.Sc. Thesis) - Christian Ulmann](https://www.research-collection.ethz.ch/handle/20.500.11850/578713)
* (01/2022)
  [A Dynamically Scheduled HLS Flow in MLIR (M.Sc. Thesis) - Morten Borup Petersen](https://infoscience.epfl.ch/record/292189)

Calyx [¶](#calyx)
-----------------

#### Relevant dialects: [¶](#relevant-dialects-1)

#### Example usage/Relevant tests [¶](#example-usagerelevant-tests-1)

#### Resources [¶](#resources-1)

 [Prev - Getting Started with the CIRCT Project](https://circt.llvm.org/docs/GettingStarted/ "Getting Started with the CIRCT Project")
[Next - Passes](https://circt.llvm.org/docs/Passes/ "Passes") 

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