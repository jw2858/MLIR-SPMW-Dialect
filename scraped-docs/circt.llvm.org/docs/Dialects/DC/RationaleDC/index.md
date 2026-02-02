DC Dialect Rationale - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

DC Dialect Rationale
====================

* [Introduction](#introduction)
* [Value (channel) semantics](#value-channel-semantics)
* [Canonicalization](#canonicalization)

Introduction [¶](#introduction)
-------------------------------

DC (**D**ynamic **C**ontrol) IR describes independent, unsynchronized processes
communicating data through First-in First-out (FIFO) communication channels.
This can be implemented in many ways, such as using synchronous logic, or with
processors.

The intention of DC is to model all operations required to represent such a
control flow language. DC aims to be strictly a control
flow dialect - as opposed to the Handshake dialect, which assigns control
semantics to *all* SSA values. As such, data values are
only present in DC where they are required to model control flow.

By having such a control language, the dialect aims to facilitate the
construction of dataflow programs where control and data
is explicitly separate. This enables optimization of the control and data side
of the program independently, as well as mapping
of the data side into functional units, pipelines, etc.. Furthermore, separating
data and control will make it easier to reason about possible critical paths of
either circuit, which may inform buffer placement.

The DC dialect has been heavily influenced by the Handshake dialect, and can
either be seen as a successor to it, or a lower level
abstraction. As of writing, DC is *fully deterministic*. This means that
non-deterministic operators such as the ones found in Handshake -
`handshake.merge, handshake.control_merge` - do **not** have a lowering to DC.
Handshake programs must therefore be converted or by construction not contain
any of these non-deterministic operators. Apart from that, **all** handshake
operations can be lowered to a combination of DC
and e.g. `arith` operations (to represent the data side semantics of any given
operation).

In DC IR, all values have implicit fork and sink semantics. That is, a DC-typed
value may be referenced multiple times, as well as it being legal that said
value is not referenced at all. This has been chosen to facilitate
canonicalization, thus removing the need for all canonicalization patterns to
view forks as opaque/a special case. If a given DC lowering requires explcit
fork/sink semantics, forks and sinks can be *materialized* throuh use of the
`--dc-materialize-forks-sinks` pass. Conversely, if one wishes to optimize DC
IR which already contains fork and sink operations, one may use the
`--dc-dematerialize-forks-sinks` pass, run canonicalization, and then re-apply
the `--dc-materialize-forks-sinks` pass.

Value (channel) semantics [¶](#value-channel-semantics)
-------------------------------------------------------

1. **Latency insensitive**:
   * Any DC-typed value (`dc.token/dc.value<T...>`) has latency insensitive
     semantics.
   * DC does **not** specify the implementation of this latency
     insensitivity, given that it strictly pertains to the **control** of latency
     insensitive values. This should reinforce the mental model that DC isn’t
     strictly a hardware construct - that is, DC values could be implemented in
     hardware by e.g ready/valid semantics or by FIFO interfaces (read/write, full, empty, …)
     or in software by e.g. message queues, RPC, or other streaming protocols.
   * In the current state of the world (CIRCT), DC uses ESI to implement its
     latency insensitive hardware protocol. By doing so, we let DC do what DC does
     best (control language) and likewise with ESI (silicon interconnect).
2. **Values are channels**:
   * Given the above latency insensitivity, it is useful to think of DC values
     as channels, wherein a channel can be arbitrarily buffered without changing the
     semantics of the program.
3. **FIFO semantics**:
   * DC-typed values have FIFO semantics, meaning that the order of values in
     the ‘channel’ is preserved (i.e. the order of values written to the channel is
     the same as the order of values read from the channel).

Canonicalization [¶](#canonicalization)
---------------------------------------

By explicitly separating data and control parts of a program, we allow for
control-only canonicalization to take place.
Here are some examples of non-trivial canonicalization patterns:

* **Transitive join closure**:
  + Taking e.g. the Handshake dialect as the source abstraction, all
    operations - unless some specific Handshake operations - will be considered as
    *unit rate actors* and have join semantics. When lowering Handshake to DC, and
    by separating the data and control paths, we can easily identify `join`
    operations which are staggered, and can be merged through a transitive closure
    of the control graph.
* **Branch to select**: Canonicalizes away a select where its inputs originate
  from a branch, and both have the same select signal.
* **Identical join**: Canonicalizes away joins where all inputs are the same
  (i.e. a single join can be used).

 [Prev - 'dc' Dialect](https://circt.llvm.org/docs/Dialects/DC/ "'dc' Dialect")
[Next - 'emit' Dialect](https://circt.llvm.org/docs/Dialects/Emit/ "'emit' Dialect") 

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
    - ['dc' Dialect-](https://circt.llvm.org/docs/Dialects/DC/)
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