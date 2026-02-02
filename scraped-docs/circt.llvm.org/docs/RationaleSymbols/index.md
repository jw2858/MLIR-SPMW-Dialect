Symbol and Inner Symbol Rationale - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

Symbol and Inner Symbol Rationale
=================================

This document describes various design points of the major CIRCT dialects
relating to the use of symbols and the introduction of inner symbols and
related types. This follows in the spirit of other
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

Introduction [¶](#introduction)
-------------------------------

Verilog and FIRRTL have, from a software compiler perspective, an unusual
number of nameable entities which can be referred to non-locally. These entities
have deep nesting in the code structures. The requirements of dealing with
these entities and references entails more complexity than provided by MLIR’s
symbols and symbol tables. Several CIRCT dialects, therefore, share a common
supplemental mechanism called “Inner Symbols” to manage these requirements.  
Inner Symbols necessarily deviate from MLIR nested symbol tables to enable
representation of the behavior of Verilog and FIRRTL.

Use of MLIR symbols [¶](#use-of-mlir-symbols)
---------------------------------------------

MLIR symbols are directly used for items in the global scope. This primarily
consists of `hw` or `firrtl` modules, though other entities, such as `sv`
interfaces and bind statements and `firrtl` non-local anchors, also share this
space. Modules and instances of them are well suited to MLIR symbols. They
are analogous in scoping and structure to functions and call instructions. The
top-level `builtin.module` or `firrtl.circuit` operations define a symbol table, and all
modules contained define symbols, with instances referring by symbol to their
instantiated module.

Inner Symbol [¶](#inner-symbol)
-------------------------------

Within a `firrtl` or `hw` module, many entities may exist which can be referenced
outside the module. Operations and ports (and memory ports), need to define
symbol-like data to allow forming non-SSA linkage between disparate elements.  
To accomplish this, an attribute named `inner_sym` is attached, providing a
scoped symbol-like name to the element. An operation with an `inner_sym`
resides in arbitrarily-nested regions of a region that defines an
`InnerSymbolTable` and a `Symbol` . `InnerSymbolTable` operations must reside
within an `InnerRefNamespace`. The `inner_sym` attribute must be an `InnerSymAttr`
which defines the inner symbols attached to the operation and its fields.
Operations containing an inner symbol must implement the `InnerSymbol` interface.

Inner Symbols are different from normal symbols due to MLIR symbol table
resolution rules. Specifically, normal symbols are resolved by first going up
to the closest parent symbol table and resolving down from there (recursing
back down for nested symbol paths). In FIRRTL and HW, modules define a symbol in a
`firrtl.circuit` or `builtin.module` symbol table. For instances to be able to resolve the
modules they instantiate, the symbol use in an instance must resolve in the
top-level symbol table. If a module were a symbol table, instances resolving a
symbol would start from their own module, never seeing other modules (since
resolution would start in the parent module of the instance and be unable to go
to the global scope). The second problem arises from nesting. Symbol
defining operations must be immediate children of a symbol table. FIRRTL and HW/SV
operations which define an `inner_sym` are grandchildren, at least, of a symbol
table and may be much further nested. Lastly, ports need to define `inner_sym`,
something not allowed by normal symbols.

Inner Symbol Reference Attribute [¶](#inner-symbol-reference-attribute)
-----------------------------------------------------------------------

An attribute `InnerRefAttr` is provided to encapsulate references to inner
symbols. This attribute stores the parent symbol and the inner symbol. This
provides a uniform type for storing and manipulating references to inner
symbols. An `InnerRefAttr` resolves in an `InnerRefNamespace`.

Operations using `InnerRefAttr` should implement the `verifyInnerRefs` method
of `InnerRefUserOpInterface` to verify these references efficiently.

Traits and Classes [¶](#traits-and-classes)
-------------------------------------------

### InnerSymbolTable [¶](#innersymboltable)

Similar to MLIR’s `SymbolTable`, `InnerSymbolTable` is both a trait and a class.

#### Trait [¶](#trait)

The trait is used by Operations to define a new scope for inner symbols
contained within. These operations must have the `Symbol` trait and be
immediate children of an `InnerRefNamespace`. Operations must use the
`InnerSymbol` interface to provide a symbol, regardless of presence of
attributes.

#### Class [¶](#class)

The class is used either manually constructed or as an analysis to track and
resolve inner symbols within an operation with the trait.

The class is also used in verification.

### InnerSymbolTableCollection [¶](#innersymboltablecollection)

This class is used to construct the inner symbol tables
for all `InnerSymbolTable`s (e.g., a Module) within an `InnerRefNamespace`
(e.g., a circuit), either on-demand or eagerly in parallel.

Use this to efficiently gather information about inner symbols.

### InnerRefNamespace [¶](#innerrefnamespace)

This is also both a class and a trait.

#### Class [¶](#class-1)

Combines `InnerSymbolTableCollection` with a `SymbolTable` for resolution of
`InnerRefAttr`s, primarily used during verification as argument to the
`verifyInnerRefs` hook which operations may use for more efficient checking of
`InnerRefAttr`s.

#### Trait [¶](#trait-1)

The `InnerRefNamespace` trait is used by Operations to define a new scope for
InnerRef’s. Operations with this trait must also be a `SymbolTable`.
Presently the only user is `CircuitOp`.

Cost [¶](#cost)
---------------

Inner symbols are more costly than normal symbols, precisely from the
relaxation of MLIR symbol constraints. Since nested regions are allowed,
finding all operations defining an `inner_sym` requires a recursive IR scan.  
Verification is likewise trickier, partly due the significant increase in
non-local references.

For this reason, verification is driven as a trait verifier on
`InnerRefNamespace` which constructs and verifies `InnerSymbolTable`s in
parallel, and uses these to conduct a per-`InnerSymbolTable` parallelized walk
to verify references by calling the `verifyInnerRefs` hook on
`InnerRefUserOpInterface` operations.

Common Use [¶](#common-use)
---------------------------

The most common use for `InnerRefAttr`s are to build paths through the instantiation
graph to use a subset of the instances of an entity in some way. This may
be reading values via SystemVerilog’s cross-module references (XMRs),
specifying SV bind constraints,
specifying placement constraints, or representing non-local attributes (FIRRTL).

The common element for building paths of instances through the instantiation
graph is with a `NameRefArrayAttr` attribute. This is used, for example, by
`hw.GlobalRefOp` and `firrtl.hierpath`.

 [Prev - Static scheduling infrastructure](https://circt.llvm.org/docs/Scheduling/ "Static scheduling infrastructure")
[Next - Using the Python Bindings](https://circt.llvm.org/docs/PythonBindings/ "Using the Python Bindings") 

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