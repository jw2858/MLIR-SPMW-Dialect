SMT Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

SMT Dialect
===========

This dialect provides types and operations modeling the SMT (Satisfiability
Modulo Theories) operations and datatypes commonly found in SMT-LIB and SMT
solvers.

* [Rationale](#rationale)
* [Dialect Structure](#dialect-structure)
* [Optimizations](#optimizations)
* [Backends](#backends)
  + [LLVM IR](#llvm-ir)
  + [SMT-LIB](#smt-lib)
  + [C/C++](#cc)
* [Handling counter-examples](#handling-counter-examples)
* [Non-Goals](#non-goals)
* [Operations](#operations)
* [Types](#types)

Rationale [¶](#rationale)
-------------------------

This dialect aims to provide a unified interface for expressing SMT problems
directly within MLIR, enabling seamless integration with other MLIR dialects and
optimization passes. It models the
[SMT-LIB standard
2.6](https://smtlib.cs.uiowa.edu/), but may also include features of commonly
used SMT solvers that are not part of the standard. In particular, the IR
constructs are designed to enable more interactive communication with the solver
(allowing if-statements, etc. to react on solver feedback).

The SMT dialect is motivated by the following advantages over directly printing
SMT-LIB or exporting to a solver:

* Reuse MLIR’s infrastructure: passes and pass managers to select different SMT
  encodings, operation builders and rewrite patterns to build SMT formulae (and
  hide the solver’s API in a provided lowering pass), common textual format to
  share rather than dumping the solver’s state, etc.
* No need to add a link-dependency on SMT solvers to CIRCT (just provide the
  path to the library as an argument to the JIT runner or manually link the
  produced binary against it).
* Using an SMT dialect as intermediary allows it to be mixed with concrete
  computations, the debug dialect, etc. This is complicated to achieve and
  reason about when building the external solver’s state directly.
* Enable easy addition of new backends
* Have a common representation and infrastructure for all SMT related efforts,
  such that people don’t have to build their own isolated tools.

The dialect follows these design principles:

* **Solver-independent**: don’t model one particular solver’s API
* **Seemless interoperability** with other dialects. E.g., to allow using the
  debug dialect to back-propagate counter examples
* **Small surface for errors**: try to keep the SMT dialect and its lowerings
  simple to avoid mistakes, implement optimizations defensively or prove them
  formally; since higher-level dialects will be lowered through the SMT dialect
  to construct formal proofs it is essential that this dialect does not
  introduce bugs
* **Interactive**: the IR should be designed such that it can be interleaved
  with operations (from other dialects) that take the current feedback of the
  solver to steer the execution of further SMT operations. It shouldn’t just
  model the very rigid SMT-LIB.
* Don’t heavily integrate the dialect itself with CIRCT to make potential
  upstreaming easy

Dialect Structure [¶](#dialect-structure)
-----------------------------------------

The SMT dialect is structured into multiple “sub-dialects”, one for each of the
following theories (this separation is also made clear in the prefix of
operation and type names as indicated in parentheses):

* Core boolean logic including quantifiers and solver interaction (`smt.*`)
* Bit-vectors (`smt.bv.*`)
* Arbitrary-precision integers (`smt.int.*`)
* Arrays (`smt.array.*`)
* Floating-point numbers (`smt.real.*`)

Several operations in the core part (e.g., quantifiers, equality, etc.) allow
operands of any SMT type (including bit-vectors, arrays, etc.). Therefore, all
type and attribute declarations are part of the core.

Certain arithmetic, bitwise, and comparison operations exist for multiple
theories. For example, there exists an AND operation for booleans and one for
bit-vectors, or there exists and ADD operation for integers and bit-vectors. In
such cases, each “sub-dialect” defines its own operation specific to its
datatypes. This simplifies the operations such that an optimization or
conversion pass only using bit-vectors does not have to take care of other
potentially supported datatypes.

Optimizations [¶](#optimizations)
---------------------------------

The primary purpose of the SMT dialect is not to optimize SMT formulae. However,
SMT solvers can exhibit significant differences in runtime, even with slight
changes in input. Improving solver performance by implementing rewrite patterns
that slightly restructure the SMT formulae may be possible.

Moreover, SMT solvers may differ in terms of built-in operators. If a solver
lacks support for advanced operators, the problem can be simplified before
passing it to the solver.

Backends [¶](#backends)
-----------------------

Having an SMT dialect instead of directly interpreting the IR and building an
SMT expression enables multiple different backends to be used in addition to the
application of SMT dialect level optimizations that rewrite the formulae for
faster and more predictable runtime performance of the solver backend (e.g.,
Z3).

In the following, we outline the different backend lowerings and their
advantages and disadvantages.

### LLVM IR [¶](#llvm-ir)

Lowering to LLVM IR that calls the C API of a given SMT solver is practical for
a few reasons:

* enables using LLVM JIT or compilation to a standalone binary
* easy to mix with Debug dialect to report back nice counter examples
* allows mixing concrete and symbolic executions (e.g., for dynamic BMC upper
  bounds, or more dynamic interaction with the solver such as extraction of
  multiple/all possible models)

However, it is solver-dependent and more complicated to implement than an
SMT-LIB printer.

### SMT-LIB [¶](#smt-lib)

The SMT-LIB format is a standardized format supported by practically all SMT
solvers and thus an ideal target to support as many solver backends as possible.
However,

* this format is quite static and does not allow easy interaction with the
  solver, in particular, it is not easily possible to make future asserts
  dependent on current solver outputs,
* providing good counter-examples to the user would mean parsing the textual
  model output of the solver and mapping it to an CIRCT-internal datastructure.
* it is impossible to mix symbolic and concrete executions, as well as debug
  constructs (see Debug Dialect).
* it is impossible to just use the LLVM JIT compiler to directly get a result,
  but instead the external solver has to be called directly, either by adding a
  compile-time dependency, or using a shell call.

### C/C++ [¶](#cc)

A C/C++ exporter that produces code which calls the C/C++ API of a given solver
could allow for easier debugging and allows to support solvers without C API
without the restrictions of SMT-LIB. However, this means the JIT compilation
flow would not be available.

Handling counter-examples [¶](#handling-counter-examples)
---------------------------------------------------------

SMT solvers check formulae for satisfiability. Typically, there are three kinds
of output a solver may give:

* **Satisfiable**: In this case a model (counter-example) can be provided by the
  solver. However, it may not be feasible to evaluate/interpret this model for a
  given SMT constant to get a constant value. This is, in particular, the case
  when the SMT encoding contains quantifiers which can lead to the model
  containing quantifiers as well. Solvers (e.g., Z3) usually don’t evaluate
  quantifiers in models (even if they are closed). If constants can be
  evaluated, a counter example can be provided and back-propagated to the
  source-code, e.g., using the debug dialect.
* **Unsatisfiable**: formal verification problems are typically encoded such
  that this output indicates correctness; a proof can be provided by the solver,
  but is often not needed
* **Unknown**: there can be various reasons why the result is unknown; a common
  one is the use of symbolic functions to represent operators and encode, e.g.,
  a LEC problem as a rewrite task of patterns of function applications. This is
  a frequent application and the unknown result is just treated like a
  satisfiable result without counter example.

Non-Goals [¶](#non-goals)
-------------------------

* The SMT Dialect does not aim to include any operations or types that model
  verification constructs not specific to SMT, i.e., things that could also be
  lowered to other kind of verification systems such as inductive theorem
  provers (e.g., Lean4).

Operations [¶](#operations)
---------------------------

Types [¶](#types)
-----------------

 [Prev - Simulation Dialect](https://circt.llvm.org/docs/Dialects/Sim/ "Simulation Dialect")
[Next - Verif Dialect](https://circt.llvm.org/docs/Dialects/Verif/ "Verif Dialect") 

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