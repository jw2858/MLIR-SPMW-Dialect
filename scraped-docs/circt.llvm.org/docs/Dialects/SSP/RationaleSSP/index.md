SSP Dialect Rationale - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

SSP Dialect Rationale
=====================

This document describes various design points of the SSP dialect, why they are
the way they are, and current status. This follows in the spirit of other
[MLIR
Rationale docs](https://mlir.llvm.org/docs/Rationale/).

Introduction [¶](#introduction)
-------------------------------

CIRCT’s
[scheduling infrastructure](https://circt.llvm.org/docs/Scheduling/) is
lightweight and dialect-agnostic, in order to fit into any lowering flow with a
need for static scheduling. However, it lacks an import/export format for
storing and exchanging problem instances. The SSP ("**S**tatic **S**cheduling
**P**roblems") dialect fills that role by defining an IR that captures problem
instances

* in full fidelity,
* in a concise syntax,
* and independent of any other “host” dialect.

The SSP dialect’s main use-cases are
[testing](#testing),
[benchmarking](#benchmarking) and
[rapid-prototyping](#rapid-prototyping). It is
strictly a companion to the existing scheduling infrastructure, and clients (HLS
flows etc.) are advised to use the C++ APIs directly.

### Testing [¶](#testing)

In order to test the scheduling infrastructure’s problem definitions (in
particular, their input checkers/solution verifiers) and algorithm
implementations, a “host” IR to store problem instances is needed. To that end,
the test-cases started out with a mix of standard and unregistered operations,
and heavy use of generic attributes, as shown in the following example (note
especially the index-based specification of auxiliary dependences):

```
func.func @canis14_fig2() attributes {
  problemInitiationInterval = 3,
  auxdeps = [ [3,0,1], [3,4] ],
  operatortypes = [
    { name = "mem_port", latency = 1, limit = 1 },
    { name = "add", latency = 1 }
  ] } {
  %0 = "dummy.load_A"() { opr = "mem_port", problemStartTime = 2 } : () -> i32
  %1 = "dummy.load_B"() { opr = "mem_port", problemStartTime = 0 } : () -> i32
  %2 = arith.addi %0, %1 { opr = "add", problemStartTime = 3 } : i32
  "dummy.store_A"(%2) { opr = "mem_port", problemStartTime = 4 } : (i32) -> ()
  return { problemStartTime = 5 }
}
```

Here is the same test-case encoded in the SSP dialect:

```
ssp.instance "canis14_fig2" of "ModuloProblem" [II<3>] {
  library {
    operator_type @MemPort [latency<1>, limit<1>]
    operator_type @Add [latency<1>]
    operator_type @Implicit [latency<0>]
  }
  graph {
    %0 = operation<@MemPort>(@store_A [dist<1>]) [t<2>]
    %1 = operation<@MemPort>() [t<0>]
    %2 = operation<@Add>(%0, %1) [t<3>]
    operation<@MemPort> @store_A(%2) [t<4>]
    operation<@Implicit>(@store_A) [t<5>]
  }
}
```

Emitting an SSP dump is also useful to test that an conversion pass correctly
constructs the scheduling problem, e.g. checking that it contains a memory
dependence to another operation:

```
// CHECK: operation<@{{.*}}>(%0, @[[store_1]])
%5 = affine.load %0[0] : memref<1xi32>
...
// CHECK: operation<@{{.*}}> @[[store_1:.*]](%7, %0)
affine.store %7, %0[0] : memref<1xi32>
```

### Benchmarking [¶](#benchmarking)

Scheduling is a hard combinatorial optimization problem that can be solved by a
variety of approaches, ranging from fast heuristics to exact formulations in
mathematical frameworks such as integer linear programs capable of computing
provably optimal solutions. It is therefore important to evaluate scheduler
implementations beyond just functional correctness testing, i.e. to assess the
scheduler’s runtime and scalability, as well as the solution quality, on sets of
representative benchmark instances.

With the SSP dialect, such instances can be saved directly from synthesis flows
using CIRCT’s scheduling infrastructure, or emitted in the textual MLIR format
by third-party tools. As the SSP IR is self-contained, it would even be viable
to store problem instances originating from out-of-tree or proprietary flows, as
their source and target IRs would not be required to load and schedule a problem
instance in a benchmark harness.

### Rapid prototyping [¶](#rapid-prototyping)

The SSP dialect also provides a path towards automatically generated Python
bindings for the scheduling infrastructure, which will ease the prototyping of
new scheduling clients and problem definitions.

### Q&A [¶](#qa)

* **Q:** Do I have to do a dialect conversion to and from this dialect to
  schedule something?

  No, use the C++ API, i.e. the problem classes and scheduler entrypoints in
  `circt::scheduling`, directly! This dialect is a one-way street in terms of
  dialect conversion, and only intended to load and store problem instances for
  the use-cases listed above.
* **Q:** Why don’t you use something like Cap’nProto to (de)serialize the
  problem instances?

  Textual MLIR is reasonably easy to write by hand, which is important for
  test-cases, and we need MLIR operations anyways, because the scheduling
  infrastructure builds on top of the MLIR def-use graph to represent its
  dependence graphs.
* **Q:** `OperationOp` doesn’t seem like a great name.

  No, you’re right. However, the SSP dialect uses the same terminology as the
  scheduling infrastructure, so any changes would have to originate there.

Rationale for selected design points [¶](#rationale-for-selected-design-points)
-------------------------------------------------------------------------------

### Use of container-like operations instead of regions in `InstanceOp` [¶](#use-of-container-like-operations-instead-of-regions-in-instanceop)

This dialect defines the `OperatorLibraryOp` and `DependenceGraphOp` to serve as
the first and second operation in an `InstanceOp`’s region. The alternative of
using two regions on the `InstanceOp` is not applicable, because the
`InstanceOp` then needs to provide a symbol table, but the upstream
`SymbolTable` trait enforces single-region ops. Lastly, we also considered using
a single graph region to hold both `OperatorTypeOp`s and `OperationOp`s, but
discarded that design because it cannot be safely roundtripped via a
`circt::scheduling::Problem` (internally, registered operator types and
operations are separate lists).

### Stand-alone use of the `OperatorLibraryOp` [¶](#stand-alone-use-of-the-operatorlibraryop)

The `OperatorLibraryOp` can be named and used outside of an `InstanceOp`. This
is useful to share operator type definitions across multiple instances. In
addition, until CIRCT gains better infrastructure to manage predefined hardware
modules and their properties, such a stand-alone `OperatorLibraryOp` can also
act as an interim solution to represent operator libraries for scheduling
clients.

### Use of SSA operands *and* symbol references to encode dependences [¶](#use-of-ssa-operands-_and_-symbol-references-to-encode-dependences)

This is required to faithfully reproduce the internal modeling in the scheduling
infrastructure, which distinguishes def-use (result to operand, tied to MLIR SSA
graph) and auxiliary (op to op, stored explicitly) dependences
(
[example](https://circt.llvm.org/docs/Scheduling/#constructing-a-problem-instance)).
To represent the former, the `OperationOp` produces an arbitrary number of
`NoneType`-typed results, and accepts an arbitrary number of operands, thus
spanning a def-use graph. Auxiliary dependences are encoded as symbol uses,
which reference the name of the dependence’s source `OperationOp`. Modeling
these dependences with symbols rather than SSA operands is a necessity because
the scheduling infrastructure implicitly considers *all* def-use edges between
registered operations. Hence, auxiliary dependences, hypothetically encoded as
SSA operands, would be counted twice.

### No attribute interface for scheduling properties [¶](#no-attribute-interface-for-scheduling-properties)

Properties are represented by dialect attributes inheriting from the base
classes in `PropertyBase.td`, which include `extraClassDeclaration`s for
`setInProblem(...)` and `getFromProblem(...)` methods that directly interact
with the C++ problem class. In order to get/set a certain property, a reference
to the concrete class is required, e.g.: a `CyclicProblem &` if we want to set a
dependence’s `distance` property.

A more obvious design would be to make these methods part of an attribute
interface. However, then the methods could only accept a `Problem &`, which
cannot be statically downcasted to the concrete class due to the use of virtual
multiple inheritance in the problem class hierarchy. If the inheritance model
were to change in the scheduling infrastructure, the use of attribute interfaces
should be reconsidered.

Import/export [¶](#importexport)
--------------------------------

The `circt/Dialect/SSP/Utilities.h` header defines methods to convert between
`ssp.InstanceOp`s and `circt::scheduling::Problem` instances. These utilities
use template parameters for the problem class and the property attribute
classes, allowing client code to load/save an instance of a certain problem
class with the given properties (but ignoring others). Incompatible properties
(e.g. `distance` on a base `Problem`, or `initiationInterval` on an operation)
will be caught at compile time as errors in the template instantiation. Note
that convenience versions that simply load/save all properties known in the
given problem class are provided as well.

Extensibility [¶](#extensibility)
---------------------------------

A key feature of the scheduling infrastructure is its extensibility. New problem
definitions in out-of-tree projects have to define attributes inheriting from
the property base classes in one of their own dialects. Due to the heavy use of
templates in the import/export utilities, these additional attributes are
supported uniformly alongside the built-in property attributes. The only
difference is that the SSP dialect provides short-form pretty printing for its
own properties, whereas externally-defined properties fall back to the generic
dialect attribute syntax.

 [Prev - 'ssp' Dialect](https://circt.llvm.org/docs/Dialects/SSP/ "'ssp' Dialect")
[Next - 'sv' Dialect](https://circt.llvm.org/docs/Dialects/SV/ "'sv' Dialect") 

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
    - ['ssp' Dialect-](https://circt.llvm.org/docs/Dialects/SSP/)
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