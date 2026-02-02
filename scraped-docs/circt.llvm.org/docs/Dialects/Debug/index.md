Debug Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

Debug Dialect
=============

This dialect provides operations and types to interleave debug information (DI)
with other parts of the IR.

* [Rationale](#rationale)
  + [Representations](#representations)
* [Representing Source Language Constructs](#representing-source-language-constructs)
* [Tracking Inlined Modules](#tracking-inlined-modules)
* [Types](#types)
  + [Overview](#overview)
  + [ArrayType](#arraytype)
  + [ScopeType](#scopetype)
  + [StructType](#structtype)
* [Operations](#operations)
  + [`dbg.array` (circt::debug::ArrayOp)](#dbgarray-circtdebugarrayop)
  + [`dbg.scope` (circt::debug::ScopeOp)](#dbgscope-circtdebugscopeop)
  + [`dbg.struct` (circt::debug::StructOp)](#dbgstruct-circtdebugstructop)
  + [`dbg.variable` (circt::debug::VariableOp)](#dbgvariable-circtdebugvariableop)

Rationale [¶](#rationale)
-------------------------

The main goal of the debug dialect is to provide a mechanism to track the
correspondence between values, types, and hierarchy of a source language and the
IR being compiled and transformed. This allows simulators, synthesizers, and
other debugging tools to reconstruct a source language view into the processed
hardware and allow for easier debugging by humans.

Debug information in CIRCT follows these principles:

* **It is best effort:** DI is meant as a tool to aid humans in their debugging
  effort, not a contractual obligation to retain all source language semantics
  through the compilation pipeline. We preserve information as well as possible
  and reasonable, but accept the fact that certain optimizations may cause
  information to be discarded.
* **It affects the output:** Enabling the tracking of DI is expected to block
  certain optimizations. We undertake an effort to minimize the impact of DI on
  the output quality, size, simulation speed, or synthesis results, but accept
  the fact that preserving visibility and observability of source language
  constructs may prevent certain optimizations from running.

### Representations [¶](#representations)

There are two mechanisms in MLIR that lend themselves to conveying debug
information:

* **Attributes** attached to existing operations. This is similar to LLVM’s
  approach of tracking DI in the operation’s metadata. Translated to MLIR, an
  operation’s location would be an obvious choice to do this tracking, since
  locations are well-preserved by passes and difficult to accidentally drop.
  MLIR currently does not support custom location attributes, which would
  require DI attributes to be attached to a `FusedLoc` as metadata.
* **Operations** interleaved with the rest of the IR. This makes DI a
  first-class citizen, but also causes debug information to potentially intefere
  with optimizations. For example, debug dialect ops introduce additional uses
  of values that might have otherwise been deleted by DCE. However, there may be
  alternative ways to dealing with such situations. For example, Verilog
  emission may simply ignore operations that are only used by debug ops,
  therefore achieving the same effect as DCE would have.

The debug dialect uses *operations* to represent debug info. This decision was
based on discussions with various people in the LLVM and MLIR community, where
DI was commonly quoted as one of LLVM’s weak points, with its living in metadata
space making it more of a second-class citizen rather than a first-class
concern. Since we want to represent source language types and constructs as
accurately as possible, and we want to track if values are type-lowered,
constant-folded, outlined, or adjusted in some other way, using operations seems
like a natural choice. MLIR ops already have all the machinery needed to refer
to values in the IR, and many passes will already do the right thing with them.

Representing Source Language Constructs [¶](#representing-source-language-constructs)
-------------------------------------------------------------------------------------

The `dbg.variable` op is the key mechanism to establish a mapping between
high-level source language values and low-level values in the IR that are
transformed by the compiler. Consider the following source language pseudocode:

```
struct Req {
  data: i42,
  valid: i1,
  ready: &i1,
}
struct Resp {
  result: i42,
  done: i1,
}
module Foo {
  parameter Depth: uint;
  const Width: uint = 2**Depth;
  input req: Req;
  output resps: Resp[2];
  let x = req;
}
```

A frontend for this language could generate the following debug variables as
part of the body of module `Foo`, in order to track the structs, arrays,
parameters, constants, and local bindings present in the source language:

```
hw.module @Foo_Width12(
  in %req_data: i42,
  in %req_valid: i1,
  out req_ready: i1,
  out resps0_result: i42,
  out resps0_done: i1,
  out resps1_result: i42,
  out resps1_done: i1
) {
  // %req_ready = ...
  // %resps0_result = ...
  // %resps0_done = ...
  // %resps1_result = ...
  // %resps1_done = ...

  // parameter Depth
  %c12_i32 = hw.constant 12 : i32
  dbg.variable "Depth", %c12_i32 : i32

  // const Width
  %c4096_i32 = hw.constant 4096 : i32
  dbg.variable "Width", %c4096_i32 : i32

  // input req: Req
  %0 = dbg.struct {"data": %req_data, "valid": %req_valid, "ready": %req_ready} : i42, i1, i1
  dbg.variable "req", %0 : !dbg.struct

  // output resps: Resp[2]
  %1 = dbg.struct {"result": %resps0_result, "done": %resps0_done} : i42, i1
  %2 = dbg.struct {"result": %resps1_result, "done": %resps1_done} : i42, i1
  %3 = dbg.array [%1, %2] : !dbg.struct, !dbg.struct
  dbg.variable "resps", %3 : !dbg.array

  // let x = req
  dbg.variable "x", %0 : !dbg.struct

  hw.output %req_ready, %resps0_result, %resps0_done, %resps1_result, %resps1_done : i1, i42, i1, i42, i1
}
```

Despite the fact that the `Req` and `Resp` structs, and `Resp[2]` array were
unrolled and lowered into separate scalar values in the IR, and the `ready: &i1`
input of `Req` having been turned into a `ready: i1` output, the `dbg.variable`
op accurately tracks how the original source language values can be
reconstructed. Note also how monomorphization has turned the `Depth` parameter
and `Width` into constants in the IR, but the corresponding `dbg.variable` ops
still expose the constant values under the name `Depth` and `Width` in the debug
info.

Tracking Inlined Modules [¶](#tracking-inlined-modules)
-------------------------------------------------------

The `dbg.scope` op can be used to track debug information about inlined modules.
By default, operations such as `hw.module` in conjunction with `hw.instance`
introduce an implicit module scope. All debug operations within a module are
added to that implicit scope, unless they have an explicit `scope` operand. This
explicit scope operand can be used to group the DI of an inlined module.
Consider the following modules:

```
hw.module @Foo(in %a: i42) {
  dbg.variable "a", %a : i42
  hw.instance "bar" @Bar(x: %a: i42)
}
hw.module @Bar(in %x: i42) {
  dbg.variable "x", %x : i42
  %0 = comb.mul %x, %x : i42
  dbg.variable "squared", %0 : i42
}
```

If we inline module `Bar`, we can introduce a `dbg.scope` operation to represent
the original instance, and group all debug variables in `Bar` under this
explicit scope:

```
hw.module @Foo(in %a: i42) {
  dbg.variable "a", %a : i42
  %0 = dbg.scope "bar", "Bar"
  dbg.variable "x", %a scope %0 : i42
  %1 = comb.mul %a, %a : i42
  dbg.variable "squared", %1 scope %0 : i42
}
```

Despite the fact that the instance op no longer exists, the explicit `dbg.scope`
op models the additional levle of hierarchy that used to exist in the input.

Types [¶](#types)
-----------------

### Overview [¶](#overview)

The debug dialect does not precisely track the type of struct and array
aggregate values. Aggregates simply return the type `!dbg.struct` and
`!dbg.array`, respectively.

Extracting and emitting the debug information of a piece of IR involves looking
through debug ops to find actually emitted values that can be used to
reconstruct the source language values. Therefore the actual structure of the
debug ops is important, but their return type is not instrumental. The
distinction between struct and array types is an arbitrary choice that can be
changed easily, either by collapsing them into one aggregate type, or by more
precisely listing field/element types and array dimensions if the need arises.

### ArrayType [¶](#arraytype)

*Debug array aggregate*

Syntax: `!dbg.array`

The result of a `dbg.array` operation.

### ScopeType [¶](#scopetype)

*Debug scope*

Syntax: `!dbg.scope`

The result of a `dbg.scope` operation.

### StructType [¶](#structtype)

*Debug struct aggregate*

Syntax: `!dbg.struct`

The result of a `dbg.struct` operation.

Operations [¶](#operations)
---------------------------

### `dbg.array` (circt::debug::ArrayOp) [¶](#dbgarray-circtdebugarrayop)

*Aggregate values into an array*

Creates an array aggregate from a list of values. The first operand is
placed at array index 0. The last operand is placed at the highest array
index. The `dbg.array` operation allows for array-like source language
values to be captured in the debug info. This includes arrays, or in the
case of SystemVerilog, packed and unpacked arrays, lists, sequences, queues,
FIFOs, channels, and vectors.

See the rationale for examples and details.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands)

| Operand | Description |
| --- | --- |
| `elements` | variadic of any type |

#### Results: [¶](#results)

| Result | Description |
| --- | --- |
| `result` | debug array aggregate |

### `dbg.scope` (circt::debug::ScopeOp) [¶](#dbgscope-circtdebugscopeop)

*Define a scope for debug values*

Syntax:

```
operation ::= `dbg.scope` $instanceName `,` $moduleName (`scope` $scope^)? attr-dict
```

Creates an additional level of hierarchy in the DI, a “scope”, which can be
used to group variables and other scopes.

Operations such as `hw.module` introduce an implicit scope. All debug
operations within a module are added to that implicit scope, unless they
have an explicit `scope` operand. Providing an explicit scope can be used to
represent inlined modules.

Scopes in DI do not necessarily have to correspond to levels of a module
hierarchy. They can also be used to model things like control flow scopes,
call stacks, and other source-language concepts.

The `scope` operand of any debug dialect operation must be defined locally
by a `dbg.scope` operation. It cannot be a block argument. (This is intended
as a temporary restriction, to be lifted in the future.)

Interfaces: `InferTypeOpInterface`

#### Attributes: [¶](#attributes)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `instanceName` | ::mlir::StringAttr | string attribute |
| `moduleName` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-1)

| Operand | Description |
| --- | --- |
| `scope` | debug scope |

#### Results: [¶](#results-1)

| Result | Description |
| --- | --- |
| `result` | debug scope |

### `dbg.struct` (circt::debug::StructOp) [¶](#dbgstruct-circtdebugstructop)

*Aggregate values into a struct*

Creates a struct aggregate from a list of names and values. The `dbg.struct`
operation allows for struct-like source language values to be captured in
the debug info. This includes structs, unions, bidirectional bundles,
interfaces, classes, and other similar structures.

See the rationale for examples and details.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `names` | ::mlir::ArrayAttr | string array attribute |

#### Operands: [¶](#operands-2)

| Operand | Description |
| --- | --- |
| `fields` | variadic of any type |

#### Results: [¶](#results-2)

| Result | Description |
| --- | --- |
| `result` | debug struct aggregate |

### `dbg.variable` (circt::debug::VariableOp) [¶](#dbgvariable-circtdebugvariableop)

*A named value to be captured in debug info*

Syntax:

```
operation ::= `dbg.variable` $name `,` $value (`scope` $scope^)? attr-dict `:` type($value)
```

Marks a value to be tracked in DI under the given name. The `dbg.variable`
operation is useful to represent named values in a source language. For
example, ports, constants, parameters, variables, nodes, or name aliases can
all be represented as a variable. In combination with `dbg.array` and
`dbg.struct`, complex aggregate source language values can be described and
reconstituted from individual IR values. The `dbg.variable` operation acts
as a tracker that follows the evolution of its assigned value throughout the
compiler’s pass pipelines. The debug info analysis uses this op to populate
a module’s scope with named source language values, and to establish how
these source language values can be reconstituted from the actual IR values
present at the end of compilation.

See the rationale for examples and details. See the `dbg.scope` operation
for additional details on how to use the `scope` operand.

#### Attributes: [¶](#attributes-2)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-3)

| Operand | Description |
| --- | --- |
| `value` | any type |
| `scope` | debug scope |

 [Prev - SystemC Dialect Rationale](https://circt.llvm.org/docs/Dialects/SystemC/RationaleSystemC/ "SystemC Dialect Rationale")
[Next - Interop Dialect](https://circt.llvm.org/docs/Dialects/Interop/ "Interop Dialect") 

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