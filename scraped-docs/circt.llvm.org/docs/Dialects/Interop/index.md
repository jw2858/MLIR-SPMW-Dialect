Interop Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

Interop Dialect
===============

*Provides interoperability between backends and tools*

This dialect defines the `interop` dialect which defines operations and
interfaces necessary to provide interoperability between backends and
and external tools without the need of writing custom pairwise interop
solutions.

* [Operations](#operations)
  + [`interop.procedural.alloc` (::circt::interop::ProceduralAllocOp)](#interopproceduralalloc-circtinteropproceduralallocop)
  + [`interop.procedural.dealloc` (::circt::interop::ProceduralDeallocOp)](#interopproceduraldealloc-circtinteropproceduraldeallocop)
  + [`interop.procedural.init` (::circt::interop::ProceduralInitOp)](#interopproceduralinit-circtinteropproceduralinitop)
  + [`interop.procedural.update` (::circt::interop::ProceduralUpdateOp)](#interopproceduralupdate-circtinteropproceduralupdateop)
  + [`interop.return` (::circt::interop::ReturnOp)](#interopreturn-circtinteropreturnop)
* [Enums](#enums)
  + [InteropMechanism](#interopmechanism)

Operations
----------

### `interop.procedural.alloc` (::circt::interop::ProceduralAllocOp)

*Represents persistent state to be allocated*

Syntax:

```
operation ::= `interop.procedural.alloc` $interopMechanism ( `:` qualified(type($states))^ )? attr-dict
```

The `interop.procedural.alloc` operation returns a variadic list of values
that represent persistent state, i.e., state that has to persist across
multiple executions of the `interop.procedural.update` operation.
For example, it can be lowered to C++ class fields that are persistent
across multiple calls of a member function, or to global simulator state
that persists over simulation cycles, etc.

Additionally, it has an attribute that specifies the interop mechanism under
which the state types are valid. This is necessary to allow bridging patterns
to map the types to valid types in the other interop mechanism, e.g., to an
opaque pointer, if it does not support the same types.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `interopMechanism` | ::InteropMechanismAttr | interface through which interoperability is achieved |

#### Results:

| Result | Description |
| --- | --- |
| `states` | variadic of any type |

### `interop.procedural.dealloc` (::circt::interop::ProceduralDeallocOp)

*Performs some deallocation logic before the state is released*

Syntax:

```
operation ::= `interop.procedural.dealloc` $interopMechanism ( $states^ `:` qualified(type($states)) )?
              attr-dict-with-keyword $deallocRegion
```

The `interop.procedural.dealloc` operation shall be executed right before
the state requested by the `interop.procedural.alloc` operation is
released. This allows the instance to do some cleanup, e.g., when the state
type was a pointer and the instance performed some `malloc`.

Structurally the operation is the same as the `interop.procedural.update`
operation, but without input and output values. The state is also passed
by value.

Traits: `NoTerminator`, `SingleBlock`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `interopMechanism` | ::InteropMechanismAttr | interface through which interoperability is achieved |

#### Operands:

| Operand | Description |
| --- | --- |
| `states` | variadic of any type |

### `interop.procedural.init` (::circt::interop::ProceduralInitOp)

*Computes the initial values for the allocated state*

Syntax:

```
operation ::= `interop.procedural.init` $interopMechanism ( $states^ `:` qualified(type($states)) )?
              attr-dict-with-keyword $initRegion
```

The `interop.procedural.init` operation takes the variadic list of states
from the `interop.procedural.alloc` operation as operands and has a body
with a `interop.return` operation that has a variadic list of operands that
matches the types of the states and represent the initial values to be
assigned to the state values.
The assignment will be inserted by the container-side lowering of the
interop operations.
The operation also has an interop mechanism attribute to allow bridging
patterns to map the types to valid types in another interop mechanism
and to wrap the operations in the body in a way to make them executable
in the other interop mechanism, e.g., wrap them in a `extern "C"` function
to make it callable from C or LLVM IR.

Traits: `SingleBlock`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `interopMechanism` | ::InteropMechanismAttr | interface through which interoperability is achieved |

#### Operands:

| Operand | Description |
| --- | --- |
| `states` | variadic of any type |

### `interop.procedural.update` (::circt::interop::ProceduralUpdateOp)

*Takes some persistent state and inputs to compute some results*

Syntax:

```
operation ::= `interop.procedural.update` $interopMechanism ( ` ` `[` $states^ `]` )? ( `(` $inputs^ `)` )? `:`
              (`[` qualified(type($states))^ `]`)? functional-type($inputs, $outputs)
              attr-dict-with-keyword $updateRegion
```

The `interop.procedural.update` operation has an interop mechanism attribute
to allow bridging patterns to map the types to valid types in another
interop mechanism and to wrap the operations in the body in a way to make
them executable using the other interop mechanism.

It takes the state values returned by the `interop.procedural.alloc`as
operands and passes them on to the body via block arguments using
pass-by-value semantics. In addition to the state values, it also takes a
variadic list of inputs and also passes them on to the body.
The `interop.return` inside the body then returns the result values after
doing some computation inside the body.

If the state needs to be mutated, it has to be a pointer type.

Traits: `AttrSizedOperandSegments`, `SingleBlock`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `interopMechanism` | ::InteropMechanismAttr | interface through which interoperability is achieved |

#### Operands:

| Operand | Description |
| --- | --- |
| `states` | variadic of any type |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `interop.return` (::circt::interop::ReturnOp)

*A return operation*

Syntax:

```
operation ::= `interop.return` attr-dict ($returnValues^ `:` type($returnValues))?
```

The `interop.return` operation lists the computed initial values when
inside the `init` operation or the computed results when inside the
`update` operation.

Traits: `HasParent<ProceduralInitOp, ProceduralUpdateOp>`, `ReturnLike`, `Terminator`

Interfaces: `RegionBranchTerminatorOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `returnValues` | variadic of any type |

Enums
-----

### InteropMechanism

*Interface through which interoperability is achieved*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| CFFI | `0` | cffi |
| CPP | `1` | cpp |

Interop Dialect Docs
--------------------

* [Interoperability Dialect Rationale](https://circt.llvm.org/docs/Dialects/Interop/RationaleInterop/)

 [Prev - Debug Dialect](https://circt.llvm.org/docs/Dialects/Debug/ "Debug Dialect")
[Next - Interoperability Dialect Rationale](https://circt.llvm.org/docs/Dialects/Interop/RationaleInterop/ "Interoperability Dialect Rationale") 

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