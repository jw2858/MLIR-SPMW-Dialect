'ssp' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'ssp' Dialect
=============

*Static scheduling problem instances and solutions.*

A dialect to abstractly represent instances and solutions of
[static
scheduling problems](https://circt.llvm.org/docs/Scheduling/), intended as
an import/export format for testing, benchmarking, and rapid prototyping of
problem definitions and algorithm implementations. See the
[rationale](https://circt.llvm.org/docs/Dialects/SSP/RationaleSSP/) for more
information.

* [Operations](#operations)
  + [`ssp.graph` (::circt::ssp::DependenceGraphOp)](#sspgraph-circtsspdependencegraphop)
  + [`ssp.instance` (::circt::ssp::InstanceOp)](#sspinstance-circtsspinstanceop)
  + [`ssp.library` (::circt::ssp::OperatorLibraryOp)](#ssplibrary-circtsspoperatorlibraryop)
  + [`ssp.operation` (::circt::ssp::OperationOp)](#sspoperation-circtsspoperationop)
  + [`ssp.operator_type` (::circt::ssp::OperatorTypeOp)](#sspoperator_type-circtsspoperatortypeop)
  + [`ssp.resource` (::circt::ssp::ResourceLibraryOp)](#sspresource-circtsspresourcelibraryop)
  + [`ssp.resource_type` (::circt::ssp::ResourceTypeOp)](#sspresource_type-circtsspresourcetypeop)
* [Attributes](#attributes-6)
  + [DependenceAttr](#dependenceattr)
  + [DistanceAttr](#distanceattr)
  + [IncomingDelayAttr](#incomingdelayattr)
  + [InitiationIntervalAttr](#initiationintervalattr)
  + [LatencyAttr](#latencyattr)
  + [LimitAttr](#limitattr)
  + [LinkedOperatorTypeAttr](#linkedoperatortypeattr)
  + [LinkedResourceTypesAttr](#linkedresourcetypesattr)
  + [OutgoingDelayAttr](#outgoingdelayattr)
  + [StartTimeInCycleAttr](#starttimeincycleattr)
  + [StartTimeAttr](#starttimeattr)

Operations
----------

### `ssp.graph` (::circt::ssp::DependenceGraphOp)

*Container for (scheduling) operations.*

Syntax:

```
operation ::= `ssp.graph` $body attr-dict
```

The dependence graph is spanned by `OperationOp`s (vertices) and a
combination of MLIR value uses and symbol references (edges).

Traits: `HasOnlyGraphRegion`, `HasParent<InstanceOp>`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `OpAsmOpInterface`

### `ssp.instance` (::circt::ssp::InstanceOp)

*Instance of a static scheduling problem.*

Syntax:

```
operation ::= `ssp.instance` ($sym_name^)? `of` $problemName custom<SSPProperties>($sspProperties) $body attr-dict
```

This operation represents an instance of a static scheduling problem,
comprised of an operator library (`OperatorLibraryOp`, a container for
`OperatorTypeOp`s), an resource library (`ResourceLibraryOp`, a container
for `ResourceTypeOp`s), and the dependence graph (`DependenceGraphOp`, a
container for `OperationOp`s). The instance and its components (operations,
operator types, resource types and dependences) can carry properties,
i.e. special MLIR attributes inheriting from the TableGen classes in
`PropertyBase.td`. The `ssp` dialect provides attribute definitions (and
short-form pretty-printing) for CIRCT’s built-in scheduling problems.

**Example**

```
ssp.instance @canis14_fig2 of "ModuloProblem" [II<3>] {
  library {
    operator_type @Memory [latency<1>]
    operator_type @Add [latency<1>]
  }
  resource {
    resource_type @ReadPort [limit<1>]
    resource_type @WritePort [limit<1>]
  }
  graph {
    %0 = operation<@Memory> @load_A(@store_A [dist<1>]) uses[@ReadPort] [t<2>]
    %1 = operation<@Memory> @load_B() uses[@ReadPort] [t<0>]
    %2 = operation<@Add> @add(%0, %1) [t<3>]  // no `resource_type` needed
    operation<@Memory> @store_A(%2) uses[@WritePort] [t<4>]
  }
}
```

Traits: `IsolatedFromAbove`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `OpAsmOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `problemName` | ::mlir::StringAttr | string attribute |
| `sspProperties` | ::mlir::ArrayAttr | array attribute |

### `ssp.library` (::circt::ssp::OperatorLibraryOp)

*Container for operator types.*

Syntax:

```
operation ::= `ssp.library` ($sym_name^)? $body attr-dict
```

The operator library abstracts the characteristics of the target
architecture/IR (onto which the source graph is scheduled), represented by
the individual `OperatorTypeOp`s. This operation may be used outside of an
`InstanceOp`.

Traits: `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `OpAsmOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `ssp.operation` (::circt::ssp::OperationOp)

*Vertex and incoming edges in the dependence graph.*

This MLIR operation represents an operation (in the terminology of CIRCT’s
scheduling infra) in a scheduling problem, or in other words, a vertex in
the surrounding instance’s dependence graph. In addition, it also encodes
the operation’s incoming dependences. In order to faithfully reproduce the
internal modeling in the scheduling infrastructure, these dependence edges
are either encoded as MLIR operands (def-use dependences) or symbol
references (auxiliary dependences). To that end, `OperationOp`s can
optionally be named, and accept/produce an arbitrary number of
operands/results. The operation and the incoming dependences can carry
properties.

The `linkedOperatorType` and `linkedResourceType` property in the root
`Problem` class are central to the problem models, because it links operations
to their properties in the target IR. Therefore, the referenced operator/resource
type symbol is parsed/printed right after the operation keyword in the custom
assembly syntax. Flat symbol references are resolved by name in the surrounding
instance’s operator/resource library. Nested references can point to arbitrary
operator/resource libraries.

**Examples**

```
// unnamed, only def-use dependences
%2 = operation<@Add>(%0, %1)

// unnamed, multiple results
%5:2 = operation<@Div>(%3, %4) // multiple results

// named, mix of def-use and auxiliary dependences
operation<@MemAccess> @store_A(%2, @store_B, @load_A) uses[@MemPort]

// dependence properties
operation<@Barrier>(%2 [dist<1>], %5#1, @store_A [dist<3>])

// operator type in stand-alone library
%7 = operation<@MathLib::@Sqrt>(%6)
```

Traits: `HasParent<DependenceGraphOp>`

Interfaces: `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `dependences` | ::mlir::ArrayAttr | dependence array attribute |
| `sspProperties` | ::mlir::ArrayAttr | array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `operands` | variadic of none type |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of none type |

### `ssp.operator_type` (::circt::ssp::OperatorTypeOp)

*Element of the target architecture/IR.*

Syntax:

```
operation ::= `ssp.operator_type` $sym_name custom<SSPProperties>($sspProperties) attr-dict
```

This operation represents an operator type, which can be augmented with a
set of problem-specific properties, and is identified through a unique name.

**Example**

```
operator_type @MemPort [latency<1>]
```

Traits: `HasParent<OperatorLibraryOp>`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `sspProperties` | ::mlir::ArrayAttr | array attribute |

### `ssp.resource` (::circt::ssp::ResourceLibraryOp)

*Container for resource types.*

Syntax:

```
operation ::= `ssp.resource` ($sym_name^)? $body attr-dict
```

The resource library represents different kinds of resource of desired
usage on the target architecture/IR. Each resource will be represented by
the individual `ResourceTypeOp`s. An `OperationOp` can be associated with
zero, one, or multiple resources. This operation may be used outside of
an `InstanceOp` so different problems can share the same resource constraints.

Traits: `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `OpAsmOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `ssp.resource_type` (::circt::ssp::ResourceTypeOp)

*Resource of desired usage on the target architecture/IR.*

Syntax:

```
operation ::= `ssp.resource_type` $sym_name custom<SSPProperties>($sspProperties) attr-dict
```

This resource represents a resource type, which can be augmented with a
set of problem-specific properties, and is identified through a unique name.

**Example**

```
resource_type @MemPort [limit<1>]
```

Traits: `HasParent<ResourceLibraryOp>`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `sspProperties` | ::mlir::ArrayAttr | array attribute |

Attributes
----------

### DependenceAttr

*Internal representation of dependence edges.*

Syntax:

```
#ssp.dependence<
  unsigned,   # operandIdx
  ::mlir::FlatSymbolRefAttr,   # sourceRef
  ::mlir::ArrayAttr   # properties
>
```

An attribute to uniformly model def-use and auxiliary
[dependences](https://circt.llvm.org/docs/Scheduling/#components) as well as
to attach
[properties](https://circt.llvm.org/docs/Scheduling/#properties)
to them. This attribute is an implementation detail of the `ssp.OperationOp`
and as such is supposed to be hidden by the custom parser/printer.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| operandIdx | `unsigned` |  |
| sourceRef | `::mlir::FlatSymbolRefAttr` |  |
| properties | `::mlir::ArrayAttr` |  |

### DistanceAttr

*Models the `Distance` property in `::circt::scheduling::CyclicProblem`.*

Syntax:

```
#ssp.dist<
  unsigned   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `unsigned` |  |

### IncomingDelayAttr

*Models the `IncomingDelay` property in `::circt::scheduling::ChainingProblem`.*

Syntax:

```
#ssp.incDelay<
  ::mlir::FloatAttr   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `::mlir::FloatAttr` |  |

### InitiationIntervalAttr

*Models the `InitiationInterval` property in `::circt::scheduling::CyclicProblem`.*

Syntax:

```
#ssp.II<
  unsigned   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `unsigned` |  |

### LatencyAttr

*Models the `Latency` property in `::circt::scheduling::Problem`.*

Syntax:

```
#ssp.latency<
  unsigned   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `unsigned` |  |

### LimitAttr

*Models the `Limit` property in `::circt::scheduling::SharedOperatorsProblem`.*

Syntax:

```
#ssp.limit<
  unsigned   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `unsigned` |  |

### LinkedOperatorTypeAttr

*Models the `LinkedOperatorType` property in `::circt::scheduling::Problem`.*

Syntax:

```
#ssp.opr<
  ::mlir::SymbolRefAttr   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `::mlir::SymbolRefAttr` |  |

### LinkedResourceTypesAttr

*Models the `LinkedResourceTypes` property in `::circt::scheduling::Problem`.*

Syntax:

```
#ssp.rsrcs<
  ::mlir::ArrayAttr   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `::mlir::ArrayAttr` |  |

### OutgoingDelayAttr

*Models the `OutgoingDelay` property in `::circt::scheduling::ChainingProblem`.*

Syntax:

```
#ssp.outDelay<
  ::mlir::FloatAttr   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `::mlir::FloatAttr` |  |

### StartTimeInCycleAttr

*Models the `StartTimeInCycle` property in `::circt::scheduling::ChainingProblem`.*

Syntax:

```
#ssp.z<
  ::mlir::FloatAttr   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `::mlir::FloatAttr` |  |

### StartTimeAttr

*Models the `StartTime` property in `::circt::scheduling::Problem`.*

Syntax:

```
#ssp.t<
  unsigned   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `unsigned` |  |

'ssp' Dialect Docs
------------------

* [SSP Dialect Rationale](https://circt.llvm.org/docs/Dialects/SSP/RationaleSSP/)

 [Prev - Seq(uential) Dialect Rationale](https://circt.llvm.org/docs/Dialects/Seq/RationaleSeq/ "Seq(uential) Dialect Rationale")
[Next - SSP Dialect Rationale](https://circt.llvm.org/docs/Dialects/SSP/RationaleSSP/ "SSP Dialect Rationale") 

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