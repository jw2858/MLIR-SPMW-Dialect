'dc' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'dc' Dialect
============

*Dynamic Control*

This is the `dc` dialect, used to represent dynamic control constructs with
handshaking semantics.

* [Operations](#operations)
  + [`dc.branch` (circt::dc::BranchOp)](#dcbranch-circtdcbranchop)
  + [`dc.buffer` (circt::dc::BufferOp)](#dcbuffer-circtdcbufferop)
  + [`dc.fork` (circt::dc::ForkOp)](#dcfork-circtdcforkop)
  + [`dc.from_esi` (circt::dc::FromESIOp)](#dcfrom_esi-circtdcfromesiop)
  + [`dc.join` (circt::dc::JoinOp)](#dcjoin-circtdcjoinop)
  + [`dc.merge` (circt::dc::MergeOp)](#dcmerge-circtdcmergeop)
  + [`dc.pack` (circt::dc::PackOp)](#dcpack-circtdcpackop)
  + [`dc.select` (circt::dc::SelectOp)](#dcselect-circtdcselectop)
  + [`dc.sink` (circt::dc::SinkOp)](#dcsink-circtdcsinkop)
  + [`dc.source` (circt::dc::SourceOp)](#dcsource-circtdcsourceop)
  + [`dc.to_esi` (circt::dc::ToESIOp)](#dcto_esi-circtdctoesiop)
  + [`dc.unpack` (circt::dc::UnpackOp)](#dcunpack-circtdcunpackop)
* [Types](#types)
  + [TokenType](#tokentype)
  + [ValueType](#valuetype)

Operations
----------

### `dc.branch` (circt::dc::BranchOp)

*Branch operation*

Syntax:

```
operation ::= `dc.branch` $condition attr-dict
```

The incoming select token is propagated to the selected output based on
the value of the condition.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `condition` | must be a !dc.value type |

#### Results:

| Result | Description |
| --- | --- |
| `trueToken` |  |
| `falseToken` |  |

### `dc.buffer` (circt::dc::BufferOp)

*Buffer operation*

Syntax:

```
operation ::= `dc.buffer` `[` $size `]` $input ($initValues^)? attr-dict `:` type($input)
```

The buffer operation may buffer a `dc.value` or `dc.token` typed SSA value.
In practice, this provides a mechanism to buffer data-side values in a
control-sensitive manner.

Example:

```
%value_out = dc.buffer [2] %value : !dc.value<i32, i1, i4>
```

**Hardware/CIRCT context note**: buffers have no dialect-side notion of
cycles/stages/implementation. It is up to the generating pass to interpret
buffer semantics - some may want to add attributes to a single buffer, some
may want to stagger `dc.buffer`s sequentially.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `size` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |
| `initValues` | ::mlir::ArrayAttr | array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | must be a !dc.value or !dc.token type |

#### Results:

| Result | Description |
| --- | --- |
| `output` | must be a !dc.value or !dc.token type |

### `dc.fork` (circt::dc::ForkOp)

*Splits the incoming token into multiple outgoing tokens*

This operator splits the incoming token into multiple outgoing tokens.

Example:

```
%0, %1 = dc.fork [2] %a : !dc.token, !dc.token
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `token` |  |

#### Results:

| Result | Description |
| --- | --- |
| `outputs` | variadic of |

### `dc.from_esi` (circt::dc::FromESIOp)

*Convert an ESI-typed value to a DC-typed value*

Syntax:

```
operation ::= `dc.from_esi` $input attr-dict `:` type($input)
```

Convert an ESI channel to a `dc.token/dc.value`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `output` | must be a !dc.value or !dc.token type |

### `dc.join` (circt::dc::JoinOp)

*Synchronizes the incoming tokens with the outgoing token*

Syntax:

```
operation ::= `dc.join` $tokens attr-dict
```

This operator synchronizes all incoming tokens. Synchronization implies applying
join semantics in between all in- and output ports.

Example:

```
%0 = dc.join %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `tokens` | variadic of |

#### Results:

| Result | Description |
| --- | --- |
| `output` |  |

### `dc.merge` (circt::dc::MergeOp)

*Merge operation*

Syntax:

```
operation ::= `dc.merge` $first `,` $second attr-dict
```

Select one of the incoming tokens and emits an output stating which token
was selected. If multiple tokens are ready to transact at the same time,
the tokens are selected with priority, from first to last (i.e. left to right
in the IR). This property ensures deterministic behavior.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `first` |  |
| `second` |  |

#### Results:

| Result | Description |
| --- | --- |
| `output` | must be a !dc.value type |

### `dc.pack` (circt::dc::PackOp)

*Pack operation*

Syntax:

```
operation ::= `dc.pack` $token `,` $input attr-dict `:` type($input)
```

An operation which packs together a !dc.token value with some other
value.

Typically, a `dc.pack` op will be used to facilitate data-dependent
control flow, wherein a `dc.value<i1>` is to be generated as a select
signal for either a `dc.branch` or `dc.select` operation.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `token` |  |
| `input` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `output` |  |

### `dc.select` (circt::dc::SelectOp)

*Select operation*

Syntax:

```
operation ::= `dc.select` $condition `,` $trueToken `,` $falseToken attr-dict
```

An input token is selected based on the value of the incoming select
signal, and propagated to the single output. Only the condition value,
the selected input, and the output will be transacted.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `condition` | must be a !dc.value type |
| `trueToken` |  |
| `falseToken` |  |

#### Results:

| Result | Description |
| --- | --- |
| `output` |  |

### `dc.sink` (circt::dc::SinkOp)

*Sink operation*

Syntax:

```
operation ::= `dc.sink` $token attr-dict
```

The sink operation will always accept any incoming tokens, and
discard them.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `token` |  |

### `dc.source` (circt::dc::SourceOp)

*Source operation*

Syntax:

```
operation ::= `dc.source` attr-dict
```

The source operation will always produce a token.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `output` |  |

### `dc.to_esi` (circt::dc::ToESIOp)

*Convert a DC-typed value to an ESI-typed value*

Syntax:

```
operation ::= `dc.to_esi` $input attr-dict `:` type($input)
```

Convert a `dc.token/dc.value` to an ESI channel.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | must be a !dc.value or !dc.token type |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `dc.unpack` (circt::dc::UnpackOp)

*Unpack operation*

Syntax:

```
operation ::= `dc.unpack` $input attr-dict `:` qualified(type($input))
```

An operation which unpacks a !dc.value value into a !dc.token value
and its constituent values.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` |  |

#### Results:

| Result | Description |
| --- | --- |
| `token` |  |
| `output` | any type |

Types
-----

### TokenType

Syntax: `!dc.token`

A `!dc.token`-typed value represents a control value with handshake semantics.

### ValueType

Syntax:

```
!dc.value<
  Type   # innerType
>
```

A `!dc.value`-typed value represents a value which is wrapped with
`!dc.token` semantics. This type is used to attach control semantics to
values. The inner value may be of any type.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| innerType | `Type` |  |

'dc' Dialect Docs
-----------------

* [DC Dialect Rationale](https://circt.llvm.org/docs/Dialects/DC/RationaleDC/)

 [Prev - 'datapath' Dialect Rationale](https://circt.llvm.org/docs/Dialects/Datapath/RationaleDatapath/ "'datapath' Dialect Rationale")
[Next - DC Dialect Rationale](https://circt.llvm.org/docs/Dialects/DC/RationaleDC/ "DC Dialect Rationale") 

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