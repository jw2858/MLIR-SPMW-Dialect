'hwarith' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'hwarith' Dialect
=================

*Types and operations for the HWArith dialect*

This dialect defines the `HWArith` dialect, modeling bit-width aware
arithmetic operations.

* [Operations](#operations)
  + [`hwarith.add` (::circt::hwarith::AddOp)](#hwarithadd-circthwarithaddop)
  + [`hwarith.cast` (::circt::hwarith::CastOp)](#hwarithcast-circthwarithcastop)
  + [`hwarith.constant` (::circt::hwarith::ConstantOp)](#hwarithconstant-circthwarithconstantop)
  + [`hwarith.div` (::circt::hwarith::DivOp)](#hwarithdiv-circthwarithdivop)
  + [`hwarith.icmp` (::circt::hwarith::ICmpOp)](#hwarithicmp-circthwarithicmpop)
  + [`hwarith.mul` (::circt::hwarith::MulOp)](#hwarithmul-circthwarithmulop)
  + [`hwarith.sub` (::circt::hwarith::SubOp)](#hwarithsub-circthwarithsubop)
* [Type constraints](#type-constraints)
  + [an arbitrary precision integer with signedness semantics](#an-arbitrary-precision-integer-with-signedness-semantics)
* [Enums](#enums)
  + [ICmpPredicate](#icmppredicate)

Operations
----------

### `hwarith.add` (::circt::hwarith::AddOp)

*Bitwidth-aware integer addition.*

Syntax:

```
operation ::= `hwarith.add` $inputs attr-dict `:` functional-type($inputs, $result)
```

The `add` operation takes two operands and returns one result. The result
type is inferred from the operand types, which may be signed or unsigned
scalar integer types of arbitrary bitwidth.

| LHS type | RHS type | Result type |
| --- | --- | --- |
| `ui<a>` | `ui<b>` | `ui<r>`, *r* = max(*a*, *b*) + 1 |
| `si<a>` | `si<b>` | `si<r>`, *r* = max(*a*, *b*) + 1 |
| `ui<a>` | `si<b>` | `si<r>`, *r* = *a* + 2 **if** *a* ≥ *b* |
|  |  | `si<r>`, *r* = *b* + 1 **if** *a* < *b* |
| `si<a>` | `ui<b>` | Same as `ui<b> + si<a>` |

Examples:

```
%0 = hwarith.add %10, %11 : (ui3, ui4) -> ui5
%1 = hwarith.add %12, %13 : (si3, si3) -> si4
%2 = hwarith.add %14, %15 : (ui3, si4) -> si5
%3 = hwarith.add %16, %17 : (si4, ui6) -> si8
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of an arbitrary precision integer with signedness semantics |

#### Results:

| Result | Description |
| --- | --- |
| `result` | an arbitrary precision integer with signedness semantics |

### `hwarith.cast` (::circt::hwarith::CastOp)

*Signedness-aware cast.*

Syntax:

```
operation ::= `hwarith.cast` $in attr-dict `:` functional-type($in, $out)
```

The `cast` operation takes one operand and returns one result. Both, the
result type and the operand type can be of any scalar integer type with
arbitrary bitwidth. However, at least one of them needs to be a
HWArithIntegerType.

| Input type | Result type | Behavior |
| --- | --- | --- |
| `ui<a>` | `ui<b>`/`si<b>`/`i<b>` | zero-extension **if** *b* ≥ *a* |
|  |  | truncation **if** *b* < *a* |
| `si<a>` | `ui<b>`/`si<b>`/`i<b>` | sign-extension **if** *b* ≥ *a* |
|  |  | truncation **if** *b* < *a* |
| `i<a>` | `ui<b>`/`si<b>` | truncation **if** *b* **≤** *a* |
| `i<a>` | `ui<b>`/`si<b>` | prohibited† **if** *b* > *a* |

†) prohibited because of the ambiguity whether a sign or a zero extension
is required.

Examples:

```
%0 = hwarith.cast %10 : (ui3) -> si5
%1 = hwarith.cast %11 : (si3) -> si4
%2 = hwarith.cast %12 : (si7) -> ui4
%3 = hwarith.cast %13 : (i7) -> si5
%3 = hwarith.cast %13 : (si14) -> i4
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `in` | integer |

#### Results:

| Result | Description |
| --- | --- |
| `out` | integer |

### `hwarith.constant` (::circt::hwarith::ConstantOp)

*Produce a constant value*

The constant operation produces a sign-aware constant value.

```
  %result = hwarith.constant 42 : t1
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`, `FirstAttrDerivedResultType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `rawValue` | ::mlir::IntegerAttr | arbitrary integer attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | an arbitrary precision integer with signedness semantics |

### `hwarith.div` (::circt::hwarith::DivOp)

*Bitwidth-aware integer division.*

Syntax:

```
operation ::= `hwarith.div` $inputs attr-dict `:` functional-type($inputs, $result)
```

The `div` operation takes two operands and returns one result. The result
type is inferred from the operand types, which may be signed or unsigned
scalar integer types of arbitrary bitwidth.

| LHS type | RHS type | Result type |
| --- | --- | --- |
| `ui<a>` | `ui<b>` | `ui<r>`, *r* = *a* |
| `si<a>` | `si<b>` | `si<r>`, *r* = *a* + 1 |
| `ui<a>` | `si<b>` | `si<r>`, *r* = *a* + 1 |
| `si<a>` | `ui<b>` | `si<r>`, *r* = *a* |

Examples:

```
%0 = hwarith.div %10, %11 : (ui3, ui4) -> ui3
%1 = hwarith.div %12, %13 : (si3, si3) -> si4
%2 = hwarith.div %14, %15 : (ui3, si4) -> si4
%3 = hwarith.div %16, %17 : (si4, ui6) -> si4
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of an arbitrary precision integer with signedness semantics |

#### Results:

| Result | Description |
| --- | --- |
| `result` | an arbitrary precision integer with signedness semantics |

### `hwarith.icmp` (::circt::hwarith::ICmpOp)

*Sign- and bitwidth-aware integer comparison.*

Syntax:

```
operation ::= `hwarith.icmp` $predicate $lhs `,` $rhs  attr-dict `:` type($lhs) `,` type($rhs)
```

The `icmp` operation compares two integers using a predicate. If the
predicate is true, returns 1, otherwise returns 0. This operation always
returns a one bit wide result of type `i1`. Both operand types may be
signed or unsigned scalar integer types of arbitrary bitwidth.

| LHS type | RHS type | Comparison type | Result type |
| --- | --- | --- | --- |
| `ui<a>` | `ui<b>` | `ui<r>`, *r* = max(*a*, *b*) | `i1` |
| `si<a>` | `si<b>` | `si<r>`, *r* = max(*a*, *b*) | `i1` |
| `ui<a>` | `si<b>` | `si<r>`, *r* = *a* + 1 **if** *a* ≥ *b* | `i1` |
|  |  | `si<r>`, *r* = *b* **if** *a* < *b* | `i1` |
| `si<a>` | `ui<b>` | Same as `ui<b> si<a>` | `i1` |

Examples:

```
%0 = hwarith.icmp lt %10, %11 : ui5, ui6
%1 = hwarith.icmp lt %12, %13 : si3, si4
%2 = hwarith.icmp lt %12, %11 : si3, ui6
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `predicate` | circt::hwarith::ICmpPredicateAttr | hwarith.icmp comparison predicate |

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | an arbitrary precision integer with signedness semantics |
| `rhs` | an arbitrary precision integer with signedness semantics |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit signless integer |

### `hwarith.mul` (::circt::hwarith::MulOp)

*Bitwidth-aware integer multiplication.*

Syntax:

```
operation ::= `hwarith.mul` $inputs attr-dict `:` functional-type($inputs, $result)
```

The `mul` operation takes two operands and returns one result. The result
type is inferred from the operand types, which may be signed or unsigned
scalar integer types of arbitrary bitwidth.

| LHS type | RHS type | Result type |
| --- | --- | --- |
| `ui<a>` | `ui<b>` | `ui<r>`, *r* = *a* + *b* |
| `si<a>` | `si<b>` | `si<r>`, *r* = *a* + *b* |
| `ui<a>` | `si<b>` | `si<r>`, *r* = *a* + *b* |
| `si<a>` | `ui<b>` | `si<r>`, *r* = *a* + *b* |

Examples:

```
%0 = hwarith.mul %10, %11 : (ui3, ui4) -> ui7
%1 = hwarith.mul %12, %13 : (si3, si3) -> si6
%2 = hwarith.mul %14, %15 : (si3, ui5) -> si8
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of an arbitrary precision integer with signedness semantics |

#### Results:

| Result | Description |
| --- | --- |
| `result` | an arbitrary precision integer with signedness semantics |

### `hwarith.sub` (::circt::hwarith::SubOp)

*Bitwidth-aware integer subtraction.*

Syntax:

```
operation ::= `hwarith.sub` $inputs attr-dict `:` functional-type($inputs, $result)
```

The `sub` operation takes two operands and returns one result. The result
type is inferred from the operand types, which may be signed or unsigned
scalar integer types of arbitrary bitwidth.

| LHS type | RHS type | Result type |
| --- | --- | --- |
| `ui<a>` | `ui<b>` | `si<r>`, *r* = max(*a*, *b*) + 1 |
| `si<a>` | `si<b>` | `si<r>`, *r* = max(*a*, *b*) + 1 |
| `ui<a>` | `si<b>` | `si<r>`, *r* = *a* + 2 **if** *a* ≥ *b* |
|  |  | `si<r>`, *r* = *b* + 1 **if** *a* < *b* |
| `si<a>` | `ui<b>` | Same as `ui<b> - si<a>` |

Examples:

```
%0 = hwarith.sub %10, %11 : (ui3, ui4) -> si5
%1 = hwarith.sub %12, %13 : (si3, si3) -> si4
%2 = hwarith.sub %14, %15 : (ui3, si4) -> si5
%3 = hwarith.sub %16, %17 : (si4, ui6) -> si8
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of an arbitrary precision integer with signedness semantics |

#### Results:

| Result | Description |
| --- | --- |
| `result` | an arbitrary precision integer with signedness semantics |

Type constraints
----------------

### an arbitrary precision integer with signedness semantics

Enums
-----

### ICmpPredicate

*Hwarith.icmp comparison predicate*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| eq | `0` | eq |
| ne | `1` | ne |
| lt | `2` | lt |
| ge | `3` | ge |
| le | `4` | le |
| gt | `5` | gt |

'hwarith' Dialect Docs
----------------------

* [HW Arith Dialect Rationale](https://circt.llvm.org/docs/Dialects/HWArith/RationaleHWArith/)

 [Prev - HW Dialect Rationale](https://circt.llvm.org/docs/Dialects/HW/RationaleHW/ "HW Dialect Rationale")
[Next - HW Arith Dialect Rationale](https://circt.llvm.org/docs/Dialects/HWArith/RationaleHWArith/ "HW Arith Dialect Rationale") 

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