'datapath' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'datapath' Dialect
==================

*Datapath Optimization*

This dialect defines the `datapath` dialect, which is about the efficient
construction of arithmetic circuits via low-level datapath optimizations.

* [Operations](#operations)
  + [`datapath.compress` (::circt::datapath::CompressOp)](#datapathcompress-circtdatapathcompressop)
  + [`datapath.partial_product` (::circt::datapath::PartialProductOp)](#datapathpartial_product-circtdatapathpartialproductop)
  + [`datapath.pos_partial_product` (::circt::datapath::PosPartialProductOp)](#datapathpos_partial_product-circtdatapathpospartialproductop)

Operations
----------

### `datapath.compress` (::circt::datapath::CompressOp)

*Reduce a set of bitvectors to a carry-save representation*

Syntax:

```
operation ::= `datapath.compress` $inputs attr-dict `:` custom<CompressFormat>(type($inputs), type($results))
```

Reduce an array of bitvectors to a smaller set of bitvectors (at least 2).
A compressor tree sums multiple bitvectors (often partial products in
multipliers or adders). Instead of adding all bitvectors sequentially, a
compressor tree reduces the number of operands in parallel stages. The
result is stored in a redundant (carry-save) representation, deferring the
compressor tree implementation to a later stage.

Example:

```
%0:2 = datapath.compress %a, %b, %c : i16 [3 -> 2]
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of a signless integer bitvector |

### `datapath.partial_product` (::circt::datapath::PartialProductOp)

*Generate partial products from multiplying the operands*

Syntax:

```
operation ::= `datapath.partial_product` $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
```

The first step in a multiplication is to generate partial products, which
when summed, yield the product of the two operands. The partial
product operator does not specify an implementation, only that summing the
results will yield the product of the two operands. The number of results
corresponds to the rows of a partial product array, which by default is
equal to the width of the inputs.

Verilog Example 4-bit multiplication:

```
partial_product[0][3:0] = {4{a[0]}} & b
...
partial_product[3][3:0] = {4{a[3]}} & b
ab[3:0] = partial_product[0] + ... + partial_product[3] // = a*b
```

Example using `datapath` dialect:

```
%0:4 = datapath.partial_product %a, %b : (i4, i4) -> (i4, i4, i4, i4)
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | a signless integer bitvector |
| `rhs` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of a signless integer bitvector |

### `datapath.pos_partial_product` (::circt::datapath::PosPartialProductOp)

*Generate partial products for (a+b)\*c*

Syntax:

```
operation ::= `datapath.pos_partial_product` $addend0  `,` $addend1 `,` $multiplicand attr-dict `:` functional-type(operands, results)
```

A partial product array which when summed produces the result (a+b)\*c.
An efficient circuit can be constructed that encodes a carry-save input
without performing a full carry-propagation. The number of results
corresponds to the rows of a partial product array, which by default is
equal to the width of the inputs.

Example using `datapath` dialect:

```
%0:3 = datapath.pos_partial_product %a, %b, %c : (i3, i3, i3) 
                                              -> (i3, i3, i3)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `addend0` | a signless integer bitvector |
| `addend1` | a signless integer bitvector |
| `multiplicand` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of a signless integer bitvector |

'datapath' Dialect Docs
-----------------------

* ['datapath' Dialect Rationale](https://circt.llvm.org/docs/Dialects/Datapath/RationaleDatapath/)

 [Prev - `comb` Dialect Rationale](https://circt.llvm.org/docs/Dialects/Comb/RationaleComb/ "`comb` Dialect Rationale")
[Next - 'datapath' Dialect Rationale](https://circt.llvm.org/docs/Dialects/Datapath/RationaleDatapath/ "'datapath' Dialect Rationale") 

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