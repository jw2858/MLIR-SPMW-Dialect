'synth' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'synth' Dialect
===============

*Synthesis dialect for logic synthesis operations*

The Synth dialect provides operations and types for logic synthesis,
including meta operations for synthesis decisions, logic representations
like AIG and MIG, and synthesis pipeline infrastructure.

* [Operations](#operations)
  + [`synth.aig.and_inv` (::circt::synth::aig::AndInverterOp)](#synthaigand_inv-circtsynthaigandinverterop)
  + [`synth.mig.maj_inv` (::circt::synth::mig::MajorityInverterOp)](#synthmigmaj_inv-circtsynthmigmajorityinverterop)

Operations
----------

### `synth.aig.and_inv` (::circt::synth::aig::AndInverterOp)

*AIG dialect AND operation*

Syntax:

```
operation ::= `synth.aig.and_inv` custom<VariadicInvertibleOperands>($inputs, type($result), $inverted, attr-dict)
```

The `synth.aig.and_inv` operation represents an And-Inverter in the AIG dialect.
Unlike `comb.and`, operands can be inverted respectively.

Example:

```
  %r1 = synth.aig.and_inv %a, %b: i3
  %r2 = synth.aig.and_inv not %a, %b, not %c : i3
  %r3 = synth.aig.and_inv not %a : i3
```

Traditionally, an And-Node in AIG has two operands. However, `synth.aig.and_inv`
extends this concept by allowing variadic operands and non-i1 integer types.
Although the final stage of the synthesis pipeline requires lowering
everything to i1-binary operands, it’s more efficient to progressively lower
the variadic multibit operations.

Variadic operands have demonstrated their utility in low-level optimizations
within the `comb` dialect. Furthermore, in synthesis, it’s common practice
to re-balance the logic path. Variadic operands enable the compiler to
select more efficient solutions without the need to traverse binary trees
multiple times.

The ability to represent multibit operations during synthesis is crucial for
scalability. This approach enables a form of vectorization, allowing for
batch processing of logic synthesis when multibit operations are constructed
in a similar manner.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inverted` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `synth.mig.maj_inv` (::circt::synth::mig::MajorityInverterOp)

*Majority-Inverter operation*

Syntax:

```
operation ::= `synth.mig.maj_inv` custom<VariadicInvertibleOperands>($inputs, type($result), $inverted,
              attr-dict)
```

The `synth.mig.maj_inv` operation represents a Majority-Inverter in the
Synth dialect. This is used to represent majority inverter graph in
synthesis. This operation computes the majority function of its inputs,
where operands can be inverted respectively.

The majority function returns 1 when more than half of the inputs are 1,
and 0 otherwise. For three inputs, it’s equivalent to:
(a & b) | (a & c) | (b & c).

Example:

```
  %r1 = synth.mig.maj_inv %a, %b, %c : i1
  %r2 = synth.mig.maj_inv not %a, %b, not %c : i1
  %r3 = synth.mig.maj_inv %a, %b, %c, %d, %e : i3
```

The number of inputs must be odd to avoid ties.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inverted` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

'synth' Dialect Docs
--------------------

* ['synth' Dialect](https://circt.llvm.org/docs/Dialects/Synth/RationaleSynth/)
* [Synth Longest Path Analysis](https://circt.llvm.org/docs/Dialects/Synth/LongestPathAnalysis/)

 [Prev - SV Dialect Rationale](https://circt.llvm.org/docs/Dialects/SV/RationaleSV/ "SV Dialect Rationale")
[Next - 'synth' Dialect](https://circt.llvm.org/docs/Dialects/Synth/RationaleSynth/ "'synth' Dialect") 

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
      * [Talks and Related Publications](https://circt.llvm.org/talks/)
      * [Getting Started](https://circt.llvm.org/getting_started/)
      * [Code Documentation-](https://circt.llvm.org/docs/)
        + [Tools+](https://circt.llvm.org/docs/Tools/)
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