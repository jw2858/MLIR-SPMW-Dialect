'pipeline' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'pipeline' Dialect
==================

* [Operations](#operations)
  + [`pipeline.latency` (::circt::pipeline::LatencyOp)](#pipelinelatency-circtpipelinelatencyop)
  + [`pipeline.latency.return` (::circt::pipeline::LatencyReturnOp)](#pipelinelatencyreturn-circtpipelinelatencyreturnop)
  + [`pipeline.return` (::circt::pipeline::ReturnOp)](#pipelinereturn-circtpipelinereturnop)
  + [`pipeline.scheduled` (::circt::pipeline::ScheduledPipelineOp)](#pipelinescheduled-circtpipelinescheduledpipelineop)
  + [`pipeline.src` (::circt::pipeline::SourceOp)](#pipelinesrc-circtpipelinesourceop)
  + [`pipeline.stage` (::circt::pipeline::StageOp)](#pipelinestage-circtpipelinestageop)
  + [`pipeline.unscheduled` (::circt::pipeline::UnscheduledPipelineOp)](#pipelineunscheduled-circtpipelineunscheduledpipelineop)

Operations
----------

### `pipeline.latency` (::circt::pipeline::LatencyOp)

*Pipeline dialect latency operation.*

Syntax:

```
operation ::= `pipeline.latency` $latency `->` `(` type($results) `)` $body attr-dict
```

The `pipeline.latency` operation represents an operation for wrapping
multi-cycle operations. The operation declares a single block
wherein any operation may be placed within. The operation is not
`IsolatedFromAbove` meaning that the operation can reference values
defined outside of the operation (subject to the materialization
phase of the parent pipeline).

Traits: `HasOnlyGraphRegion`, `HasParent<UnscheduledPipelineOp, ScheduledPipelineOp>`, `SingleBlockImplicitTerminator<LatencyReturnOp>`, `SingleBlock`

Interfaces: `RegionKindInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `latency` | ::mlir::IntegerAttr | 32-bit signless integer attribute whose minimum value is 1 |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `pipeline.latency.return` (::circt::pipeline::LatencyReturnOp)

*Pipeline latency return operation.*

Syntax:

```
operation ::= `pipeline.latency.return` ($inputs^)? attr-dict (`:` type($inputs)^)?
```

The `pipeline.latency.return` operation represents a terminator of a
`pipeline.latency` operation.

Traits: `HasParent<LatencyOp>`, `Terminator`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

### `pipeline.return` (::circt::pipeline::ReturnOp)

*Pipeline dialect return.*

Syntax:

```
operation ::= `pipeline.return` ($inputs^)? attr-dict (`:` type($inputs)^)?
```

The “return” operation represents a terminator of a `pipeline.pipeline`.

Traits: `HasParent<UnscheduledPipelineOp, ScheduledPipelineOp>`, `Terminator`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

### `pipeline.scheduled` (::circt::pipeline::ScheduledPipelineOp)

*Scheduled pipeline operation*

The `pipeline.scheduled` operation represents a scheduled pipeline.
The pipeline contains a single block representing a graph region.

A `pipeline.scheduled` operation can exist in multiple phases, mainly
pertaining to when registers have been materialized (made explicit).
For an in-depth explanation, please refer to the Pipeline dialect rationale.

A `pipeline.scheduled` supports a `stall` input. This signal is intended to
connect to all stages within the pipeline, and is used to stall the entirety
of the pipeline. It is lowering defined how stages choose to use this signal,
although in the common case, a `stall` signal would typically connect to
the clock-enable input of the stage-separating registers.

The `go` input is used to start the pipeline. This value is fed through
the stages as the current stage valid/next stage enable signal.
Note: the op is currently only designed for pipelines with II=1. For
pipelines with II>1, a user must themselves maintain state about when
the pipeline is ready to accept new inputs. We plan to add support for
head-of-pipeline backpressure in the future.

Any value defined outside the pipeline is considered an external input. An
external input will *not* be registered.

The pipeline may optionally be provided with an array of bits `stallability`
which is used to determine which stages are stallable.

* If not provided and the pipeline has a stall signal, all stages are stallable.
* If provided, and the pipeline has a stall signal, the number of bits must
  match the number of stages in the pipeline. Each bit represents a stage,
  in the order of which the stages appear wrt. the `pipeline.stage` operations.
  A bit set to 1 indicates that the stage is stallable, and 0 indicates that
  the stage is not stallable.

The exit (non-registered) stage of a pipeline cannot be non-stallable, and
will always follow the stallability of the parent pipeline.

For more information about non-stallable stages, and how these are lowered,
please refer to the Pipeline dialect rationale.

Traits: `AttrSizedOperandSegments`

Interfaces: `OpAsmOpInterface`, `RegionKindInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `inputNames` | ::mlir::ArrayAttr | string array attribute |
| `outputNames` | ::mlir::ArrayAttr | string array attribute |
| `stallability` | ::mlir::ArrayAttr | 1-bit boolean array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |
| `stall` | 1-bit signless integer |
| `clock` | A type for clock-carrying wires |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `dataOutputs` | variadic of any type |
| `done` | 1-bit signless integer |

### `pipeline.src` (::circt::pipeline::SourceOp)

*Pipeline source operation*

Syntax:

```
operation ::= `pipeline.src` $input `:` type($input) attr-dict
```

The `pipeline.src` operation represents a source operation in a scheduled,
non-register materialized pipeline.
It is used as a canonicalization barrier to prevent cross-block canonicalization
of operations that are not allowed to be moved or mutated across pipeline
stages (i.e. MLIR blocks).

To facilitate this, the operation is *not* marked as `Pure`.

Traits: `HasParent<ScheduledPipelineOp>`

Interfaces: `InferTypeOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `output` | any type |

### `pipeline.stage` (::circt::pipeline::StageOp)

*Pipeline stage terminator.*

Syntax:

```
operation ::= `pipeline.stage` $nextStage
              custom<StageRegisters>($registers, type($registers), $clockGates, $clockGatesPerRegister, $registerNames)
              custom<Passthroughs>($passthroughs, type($passthroughs), $passthroughNames)
              attr-dict
```

The `pipeline.stage` operation represents a stage terminator. It is used
to communicate:

1. which stage (block) to transition to next
2. which registers to build at this stage boundary
3. which values to pass through to the next stage without registering
4. An optional hierarchy of boolean values to be used for clock gates for
   each register.

* The implicit ‘!stalled’ gate will always be the first signal in the
  hierarchy. Further signals are added to the hierarchy from left to
  right.

Example:

```
pipeline.stage ^bb1 regs(%a : i32 gated by [%foo, %bar], %b : i1) pass(%c : i32)
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HasParent<ScheduledPipelineOp>`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `clockGatesPerRegister` | ::mlir::ArrayAttr | 64-bit integer array attribute |
| `registerNames` | ::mlir::ArrayAttr | string array attribute |
| `passthroughNames` | ::mlir::ArrayAttr | string array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `registers` | variadic of any type |
| `passthroughs` | variadic of any type |
| `clockGates` | variadic of 1-bit signless integer |

#### Successors:

| Successor | Description |
| --- | --- |
| `nextStage` | any successor |

### `pipeline.unscheduled` (::circt::pipeline::UnscheduledPipelineOp)

*Unscheduled pipeline operation*

The “pipeline.unscheduled” operation represents a pipeline that has not yet
been scheduled. It contains a single block representing a graph region of
operations to-be-scheduled into a pipeline.
Mainly serves as a container and entrypoint for scheduling.

The interface of a `pipeline.unscheduled` is similar to that of a
`pipeline.scheduled`. Please refer to this op for further documentation
about the interface signals.

Traits: `AttrSizedOperandSegments`, `HasOnlyGraphRegion`, `SingleBlockImplicitTerminator<ReturnOp>`, `SingleBlock`

Interfaces: `OpAsmOpInterface`, `RegionKindInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `inputNames` | ::mlir::ArrayAttr | string array attribute |
| `outputNames` | ::mlir::ArrayAttr | string array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |
| `stall` | 1-bit signless integer |
| `clock` | A type for clock-carrying wires |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `dataOutputs` | variadic of any type |
| `done` | 1-bit signless integer |

'pipeline' Dialect Docs
-----------------------

* [Pipeline Dialect Rationale](https://circt.llvm.org/docs/Dialects/Pipeline/RationalePipeline/)

 [Prev - Object Model Dialect Rationale](https://circt.llvm.org/docs/Dialects/OM/RationaleOM/ "Object Model Dialect Rationale")
[Next - Pipeline Dialect Rationale](https://circt.llvm.org/docs/Dialects/Pipeline/RationalePipeline/ "Pipeline Dialect Rationale") 

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