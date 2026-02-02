'loopschedule' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'loopschedule' Dialect
======================

*Representation of scheduled loops*

* [Operations](#operations)
  + [`loopschedule.pipeline` (::circt::loopschedule::LoopSchedulePipelineOp)](#loopschedulepipeline-circtloopscheduleloopschedulepipelineop)
  + [`loopschedule.pipeline.stage` (::circt::loopschedule::LoopSchedulePipelineStageOp)](#loopschedulepipelinestage-circtloopscheduleloopschedulepipelinestageop)
  + [`loopschedule.register` (::circt::loopschedule::LoopScheduleRegisterOp)](#loopscheduleregister-circtloopscheduleloopscheduleregisterop)
  + [`loopschedule.terminator` (::circt::loopschedule::LoopScheduleTerminatorOp)](#loopscheduleterminator-circtloopscheduleloopscheduleterminatorop)

Operations
----------

### `loopschedule.pipeline` (::circt::loopschedule::LoopSchedulePipelineOp)

*LoopSchedule dialect pipeline-loop.*

The `loopschedule.pipeline` operation represents a statically scheduled
pipeline stucture that executes while a condition is true. For more details,
see:
<https://llvm.discourse.group/t/rfc-representing-pipelined-loops/4171>.

A pipeline captures the result of scheduling, and is not generally safe to
transform, besides lowering to hardware dialects. For more discussion about
relaxing this, see:
<https://github.com/llvm/circt/issues/2204>.

This is the top-level operation representing a high-level pipeline. It is
not isolated from above, but could be if this is helpful. A pipeline
contains two regions: `condition` and `stages`.

The pipeline may accept an optional `iter_args`, similar to the SCF dialect,
for representing loop-carried values like induction variables or reductions.
When the pipeline starts execution, the registers indicated as `iter_args`
by `pipeline.terminator` should be initialized to the initial
values specified in the `iter_args` section here. The `iter_args` relate to
the initiation interval of the loop. The maximum distance in stages between
where an `iter_arg` is used and where that `iter_arg` is registered must be
less than the loop’s initiation interval. For example, with II=1, each
`iter_arg` must be used and registered in the same stage.

The single-block `condition` region dictates the condition under which the
pipeline should execute. It has a `register` terminator, and the
pipeline initiates new iterations while the registered value is `true : i1`.
It may access SSA values dominating the pipeline, as well as `iter_args`,
which are block arguments. The body of the block may only contain
“combinational” operations, which are currently defined to be simple
arithmetic, comparisons, and selects from the `Standard` dialect.

The single-block `stages` region wraps `loopschedule.pipeline.stage`
operations. It has a `loopschedule.terminator` terminator, which can
both return results from the pipeline and register `iter_args`. Stages may
access SSA values dominating the pipeline, as well as `iter_args`, which are
block arguments.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `II` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `tripCount` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `iterArgs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `loopschedule.pipeline.stage` (::circt::loopschedule::LoopSchedulePipelineStageOp)

*LoopSchedule dialect pipeline stage.*

Syntax:

```
operation ::= `loopschedule.pipeline.stage` `start` `=` $start (`when` $when^)? $body (`:` qualified(type($results))^)? attr-dict
```

This operation has a single-block region which dictates the operations that
may occur concurrently.

It has a `start` attribute, which indicates the start cycle for this stage.

It may have an optional `when` predicate, which supports conditional
execution for each stage. This is in addition to the `condition` region that
controls the execution of the whole pipeline. A stage with a `when`
predicate should only execute when the predicate is `true : i1`, and push a
bubble through the pipeline otherwise.

It has a `register` terminator, which passes the concurrently
computed values forward to the next stage.

Any stage may access `iter_args`. If a stage accesses an `iter_arg` after
the stage in which it is defined, it is up to lowering passes to preserve
this value until the last stage that needs it.

Other than `iter_args`, stages may only access SSA values dominating the
pipeline or SSA values computed by any previous stage. This ensures the
stages capture the coarse-grained schedule of the pipeline and how values
feed forward and backward.

Traits: `HasParent<LoopSchedulePipelineOp>`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `start` | ::mlir::IntegerAttr | 64-bit signed integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `when` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `loopschedule.register` (::circt::loopschedule::LoopScheduleRegisterOp)

*LoopSchedule dialect loop register.*

Syntax:

```
operation ::= `loopschedule.register` $operands (`:` qualified(type($operands))^)? attr-dict
```

The `loopschedule.register` terminates a pipeline stage and
“registers” the values specified as operands. These values become the
results of the stage.

Traits: `HasParent<LoopSchedulePipelineOp, LoopSchedulePipelineStageOp>`, `Terminator`

#### Operands:

| Operand | Description |
| --- | --- |
| `operands` | variadic of any type |

### `loopschedule.terminator` (::circt::loopschedule::LoopScheduleTerminatorOp)

*LoopSchedule dialect terminator.*

Syntax:

```
operation ::= `loopschedule.terminator` `iter_args` `(` $iter_args `)` `,`
              `results` `(` $results `)` `:`
              functional-type($iter_args, $results) attr-dict
```

The `loopschedule.terminator` operation represents the terminator of
a `loopschedule.pipeline`.

The `results` section accepts a variadic list of values which become the
pipeline’s return values. These must be results of a stage, and their types
must match the pipeline’s return types. The results need not be defined in
the final stage, and it is up to lowering passes to preserve these values
until the final stage is complete.

The `iter_args` section accepts a variadic list of values which become the
next iteration’s `iter_args`. These may be the results of any stage, and
their types must match the pipeline’s `iter_args` types.

Traits: `AttrSizedOperandSegments`, `HasParent<LoopSchedulePipelineOp>`, `Terminator`

#### Operands:

| Operand | Description |
| --- | --- |
| `iter_args` | variadic of any type |
| `results` | variadic of any type |

'loopschedule' Dialect Docs
---------------------------

* [LoopSchedule Dialect Rationale](https://circt.llvm.org/docs/Dialects/LoopSchedule/LoopSchedule/)

 [Prev - `kanagawa` Dialect Rationale](https://circt.llvm.org/docs/Dialects/Kanagawa/RationaleKanagawa/ "`kanagawa` Dialect Rationale")
[Next - LoopSchedule Dialect Rationale](https://circt.llvm.org/docs/Dialects/LoopSchedule/LoopSchedule/ "LoopSchedule Dialect Rationale") 

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