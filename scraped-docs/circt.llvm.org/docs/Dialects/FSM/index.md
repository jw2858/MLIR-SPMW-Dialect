'fsm' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'fsm' Dialect
=============

*Types and operations for FSM dialect*

This dialect defines the `fsm` dialect, which is intended to represent
finite-state machines.

* [Operations](#operations)
  + [`fsm.hw_instance` (::circt::fsm::HWInstanceOp)](#fsmhw_instance-circtfsmhwinstanceop)
  + [`fsm.instance` (::circt::fsm::InstanceOp)](#fsminstance-circtfsminstanceop)
  + [`fsm.machine` (::circt::fsm::MachineOp)](#fsmmachine-circtfsmmachineop)
  + [`fsm.output` (::circt::fsm::OutputOp)](#fsmoutput-circtfsmoutputop)
  + [`fsm.return` (::circt::fsm::ReturnOp)](#fsmreturn-circtfsmreturnop)
  + [`fsm.state` (::circt::fsm::StateOp)](#fsmstate-circtfsmstateop)
  + [`fsm.transition` (::circt::fsm::TransitionOp)](#fsmtransition-circtfsmtransitionop)
  + [`fsm.trigger` (::circt::fsm::TriggerOp)](#fsmtrigger-circtfsmtriggerop)
  + [`fsm.update` (::circt::fsm::UpdateOp)](#fsmupdate-circtfsmupdateop)
  + [`fsm.variable` (::circt::fsm::VariableOp)](#fsmvariable-circtfsmvariableop)
* [Types](#types)
  + [InstanceType](#instancetype)

Operations
----------

### `fsm.hw_instance` (::circt::fsm::HWInstanceOp)

*Create a hardware-style instance of a state machine*

Syntax:

```
operation ::= `fsm.hw_instance` $name $machine attr-dict `(` $inputs `)`
              `,` `clock` $clock `,` `reset` $reset `:` functional-type($inputs, $outputs)
```

`fsm.hw_instance` represents a hardware-style instance of a state machine,
including an instance name and a symbol reference of the machine. The inputs
and outputs are correponding to the inputs and outputs of the referenced
machine.

Interfaces: `InstanceOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `machine` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |
| `clock` | A type for clock-carrying wires |
| `reset` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `fsm.instance` (::circt::fsm::InstanceOp)

*Create an instance of a state machine*

Syntax:

```
operation ::= `fsm.instance` $name $machine attr-dict
```

`fsm.instance` represents an instance of a state machine, including an
instance name and a symbol reference of the machine.

Interfaces: `HasCustomSSAName`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `machine` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Results:

| Result | Description |
| --- | --- |
| `instance` | An FSM instance type |

### `fsm.machine` (::circt::fsm::MachineOp)

*Define a finite-state machine*

`fsm.machine` represents a finite-state machine, including a machine name,
the type of machine state, and the types of inputs and outputs. This op also
includes a `$body` region that contains internal variables and states.

Traits: `NoTerminator`, `SymbolTable`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallableOpInterface`, `FunctionOpInterface`, `ModuleOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `initialState` | ::mlir::StringAttr | string attribute |
| `function_type` | ::mlir::TypeAttr | type attribute of function type |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `argNames` | ::mlir::ArrayAttr | string array attribute |
| `resNames` | ::mlir::ArrayAttr | string array attribute |

### `fsm.output` (::circt::fsm::OutputOp)

*Output values from a state machine*

Syntax:

```
operation ::= `fsm.output` attr-dict ($operands^ `:` qualified(type($operands)))?
```

“fsm.output” represents the outputs of a machine under a specific state. The
types of `$operands` should be consistent with the output types of the state
machine.

Traits: `HasParent<StateOp>`, `ReturnLike`, `Terminator`

Interfaces: `RegionBranchTerminatorOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `operands` | variadic of any type |

### `fsm.return` (::circt::fsm::ReturnOp)

*Return values from a region*

Syntax:

```
operation ::= `fsm.return` attr-dict ($operand^)?
```

“fsm.return” marks the end of a region of `fsm.transition` and return
values if the parent region is a `$guard` region.

Traits: `HasParent<TransitionOp>`, `ReturnLike`, `Terminator`

Interfaces: `RegionBranchTerminatorOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `operand` | 1-bit signless integer |

### `fsm.state` (::circt::fsm::StateOp)

*Define a state of a machine*

Syntax:

```
operation ::= `fsm.state` $sym_name attr-dict (`output` $output^)? (`transitions` $transitions^)?
```

`fsm.state` represents a state of a state machine. This op includes an
`$output` region with an `fsm.output` as terminator to define the machine
outputs under this state. This op also includes a `transitions` region that
contains all the transitions of this state.

Traits: `HasParent<MachineOp>`, `NoTerminator`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `fsm.transition` (::circt::fsm::TransitionOp)

*Define a transition of a state*

Syntax:

```
operation ::= `fsm.transition` $nextState attr-dict (`guard` $guard^)? (`action` $action^)?
```

`fsm.transition` represents a transition of a state with a symbol reference
of the next state. This op includes an optional `$guard` region with an `fsm.return`
as terminator that returns a Boolean value indicating the guard condition of
this transition. This op also includes an optional `$action` region that represents
the actions to be executed when this transition is taken.

Traits: `HasParent<StateOp>`, `NoTerminator`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `nextState` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

### `fsm.trigger` (::circt::fsm::TriggerOp)

*Trigger an instance*

Syntax:

```
operation ::= `fsm.trigger` $instance attr-dict `(` $inputs `)` `:` functional-type($inputs, $outputs)
```

`fsm.trigger` triggers a state machine instance. The inputs and outputs are
correponding to the inputs and outputs of the referenced machine of the
instance.

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |
| `instance` | An FSM instance type |

#### Results:

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `fsm.update` (::circt::fsm::UpdateOp)

*Update a variable in a state machine*

Syntax:

```
operation ::= `fsm.update` attr-dict $variable `,` $value `:` qualified(type($value))
```

`fsm.update` updates the `$variable` with the `$value`. The definition op of
`$variable` should be an `fsm.variable`. This op should *only* appear in the
`action` region of a transtion.

Traits: `HasParent<TransitionOp>`, `SameTypeOperands`

#### Operands:

| Operand | Description |
| --- | --- |
| `variable` | any type |
| `value` | any type |

### `fsm.variable` (::circt::fsm::VariableOp)

*Declare a variable in a state machine*

Syntax:

```
operation ::= `fsm.variable` $name attr-dict `:` qualified(type($result))
```

`fsm.variable` represents an internal variable in a state machine with an
initialization value.

Traits: `FirstAttrDerivedResultType`, `HasParent<MachineOp>`

Interfaces: `HasCustomSSAName`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `initValue` | ::mlir::Attribute | any attribute |
| `name` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

Types
-----

### InstanceType

*An FSM instance type*

Syntax: `!fsm.instance`

Represents an FSM instance.

'fsm' Dialect Docs
------------------

* [FSM Dialect Rationale](https://circt.llvm.org/docs/Dialects/FSM/RationaleFSM/)

 [Prev - Intrinsics](https://circt.llvm.org/docs/Dialects/FIRRTL/FIRRTLIntrinsics/ "Intrinsics")
[Next - FSM Dialect Rationale](https://circt.llvm.org/docs/Dialects/FSM/RationaleFSM/ "FSM Dialect Rationale") 

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