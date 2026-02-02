FSM Dialect Rationale - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

FSM Dialect Rationale
=====================

This document describes various design points of the FSM dialect, why they are
the way they are, and current status. This follows in the spirit of other
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

Introduction [¶](#introduction)
-------------------------------

[Finite-state machine (FSM)](https://en.wikipedia.org/wiki/Finite-state_machine)
is an abstract machine that can be in exactly one of a finite number of states
at any given time. The FSM can change from one state to another in response to
some inputs; the change from one state to another is called a transition.
Verification, Hardware IP block control, system state control, hardware design,
and software design have aspects that are succinctly described as some form of
FSM. For integrated development purposes, late design-time choice, and
per-project choice, we want to encode system descriptions in an FSM form. We
want a compiler to be able to manipulate, query, and generate code in multiple
domains, such as SW drivers, firmware, RTL hardware, and verification.

The FSM dialect in CIRCT is designed to provide a set of abstractions for FSM
with the following features:

1. Provide explicit and structural representations of states, transitions, and
   internal variables of an FSM, allowing convenient analysis and transformation.
2. Provide a target-agnostic representation of FSM, allowing the state machine
   to be instantiated and attached to other dialects from different domains.
3. By cooperating with two conversion passes, FSMToSV and FSMToStandard, allow
   to lower the FSM abstraction into HW+Comb+SV (Hardware) and Standard+SCF+MemRef
   (Software) dialects for the purposes of simulation, code generation, etc.

Operations [¶](#operations)
---------------------------

### Two ways of instantiation [¶](#two-ways-of-instantiation)

A state machine is defined by an `fsm.machine` operation, which contains all
the states and transitions of the state machine. `fsm.machine` has a list of
inputs and outputs and explicit state type:

```
fsm.machine @foo(%arg0: i1) -> i1 attributes {stateType = i1} {
  ...
}
```

FSM dialect provides two ways to instantiate a state machine: `fsm.hw_instance`
is intended for use in HW context (usually described by graph regions) and
`fsm.instance`+`fsm.trigger` is intended for use in a SW context (usually
described by CFG regions). In HW IRs (such as HW+Comb+SV and FIRRTL), although
an MLIR value is only defined once in the IR, it is actually “driven” by its
predecessors continuously during the runtime and can “hold” different values at
different moments. However, in the world of SW IRs (such as Standard+SCF), we
don’t have such a semantics – SW IRs “run” sequentially.

Here we define that each *trigger* causes the possibility of a transition from
one state to another state through exactly one transition. In a SW context,
`fsm.instance` generates an `InstanceType` value to represent a state machine
instance. Each `fsm.trigger` targets a machine instance and explicitly causes a
*trigger*. Therefore, `fsm.trigger` may change the state of the machine
instance thus is a side-effecting operation. The following MLIR code shows an
example of instantiating and triggering the state machine defined above:

```
func @bar() {
  %foo_inst = fsm.instance "foo_inst" @foo
  %in0 = ...
  %out0 = fsm.trigger %foo_inst(%in0) : (i1) -> i1
  ...
  %in1 = ...
  %out1 = fsm.trigger %foo_inst(%in1) : (i1) -> i1
  return
}
```

In the contrast, to comply with the HW semantics, `fsm.hw_instance` directly
consumes inputs and generates results. The operand and result types must align
with the type of the referenced `fsm.machine`. In a HW context, *trigger*s are
implicitly initiated by the processors of `fsm.hw_instance`. The following
MLIR code shows an example of instantiating the same state machine in HW IRs:

```
hw.module @qux() {
  %in = ...
  %out = fsm.hw_instance "foo_inst" @foo(%in) : (i1) -> i1
}
```

### Explicit state and transition representation [¶](#explicit-state-and-transition-representation)

Each state of an FSM is represented explicitly with an `fsm.state` operation.
`fsm.state` must have an `output` CFG region representing the combinational
logic of generating the state machine’s outputs. The output region must have an
`fsm.output` operation as terminator and the operand types of the `fsm.output`
must align with the result types of the state machine. `fsm.state` also
contains a list of `fsm.transition` operations representing the outgoing
transitions that can be triggered from the current state. `fsm.state` also has
a symbol name that can be referred to by `fsm.transition`s as the next state.
The following MLIR code shows a running example:

```
fsm.machine @foo(%arg0: i1) -> i1 attributes {stateType = i1} {
  fsm.state "IDLE" output  {
    %true = constant true
    fsm.output %true : i1
  } transitions  {
    fsm.transition @BUSY ...
  }

  fsm.state "BUSY" output  {
    %false = constant false
    fsm.output %false : i1
  } transitions  {
    fsm.transition @BUSY ...
    fsm.transition @IDLE ...
  }
}
```

### Guard region of transitions [¶](#guard-region-of-transitions)

`fsm.transition` has an optional `guard` CFG region, which must be terminated
with an `fsm.return` operation returning a Boolean value to indicate whether
the transition is taken:

```
fsm.machine @foo(%arg0: i1) -> i1 attributes {stateType = i1} {
  fsm.state "IDLE" output  {
    %true = constant true
    fsm.output %true : i1
  } transitions  {
    fsm.transition @BUSY guard  {
      fsm.return %arg0 : i1
    }
  }
  ...
}
```

If a state has more than one transition, multiple transitions are prioritized
in the order that they appear in the transitions region. Guards must also not
contain any operations with side effects, enabling the evaluation of guards to
be parallelized. Note that an empty guard region is evaluated as true, which
means the corresponding transition is always taken.

### Action region of transitions and internal variables [¶](#action-region-of-transitions-and-internal-variables)

To avoid *state explosion*, we introduce `fsm.variable` operation (similar to
the
[extended state](https://en.wikipedia.org/wiki/UML_state_machine) in UML
state machine) which represents a variable associated with an FSM instance and
can hold a value of any type, which can be updated through `fsm.update`
operations.

`fsm.transition` has an optional `action` CFG region representing the actions
associated with the current transition. The action region can contain
side-effecting operations. `fsm.update` must be contained by the action region
of a transition. The following MLIR code shows a running example:

```
fsm.machine @foo(%arg0: i1) -> i1 attributes {stateType = i1} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16
  fsm.state "IDLE" output  {
    %true = constant true
    fsm.output %true : i1
  } transitions  {
    fsm.transition @BUSY guard  {
      fsm.return %arg0 : i1
    } action  {
      %c256_i16 = constant 256 : i16
      fsm.update %cnt, %c256_i16 : i16
    }
  }
  ...
}
```

 [Prev - 'fsm' Dialect](https://circt.llvm.org/docs/Dialects/FSM/ "'fsm' Dialect")
[Next - 'handshake' Dialect](https://circt.llvm.org/docs/Dialects/Handshake/ "'handshake' Dialect") 

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
    - ['fsm' Dialect-](https://circt.llvm.org/docs/Dialects/FSM/)
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