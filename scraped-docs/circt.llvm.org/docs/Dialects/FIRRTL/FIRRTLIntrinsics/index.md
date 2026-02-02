Intrinsics - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

Intrinsics
==========

Intrinsics provide an implementation-specific way to extend the FIRRTL language
with new operations.

* [Motivation](#motivation)
* [Supported Intrinsics](#supported-intrinsics)
  + [circt\_plusargs\_value](#circt_plusargs_value)
  + [circt\_plusargs\_test](#circt_plusargs_test)
  + [circt\_clock\_gate](#circt_clock_gate)
  + [circt\_chisel\_assert](#circt_chisel_assert)
  + [circt\_chisel\_ifelsefatal](#circt_chisel_ifelsefatal)
  + [circt\_chisel\_assume](#circt_chisel_assume)
  + [circt\_chisel\_cover](#circt_chisel_cover)
  + [circt\_unclocked\_assume](#circt_unclocked_assume)
  + [circt\_dpi\_call](#circt_dpi_call)
  + [circt\_view](#circt_view)
  + [circt\_verif\_assert](#circt_verif_assert)
  + [circt\_verif\_assume](#circt_verif_assume)
  + [circt\_verif\_cover](#circt_verif_cover)
  + [circt\_verif\_require](#circt_verif_require)
  + [circt\_verif\_ensure](#circt_verif_ensure)

Motivation [¶](#motivation)
---------------------------

Intrinsics provide a way to add functionality to FIRRTL without having to extend
the FIRRTL language. This allows a fast path for prototyping new operations to
rapidly respond to output requirements. Intrinsics maintain strict definitions
and type checking.

Supported Intrinsics [¶](#supported-intrinsics)
-----------------------------------------------

### circt\_plusargs\_value [¶](#circt_plusargs_value)

Tests and extracts a value from simulator command line options with SystemVerilog
`$value$plusargs`. This is described in SystemVerilog 2012 section 21.6.

We do not currently check that the format string substitution flag matches the
type of the result.

| Parameter | Type | Description |
| --- | --- | --- |
| FORMAT | string | Format string per SV 21.6 |

| Result | Type | Description |
| --- | --- | --- |
| found | UInt<1> | found in args |
| result | AnyType | value of the argument |

### circt\_plusargs\_test [¶](#circt_plusargs_test)

Tests simulator command line options with SystemVerilog `$test$plusargs`. This
is described in SystemVerilog 2012 section 21.6.

| Parameter | Type | Description |
| --- | --- | --- |
| FORMAT | string | Format string per SV 21.6 |

| Result | Type | Description |
| --- | --- | --- |
| found | UInt<1> | found in args |

### circt\_clock\_gate [¶](#circt_clock_gate)

Enables and disables a clock safely, without glitches, based on a boolean enable value. If the enable input is 1, the output clock produced by the clock gate is identical to the input clock. If the enable input is 0, the output clock is a constant zero.

The enable input is sampled at the rising edge of the input clock; any changes on the enable before or after that edge are ignored and do not affect the output clock.

| Argument | Type | Description |
| --- | --- | --- |
| in | Clock | input clock |
| en | UInt<1> | enable for the output clock |

| Result | Type | Description |
| --- | --- | --- |
| out | Clock | gated output clock |

### circt\_chisel\_assert [¶](#circt_chisel_assert)

Generate a clocked SV assert statement, with optional formatted error message.

| Parameter | Type | Description |
| --- | --- | --- |
| format | string | FIRRTL format string. Optional. |
| label | string | Label for assert/assume. Optional. |
| guards | string | Semicolon-delimited list of pre-processor tokens to use as ifdef guards. Optional. |

| Argument | Type | Description |
| --- | --- | --- |
| clock | Clock | input clock |
| predicate | UInt<1> | predicate to assert/assume |
| enable | UInt<1> | enable signal |
| … | Signals | arguments to format string |

Example output:

```
wire _GEN = ~enable | cond;
assert__label: assert property (@(posedge clock) _GEN) else $error("message");
```

### circt\_chisel\_ifelsefatal [¶](#circt_chisel_ifelsefatal)

Generate a particular Verilog sequence that’s similar to an assertion.

Has legacy special behavior and should not be used by new code.

| Parameter | Type | Description |
| --- | --- | --- |
| format | string | FIRRTL format string. Optional. |

This intrinsic also accepts the `label` and `guard` parameters which
are recorded but not used in the normal emission.

| Argument | Type | Description |
| --- | --- | --- |
| clock | Clock | input clock |
| predicate | UInt<1> | predicate to check |
| enable | UInt<1> | enable signal |
| … | Signals | arguments to format string |

Example SV output:

```
`ifndef SYNTHESIS
  always @(posedge clock) begin
    if (enable & ~cond) begin
      if (`ASSERT_VERBOSE_COND_)
        $error("message");
      if (`STOP_COND_)
        $fatal;
    end
  end // always @(posedge)
`endif // not def SYNTHESIS
```

### circt\_chisel\_assume [¶](#circt_chisel_assume)

Generate a clocked SV assume statement, with optional formatted error message.

| Parameter | Type | Description |
| --- | --- | --- |
| format | string | FIRRTL format string. Optional. |
| label | string | Label for assume statement. Optional. |
| guards | string | Semicolon-delimited list of pre-processor tokens to use as ifdef guards. Optional. |

| Argument | Type | Description |
| --- | --- | --- |
| clock | Clock | input clock |
| predicate | UInt<1> | predicate to assume |
| enable | UInt<1> | enable signal |
| … | Signals | arguments to format string |

Example SV output:

```
assume__label: assume property (@(posedge clock) ~enable | cond) else $error("message");	
```

### circt\_chisel\_cover [¶](#circt_chisel_cover)

Generate a clocked SV cover statement.

| Parameter | Type | Description |
| --- | --- | --- |
| label | string | Label for cover statement. Optional. |
| guards | string | Semicolon-delimited list of pre-processor tokens to use as ifdef guards. Optional. |

| Argument | Type | Description |
| --- | --- | --- |
| clock | Clock | input clock |
| predicate | UInt<1> | predicate to cover |
| enable | UInt<1> | enable signal |

Example SV output:

```
cover__label: cover property (@(posedge clock) enable & cond);
```

### circt\_unclocked\_assume [¶](#circt_unclocked_assume)

Generate a SV assume statement whose predicate is used in a sensitivity list of the enclosing always block.

| Parameter | Type | Description |
| --- | --- | --- |
| format | string | FIRRTL format string. Optional. |
| label | string | Label for assume statement. Optional. |
| guards | string | Semicolon-delimited list of pre-processor tokens to use as ifdef guards. Optional. |

| Argument | Type | Description |
| --- | --- | --- |
| predicate | UInt<1> | predicate to assume |
| enable | UInt<1> | enable signal |
| … | Signals | arguments to format string |

Example SV output:

```
ifdef USE_FORMAL_ONLY_CONSTRAINTS
 `ifdef USE_UNR_ONLY_CONSTRAINTS
   wire _GEN = ~enable | pred;
   always @(edge _GEN)
     assume_label: assume(_GEN) else $error("Conditional compilation example for UNR-only and formal-only assert");
 `endif // USE_UNR_ONLY_CONSTRAINTS
endif // USE_FORMAL_ONLY_CONSTRAINTS
```

### circt\_dpi\_call [¶](#circt_dpi_call)

Call a DPI function. `clock` is optional and if `clock` is not provided,
the callee is invoked when input values are changed.
If provided, the dpi function is called at clock’s posedge. The result values behave
like registers and the DPI function is used as a state transfer function of them.

`enable` operand is used to conditionally call the DPI since DPI call could be quite
more expensive than native constructs. When `enable` is low, results of unclocked
calls are undefined and evaluated into `X`. Users are expected to gate result values
by another `enable` to model a default value of results.

For clocked calls, a low enable means that its register state transfer function is
not called. Hence their values will not be modify in that clock.

| Parameter | Type | Description |
| --- | --- | --- |
| isClocked | int | Set 1 if the dpi call is clocked. |
| functionName | string | Specify the function name. |
| inputNames | string | Semicolon-delimited list of input names. Optional. |
| outputName | string | Output name. Optional. |

| Argument | Type | Description |
| --- | --- | --- |
| clock (optional) | Clock | Optional clock operand |
| enable | UInt<1> | Enable signal |
| … | Signals | Arguments to DPI function call |

| Result | Type | Description |
| --- | --- | --- |
| result (optional) | Signal | Optional result of the dpi call |

#### DPI Intrinsic ABI [¶](#dpi-intrinsic-abi)

Function Declaration:

* Imported DPI function must be a void function that has input arguments which correspond to operand types, and an output argument which correspond to a result type.
* Output argument must be a last argument.

Types:

* Operand and result types must be passive.
* A vector is lowered to an unpacked open array type, e.g. `a: Vec<4, UInt<8>>` to `byte a []`.
* A bundle is lowered to a packed struct.
* Integer types are lowered into into 2-state types.
* Small integer types (< 64 bit) must be compatible to C-types and arguments are passed by values. Users are required to use specific integer types for small integers shown in the table below. Large integers are lowered to `bit` and passed by a reference.

| Width | Verilog Type | Argument Passing Modes |
| --- | --- | --- |
| 1 | bit | value |
| 8 | byte | value |
| 16 | shortint | value |
| 32 | int | value |
| 64 | longint | value |
| > 64 | bit [w-1:0] | reference |

Example SV output:

```
node result = intrinsic(circt_dpi_call<isClocked = 1, functionName="dpi_func"> : UInt<64>, clock, enable, uint_8_value, uint_32_value, uint_8_vector)
```

```
import "DPI-C" function void dpi_func(
  input  byte    in_0,
         int     in_1,
         byte    in_2[],
  output longint out_0
);

...

logic [63:0] _dpi_func_0;
reg   [63:0] _GEN;
always @(posedge clock) begin
  if (enable) begin
    dpi_func(in1, in2, _dpi_func_0);
    _GEN <= _dpi_func_0;
  end
end
```

### circt\_view [¶](#circt_view)

This will become a SystemVerilog Interface that is driven by its arguments.
This is *not* a true SystemVerilog Interface, it is only lowered to one.

| Parameter | Type | Description |
| --- | --- | --- |
| name | string | Instance name of the view. |
| info | string | JSON encoding the view structure. |
| yaml | string | Optional path to emit YAML description. |

| Argument | Type | Description |
| --- | --- | --- |
| … | Ground | Leaf ground values for the view |

The structure of the view is encoded using JSON, with the top-level object
required to be an `AugmentedBundleType`.

The intrinsic operands correspond to the `AugmentedGroundType` leaves,
and must be of ground type.

This encoding is a trimmed version of what’s used for the old GrandCentral View
annotation.

Example usage:

```
circuit ViewExample:
  public module ViewExample:
    input in : { x : UInt<2>, y : { z : UInt<3>[2] } }
    intrinsic(circt_view<name="view", info="{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\",\"defName\":\"ViewName\",\"elements\":[{\"description\":\"X marks the spot\",\"name\":\"x\",\"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}},{\"description\":\"y bundle\",\"name\":\"y\",\"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\",\"defName\":\"YView\",\"elements\":[{\"name\":\"z\",\"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedVectorType\",\"elements\":[{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"},{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}]}}]}}]}">, in.x, in.y.z[0], in.y.z[1])
```

Example Output:

```
module ViewExample(
  input [1:0] in_x,
  input [2:0] in_y_z_0,
              in_y_z_1
);

  ViewName view();
  assign view.x = in_x;
  assign view.y.z[0] = in_y_z_0;
  assign view.y.z[1] = in_y_z_1;
endmodule

// VCS coverage exclude_file
interface ViewName;
  // X marks the spot
  logic [1:0] x;
  // y bundle
  YView y();
endinterface

// VCS coverage exclude_file
interface YView;
  logic [2:0] z[0:1];
endinterface
```

#### AugmentedGroundType [¶](#augmentedgroundtype)

| Property | Type | Description |
| --- | --- | --- |
| class | string | `sifive.enterprise.grandcentral.AugmentedGroundType` |

Creates a SystemVerilog logic type.

Each ground type corresponds to an operand to the view intrinsic.

Example:

```
{
  "class": "sifive.enterprise.grandcentral.AugmentedGroundType"
}
```

#### AugmentedVectorType [¶](#augmentedvectortype)

| Property | Type | Description |
| --- | --- | --- |
| class | string | `sifive.enterprise.grandcentral.AugmentedVectorType` |
| elements | array | List of augmented types. |

Creates a SystemVerilog unpacked array.

Example:

```
{
  "class": "sifive.enterprise.grandcentral.AugmentedVectorType",
  "elements": [
    {
      "class": "sifive.enterprise.grandcentral.AugmentedGroundType"
    },
    {
      "class": "sifive.enterprise.grandcentral.AugmentedGroundType"
    }
  ]
}
```

#### AugmentedField [¶](#augmentedfield)

| Property | Type | Description |
| --- | --- | --- |
| name | string | Name of the field |
| description | string | A textual description of this type |
| tpe | string | A nested augmented type |

A field in an augmented bundle type. This can provide a small description of
what the field in the bundle is.

#### AugmentedBundleType [¶](#augmentedbundletype)

| Property | Type | Description |
| --- | --- | --- |
| class | string | sifive.enterprise.grandcentral.AugmentedBundleType |
| defName | string | The name of the SystemVerilog interface. May be renamed. |
| elements | array | List of AugmentedFields |

Creates a SystemVerilog interface for each bundle type.

### circt\_verif\_assert [¶](#circt_verif_assert)

Asserts that a property holds.
The property may be an boolean, sequence, or property.
Booleans are represented as `UInt<1>` values.
Sequences and properties are defined by the corresponding `circt_ltl_*` intrinsics and are also represented as `UInt<1>`, but are converted into dedicated sequence and property types later in the compiler.

| Parameter | Type | Description |
| --- | --- | --- |
| label | string | Optional user-defined label |

| Argument | Type | Description |
| --- | --- | --- |
| property | UInt<1> | A property to be checked. |
| enable | UInt<1> | Optional enable condition. |
|  |  | If 0, behaves as if the assert was not present. |

### circt\_verif\_assume [¶](#circt_verif_assume)

Assumes that a property holds.
Otherwise behaves like
[`circt_verif_assert`](#circt_verif_assert).

### circt\_verif\_cover [¶](#circt_verif_cover)

Checks that a property holds at least once, or can hold at all.
Otherwise behaves like
[`circt_verif_assert`](#circt_verif_assert).

### circt\_verif\_require [¶](#circt_verif_require)

Requires that a property holds as a pre-condition to a contract.
Gets converted into an assert if used outside of a FIRRTL `contract`.
Otherwise behaves like
[`circt_verif_assert`](#circt_verif_assert).

### circt\_verif\_ensure [¶](#circt_verif_ensure)

Ensures that a property holds as a post-condition of a contract.
Gets converted into an assert if used outside of a FIRRTL `contract`.
Otherwise behaves like
[`circt_verif_assert`](#circt_verif_assert).

 [Prev - FIRRTL Dialect Rationale](https://circt.llvm.org/docs/Dialects/FIRRTL/RationaleFIRRTL/ "FIRRTL Dialect Rationale")
[Next - 'fsm' Dialect](https://circt.llvm.org/docs/Dialects/FSM/ "'fsm' Dialect") 

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
    - ['firrtl' Dialect-](https://circt.llvm.org/docs/Dialects/FIRRTL/)
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