'msft' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'msft' Dialect
==============

*Microsoft internal support dialect*

Umbrella dialect for everything needed to support Microsoft development
but not thoroughly discussed. Most (if not everything) in this dialect is
a candidate for generalization and re-homing.

* [Operations](#operations)
  + [`msft.hlc.linear` (::circt::msft::LinearOp)](#msfthlclinear-circtmsftlinearop)
  + [`msft.instance.dynamic` (::circt::msft::DynamicInstanceOp)](#msftinstancedynamic-circtmsftdynamicinstanceop)
  + [`msft.instance.hierarchy` (::circt::msft::InstanceHierarchyOp)](#msftinstancehierarchy-circtmsftinstancehierarchyop)
  + [`msft.instance.verb_attr` (::circt::msft::DynamicInstanceVerbatimAttrOp)](#msftinstanceverb_attr-circtmsftdynamicinstanceverbatimattrop)
  + [`msft.output` (::circt::msft::OutputOp)](#msftoutput-circtmsftoutputop)
  + [`msft.pd.location` (::circt::msft::PDPhysLocationOp)](#msftpdlocation-circtmsftpdphyslocationop)
  + [`msft.pd.multicycle` (::circt::msft::PDMulticycleOp)](#msftpdmulticycle-circtmsftpdmulticycleop)
  + [`msft.pd.physregion` (::circt::msft::PDPhysRegionOp)](#msftpdphysregion-circtmsftpdphysregionop)
  + [`msft.pd.reg_location` (::circt::msft::PDRegPhysLocationOp)](#msftpdreg_location-circtmsftpdregphyslocationop)
  + [`msft.pe.output` (::circt::msft::PEOutputOp)](#msftpeoutput-circtmsftpeoutputop)
  + [`msft.physical_region` (::circt::msft::DeclPhysicalRegionOp)](#msftphysical_region-circtmsftdeclphysicalregionop)
  + [`msft.systolic.array` (::circt::msft::SystolicArrayOp)](#msftsystolicarray-circtmsftsystolicarrayop)
* [Attributes](#attributes-8)
  + [LocationVectorAttr](#locationvectorattr)
  + [PhysLocationAttr](#physlocationattr)
  + [PhysicalBoundsAttr](#physicalboundsattr)
* [Enums](#enums)
  + [PrimitiveType](#primitivetype)

Operations [¶](#operations)
---------------------------

### `msft.hlc.linear` (::circt::msft::LinearOp) [¶](#msfthlclinear-circtmsftlinearop)

*Model of a linear datapath which can be arbitrarily pipelined*

Syntax:

```
operation ::= `msft.hlc.linear` `clock` $clock attr-dict `:` type($outs) $datapath
```

Defines a feed-forward datapath which can be scheduled into a pipeline.
Due to the feed-forwardness, the inner region is NOT a graph region.
Internally, only combinational operations (`comb`, `msft`, `hw`) are allowed.

Example:

```
msft.module @foo(%in0 : i32, %in1 : i32, %in2 : i32, %clk : i1) -> (out: i32) -> {
  %0 = msft.hlc.linear(%a = %in0, %b = %in1, %c = %in2) clock %clk (i32, i32, i32) -> (i32) {
    %0 = comb.mul %a, %b : i32
    %1 = comb.add %0, %c : i32
    msft.output %1 : i32
  }
}
```

Traits: `SingleBlockImplicitTerminator<OutputOp>`, `SingleBlock`

#### Operands: [¶](#operands)

| Operand | Description |
| --- | --- |
| `clock` | A type for clock-carrying wires |

#### Results: [¶](#results)

| Result | Description |
| --- | --- |
| `outs` | variadic of any type |

### `msft.instance.dynamic` (::circt::msft::DynamicInstanceOp) [¶](#msftinstancedynamic-circtmsftdynamicinstanceop)

*A module instance in the instance hierarchy*

Syntax:

```
operation ::= `msft.instance.dynamic` custom<ImplicitInnerRef>($instanceRef) $body attr-dict
```

Represents an instance (as in instance in the instance hierarchy) referred
to henceforth as a dynamic instance. Specified with a path through the
instance hierarchy (which in the future will be replaced with an AppID).
Lowers to a `hw.hierpath` but unlike a global ref, does not require all of
the ops participating in the hierpath to contain a back pointer attribute.
Allows users to efficiently add placements to a large number of dynamic
instances which happen to map to a small number of static instances by
bulk-adding the necessary `hw.hierpath` attributes.

During the lowering, moves the operations in the body to the top level and
gives them the symbol of the hierpath which was created to replace the
dynamic instance.

Traits: `HasParent<circt::msft::InstanceHierarchyOp, circt::msft::DynamicInstanceOp>`, `NoTerminator`

#### Attributes: [¶](#attributes)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `instanceRef` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |

### `msft.instance.hierarchy` (::circt::msft::InstanceHierarchyOp) [¶](#msftinstancehierarchy-circtmsftinstancehierarchyop)

*The root of an instance hierarchy*

Syntax:

```
operation ::= `msft.instance.hierarchy` $topModuleRef ($instName^)? $body attr-dict
```

Models the “root” / “top” of an instance hierarchy. `DynamicInstanceOp`s
must be contained by this op. Specifies the top module and (optionally) an
“instance” name in the case where there are multiple instances of a
particular module in a design. (As is often the case where one isn’t
producing the design’s “top” module but a subdesign.)

Traits: `HasParent<mlir::ModuleOp>`, `NoTerminator`

#### Attributes: [¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `topModuleRef` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `instName` | ::mlir::StringAttr | string attribute |

### `msft.instance.verb_attr` (::circt::msft::DynamicInstanceVerbatimAttrOp) [¶](#msftinstanceverb_attr-circtmsftdynamicinstanceverbatimattrop)

*Specify an arbitrary attribute attached to a dynamic instance*

Syntax:

```
operation ::= `msft.instance.verb_attr` ($ref^)? `name` `:` $name `value` `:` $value (`path` `:` $subPath^)? attr-dict
```

Allows a user to specify a custom attribute name and value which is attached
to a dynamic instance.

For Quartus tcl, translates to:
set\_instance\_assignment -name $name $value -to $parent|<instance\_path>

Interfaces: `DynInstDataOpInterface`, `UnaryDynInstDataOpInterface`

#### Attributes: [¶](#attributes-2)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `value` | ::mlir::StringAttr | string attribute |
| `subPath` | ::mlir::StringAttr | string attribute |
| `ref` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

### `msft.output` (::circt::msft::OutputOp) [¶](#msftoutput-circtmsftoutputop)

*Termination operation*

Syntax:

```
operation ::= `msft.output` attr-dict ($operands^ `:` qualified(type($operands)))?
```

Traits: `AlwaysSpeculatableImplTrait`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-1)

| Operand | Description |
| --- | --- |
| `operands` | variadic of any type |

### `msft.pd.location` (::circt::msft::PDPhysLocationOp) [¶](#msftpdlocation-circtmsftpdphyslocationop)

*Specify a location for an instance*

Syntax:

```
operation ::= `msft.pd.location` ($ref^)? custom<PhysLoc>($loc) (`path` `:` $subPath^)? attr-dict
```

Used to specify a specific location on an FPGA to place a dynamic instance.
Supports specifying the location of a subpath for extern modules and device
primitives. Intended to live as a child of `instance.dynamic` initially
without the `ref` field. The dynamic instance lowering will fill in `ref`
with the symol of the `hw.hierpath` op corresponding to the lowered dynamic
instance.

Interfaces: `DynInstDataOpInterface`, `UnaryDynInstDataOpInterface`

#### Attributes: [¶](#attributes-3)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `loc` | ::circt::msft::PhysLocationAttr | Descibes a physical location on a device |
| `subPath` | ::mlir::StringAttr | string attribute |
| `ref` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

### `msft.pd.multicycle` (::circt::msft::PDMulticycleOp) [¶](#msftpdmulticycle-circtmsftpdmulticycleop)

*Specify a multicycle constraint*

Syntax:

```
operation ::= `msft.pd.multicycle` $cycles $source `->` $dest attr-dict
```

Specifies a multicycle constraint in between two registers.
`source` and `dest` symbols reference `HierPathOp` symbols denoting the
exact registers in the instance hierarchy to which the constraint applies.

Interfaces: `DynInstDataOpInterface`

#### Attributes: [¶](#attributes-4)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `source` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `dest` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `cycles` | ::mlir::IntegerAttr | 32-bit signless integer attribute whose minimum value is 1 |

### `msft.pd.physregion` (::circt::msft::PDPhysRegionOp) [¶](#msftpdphysregion-circtmsftpdphysregionop)

*Specify a physical region for an instance*

Syntax:

```
operation ::= `msft.pd.physregion` ($ref^)? $physRegionRef (`path` `:` $subPath^)? attr-dict
```

Annotate a particular entity within an op with the region of the devices
on an FPGA to which it should mapped. The physRegionRef must refer to a
DeclPhysicalRegion operation.

Interfaces: `DynInstDataOpInterface`, `UnaryDynInstDataOpInterface`

#### Attributes: [¶](#attributes-5)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `physRegionRef` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `subPath` | ::mlir::StringAttr | string attribute |
| `ref` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

### `msft.pd.reg_location` (::circt::msft::PDRegPhysLocationOp) [¶](#msftpdreg_location-circtmsftpdregphyslocationop)

*Specify register locations*

Syntax:

```
operation ::= `msft.pd.reg_location` (`ref` $ref^)? custom<ListOptionalRegLocList>($locs) attr-dict
```

A version of “PDPhysLocationOp” specialized for registers, which have one
location per bit.

Interfaces: `DynInstDataOpInterface`, `UnaryDynInstDataOpInterface`

#### Attributes: [¶](#attributes-6)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `locs` | ::circt::msft::LocationVectorAttr | Vector of optional locations corresponding to bits in a type |
| `ref` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

### `msft.pe.output` (::circt::msft::PEOutputOp) [¶](#msftpeoutput-circtmsftpeoutputop)

*Set the outputs from a PE block*

Syntax:

```
operation ::= `msft.pe.output` $output attr-dict `:` type($output)
```

Traits: `Terminator`

#### Operands: [¶](#operands-2)

| Operand | Description |
| --- | --- |
| `output` | any type |

### `msft.physical_region` (::circt::msft::DeclPhysicalRegionOp) [¶](#msftphysical_region-circtmsftdeclphysicalregionop)

Syntax:

```
operation ::= `msft.physical_region` $sym_name `,` $bounds attr-dict
```

Traits: `HasParent<mlir::ModuleOp>`

Interfaces: `Symbol`

#### Attributes: [¶](#attributes-7)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `bounds` | ::mlir::ArrayAttr | array of PhysicalBounds |

### `msft.systolic.array` (::circt::msft::SystolicArrayOp) [¶](#msftsystolicarray-circtmsftsystolicarrayop)

*Model of a row/column broadcast systolic array*

Note: the PE region is NOT a graph region. This was intentional since
systolic arrays are entirely feed-forward.

Traits: `SingleBlockImplicitTerminator<PEOutputOp>`, `SingleBlock`

#### Operands: [¶](#operands-3)

| Operand | Description |
| --- | --- |
| `rowInputs` | an ArrayType |
| `colInputs` | an ArrayType |

#### Results: [¶](#results-1)

| Result | Description |
| --- | --- |
| `peOutputs` | an array of arrays |

Attributes [¶](#attributes-8)
-----------------------------

### LocationVectorAttr [¶](#locationvectorattr)

*Vector of optional locations corresponding to bits in a type*

#### Parameters: [¶](#parameters)

| Parameter | C++ type | Description |
| --- | --- | --- |
| type | `::mlir::TypeAttr` |  |
| locs | `::llvm::ArrayRef<::circt::msft::PhysLocationAttr>` |  |

### PhysLocationAttr [¶](#physlocationattr)

*Descibes a physical location on a device*

Annotate a particular entity within an op with the location of the device
on an FPGA to which it should mapped. The coordinates in this attribute
are absolute locations on the device, so if there are two instances of a
module with this annotation incorrect results will be generated. How to
solve this is a more general, open problem.

#### Parameters: [¶](#parameters-1)

| Parameter | C++ type | Description |
| --- | --- | --- |
| primitiveType | `PrimitiveTypeAttr` |  |
| x | `uint64_t` |  |
| y | `uint64_t` |  |
| num | `uint64_t` |  |

### PhysicalBoundsAttr [¶](#physicalboundsattr)

*Describes a rectangle bounding a physical region on a device*

Describes a rectangular bound within a device. The lower and upper bounds
must be specified for both the X and Y axis. The bounds are inclusive.

#### Parameters: [¶](#parameters-2)

| Parameter | C++ type | Description |
| --- | --- | --- |
| xMin | `uint64_t` |  |
| xMax | `uint64_t` |  |
| yMin | `uint64_t` |  |
| yMax | `uint64_t` |  |

Enums [¶](#enums)
-----------------

### PrimitiveType [¶](#primitivetype)

*Type of device at physical location*

#### Cases: [¶](#cases)

| Symbol | Value | String |
| --- | --- | --- |
| M20K | `1` | M20K |
| DSP | `2` | DSP |
| FF | `3` | FF |

 [Prev - LoopSchedule Dialect Rationale](https://circt.llvm.org/docs/Dialects/LoopSchedule/LoopSchedule/ "LoopSchedule Dialect Rationale")
[Next - 'om' Dialect](https://circt.llvm.org/docs/Dialects/OM/ "'om' Dialect") 

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