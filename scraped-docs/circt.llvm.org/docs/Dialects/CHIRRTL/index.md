'chirrtl' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'chirrtl' Dialect
=================

*Types and operations for the chirrtl dialect*

This dialect defines the `chirrtl` dialect, which contains high-level
memory defintions which can be lowered to FIRRTL.

* [Operations](#operations)
  + [`chirrtl.combmem` (::circt::chirrtl::CombMemOp)](#chirrtlcombmem-circtchirrtlcombmemop)
  + [`chirrtl.debugport` (::circt::chirrtl::MemoryDebugPortOp)](#chirrtldebugport-circtchirrtlmemorydebugportop)
  + [`chirrtl.memoryport` (::circt::chirrtl::MemoryPortOp)](#chirrtlmemoryport-circtchirrtlmemoryportop)
  + [`chirrtl.memoryport.access` (::circt::chirrtl::MemoryPortAccessOp)](#chirrtlmemoryportaccess-circtchirrtlmemoryportaccessop)
  + [`chirrtl.seqmem` (::circt::chirrtl::SeqMemOp)](#chirrtlseqmem-circtchirrtlseqmemop)
* [Types](#types)
  + [CMemoryPortType](#cmemoryporttype)
  + [CMemoryType](#cmemorytype)
* [Enums](#enums)
  + [Convention](#convention)
  + [EventControl](#eventcontrol)
  + [LayerConvention](#layerconvention)
  + [LayerSpecialization](#layerspecialization)
  + [MemDirAttr](#memdirattr)
  + [NameKindEnum](#namekindenum)
  + [RUWBehavior](#ruwbehavior)
  + [TargetKind](#targetkind)

Operations [¶](#operations)
---------------------------

### `chirrtl.combmem` (::circt::chirrtl::CombMemOp) [¶](#chirrtlcombmem-circtchirrtlcombmemop)

*Define a new combinational memory*

Syntax:

```
operation ::= `chirrtl.combmem` (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
              `` custom<CombMemOp>(attr-dict) `:` qualified(type($result))
```

Define a new behavioral combinational memory. Combinational memories have a
write latency of 1 and a read latency of 0.

Interfaces: `FNamableOp`, `HasCustomSSAName`, `InnerSymbolOpInterface`

#### Attributes: [¶](#attributes)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `nameKind` | ::circt::firrtl::NameKindEnumAttr | name kind |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `init` | ::circt::firrtl::MemoryInitAttr | Information about the initial state of a memory |
| `prefix` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results)

| Result | Description |
| --- | --- |
| `result` | a behavioral memory |

### `chirrtl.debugport` (::circt::chirrtl::MemoryDebugPortOp) [¶](#chirrtldebugport-circtchirrtlmemorydebugportop)

*Defines a debug memory port on CHIRRTL memory*

Syntax:

```
operation ::= `chirrtl.debugport` $memory custom<MemoryDebugPortOp>(attr-dict) `:`
              functional-type(operands, results)
```

This operation defines a new debug memory port on a `combmem`CHISEL.
`data` is the data returned from the memory port.

Interfaces: `HasCustomSSAName`, `InferTypeOpInterface`

#### Attributes: [¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |

#### Operands: [¶](#operands)

| Operand | Description |
| --- | --- |
| `memory` | a behavioral memory |

#### Results: [¶](#results-1)

| Result | Description |
| --- | --- |
| `data` | reference type |

### `chirrtl.memoryport` (::circt::chirrtl::MemoryPortOp) [¶](#chirrtlmemoryport-circtchirrtlmemoryportop)

*Defines a memory port on CHIRRTL memory*

Syntax:

```
operation ::= `chirrtl.memoryport` $direction $memory `` custom<MemoryPortOp>(attr-dict) `:`
              functional-type(operands, results)
```

This operation defines a new memory port on a `seqmem` or `combmem`CHISEL.
`data` is the data returned from the memory port.

The memory port requires an access point, which sets the enable condition
of the port, the clock, and the address. This is done by passing the the
`port` argument to a `chirrtl.memoryport.access operation`.

Interfaces: `HasCustomSSAName`, `InferTypeOpInterface`

#### Attributes: [¶](#attributes-2)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `direction` | ::circt::firrtl::MemDirAttrAttr | Memory Direction Enum |
| `name` | ::mlir::StringAttr | string attribute |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |

#### Operands: [¶](#operands-1)

| Operand | Description |
| --- | --- |
| `memory` | a behavioral memory |

#### Results: [¶](#results-2)

| Result | Description |
| --- | --- |
| `data` | a base type |
| `port` | a behavioral memory port |

### `chirrtl.memoryport.access` (::circt::chirrtl::MemoryPortAccessOp) [¶](#chirrtlmemoryportaccess-circtchirrtlmemoryportaccessop)

*Enables a memory port*

Syntax:

```
operation ::= `chirrtl.memoryport.access` $port `[` $index `]` `,` $clock attr-dict `:` qualified(type(operands))
```

This operation is used to conditionally enable a memory port, and associate
it with a `clock` and `index`. The memory port will be actuve on the
positive edge of the clock. The index is the address of the memory
accessed. See the FIRRTL rational for more information about why this
operation exists.

#### Operands: [¶](#operands-2)

| Operand | Description |
| --- | --- |
| `port` | a behavioral memory port |
| `index` | sint or uint type |
| `clock` | clock |

### `chirrtl.seqmem` (::circt::chirrtl::SeqMemOp) [¶](#chirrtlseqmem-circtchirrtlseqmemop)

*Define a new sequential memory*

Syntax:

```
operation ::= `chirrtl.seqmem` (`sym` $inner_sym^)? `` custom<NameKind>($nameKind) $ruw
              custom<SeqMemOp>(attr-dict) `:` qualified(type($result))
```

Define a new behavioral sequential memory. Sequential memories have a
write latency and a read latency of 1.

Interfaces: `FNamableOp`, `HasCustomSSAName`, `InnerSymbolOpInterface`

#### Attributes: [¶](#attributes-3)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `ruw` | ::circt::firrtl::RUWBehaviorAttr | read under write behavior |
| `name` | ::mlir::StringAttr | string attribute |
| `nameKind` | ::circt::firrtl::NameKindEnumAttr | name kind |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `init` | ::circt::firrtl::MemoryInitAttr | Information about the initial state of a memory |
| `prefix` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-3)

| Result | Description |
| --- | --- |
| `result` | a behavioral memory |

Types [¶](#types)
-----------------

### CMemoryPortType [¶](#cmemoryporttype)

*A behavioral memory port*

Syntax: `!chirrtl.cmemoryport`

Syntax:

```
cmemoryport-type ::= `cmemoryport`
```

The value of a `cmemoryport` type represents a port which has been
declared on a `cmemory`. This value is used to set the memory port access
conditions.

### CMemoryType [¶](#cmemorytype)

*A behavioral memory*

Syntax:

```
cmemory-type ::= `cmemory` `<` element-type, element-count `>`
```

The value of a `cmemory` type represents a behavioral memory with unknown
ports. This is produced by `combmem` and `seqmem` declarations and used by
`memoryport` declarations to define memories and their ports. A CMemory is
similar to a vector of passive element types.

Examples:

```
!chirrtl.cmemory<uint<32>, 16>
!chirrtl.cmemory<bundle<a : uint<1>>, 16>
```

#### Parameters: [¶](#parameters)

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `firrtl::FIRRTLBaseType` |  |
| numElements | `uint64_t` |  |

Enums [¶](#enums)
-----------------

### Convention [¶](#convention)

*Lowering convention*

#### Cases: [¶](#cases)

| Symbol | Value | String |
| --- | --- | --- |
| Internal | `0` | internal |
| Scalarized | `1` | scalarized |

### EventControl [¶](#eventcontrol)

*Edge control trigger*

#### Cases: [¶](#cases-1)

| Symbol | Value | String |
| --- | --- | --- |
| AtPosEdge | `0` | posedge |
| AtNegEdge | `1` | negedge |
| AtEdge | `2` | edge |

### LayerConvention [¶](#layerconvention)

*Layer convention*

#### Cases: [¶](#cases-2)

| Symbol | Value | String |
| --- | --- | --- |
| Bind | `0` | bind |
| Inline | `1` | inline |

### LayerSpecialization [¶](#layerspecialization)

*Layer specialization*

#### Cases: [¶](#cases-3)

| Symbol | Value | String |
| --- | --- | --- |
| Enable | `0` | enable |
| Disable | `1` | disable |

### MemDirAttr [¶](#memdirattr)

*Memory Direction Enum*

#### Cases: [¶](#cases-4)

| Symbol | Value | String |
| --- | --- | --- |
| Infer | `0` | Infer |
| Read | `1` | Read |
| Write | `2` | Write |
| ReadWrite | `3` | ReadWrite |

### NameKindEnum [¶](#namekindenum)

*Name kind*

#### Cases: [¶](#cases-5)

| Symbol | Value | String |
| --- | --- | --- |
| DroppableName | `0` | droppable\_name |
| InterestingName | `1` | interesting\_name |

### RUWBehavior [¶](#ruwbehavior)

*Read under write behavior*

#### Cases: [¶](#cases-6)

| Symbol | Value | String |
| --- | --- | --- |
| Undefined | `0` | Undefined |
| Old | `1` | Old |
| New | `2` | New |

### TargetKind [¶](#targetkind)

*Object model target kind*

#### Cases: [¶](#cases-7)

| Symbol | Value | String |
| --- | --- | --- |
| DontTouch | `0` | dont\_touch |
| Instance | `1` | instance |
| MemberInstance | `2` | member\_instance |
| MemberReference | `3` | member\_reference |
| Reference | `4` | reference |

 [Prev - 'calyx' Dialect](https://circt.llvm.org/docs/Dialects/Calyx/ "'calyx' Dialect")
[Next - 'comb' Dialect](https://circt.llvm.org/docs/Dialects/Comb/ "'comb' Dialect") 

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