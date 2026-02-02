'calyx' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'calyx' Dialect
===============

*Types and operations for the Calyx dialect*

Calyx is an intermediate language and infrastructure for building
compilers that generate custom hardware accelerators. For more
information, visit the
[documentation](https://capra.cs.cornell.edu/calyx/).

* [Operations](#operations)
  + [`calyx.assign` (::circt::calyx::AssignOp)](#calyxassign-circtcalyxassignop)
  + [`calyx.comb_component` (::circt::calyx::CombComponentOp)](#calyxcomb_component-circtcalyxcombcomponentop)
  + [`calyx.comb_group` (::circt::calyx::CombGroupOp)](#calyxcomb_group-circtcalyxcombgroupop)
  + [`calyx.component` (::circt::calyx::ComponentOp)](#calyxcomponent-circtcalyxcomponentop)
  + [`calyx.constant` (::circt::calyx::ConstantOp)](#calyxconstant-circtcalyxconstantop)
  + [`calyx.control` (::circt::calyx::ControlOp)](#calyxcontrol-circtcalyxcontrolop)
  + [`calyx.cycle` (::circt::calyx::CycleOp)](#calyxcycle-circtcalyxcycleop)
  + [`calyx.enable` (::circt::calyx::EnableOp)](#calyxenable-circtcalyxenableop)
  + [`calyx.group` (::circt::calyx::GroupOp)](#calyxgroup-circtcalyxgroupop)
  + [`calyx.group_done` (::circt::calyx::GroupDoneOp)](#calyxgroup_done-circtcalyxgroupdoneop)
  + [`calyx.group_go` (::circt::calyx::GroupGoOp)](#calyxgroup_go-circtcalyxgroupgoop)
  + [`calyx.ieee754.add` (::circt::calyx::AddFOpIEEE754)](#calyxieee754add-circtcalyxaddfopieee754)
  + [`calyx.ieee754.compare` (::circt::calyx::CompareFOpIEEE754)](#calyxieee754compare-circtcalyxcomparefopieee754)
  + [`calyx.ieee754.divSqrt` (::circt::calyx::DivSqrtOpIEEE754)](#calyxieee754divsqrt-circtcalyxdivsqrtopieee754)
  + [`calyx.ieee754.fpToInt` (::circt::calyx::FpToIntOpIEEE754)](#calyxieee754fptoint-circtcalyxfptointopieee754)
  + [`calyx.ieee754.intToFp` (::circt::calyx::IntToFpOpIEEE754)](#calyxieee754inttofp-circtcalyxinttofpopieee754)
  + [`calyx.ieee754.mul` (::circt::calyx::MulFOpIEEE754)](#calyxieee754mul-circtcalyxmulfopieee754)
  + [`calyx.if` (::circt::calyx::IfOp)](#calyxif-circtcalyxifop)
  + [`calyx.instance` (::circt::calyx::InstanceOp)](#calyxinstance-circtcalyxinstanceop)
  + [`calyx.invoke` (::circt::calyx::InvokeOp)](#calyxinvoke-circtcalyxinvokeop)
  + [`calyx.memory` (::circt::calyx::MemoryOp)](#calyxmemory-circtcalyxmemoryop)
  + [`calyx.par` (::circt::calyx::ParOp)](#calyxpar-circtcalyxparop)
  + [`calyx.primitive` (::circt::calyx::PrimitiveOp)](#calyxprimitive-circtcalyxprimitiveop)
  + [`calyx.register` (::circt::calyx::RegisterOp)](#calyxregister-circtcalyxregisterop)
  + [`calyx.repeat` (::circt::calyx::RepeatOp)](#calyxrepeat-circtcalyxrepeatop)
  + [`calyx.seq` (::circt::calyx::SeqOp)](#calyxseq-circtcalyxseqop)
  + [`calyx.seq_mem` (::circt::calyx::SeqMemoryOp)](#calyxseq_mem-circtcalyxseqmemoryop)
  + [`calyx.static_group` (::circt::calyx::StaticGroupOp)](#calyxstatic_group-circtcalyxstaticgroupop)
  + [`calyx.static_if` (::circt::calyx::StaticIfOp)](#calyxstatic_if-circtcalyxstaticifop)
  + [`calyx.static_par` (::circt::calyx::StaticParOp)](#calyxstatic_par-circtcalyxstaticparop)
  + [`calyx.static_repeat` (::circt::calyx::StaticRepeatOp)](#calyxstatic_repeat-circtcalyxstaticrepeatop)
  + [`calyx.static_seq` (::circt::calyx::StaticSeqOp)](#calyxstatic_seq-circtcalyxstaticseqop)
  + [`calyx.std_add` (::circt::calyx::AddLibOp)](#calyxstd_add-circtcalyxaddlibop)
  + [`calyx.std_and` (::circt::calyx::AndLibOp)](#calyxstd_and-circtcalyxandlibop)
  + [`calyx.std_divs_pipe` (::circt::calyx::DivSPipeLibOp)](#calyxstd_divs_pipe-circtcalyxdivspipelibop)
  + [`calyx.std_divu_pipe` (::circt::calyx::DivUPipeLibOp)](#calyxstd_divu_pipe-circtcalyxdivupipelibop)
  + [`calyx.std_eq` (::circt::calyx::EqLibOp)](#calyxstd_eq-circtcalyxeqlibop)
  + [`calyx.std_ge` (::circt::calyx::GeLibOp)](#calyxstd_ge-circtcalyxgelibop)
  + [`calyx.std_gt` (::circt::calyx::GtLibOp)](#calyxstd_gt-circtcalyxgtlibop)
  + [`calyx.std_le` (::circt::calyx::LeLibOp)](#calyxstd_le-circtcalyxlelibop)
  + [`calyx.std_lsh` (::circt::calyx::LshLibOp)](#calyxstd_lsh-circtcalyxlshlibop)
  + [`calyx.std_lt` (::circt::calyx::LtLibOp)](#calyxstd_lt-circtcalyxltlibop)
  + [`calyx.std_mult_pipe` (::circt::calyx::MultPipeLibOp)](#calyxstd_mult_pipe-circtcalyxmultpipelibop)
  + [`calyx.std_mux` (::circt::calyx::MuxLibOp)](#calyxstd_mux-circtcalyxmuxlibop)
  + [`calyx.std_neq` (::circt::calyx::NeqLibOp)](#calyxstd_neq-circtcalyxneqlibop)
  + [`calyx.std_not` (::circt::calyx::NotLibOp)](#calyxstd_not-circtcalyxnotlibop)
  + [`calyx.std_or` (::circt::calyx::OrLibOp)](#calyxstd_or-circtcalyxorlibop)
  + [`calyx.std_pad` (::circt::calyx::PadLibOp)](#calyxstd_pad-circtcalyxpadlibop)
  + [`calyx.std_rems_pipe` (::circt::calyx::RemSPipeLibOp)](#calyxstd_rems_pipe-circtcalyxremspipelibop)
  + [`calyx.std_remu_pipe` (::circt::calyx::RemUPipeLibOp)](#calyxstd_remu_pipe-circtcalyxremupipelibop)
  + [`calyx.std_rsh` (::circt::calyx::RshLibOp)](#calyxstd_rsh-circtcalyxrshlibop)
  + [`calyx.std_seq` (::circt::calyx::SeqLibOp)](#calyxstd_seq-circtcalyxseqlibop)
  + [`calyx.std_sge` (::circt::calyx::SgeLibOp)](#calyxstd_sge-circtcalyxsgelibop)
  + [`calyx.std_sgt` (::circt::calyx::SgtLibOp)](#calyxstd_sgt-circtcalyxsgtlibop)
  + [`calyx.std_shru` (::circt::calyx::ShruLibOp)](#calyxstd_shru-circtcalyxshrulibop)
  + [`calyx.std_signext` (::circt::calyx::ExtSILibOp)](#calyxstd_signext-circtcalyxextsilibop)
  + [`calyx.std_sle` (::circt::calyx::SleLibOp)](#calyxstd_sle-circtcalyxslelibop)
  + [`calyx.std_slice` (::circt::calyx::SliceLibOp)](#calyxstd_slice-circtcalyxslicelibop)
  + [`calyx.std_slt` (::circt::calyx::SltLibOp)](#calyxstd_slt-circtcalyxsltlibop)
  + [`calyx.std_sneq` (::circt::calyx::SneqLibOp)](#calyxstd_sneq-circtcalyxsneqlibop)
  + [`calyx.std_srsh` (::circt::calyx::SrshLibOp)](#calyxstd_srsh-circtcalyxsrshlibop)
  + [`calyx.std_sub` (::circt::calyx::SubLibOp)](#calyxstd_sub-circtcalyxsublibop)
  + [`calyx.std_wire` (::circt::calyx::WireLibOp)](#calyxstd_wire-circtcalyxwirelibop)
  + [`calyx.std_xor` (::circt::calyx::XorLibOp)](#calyxstd_xor-circtcalyxxorlibop)
  + [`calyx.undef` (::circt::calyx::UndefinedOp)](#calyxundef-circtcalyxundefinedop)
  + [`calyx.undefined` (::circt::calyx::UndefLibOp)](#calyxundefined-circtcalyxundeflibop)
  + [`calyx.while` (::circt::calyx::WhileOp)](#calyxwhile-circtcalyxwhileop)
  + [`calyx.wires` (::circt::calyx::WiresOp)](#calyxwires-circtcalyxwiresop)

Operations [¶](#operations)
---------------------------

### `calyx.assign` (::circt::calyx::AssignOp) [¶](#calyxassign-circtcalyxassignop)

*Calyx Assignment*

The “calyx.assign” operation represents a non-blocking
assignment. An assignment may optionally be guarded,
which controls when the assignment should be active.
This operation should only be instantiated in the
“calyx.wires” section or a “calyx.group”.

```
  calyx.assign %dest = %src : i16
  calyx.assign %dest = %guard ? %src : i16
```

Traits: `HasParent<GroupOp, CombGroupOp, StaticGroupOp, WiresOp>`

#### Operands: [¶](#operands)

| Operand | Description |
| --- | --- |
| `dest` | any type |
| `src` | any type |
| `guard` | 1-bit signless integer |

### `calyx.comb_component` (::circt::calyx::CombComponentOp) [¶](#calyxcomb_component-circtcalyxcombcomponentop)

*Calyx Combinational Component*

The “calyx.comb\_component” operation represents a Calyx combinational component containing:
(1) In- and output port definitions that define the interface.
(2) Combinational cells and wires.

```
  calyx.comb_component @C(%in: i32) -> (%out: i16) {
    ...
    calyx.wires { ... }
  }
```

Traits: `HasParent<mlir::ModuleOp>`, `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallableOpInterface`, `ComponentInterface`, `FunctionOpInterface`, `OpAsmOpInterface`, `Symbol`

#### Attributes: [¶](#attributes)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `function_type` | ::mlir::TypeAttr | type attribute of function type |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `portNames` | ::mlir::ArrayAttr | array attribute |
| `portAttributes` | ::mlir::ArrayAttr | array attribute |
| `portDirections` | ::mlir::IntegerAttr | arbitrary integer attribute |

### `calyx.comb_group` (::circt::calyx::CombGroupOp) [¶](#calyxcomb_group-circtcalyxcombgroupop)

*Calyx Combinational Group*

Syntax:

```
operation ::= `calyx.comb_group` $sym_name $body attr-dict
```

Represents a Calyx combinational group, which is a collection
of combinational assignments that are only active when the group
is run from the control execution schedule.
A combinational group does not have group\_go or group\_done operators.

```
  calyx.comb_group @MyCombGroup {
    calyx.assign %1 = %2 : i32
  }
```

Traits: `HasParent<WiresOp>`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

Interfaces: `GroupInterface`, `RegionKindInterface`, `Symbol`

#### Attributes: [¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `calyx.component` (::circt::calyx::ComponentOp) [¶](#calyxcomponent-circtcalyxcomponentop)

*Calyx Component*

The “calyx.component” operation represents an overall Calyx component containing:
(1) In- and output port definitions that define the interface.
(2) The cells, wires, and control schedule.

A Calyx component requires attributes `clk`, `go`, and `reset` on separate input ports,
and `done` on an output port.

```
  calyx.component @C(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i16, %done: i1 {done}) {
    ...
    calyx.wires { ... }
    calyx.control { ... }
  }
```

Traits: `HasParent<mlir::ModuleOp>`, `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallableOpInterface`, `ComponentInterface`, `FunctionOpInterface`, `OpAsmOpInterface`, `RegionKindInterface`, `Symbol`

#### Attributes: [¶](#attributes-2)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `function_type` | ::mlir::TypeAttr | type attribute of function type |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `portNames` | ::mlir::ArrayAttr | array attribute |
| `portAttributes` | ::mlir::ArrayAttr | array attribute |
| `portDirections` | ::mlir::IntegerAttr | arbitrary integer attribute |

### `calyx.constant` (::circt::calyx::ConstantOp) [¶](#calyxconstant-circtcalyxconstantop)

*Constant capable of representing an integer or floating point value*

Syntax:

```
operation ::= `calyx.constant` $sym_name ` ` `<` $value `>` attr-dict `:` qualified(type($out))
```

The `constant` operation is a wrapper around bit vectors with fixed-size number of bits.
Specific value and intended type should be specified via attribute only.

Example:

```
// Integer constant
%1 = calyx.constant <42 : i32> : i32

// Floating point constant
%1 = calyx.constant <4.2 : f32> : i32
```

Traits: `ConstantLike`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-3)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `value` | ::mlir::TypedAttr | TypedAttr instance |

#### Results: [¶](#results)

| Result | Description |
| --- | --- |
| `out` | signless integer |

### `calyx.control` (::circt::calyx::ControlOp) [¶](#calyxcontrol-circtcalyxcontrolop)

*Calyx Control*

Syntax:

```
operation ::= `calyx.control` $body attr-dict
```

The “calyx.control” operation represents the
execution schedule defined for the given
component, i.e. when each group executes.

```
  calyx.control {
    calyx.seq {
      calyx.enable @GroupA
    }
  }
```

Traits: `HasParent<ComponentOp>`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

### `calyx.cycle` (::circt::calyx::CycleOp) [¶](#calyxcycle-circtcalyxcycleop)

*Calyx Static Cycle Op*

Returns an I1 signal that is active `start` cycles after the static
group starts, and inactive all other times. Optional `end`
attribute allows the signal to be active from until `end` cycles
after the static group starts. Must be used within a static group.

```
  calyx.static_group latency<2> @MyStaticGroup {
    %1 = calyx.cycle 1 : i1
  }
```

Traits: `HasParent<StaticGroupOp>`

#### Attributes: [¶](#attributes-4)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `start` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `end` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Results: [¶](#results-1)

| Result | Description |
| --- | --- |
| `active` | 1-bit signless integer |

### `calyx.enable` (::circt::calyx::EnableOp) [¶](#calyxenable-circtcalyxenableop)

*Calyx Enable*

Syntax:

```
operation ::= `calyx.enable` $groupName attr-dict
```

The “calyx.enable” operation represents the execution of
a group defined explicitly in the “calyx.wires” section.

The ‘compiledGroups’ attribute is used in the Compile
Control pass to track which groups are compiled within
the new compilation group.

```
  calyx.enable @SomeGroup
```

Traits: `ControlLike`

#### Attributes: [¶](#attributes-5)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `groupName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `compiledGroups` | ::mlir::ArrayAttr | array attribute |

### `calyx.group` (::circt::calyx::GroupOp) [¶](#calyxgroup-circtcalyxgroupop)

*Calyx Group*

Syntax:

```
operation ::= `calyx.group` $sym_name $body attr-dict
```

Represents a Calyx group, which is a collection
of assignments that are only active when the group
is run from the control execution schedule. A group
signifies its termination with a special port named
a “done” port.

```
  calyx.group @MyGroup {
    calyx.assign %1 = %2 : i32
    calyx.group_done %3 : i1
  }
```

Traits: `HasParent<WiresOp>`, `NoRegionArguments`, `SingleBlock`

Interfaces: `GroupInterface`, `RegionKindInterface`, `Symbol`

#### Attributes: [¶](#attributes-6)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `calyx.group_done` (::circt::calyx::GroupDoneOp) [¶](#calyxgroup_done-circtcalyxgroupdoneop)

*Calyx Group Done Port*

The “calyx.group\_done” operation represents a port on a
Calyx group that signifies when the group is finished.
A done operation may optionally be guarded, which controls
when the group’s done operation should be active.

```
  calyx.group_done %src : i1
  calyx.group_done %guard ? %src : i1
```

Traits: `HasParent<GroupOp>`, `Terminator`

#### Operands: [¶](#operands-1)

| Operand | Description |
| --- | --- |
| `src` | 1-bit signless integer |
| `guard` | 1-bit signless integer |

### `calyx.group_go` (::circt::calyx::GroupGoOp) [¶](#calyxgroup_go-circtcalyxgroupgoop)

*Calyx Group Go Port*

The “calyx.group\_go” operation represents a port on a
Calyx group that signifies when the group begins.
A go operation may optionally be guarded, which
controls when the group’s go operation should be
active. The go operation should only be inserted
during the Go Insertion pass. It does not
receive a source until the Compile Control pass.

```
  %group_name1.go = calyx.group_go %src : i1
  %group_name2.go = calyx.group_go %guard ? %src : i1
```

Traits: `HasParent<GroupOp>`

Interfaces: `OpAsmOpInterface`

#### Operands: [¶](#operands-2)

| Operand | Description |
| --- | --- |
| `src` | 1-bit signless integer |
| `guard` | 1-bit signless integer |

#### Results: [¶](#results-2)

| Result | Description |
| --- | --- |
| «unnamed» | 1-bit signless integer |

### `calyx.ieee754.add` (::circt::calyx::AddFOpIEEE754) [¶](#calyxieee754add-circtcalyxaddfopieee754)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.ieee754.add` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Interfaces: `CellInterface`, `FloatingPointOpInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-7)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-3)

| Result | Description |
| --- | --- |
| `clk` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |
| `control` | 1-bit signless integer |
| `subOp` | 1-bit signless integer |
| `left` | signless integer |
| `right` | signless integer |
| `roundingMode` | 3-bit signless integer |
| `out` | signless integer |
| `exceptionalFlags` | 5-bit signless integer |
| `done` | 1-bit signless integer |

### `calyx.ieee754.compare` (::circt::calyx::CompareFOpIEEE754) [¶](#calyxieee754compare-circtcalyxcomparefopieee754)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.ieee754.compare` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Interfaces: `CellInterface`, `FloatingPointOpInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-8)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-4)

| Result | Description |
| --- | --- |
| `clk` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |
| `left` | signless integer |
| `right` | signless integer |
| `signaling` | 1-bit signless integer |
| `lt` | 1-bit signless integer |
| `eq` | 1-bit signless integer |
| `gt` | 1-bit signless integer |
| `unordered` | 1-bit signless integer |
| `exceptionalFlags` | 5-bit signless integer |
| `done` | 1-bit signless integer |

### `calyx.ieee754.divSqrt` (::circt::calyx::DivSqrtOpIEEE754) [¶](#calyxieee754divsqrt-circtcalyxdivsqrtopieee754)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.ieee754.divSqrt` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Interfaces: `CellInterface`, `FloatingPointOpInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-9)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-5)

| Result | Description |
| --- | --- |
| `clk` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |
| `control` | 1-bit signless integer |
| `sqrtOp` | 1-bit signless integer |
| `left` | signless integer |
| `right` | signless integer |
| `roundingMode` | 3-bit signless integer |
| `out` | signless integer |
| `exceptionalFlags` | 5-bit signless integer |
| `done` | 1-bit signless integer |

### `calyx.ieee754.fpToInt` (::circt::calyx::FpToIntOpIEEE754) [¶](#calyxieee754fptoint-circtcalyxfptointopieee754)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.ieee754.fpToInt` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `FloatingPointOpInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-10)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-6)

| Result | Description |
| --- | --- |
| `clk` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |
| `in` | signless integer |
| `signedOut` | 1-bit signless integer |
| `out` | signless integer |
| `done` | 1-bit signless integer |

### `calyx.ieee754.intToFp` (::circt::calyx::IntToFpOpIEEE754) [¶](#calyxieee754inttofp-circtcalyxinttofpopieee754)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.ieee754.intToFp` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `FloatingPointOpInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-11)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-7)

| Result | Description |
| --- | --- |
| `clk` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |
| `in` | signless integer |
| `signedIn` | 1-bit signless integer |
| `out` | signless integer |
| `done` | 1-bit signless integer |

### `calyx.ieee754.mul` (::circt::calyx::MulFOpIEEE754) [¶](#calyxieee754mul-circtcalyxmulfopieee754)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.ieee754.mul` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Interfaces: `CellInterface`, `FloatingPointOpInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-12)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-8)

| Result | Description |
| --- | --- |
| `clk` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |
| `control` | 1-bit signless integer |
| `left` | signless integer |
| `right` | signless integer |
| `roundingMode` | 3-bit signless integer |
| `out` | signless integer |
| `exceptionalFlags` | 5-bit signless integer |
| `done` | 1-bit signless integer |

### `calyx.if` (::circt::calyx::IfOp) [¶](#calyxif-circtcalyxifop)

*Calyx If*

Syntax:

```
operation ::= `calyx.if` $cond (`with` $groupName^)? $thenRegion (`else` $elseRegion^)? attr-dict
```

The “calyx.if” operation represents and if-then-else construct for
conditionally executing two Calyx groups. The operands to an if operation is
a 1-bit port and an optional combinational group under which this port is driven.

Note: The native and CIRCT Calyx IRs may diverge wrt. ‘with’ execution, see:
<https://github.com/cucapra/calyx/discussions/588>

```
  calyx.if %1 with @G1 {
    calyx.enable @G2
    ...
  } else {
    calyx.enable @G3
    ...
  }
  calyx.if %1 {
    calyx.enable @G2
    ...
  }
```

Traits: `ControlLike`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

Interfaces: `IfInterface`

#### Attributes: [¶](#attributes-13)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `groupName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands: [¶](#operands-3)

| Operand | Description |
| --- | --- |
| `cond` | 1-bit signless integer |

### `calyx.instance` (::circt::calyx::InstanceOp) [¶](#calyxinstance-circtcalyxinstanceop)

*Calyx Component Instance*

Syntax:

```
operation ::= `calyx.instance` $sym_name `of` $componentName attr-dict (`:` qualified(type($results))^)?
```

Represents an instance of a Calyx component, which may include state.

```
  %c.in, %c.out = calyx.instance @c of @MyComponent : i64, i16
```

Interfaces: `CellInterface`, `OpAsmOpInterface`, `SymbolUserOpInterface`

#### Attributes: [¶](#attributes-14)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `componentName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Results: [¶](#results-9)

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `calyx.invoke` (::circt::calyx::InvokeOp) [¶](#calyxinvoke-circtcalyxinvokeop)

*Calyx Invoke*

calyx.invoke is similar to the behavior of a function
call, which invokes a given component.

The ‘callee’ attribute is the name of the component,
the ‘ports’ attribute specifies the input port of the component when it is invoked,
the ‘inputs’ attribute specifies the assignment on the corresponding port,
and the ‘refCellsMap’ attribute maps the reference cells in the `callee` with
the original cells in the caller.

```
  calyx.component @identity {
    %mem.addr0, %mem.clk, ... = calyx.seq_mem @mem <[1] x 32> [1] {external = false}
  }
  %id.in, %id.out, ... = calyx.instance @id of @identity : i32, i32, ...
  %r.in, ... = calyx.register @r : i32, ...
  %mem_1.addr0, %mem_1.clk, ... = calyx.seq_mem @mem_1 <[1] x 32> [1] {external = true}
  ...
  calyx.control {
    calyx.seq {
      calyx.invoke @id[mem = mem_1](%id.in = %c1_10, %r.in = %id.out) -> (i32, i32)
    }
  }
```

Traits: `ControlLike`, `SameVariadicOperandSize`

#### Attributes: [¶](#attributes-15)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `callee` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `refCellsMap` | ::mlir::ArrayAttr | array attribute |
| `portNames` | ::mlir::ArrayAttr | array attribute |
| `inputNames` | ::mlir::ArrayAttr | array attribute |

#### Operands: [¶](#operands-4)

| Operand | Description |
| --- | --- |
| `ports` | variadic of any type |
| `inputs` | variadic of any type |

### `calyx.memory` (::circt::calyx::MemoryOp) [¶](#calyxmemory-circtcalyxmemoryop)

*Defines a memory*

Syntax:

```
operation ::= `calyx.memory` $sym_name ` ` `<` $sizes `x` $width `>` $addrSizes attr-dict `:` qualified(type($results))
```

The “calyx.memory” op defines a memory. Memories can have any number of
dimensions, as specified by the length of the `$sizes` and `$addrSizes`
arrays. The `$addrSizes` specify the bitwidth of each dimension’s address,
and should be wide enough to address the range of the corresponding
dimension in `$sizes`. The `$width` attribute dictates the width of a single
element.

See
<https://docs.calyxir.org/libraries/core.html#memories> for
more information.

```
  // A 1-dimensional, 32-bit memory with size dimension 1. Equivalent representation in the native compiler:
  // `m1 = std_mem_d1(32, 1, 1)`
  %m1.addr0, %m1.write_data, %m1.write_en, %m1.clk, %m1.read_data, %m1.done = calyx.memory @m1 <[1] x 32> [1] : i1, i32, i1, i1, i32, i1

  // A 2-dimensional, 8-bit memory with size dimensions 64 x 64. Equivalent representation in the native compiler:
  // `m2 = std_mem_d2(8, 64, 64, 6, 6)`
  %m2.addr0, %m2.addr1, %m2.write_data, %m2.write_en, %m2.clk, %m2.read_data, %m2.done = calyx.memory @m2 <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
```

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-16)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `width` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `sizes` | ::mlir::ArrayAttr | array attribute |
| `addrSizes` | ::mlir::ArrayAttr | array attribute |

#### Results: [¶](#results-10)

| Result | Description |
| --- | --- |
| `results` | variadic of signless integer |

### `calyx.par` (::circt::calyx::ParOp) [¶](#calyxpar-circtcalyxparop)

*Calyx Parallel*

Syntax:

```
operation ::= `calyx.par` $body attr-dict
```

The “calyx.par” operation executes the
control within its region in parallel.

```
  calyx.par {
    // G1 and G2 will execute in parallel.
    // The region is complete when both
    // G1 and G2 are done.
    calyx.enable @G1
    calyx.enable @G2
  }
```

Traits: `ControlLike`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

### `calyx.primitive` (::circt::calyx::PrimitiveOp) [¶](#calyxprimitive-circtcalyxprimitiveop)

*Calyx Primitive Instance*

Syntax:

```
operation ::= `calyx.primitive` $sym_name `of` $primitiveName `` custom<ParameterList>($parameters) attr-dict (`:` qualified(type($results))^)?
```

Represents an instance of a Calyx primitive.

```
  %c.in, %c.out = calyx.primitive @c of @MyExternalPrimitive : i64, i16
```

Interfaces: `CellInterface`, `OpAsmOpInterface`, `SymbolUserOpInterface`

#### Attributes: [¶](#attributes-17)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `parameters` | ::mlir::ArrayAttr | parameter array |
| `primitiveName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Results: [¶](#results-11)

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `calyx.register` (::circt::calyx::RegisterOp) [¶](#calyxregister-circtcalyxregisterop)

*Defines a register*

Syntax:

```
operation ::= `calyx.register` $sym_name attr-dict `:` qualified(type(results))
```

The “calyx.register” op defines a register.

```
  // A 32-bit register.
  %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
```

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-18)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-12)

| Result | Description |
| --- | --- |
| `in` | any type |
| `write_en` | 1-bit signless integer |
| `clk` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `out` | any type |
| `done` | 1-bit signless integer |

### `calyx.repeat` (::circt::calyx::RepeatOp) [¶](#calyxrepeat-circtcalyxrepeatop)

*Calyx Dynamic Repeat*

Syntax:

```
operation ::= `calyx.repeat` $count $body attr-dict
```

The “calyx.repeat” operation represents the repeated execution of
the control within its region.
The key difference with static repeat is that the body (unlike with static
repeat) can be dynamically timed.

```
  calyx.repeat 10 {
    calyx.enable @G1
    ...
  }
```

Traits: `ControlLike`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

#### Attributes: [¶](#attributes-19)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `count` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

### `calyx.seq` (::circt::calyx::SeqOp) [¶](#calyxseq-circtcalyxseqop)

*Calyx Sequential*

Syntax:

```
operation ::= `calyx.seq` $body attr-dict
```

The “calyx.seq” operation executes the
control within its region sequentially.

```
  calyx.seq {
    // G2 will not begin execution until G1 is done.
    calyx.enable @G1
    calyx.enable @G2
  }
```

Traits: `ControlLike`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

### `calyx.seq_mem` (::circt::calyx::SeqMemoryOp) [¶](#calyxseq_mem-circtcalyxseqmemoryop)

*Defines a memory with sequential read*

Syntax:

```
operation ::= `calyx.seq_mem` $sym_name ` ` `<` $sizes `x` $width `>` $addrSizes attr-dict `:` qualified(type($results))
```

The “calyx.seq\_mem” op defines a memory with sequential reads. Memories can
have any number of dimensions, as specified by the length of the `$sizes` and
`$addrSizes` arrays. The `$addrSizes` specify the bitwidth of each dimension’s
address, and should be wide enough to address the range of the corresponding
dimension in `$sizes`. The `$width` attribute dictates the width of a single
element.

See
<https://docs.calyxir.org/libraries/core.html#memories> for
more information.

```
  // A 1-dimensional, 32-bit memory with size dimension 1. Equivalent representation in the native compiler:
  // `m1 = seq_mem_d1(32, 1, 1)`
  %m1.addr0, %m1.write_data, %m1.write_en, %m1.clk, %m1.read_data, %m1.read_en, %m1.done = calyx.memory @m1 <[1] x 32> [1] : i1, i32, i1, i1, i32, i1

  // A 2-dimensional, 8-bit memory with size dimensions 64 x 64. Equivalent representation in the native compiler:
  // `m2 = seq_mem_d2(8, 64, 64, 6, 6)`
  %m2.addr0, %m2.addr1, %m2.write_data, %m2.write_en, %m2.write_done, %m2.clk, %m2.read_data, %m2.read_en, %m2.read_done = calyx.memory @m2 <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
```

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-20)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `width` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `sizes` | ::mlir::ArrayAttr | array attribute |
| `addrSizes` | ::mlir::ArrayAttr | array attribute |

#### Results: [¶](#results-13)

| Result | Description |
| --- | --- |
| `results` | variadic of signless integer |

### `calyx.static_group` (::circt::calyx::StaticGroupOp) [¶](#calyxstatic_group-circtcalyxstaticgroupop)

*Calyx Static Group*

Syntax:

```
operation ::= `calyx.static_group` `latency` `<` $latency `>` $sym_name $body attr-dict
```

Represents a Calyx static group, which is a collection
of assignments that are only active when the group
is run from the control execution schedule. A group
signifies its termination with a special port named
a “done” port.

```
  calyx.static_group latency<1> @MyStaticGroup {
    calyx.assign %1 = %2 : i32
  }
```

Traits: `HasParent<WiresOp>`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

Interfaces: `GroupInterface`, `RegionKindInterface`, `Symbol`

#### Attributes: [¶](#attributes-21)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `latency` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

### `calyx.static_if` (::circt::calyx::StaticIfOp) [¶](#calyxstatic_if-circtcalyxstaticifop)

*Calyx Static If*

Syntax:

```
operation ::= `calyx.static_if` $cond $thenRegion (`else` $elseRegion^)? attr-dict
```

The “calyx.static\_if” operation represents an if-then-else construct for
conditionally executing two Calyx groups. The operands to an if operation is
a 1-bit port.

```
  calyx.static_if %1 {
    calyx.enable @G2
    ...
  } else {
    calyx.enable @G3
    ...
  }
  calyx.if %1 {
    calyx.enable @G2
    ...
  }
```

Traits: `ControlLike`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

Interfaces: `IfInterface`

#### Operands: [¶](#operands-5)

| Operand | Description |
| --- | --- |
| `cond` | 1-bit signless integer |

### `calyx.static_par` (::circt::calyx::StaticParOp) [¶](#calyxstatic_par-circtcalyxstaticparop)

*Calyx Static Parallel*

Syntax:

```
operation ::= `calyx.static_par` $body attr-dict
```

The “calyx.static\_par” operation executes the
control within its region in parallel.

```
  calyx.static_par {
    // G1 and G2 will execute in parallel.
    // The region is complete when both
    // G1 and G2 are done.
    calyx.enable @G1
    calyx.enable @G2
  }
```

Traits: `ControlLike`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

### `calyx.static_repeat` (::circt::calyx::StaticRepeatOp) [¶](#calyxstatic_repeat-circtcalyxstaticrepeatop)

*Calyx Static Repeat*

Syntax:

```
operation ::= `calyx.static_repeat` $count $body attr-dict
```

The “calyx.static\_repeat” operation represents the repeated execution of
the control within its region. All control within the region must be static.

```
  calyx.static_repeat 10 {
    calyx.enable @G1
    ...
  }
```

Traits: `ControlLike`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

#### Attributes: [¶](#attributes-22)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `count` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

### `calyx.static_seq` (::circt::calyx::StaticSeqOp) [¶](#calyxstatic_seq-circtcalyxstaticseqop)

*Calyx Static Seq*

Syntax:

```
operation ::= `calyx.static_seq` $body attr-dict
```

The “calyx.static\_seq” operation executes the
control within its region sequentially.

```
  calyx.static_seq {
    // G2 will not begin execution until G1 is done.
    calyx.enable @G1
    calyx.enable @G2
  }
```

Traits: `ControlLike`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

### `calyx.std_add` (::circt::calyx::AddLibOp) [¶](#calyxstd_add-circtcalyxaddlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_add` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-23)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-14)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | any type |

### `calyx.std_and` (::circt::calyx::AndLibOp) [¶](#calyxstd_and-circtcalyxandlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_and` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-24)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-15)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | any type |

### `calyx.std_divs_pipe` (::circt::calyx::DivSPipeLibOp) [¶](#calyxstd_divs_pipe-circtcalyxdivspipelibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_divs_pipe` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-25)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-16)

| Result | Description |
| --- | --- |
| `clk` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |
| `left` | any type |
| `right` | any type |
| `out` | any type |
| `done` | 1-bit signless integer |

### `calyx.std_divu_pipe` (::circt::calyx::DivUPipeLibOp) [¶](#calyxstd_divu_pipe-circtcalyxdivupipelibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_divu_pipe` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-26)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-17)

| Result | Description |
| --- | --- |
| `clk` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |
| `left` | any type |
| `right` | any type |
| `out` | any type |
| `done` | 1-bit signless integer |

### `calyx.std_eq` (::circt::calyx::EqLibOp) [¶](#calyxstd_eq-circtcalyxeqlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_eq` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-27)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-18)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | 1-bit signless integer |

### `calyx.std_ge` (::circt::calyx::GeLibOp) [¶](#calyxstd_ge-circtcalyxgelibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_ge` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-28)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-19)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | 1-bit signless integer |

### `calyx.std_gt` (::circt::calyx::GtLibOp) [¶](#calyxstd_gt-circtcalyxgtlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_gt` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-29)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-20)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | 1-bit signless integer |

### `calyx.std_le` (::circt::calyx::LeLibOp) [¶](#calyxstd_le-circtcalyxlelibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_le` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-30)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-21)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | 1-bit signless integer |

### `calyx.std_lsh` (::circt::calyx::LshLibOp) [¶](#calyxstd_lsh-circtcalyxlshlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_lsh` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-31)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-22)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | any type |

### `calyx.std_lt` (::circt::calyx::LtLibOp) [¶](#calyxstd_lt-circtcalyxltlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_lt` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-32)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-23)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | 1-bit signless integer |

### `calyx.std_mult_pipe` (::circt::calyx::MultPipeLibOp) [¶](#calyxstd_mult_pipe-circtcalyxmultpipelibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_mult_pipe` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-33)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-24)

| Result | Description |
| --- | --- |
| `clk` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |
| `left` | any type |
| `right` | any type |
| `out` | any type |
| `done` | 1-bit signless integer |

### `calyx.std_mux` (::circt::calyx::MuxLibOp) [¶](#calyxstd_mux-circtcalyxmuxlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_mux` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-34)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-25)

| Result | Description |
| --- | --- |
| `cond` | 1-bit signless integer |
| `tru` | any type |
| `fal` | any type |
| `out` | any type |

### `calyx.std_neq` (::circt::calyx::NeqLibOp) [¶](#calyxstd_neq-circtcalyxneqlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_neq` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-35)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-26)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | 1-bit signless integer |

### `calyx.std_not` (::circt::calyx::NotLibOp) [¶](#calyxstd_not-circtcalyxnotlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_not` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-36)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-27)

| Result | Description |
| --- | --- |
| `in` | integer |
| `out` | integer |

### `calyx.std_or` (::circt::calyx::OrLibOp) [¶](#calyxstd_or-circtcalyxorlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_or` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-37)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-28)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | any type |

### `calyx.std_pad` (::circt::calyx::PadLibOp) [¶](#calyxstd_pad-circtcalyxpadlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_pad` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-38)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-29)

| Result | Description |
| --- | --- |
| `in` | integer |
| `out` | integer |

### `calyx.std_rems_pipe` (::circt::calyx::RemSPipeLibOp) [¶](#calyxstd_rems_pipe-circtcalyxremspipelibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_rems_pipe` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-39)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-30)

| Result | Description |
| --- | --- |
| `clk` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |
| `left` | any type |
| `right` | any type |
| `out` | any type |
| `done` | 1-bit signless integer |

### `calyx.std_remu_pipe` (::circt::calyx::RemUPipeLibOp) [¶](#calyxstd_remu_pipe-circtcalyxremupipelibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_remu_pipe` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-40)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-31)

| Result | Description |
| --- | --- |
| `clk` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |
| `left` | any type |
| `right` | any type |
| `out` | any type |
| `done` | 1-bit signless integer |

### `calyx.std_rsh` (::circt::calyx::RshLibOp) [¶](#calyxstd_rsh-circtcalyxrshlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_rsh` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-41)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-32)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | any type |

### `calyx.std_seq` (::circt::calyx::SeqLibOp) [¶](#calyxstd_seq-circtcalyxseqlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_seq` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-42)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-33)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | 1-bit signless integer |

### `calyx.std_sge` (::circt::calyx::SgeLibOp) [¶](#calyxstd_sge-circtcalyxsgelibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_sge` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-43)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-34)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | 1-bit signless integer |

### `calyx.std_sgt` (::circt::calyx::SgtLibOp) [¶](#calyxstd_sgt-circtcalyxsgtlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_sgt` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-44)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-35)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | 1-bit signless integer |

### `calyx.std_shru` (::circt::calyx::ShruLibOp) [¶](#calyxstd_shru-circtcalyxshrulibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_shru` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-45)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-36)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | any type |

### `calyx.std_signext` (::circt::calyx::ExtSILibOp) [¶](#calyxstd_signext-circtcalyxextsilibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_signext` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-46)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-37)

| Result | Description |
| --- | --- |
| `in` | integer |
| `out` | integer |

### `calyx.std_sle` (::circt::calyx::SleLibOp) [¶](#calyxstd_sle-circtcalyxslelibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_sle` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-47)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-38)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | 1-bit signless integer |

### `calyx.std_slice` (::circt::calyx::SliceLibOp) [¶](#calyxstd_slice-circtcalyxslicelibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_slice` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-48)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-39)

| Result | Description |
| --- | --- |
| `in` | integer |
| `out` | integer |

### `calyx.std_slt` (::circt::calyx::SltLibOp) [¶](#calyxstd_slt-circtcalyxsltlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_slt` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-49)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-40)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | 1-bit signless integer |

### `calyx.std_sneq` (::circt::calyx::SneqLibOp) [¶](#calyxstd_sneq-circtcalyxsneqlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_sneq` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-50)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-41)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | 1-bit signless integer |

### `calyx.std_srsh` (::circt::calyx::SrshLibOp) [¶](#calyxstd_srsh-circtcalyxsrshlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_srsh` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-51)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-42)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | any type |

### `calyx.std_sub` (::circt::calyx::SubLibOp) [¶](#calyxstd_sub-circtcalyxsublibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_sub` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-52)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-43)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | any type |

### `calyx.std_wire` (::circt::calyx::WireLibOp) [¶](#calyxstd_wire-circtcalyxwirelibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_wire` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-53)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-44)

| Result | Description |
| --- | --- |
| `in` | integer |
| `out` | integer |

### `calyx.std_xor` (::circt::calyx::XorLibOp) [¶](#calyxstd_xor-circtcalyxxorlibop)

*Defines an operation which maps to a Calyx library primitive*

Syntax:

```
operation ::= `calyx.std_xor` $sym_name attr-dict `:` qualified(type(results))
```

This operation represents an instance of a Calyx library primitive.
A library primitive maps to some hardware-implemented component within the
native Calyx compiler.

```
  // A 32-bit adder. This falls under the binary library operations.
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  // An 8-bit comparison operator (with a 1-bit output). This falls under
  // the boolean binary library operations.
  %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

  // An 8-bit to 16-bit pad operator. This falls under the unary
  // library operations.
  %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
```

Traits: `Combinational`

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-54)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-45)

| Result | Description |
| --- | --- |
| `left` | any type |
| `right` | any type |
| `out` | any type |

### `calyx.undef` (::circt::calyx::UndefinedOp) [¶](#calyxundef-circtcalyxundefinedop)

*Calyx Undefined Value*

Syntax:

```
operation ::= `calyx.undef` attr-dict `:` qualified(type($res))
```

The “undef” operation represents an undefined value
that may be used when a specific source or destination
does not have an assignment yet. This is used to avoid
pulling in the entire LLVMIR dialect for a single
operation.

```
  %0 = calyx.undef : i1
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results: [¶](#results-46)

| Result | Description |
| --- | --- |
| `res` | any type |

### `calyx.undefined` (::circt::calyx::UndefLibOp) [¶](#calyxundefined-circtcalyxundeflibop)

*An undefined signal*

Syntax:

```
operation ::= `calyx.undefined` $sym_name attr-dict `:` qualified(type(results))
```

The “calyx.undef” op defines an undefined signal. An undefined signal can be
replaced with any value representable in n-bits.

```
  // A 32-bit undefined value
  %undef.out = calyx.undef @undef : i32
```

Interfaces: `CellInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-55)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-47)

| Result | Description |
| --- | --- |
| `out` | any type |

### `calyx.while` (::circt::calyx::WhileOp) [¶](#calyxwhile-circtcalyxwhileop)

*Calyx While*

Syntax:

```
operation ::= `calyx.while` $cond (`with` $groupName^)? $body attr-dict
```

The “calyx.while” operation represents a construct for continuously
executing the inner groups of the ‘while’ operation while the condition port
evaluates to true. The operands to a while operation is a 1-bit port and an
optional combinational group under which this port is driven.

Note: The native and CIRCT Calyx IRs may diverge wrt. ‘with’ execution, see:
<https://github.com/cucapra/calyx/discussions/588>

```
  calyx.while %1 with @G1 {
    calyx.enable @G2
    ...
  }
  calyx.while %1 {
    calyx.enable @G2
    ...
  }
```

Traits: `ControlLike`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

#### Attributes: [¶](#attributes-56)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `groupName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands: [¶](#operands-6)

| Operand | Description |
| --- | --- |
| `cond` | 1-bit signless integer |

### `calyx.wires` (::circt::calyx::WiresOp) [¶](#calyxwires-circtcalyxwiresop)

*Calyx Wires*

Syntax:

```
operation ::= `calyx.wires` $body attr-dict
```

The “calyx.wires” operation represents a set of
guarded connections between component instances,
which may be placed within groups.

```
  calyx.wires {
    calyx.group @A { ... }
    calyx.assign %1 = %2 : i16
  }
```

Traits: `HasParent<ComponentOp, CombComponentOp>`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `RegionKindInterface`

 [Prev - 'arc' Dialect](https://circt.llvm.org/docs/Dialects/Arc/ "'arc' Dialect")
[Next - 'chirrtl' Dialect](https://circt.llvm.org/docs/Dialects/CHIRRTL/ "'chirrtl' Dialect") 

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