'sv' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'sv' Dialect
============

*Types and operations for SV dialect*

This dialect defines the `sv` dialect, which represents various
SystemVerilog-specific constructs in an AST-like representation.

* [Operations](#operations)
  + [`sv.alias` (::circt::sv::AliasOp)](#svalias-circtsvaliasop)
  + [`sv.always` (::circt::sv::AlwaysOp)](#svalways-circtsvalwaysop)
  + [`sv.alwayscomb` (::circt::sv::AlwaysCombOp)](#svalwayscomb-circtsvalwayscombop)
  + [`sv.alwaysff` (::circt::sv::AlwaysFFOp)](#svalwaysff-circtsvalwaysffop)
  + [`sv.array_index_inout` (::circt::sv::ArrayIndexInOutOp)](#svarray_index_inout-circtsvarrayindexinoutop)
  + [`sv.assert` (::circt::sv::AssertOp)](#svassert-circtsvassertop)
  + [`sv.assert.concurrent` (::circt::sv::AssertConcurrentOp)](#svassertconcurrent-circtsvassertconcurrentop)
  + [`sv.assert_property` (::circt::sv::AssertPropertyOp)](#svassert_property-circtsvassertpropertyop)
  + [`sv.assign` (::circt::sv::AssignOp)](#svassign-circtsvassignop)
  + [`sv.assume` (::circt::sv::AssumeOp)](#svassume-circtsvassumeop)
  + [`sv.assume.concurrent` (::circt::sv::AssumeConcurrentOp)](#svassumeconcurrent-circtsvassumeconcurrentop)
  + [`sv.assume_property` (::circt::sv::AssumePropertyOp)](#svassume_property-circtsvassumepropertyop)
  + [`sv.bind` (::circt::sv::BindOp)](#svbind-circtsvbindop)
  + [`sv.bind.interface` (::circt::sv::BindInterfaceOp)](#svbindinterface-circtsvbindinterfaceop)
  + [`sv.bpassign` (::circt::sv::BPAssignOp)](#svbpassign-circtsvbpassignop)
  + [`sv.case` (::circt::sv::CaseOp)](#svcase-circtsvcaseop)
  + [`sv.constantStr` (::circt::sv::ConstantStrOp)](#svconstantstr-circtsvconstantstrop)
  + [`sv.constantX` (::circt::sv::ConstantXOp)](#svconstantx-circtsvconstantxop)
  + [`sv.constantZ` (::circt::sv::ConstantZOp)](#svconstantz-circtsvconstantzop)
  + [`sv.cover` (::circt::sv::CoverOp)](#svcover-circtsvcoverop)
  + [`sv.cover.concurrent` (::circt::sv::CoverConcurrentOp)](#svcoverconcurrent-circtsvcoverconcurrentop)
  + [`sv.cover_property` (::circt::sv::CoverPropertyOp)](#svcover_property-circtsvcoverpropertyop)
  + [`sv.error` (::circt::sv::ErrorOp)](#sverror-circtsverrorop)
  + [`sv.exit` (::circt::sv::ExitOp)](#svexit-circtsvexitop)
  + [`sv.fatal` (::circt::sv::FatalOp)](#svfatal-circtsvfatalop)
  + [`sv.fflush` (::circt::sv::FFlushOp)](#svfflush-circtsvfflushop)
  + [`sv.finish` (::circt::sv::FinishOp)](#svfinish-circtsvfinishop)
  + [`sv.for` (::circt::sv::ForOp)](#svfor-circtsvforop)
  + [`sv.force` (::circt::sv::ForceOp)](#svforce-circtsvforceop)
  + [`sv.func` (::circt::sv::FuncOp)](#svfunc-circtsvfuncop)
  + [`sv.func.call` (::circt::sv::FuncCallOp)](#svfunccall-circtsvfunccallop)
  + [`sv.func.call.procedural` (::circt::sv::FuncCallProceduralOp)](#svfunccallprocedural-circtsvfunccallproceduralop)
  + [`sv.func.dpi.import` (::circt::sv::FuncDPIImportOp)](#svfuncdpiimport-circtsvfuncdpiimportop)
  + [`sv.fwrite` (::circt::sv::FWriteOp)](#svfwrite-circtsvfwriteop)
  + [`sv.generate` (::circt::sv::GenerateOp)](#svgenerate-circtsvgenerateop)
  + [`sv.generate.case` (::circt::sv::GenerateCaseOp)](#svgeneratecase-circtsvgeneratecaseop)
  + [`sv.if` (::circt::sv::IfOp)](#svif-circtsvifop)
  + [`sv.ifdef` (::circt::sv::IfDefOp)](#svifdef-circtsvifdefop)
  + [`sv.ifdef.procedural` (::circt::sv::IfDefProceduralOp)](#svifdefprocedural-circtsvifdefproceduralop)
  + [`sv.include` (::circt::sv::IncludeOp)](#svinclude-circtsvincludeop)
  + [`sv.indexed_part_select` (::circt::sv::IndexedPartSelectOp)](#svindexed_part_select-circtsvindexedpartselectop)
  + [`sv.indexed_part_select_inout` (::circt::sv::IndexedPartSelectInOutOp)](#svindexed_part_select_inout-circtsvindexedpartselectinoutop)
  + [`sv.info` (::circt::sv::InfoOp)](#svinfo-circtsvinfoop)
  + [`sv.initial` (::circt::sv::InitialOp)](#svinitial-circtsvinitialop)
  + [`sv.interface` (::circt::sv::InterfaceOp)](#svinterface-circtsvinterfaceop)
  + [`sv.interface.instance` (::circt::sv::InterfaceInstanceOp)](#svinterfaceinstance-circtsvinterfaceinstanceop)
  + [`sv.interface.modport` (::circt::sv::InterfaceModportOp)](#svinterfacemodport-circtsvinterfacemodportop)
  + [`sv.interface.signal` (::circt::sv::InterfaceSignalOp)](#svinterfacesignal-circtsvinterfacesignalop)
  + [`sv.interface.signal.assign` (::circt::sv::AssignInterfaceSignalOp)](#svinterfacesignalassign-circtsvassigninterfacesignalop)
  + [`sv.interface.signal.read` (::circt::sv::ReadInterfaceSignalOp)](#svinterfacesignalread-circtsvreadinterfacesignalop)
  + [`sv.localparam` (::circt::sv::LocalParamOp)](#svlocalparam-circtsvlocalparamop)
  + [`sv.logic` (::circt::sv::LogicOp)](#svlogic-circtsvlogicop)
  + [`sv.macro.decl` (::circt::sv::MacroDeclOp)](#svmacrodecl-circtsvmacrodeclop)
  + [`sv.macro.def` (::circt::sv::MacroDefOp)](#svmacrodef-circtsvmacrodefop)
  + [`sv.macro.error` (::circt::sv::MacroErrorOp)](#svmacroerror-circtsvmacroerrorop)
  + [`sv.macro.ref` (::circt::sv::MacroRefOp)](#svmacroref-circtsvmacrorefop)
  + [`sv.macro.ref.expr` (::circt::sv::MacroRefExprOp)](#svmacrorefexpr-circtsvmacrorefexprop)
  + [`sv.macro.ref.expr.se` (::circt::sv::MacroRefExprSEOp)](#svmacrorefexprse-circtsvmacrorefexprseop)
  + [`sv.modport.get` (::circt::sv::GetModportOp)](#svmodportget-circtsvgetmodportop)
  + [`sv.nonstandard.deposit` (::circt::sv::DepositOp)](#svnonstandarddeposit-circtsvdepositop)
  + [`sv.ordered` (::circt::sv::OrderedOutputOp)](#svordered-circtsvorderedoutputop)
  + [`sv.passign` (::circt::sv::PAssignOp)](#svpassign-circtsvpassignop)
  + [`sv.read_inout` (::circt::sv::ReadInOutOp)](#svread_inout-circtsvreadinoutop)
  + [`sv.readmem` (::circt::sv::ReadMemOp)](#svreadmem-circtsvreadmemop)
  + [`sv.reg` (::circt::sv::RegOp)](#svreg-circtsvregop)
  + [`sv.release` (::circt::sv::ReleaseOp)](#svrelease-circtsvreleaseop)
  + [`sv.reserve_names` (::circt::sv::ReserveNamesOp)](#svreserve_names-circtsvreservenamesop)
  + [`sv.return` (::circt::sv::ReturnOp)](#svreturn-circtsvreturnop)
  + [`sv.sformatf` (::circt::sv::SFormatFOp)](#svsformatf-circtsvsformatfop)
  + [`sv.stop` (::circt::sv::StopOp)](#svstop-circtsvstopop)
  + [`sv.struct_field_inout` (::circt::sv::StructFieldInOutOp)](#svstruct_field_inout-circtsvstructfieldinoutop)
  + [`sv.system` (::circt::sv::SystemFunctionOp)](#svsystem-circtsvsystemfunctionop)
  + [`sv.system.sampled` (::circt::sv::SampledOp)](#svsystemsampled-circtsvsampledop)
  + [`sv.system.stime` (::circt::sv::STimeOp)](#svsystemstime-circtsvstimeop)
  + [`sv.system.time` (::circt::sv::TimeOp)](#svsystemtime-circtsvtimeop)
  + [`sv.unpacked_array_create` (::circt::sv::UnpackedArrayCreateOp)](#svunpacked_array_create-circtsvunpackedarraycreateop)
  + [`sv.unpacked_open_array_cast` (::circt::sv::UnpackedOpenArrayCastOp)](#svunpacked_open_array_cast-circtsvunpackedopenarraycastop)
  + [`sv.verbatim` (::circt::sv::VerbatimOp)](#svverbatim-circtsvverbatimop)
  + [`sv.verbatim.expr` (::circt::sv::VerbatimExprOp)](#svverbatimexpr-circtsvverbatimexprop)
  + [`sv.verbatim.expr.se` (::circt::sv::VerbatimExprSEOp)](#svverbatimexprse-circtsvverbatimexprseop)
  + [`sv.verbatim.module` (::circt::sv::SVVerbatimModuleOp)](#svverbatimmodule-circtsvsvverbatimmoduleop)
  + [`sv.verbatim.source` (::circt::sv::SVVerbatimSourceOp)](#svverbatimsource-circtsvsvverbatimsourceop)
  + [`sv.warning` (::circt::sv::WarningOp)](#svwarning-circtsvwarningop)
  + [`sv.wire` (::circt::sv::WireOp)](#svwire-circtsvwireop)
  + [`sv.xmr` (::circt::sv::XMROp)](#svxmr-circtsvxmrop)
  + [`sv.xmr.ref` (::circt::sv::XMRRefOp)](#svxmrref-circtsvxmrrefop)
* [Attributes](#attributes-63)
  + [CaseExprPatternAttr](#caseexprpatternattr)
  + [IncludeStyleAttr](#includestyleattr)
  + [MacroIdentAttr](#macroidentattr)
  + [ModportDirectionAttr](#modportdirectionattr)
  + [ModportStructAttr](#modportstructattr)
  + [SVAttributeAttr](#svattributeattr)
  + [ValidationQualifierTypeEnumAttr](#validationqualifiertypeenumattr)
* [Type constraints](#type-constraints)
  + [an Unpacked Open ArrayType](#an-unpacked-open-arraytype)
* [Types](#types)
  + [InterfaceType](#interfacetype)
  + [ModportType](#modporttype)
  + [UnpackedOpenArrayType](#unpackedopenarraytype)
* [Enums](#enums)
  + [CaseStmtType](#casestmttype)
  + [DeferAssert](#deferassert)
  + [EventControl](#eventcontrol)
  + [IncludeStyle](#includestyle)
  + [MemBaseTypeAttr](#membasetypeattr)
  + [ModportDirection](#modportdirection)
  + [ResetType](#resettype)
  + [ValidationQualifierTypeEnum](#validationqualifiertypeenum)

Operations
----------

### `sv.alias` (::circt::sv::AliasOp)

*SystemVerilog ‘alias’ statement*

Syntax:

```
operation ::= `sv.alias` $aliases attr-dict `:` qualified(type($aliases))
```

An alias statement declares multiple names for the same physical net, or
bits within a net. Aliases always have at least two operands.

#### Operands:

| Operand | Description |
| --- | --- |
| `aliases` | variadic of InOutType |

### `sv.always` (::circt::sv::AlwaysOp)

*‘always @’ block*

Syntax:

```
operation ::= `sv.always` custom<EventList>($events, $clocks) $body attr-dict
```

See SV Spec 9.2, and 9.4.2.2.

Traits: `NoRegionArguments`, `NoTerminator`, `NonProceduralOp`, `ProceduralRegion`, `RecursiveMemoryEffects`, `RecursivelySpeculatableImplTrait`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `events` | ::mlir::ArrayAttr | events |

#### Operands:

| Operand | Description |
| --- | --- |
| `clocks` | variadic of 1-bit signless integer |

### `sv.alwayscomb` (::circt::sv::AlwaysCombOp)

*‘alwayscomb block*

Syntax:

```
operation ::= `sv.alwayscomb` $body attr-dict
```

See SV Spec 9.2, and 9.2.2.2.

Traits: `NoRegionArguments`, `NoTerminator`, `NonProceduralOp`, `ProceduralRegion`, `RecursiveMemoryEffects`, `RecursivelySpeculatableImplTrait`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`

### `sv.alwaysff` (::circt::sv::AlwaysFFOp)

*‘alwaysff @’ block with optional reset*

Syntax:

```
operation ::= `sv.alwaysff` `(` $clockEdge $clock `)` $bodyBlk
              ( `(` $resetStyle `:` $resetEdge^ $reset `)` $resetBlk )? attr-dict
```

alwaysff blocks represent always\_ff verilog nodes, which enforce inference
of registers. This block takes a clock signal and edge sensitivity and
reset type. If the reset type is anything but ’noreset’, the block takes a
reset signal, reset sensitivity, and reset block. Appropriate if conditions
are generated in the output code based on the reset type. A negative-edge,
asynchronous reset will check the inverse of the reset condition
(if (!reset) begin resetblock end) to match the sensitivity.

Traits: `NoRegionArguments`, `NoTerminator`, `NonProceduralOp`, `ProceduralRegion`, `RecursiveMemoryEffects`, `RecursivelySpeculatableImplTrait`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `clockEdge` | circt::sv::EventControlAttr | edge control trigger |
| `resetStyle` | ::circt::sv::ResetTypeAttr | reset type |
| `resetEdge` | circt::sv::EventControlAttr | edge control trigger |

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | 1-bit signless integer |
| `reset` | 1-bit signless integer |

### `sv.array_index_inout` (::circt::sv::ArrayIndexInOutOp)

*Index an inout memory to produce an inout element*

Syntax:

```
operation ::= `sv.array_index_inout` $input`[`$index`]` attr-dict `:` qualified(type($input)) `,` qualified(type($index))
```

See SV Spec 11.5.2.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an inout type with array element |
| `index` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | InOutType |

### `sv.assert` (::circt::sv::AssertOp)

*Immediate assertion statement*

Syntax:

```
operation ::= `sv.assert` $expression `,` $defer
              (`label` $label^)?
              (`message` $message^ (`(` $substitutions^ `)` `:`
              qualified(type($substitutions)))?)? attr-dict
```

Specify that a Boolean expression is always true. This can be used to both
document the behavior of the design and to test that the design behaves as
expected. See Section 16.3 of the SystemVerilog 2017 specification.

Traits: `ProceduralOp`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `defer` | circt::sv::DeferAssertAttr | assertion deferring mode |
| `label` | ::mlir::StringAttr | string attribute |
| `message` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `expression` | 1-bit signless integer |
| `substitutions` | variadic of any type |

### `sv.assert.concurrent` (::circt::sv::AssertConcurrentOp)

*Concurrent assertion statement, i.e., assert property*

Syntax:

```
operation ::= `sv.assert.concurrent` $event $clock `,` $property
              (`label` $label^)?
              (`message` $message^ (`(` $substitutions^ `)` `:`
              qualified(type($substitutions)))?)? attr-dict
```

Specify that a property of the hardware design is true whenever the property
is evaluated. This can be used to both document the behavior of the design
and to test that the design behaves as expected. See section 16.5 of the
SystemVerilog 2017 specification.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `event` | circt::sv::EventControlAttr | edge control trigger |
| `label` | ::mlir::StringAttr | string attribute |
| `message` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | 1-bit signless integer |
| `property` | 1-bit signless integer |
| `substitutions` | variadic of any type |

### `sv.assert_property` (::circt::sv::AssertPropertyOp)

*Property assertion – can be disabled and clocked*

Syntax:

```
operation ::= `sv.assert_property` $property (`on` $event^)? ($clock^)? (`disable_iff` $disable^)?
              (`label` $label^)? attr-dict `:` type($property)
```

Assert that a given SVA-style property holds. This is only checked when
the disable signal is low and a clock event occurs. This is analogous to
the verif.assert operation, but with a flipped enable polarity.

Traits: `AttrSizedOperandSegments`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `event` | circt::sv::EventControlAttr | edge control trigger |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `property` | 1-bit signless integer or LTL sequence type or LTL property type |
| `clock` | 1-bit signless integer |
| `disable` | 1-bit signless integer |

### `sv.assign` (::circt::sv::AssignOp)

*Continuous assignment*

Syntax:

```
operation ::= `sv.assign` $dest `,` $src attr-dict `:` qualified(type($src))
```

A SystemVerilog assignment statement ‘x = y;’.
These occur in module scope. See SV Spec 10.3.2.

Traits: `NonProceduralOp`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | InOutType |
| `src` | a valid inout element |

### `sv.assume` (::circt::sv::AssumeOp)

*Immediate assume statement*

Syntax:

```
operation ::= `sv.assume` $expression `,` $defer
              (`label` $label^)?
              (`message` $message^ (`(` $substitutions^ `)` `:`
              qualified(type($substitutions)))?)? attr-dict
```

Specify that a Boolean expression is assumed to always be true. This can
either be used as an assertion-like check that the expression is, in fact,
always true or to bound legal input values during testing. See Section 16.3
of the SystemVerilog 2017 specification.

Traits: `ProceduralOp`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `defer` | circt::sv::DeferAssertAttr | assertion deferring mode |
| `label` | ::mlir::StringAttr | string attribute |
| `message` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `expression` | 1-bit signless integer |
| `substitutions` | variadic of any type |

### `sv.assume.concurrent` (::circt::sv::AssumeConcurrentOp)

*Concurrent assume statement, i.e., assume property*

Syntax:

```
operation ::= `sv.assume.concurrent` $event $clock `,` $property
              (`label` $label^)?
              (`message` $message^ (`(` $substitutions^ `)` `:`
              qualified(type($substitutions)))?)? attr-dict
```

Specify that a property is assumed to be true whenever the property is
evaluated. This can be used to both document the behavior of the design and
to test that the design behaves as expected. See section 16.5 of the
SystemVerilog 2017 specification.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `event` | circt::sv::EventControlAttr | edge control trigger |
| `label` | ::mlir::StringAttr | string attribute |
| `message` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | 1-bit signless integer |
| `property` | 1-bit signless integer |
| `substitutions` | variadic of any type |

### `sv.assume_property` (::circt::sv::AssumePropertyOp)

*Property assumption – can be disabled and clocked*

Syntax:

```
operation ::= `sv.assume_property` $property (`on` $event^)? ($clock^)? (`disable_iff` $disable^)?
              (`label` $label^)? attr-dict `:` type($property)
```

Assume that a given SVA-style property holds. This is only considered when
the disable signal is low and a clock event occurs. This is analogous to
the verif.assume operation, but with a flipped enable polarity.

Traits: `AttrSizedOperandSegments`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `event` | circt::sv::EventControlAttr | edge control trigger |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `property` | 1-bit signless integer or LTL sequence type or LTL property type |
| `clock` | 1-bit signless integer |
| `disable` | 1-bit signless integer |

### `sv.bind` (::circt::sv::BindOp)

*Indirect instantiation statement*

Syntax:

```
operation ::= `sv.bind` $instance attr-dict
```

Indirectly instantiate a module from the context of another module. BindOp
pairs with a `hw.instance` (identified by a `boundInstance` symbol) which
tracks all information except the emission point for the bind. BindOp also
tracks the `instanceModule` symbol for the `hw.module` that contains the
`hw.instance` to accelerate symbol lookup.

See 23.11 of SV 2017 spec for more information about bind.

Interfaces: `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `instance` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |

### `sv.bind.interface` (::circt::sv::BindInterfaceOp)

*Indirectly instantiate an interface*

Syntax:

```
operation ::= `sv.bind.interface` $instance attr-dict
```

Indirectly instantiate an interface in the context of another module. This
operation must pair with a `sv.interface.instance`.

Interfaces: `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `instance` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |

### `sv.bpassign` (::circt::sv::BPAssignOp)

*Blocking procedural assignment*

Syntax:

```
operation ::= `sv.bpassign` $dest `,` $src  attr-dict `:` qualified(type($src))
```

A SystemVerilog blocking procedural assignment statement ‘x = y;’. These
occur in initial, always, task, and function blocks. The statement is
executed before any following statements are. See SV Spec 10.4.1.

Traits: `ProceduralOp`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | InOutType |
| `src` | a valid inout element |

### `sv.case` (::circt::sv::CaseOp)

*‘case (cond)’ block*

See SystemVerilog 2017 12.5.

Traits: `NoRegionArguments`, `NoTerminator`, `ProceduralOp`, `ProceduralRegion`, `SingleBlock`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `caseStyle` | ::CaseStmtTypeAttr | case type |
| `casePatterns` | ::mlir::ArrayAttr | array attribute |
| `validationQualifier` | ::circt::sv::ValidationQualifierTypeEnumAttr | validation qualifier type |

#### Operands:

| Operand | Description |
| --- | --- |
| `cond` | any type |
| `caseValues` | variadic of any type |

### `sv.constantStr` (::circt::sv::ConstantStrOp)

*A constant of string value*

Syntax:

```
operation ::= `sv.constantStr` $str attr-dict
```

This operation produces a constant string literal.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `str` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a HW string |

### `sv.constantX` (::circt::sv::ConstantXOp)

*A constant of value ‘x’*

Syntax:

```
operation ::= `sv.constantX` attr-dict `:` qualified(type($result))
```

This operation produces a constant value of ‘x’. This ‘x’ follows the
System Verilog rules for ‘x’ propagation.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `result` | a known primitive element |

### `sv.constantZ` (::circt::sv::ConstantZOp)

*A constant of value ‘z’*

Syntax:

```
operation ::= `sv.constantZ` attr-dict `:` qualified(type($result))
```

This operation produces a constant value of ‘z’. This ‘z’ follows the
System Verilog rules for ‘z’ propagation.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `result` | a known primitive element |

### `sv.cover` (::circt::sv::CoverOp)

*Immediate cover statement*

Syntax:

```
operation ::= `sv.cover` $expression `,` $defer
              (`label` $label^)?
              (`message` $message^ (`(` $substitutions^ `)` `:`
              qualified(type($substitutions)))?)? attr-dict
```

Specify that a Boolean expression should be monitored for coverage, i.e., a
simulator will watch if it occurs and how many times it occurs. See section
16.3 of the SystemVerilog 2017 specification.

Traits: `ProceduralOp`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `defer` | circt::sv::DeferAssertAttr | assertion deferring mode |
| `label` | ::mlir::StringAttr | string attribute |
| `message` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `expression` | 1-bit signless integer |
| `substitutions` | variadic of any type |

### `sv.cover.concurrent` (::circt::sv::CoverConcurrentOp)

*Concurrent cover statement, i.e., cover property*

Syntax:

```
operation ::= `sv.cover.concurrent` $event $clock `,` $property
              (`label` $label^)?
              (`message` $message^ (`(` $substitutions^ `)` `:`
              qualified(type($substitutions)))?)? attr-dict
```

Specify that a specific property should be monitored for coverage, i.e., a
simulation will watch if it occurrs and how many times it occurs. See
section 16.5 of the SystemVerilog 2017 specification.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `event` | circt::sv::EventControlAttr | edge control trigger |
| `label` | ::mlir::StringAttr | string attribute |
| `message` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | 1-bit signless integer |
| `property` | 1-bit signless integer |
| `substitutions` | variadic of any type |

### `sv.cover_property` (::circt::sv::CoverPropertyOp)

*Property cover point – can be disabled and clocked*

Syntax:

```
operation ::= `sv.cover_property` $property (`on` $event^)? ($clock^)? (`disable_iff` $disable^)?
              (`label` $label^)? attr-dict `:` type($property)
```

Cover when a given SVA-style property holds. This is only checked when
the disable signal is low and a clock event occurs. This is analogous to
the verif.cover operation, but with a flipped enable polarity.

Traits: `AttrSizedOperandSegments`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `event` | circt::sv::EventControlAttr | edge control trigger |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `property` | 1-bit signless integer or LTL sequence type or LTL property type |
| `clock` | 1-bit signless integer |
| `disable` | 1-bit signless integer |

### `sv.error` (::circt::sv::ErrorOp)

*`$error` severity message task*

Syntax:

```
operation ::= `sv.error` ($message^ (`(` $substitutions^ `)` `:` qualified(type($substitutions)))?)?
              attr-dict
```

This system task indicates a run-time error.

If present, the optional message is printed with any additional operands
interpolated into the message string.

Traits: `ProceduralOp`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `message` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `substitutions` | variadic of any type |

### `sv.exit` (::circt::sv::ExitOp)

*`$exit` system task*

Syntax:

```
operation ::= `sv.exit` attr-dict
```

Waits for all `program` blocks to complete and then makes an implicit call
to `$finish` with default verbosity (level 1) to conclude the simulation.

Traits: `ProceduralOp`

### `sv.fatal` (::circt::sv::FatalOp)

*`$fatal` severity message task*

Syntax:

```
operation ::= `sv.fatal` $verbosity
              (`,` $message^ (`(` $substitutions^ `)` `:`
              qualified(type($substitutions)))?)? attr-dict
```

Generates a run-time fatal error which terminates the simulation with an
error code. Makes an implicit call to `$finish`, forwarding the `verbosity`
operand. If present, the optional message is printed with any additional
operands interpolated into the message string.

Traits: `ProceduralOp`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `verbosity` | ::mlir::IntegerAttr | 8-bit signless integer attribute whose minimum value is 0 whose maximum value is 2 |
| `message` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `substitutions` | variadic of any type |

### `sv.fflush` (::circt::sv::FFlushOp)

*’$fflush’ statement*

Syntax:

```
operation ::= `sv.fflush` (`fd` $fd^)? attr-dict
```

The $fflush system task flushes the output buffer of the specified file
descriptor. If no file descriptor is specified, all open files are flushed.

See IEEE 1800-2023 Section 21.3.6.

Traits: `ProceduralOp`

#### Operands:

| Operand | Description |
| --- | --- |
| `fd` | 32-bit signless integer |

### `sv.finish` (::circt::sv::FinishOp)

*`$finish` system task*

Syntax:

```
operation ::= `sv.finish` $verbosity attr-dict
```

Stops the simulation and exits/terminates the simulator process. In practice
most GUI-based simulators will show a prompt to the user offering them an
opportunity to not close the simulator altogether.

Other tasks such as `$exit` or `$fatal` implicitly call this system task.

The optional `verbosity` parameter controls how much diagnostic information
is printed when the system task is executed (see section 20.2 of IEEE
1800-2017):

* `0`: Prints nothing
* `1`: Prints simulation time and location (default)
* `2`: Prints simulation time, location, and statistics about the memory and
  CPU time used in simulation

Traits: `ProceduralOp`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `verbosity` | ::mlir::IntegerAttr | 8-bit signless integer attribute whose minimum value is 0 whose maximum value is 2 |

### `sv.for` (::circt::sv::ForOp)

*System verilog for loop*

The `sv.for` operation in System Verilog defines a for statement that requires
three SSA operands: `lowerBounds`, `upperBound`, and `step`. It functions
similarly to `scf.for`, where the loop iterates the induction variable from
`lowerBound` to `upperBound` with a step size of `step`, i.e:

```
 for (logic ... indVar = lowerBound; indVar < upperBound; indVar += step) begin
 end
```

It’s important to note that since we are using a bit precise type instead of a Verilog
`integer` type, users must be cautious about potential overflow. For example, if
you wish to iterate over all 2-bit values, you must use a 3-bit value as the
induction variable type.

Traits: `NoTerminator`, `ProceduralOp`, `ProceduralRegion`, `SingleBlock`

Interfaces: `OpAsmOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inductionVarName` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `lowerBound` | a signless integer bitvector |
| `upperBound` | a signless integer bitvector |
| `step` | a signless integer bitvector |

### `sv.force` (::circt::sv::ForceOp)

*Force procedural statement*

Syntax:

```
operation ::= `sv.force` $dest `,` $src  attr-dict `:` qualified(type($src))
```

A SystemVerilog force procedural statement ‘force x = y;’. These
occur in initial, always, task, and function blocks.
A force statement shall override a procedural assignment until
a release statement is executed on the variable.
The left-hand side of the assignment can be a variable, a net,
a constant bit-select of a vector net, a part-select of a vector
net or a concatenation. It cannot be a memory word or a bit-select
or part-select of a vector variable. See SV Spec 10.6.2.

Traits: `ProceduralOp`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | InOutType |
| `src` | a valid inout element |

### `sv.func` (::circt::sv::FuncOp)

*A SystemVerilog function*

`sv.func` represents SystemVerilog function in IEEE 1800-2017 section 13.4
“Functions”. Similar to HW module, it’s allowed to mix the order of input
and output arguments. `sv.func` can be used for both function
declaration and definition, i.e. a function without a body
region is a declaration.

In SV there are two representations for function results,
“output argument” and “return value”. Currently an output argument
is considered as as a return value if it’s is the last argument
and has a special attribute `sv.func.explicitly_returned`.

Traits: `HasParent<mlir::ModuleOp>`, `IsolatedFromAbove`, `ProceduralRegion`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallableOpInterface`, `Emittable`, `FunctionOpInterface`, `HWEmittableModuleLike`, `HWModuleLike`, `InstanceGraphModuleOpInterface`, `OpAsmOpInterface`, `PortList`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `module_type` | ::mlir::TypeAttr | type attribute of a module |
| `per_argument_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `input_locs` | ::mlir::ArrayAttr | location array attribute |
| `result_locs` | ::mlir::ArrayAttr | location array attribute |
| `verilogName` | ::mlir::StringAttr | string attribute |

### `sv.func.call` (::circt::sv::FuncCallOp)

*Function call in a non-procedural region*

Syntax:

```
operation ::= `sv.func.call` $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
```

This op represents a function call in a non-procedural region.
A function call in a non-procedural region must have a return
value and no output argument.

Traits: `NonProceduralOp`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `callee` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| «unnamed» | variadic of any type |

### `sv.func.call.procedural` (::circt::sv::FuncCallProceduralOp)

*Function call in a procedural region*

Syntax:

```
operation ::= `sv.func.call.procedural` $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
```

Traits: `ProceduralOp`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `callee` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| «unnamed» | variadic of any type |

### `sv.func.dpi.import` (::circt::sv::FuncDPIImportOp)

*DPI import statement*

Syntax:

```
operation ::= `sv.func.dpi.import` (`linkage` $linkage_name^)? $callee attr-dict
```

`sv.func.dpi.import` represents DPI function import statement defined in
IEEE 1800-2017 section 35.4.

Interfaces: `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `callee` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `linkage_name` | ::mlir::StringAttr | string attribute |

### `sv.fwrite` (::circt::sv::FWriteOp)

*’$fwrite’ statement*

Syntax:

```
operation ::= `sv.fwrite` $fd `,` $format_string attr-dict (`(` $substitutions^ `)` `:`
              qualified(type($substitutions)))?
```

Traits: `ProceduralOp`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `format_string` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `fd` | 32-bit signless integer |
| `substitutions` | variadic of any type |

### `sv.generate` (::circt::sv::GenerateOp)

*A generate block*

Syntax:

```
operation ::= `sv.generate` $sym_name attr-dict `:` $body
```

See SystemVerilog 2017 27.

Traits: `NoRegionArguments`, `NoTerminator`, `SingleBlock`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `sv.generate.case` (::circt::sv::GenerateCaseOp)

*A ‘case’ statement inside of a generate block*

Syntax:

```
operation ::= `sv.generate.case` $cond attr-dict ` ` `[`
              custom<CaseRegions>($casePatterns, $caseNames, $caseRegions)
              `]`
```

See SystemVerilog 2017 27.5.

Traits: `HasParent<GenerateOp>`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `cond` | ::mlir::TypedAttr | TypedAttr instance |
| `casePatterns` | ::mlir::ArrayAttr | case pattern array |
| `caseNames` | ::mlir::ArrayAttr | string array attribute |

### `sv.if` (::circt::sv::IfOp)

*‘if (cond)’ block*

Syntax:

```
operation ::= `sv.if` $cond $thenRegion (`else` $elseRegion^)? attr-dict
```

Traits: `NoRegionArguments`, `NoTerminator`, `ProceduralOp`, `ProceduralRegion`, `SingleBlock`

#### Operands:

| Operand | Description |
| --- | --- |
| `cond` | 1-bit signless integer |

### `sv.ifdef` (::circt::sv::IfDefOp)

*‘ifdef MACRO’ block*

Syntax:

```
operation ::= `sv.ifdef` $cond $thenRegion (`else` $elseRegion^)? attr-dict
```

This operation is an #ifdef block, which has a “then” and “else” region.
This operation is for non-procedural regions and its body is non-procedural.

Traits: `HasOnlyGraphRegion`, `NoRegionArguments`, `NoTerminator`, `NonProceduralOp`, `SingleBlock`

Interfaces: `RegionKindInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `cond` | ::circt::sv::MacroIdentAttr | Macro identifier |

### `sv.ifdef.procedural` (::circt::sv::IfDefProceduralOp)

*‘ifdef MACRO’ block for procedural regions*

Syntax:

```
operation ::= `sv.ifdef.procedural` $cond $thenRegion (`else` $elseRegion^)? attr-dict
```

This operation is an #ifdef block, which has a “then” and “else” region.
This operation is for procedural regions and its body is procedural.

Traits: `NoRegionArguments`, `NoTerminator`, `ProceduralOp`, `ProceduralRegion`, `SingleBlock`

Interfaces: `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `cond` | ::circt::sv::MacroIdentAttr | Macro identifier |

### `sv.include` (::circt::sv::IncludeOp)

*Preprocessor `include directive*

Syntax:

```
operation ::= `sv.include` $style $target attr-dict
```

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `style` | ::circt::sv::IncludeStyleAttr | Double-quoted local include, or angle bracketed system include |
| `target` | ::mlir::StringAttr | string attribute |

### `sv.indexed_part_select` (::circt::sv::IndexedPartSelectOp)

*Read several contiguous bits of an int type.This is an indexed part-select operator.The base is an integer expression and the width is an integer constant. The bits start from base and the number of bits selected is equal to width. If $decrement is true, then part select decrements starting from $base.See SV Spec 11.5.1.*

Syntax:

```
operation ::= `sv.indexed_part_select` $input`[`$base (`decrement` $decrement^)?`:` $width`]` attr-dict `:` qualified(type($input)) `,` qualified(type($base))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `width` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `decrement` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a signless integer bitvector |
| `base` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `sv.indexed_part_select_inout` (::circt::sv::IndexedPartSelectInOutOp)

*Address several contiguous bits of an inout type (e.g. a wire or inout port). This is an indexed part-select operator.The base is an integer expression and the width is an integer constant. The bits start from base and the number of bits selected is equal to width. If $decrement is true, then part select decrements starting from $base.See SV Spec 11.5.1.*

Syntax:

```
operation ::= `sv.indexed_part_select_inout` $input`[`$base (`decrement` $decrement^)?`:` $width`]` attr-dict `:` qualified(type($input)) `,` qualified(type($base))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `width` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `decrement` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | InOutType |
| `base` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | InOutType |

### `sv.info` (::circt::sv::InfoOp)

*`$info` severity message task*

Syntax:

```
operation ::= `sv.info` ($message^ (`(` $substitutions^ `)` `:` qualified(type($substitutions)))?)?
              attr-dict
```

This system task indicates a message with no specific severity.

If present, the optional message is printed with any additional operands
interpolated into the message string.

Traits: `ProceduralOp`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `message` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `substitutions` | variadic of any type |

### `sv.initial` (::circt::sv::InitialOp)

*‘initial’ block*

Syntax:

```
operation ::= `sv.initial` $body attr-dict
```

See SV Spec 9.2.1.

Traits: `NoRegionArguments`, `NoTerminator`, `NonProceduralOp`, `ProceduralRegion`, `RecursiveMemoryEffects`, `RecursivelySpeculatableImplTrait`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`

### `sv.interface` (::circt::sv::InterfaceOp)

*Operation to define a SystemVerilog interface*

Syntax:

```
operation ::= `sv.interface` $sym_name attr-dict-with-keyword $body
```

This operation defines a named interface. Its name is a symbol that can
be looked up when declared inside a SymbolTable operation. This operation is
also a SymbolTable itself, so the symbols in its region can be looked up.

Example:

```
sv.interface @myinterface {
  sv.interface.signal @data : i32
  sv.interface.modport @input_port (input @data)
  sv.interface.modport @output_port (output @data)
}
```

Traits: `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `comment` | ::mlir::StringAttr | string attribute |

### `sv.interface.instance` (::circt::sv::InterfaceInstanceOp)

*Instantiate an interface*

Syntax:

```
operation ::= `sv.interface.instance` (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) attr-dict
              `:` qualified(type($result))
```

Use this to declare an instance of an interface:

```
%iface = sv.interface.instance : !sv.interface<@handshake_example>
```

Interfaces: `HasCustomSSAName`, `InnerSymbolOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `doNotPrint` | ::mlir::UnitAttr | unit attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | SystemVerilog interface type pointing to an InterfaceOp |

### `sv.interface.modport` (::circt::sv::InterfaceModportOp)

*Operation to define a SystemVerilog modport for interfaces*

Syntax:

```
operation ::= `sv.interface.modport` attr-dict $sym_name custom<ModportStructs>($ports)
```

This operation defines a named modport within an interface. Its name is a
symbol that can be looked up inside its parent interface. There is an array
of structs that contains two fields: an enum to indicate the direction of
the signal in the modport, and a symbol reference to refer to the signal.

Example:

```
sv.interface.modport @input_port (input @data)
sv.interface.modport @output_port (output @data)
```

Traits: `HasParent<InterfaceOp>`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `ports` | ::mlir::ArrayAttr | array of modport structs |

### `sv.interface.signal` (::circt::sv::InterfaceSignalOp)

*Operation to define a SystemVerilog signal for interfaces*

Syntax:

```
operation ::= `sv.interface.signal` attr-dict $sym_name `:` $type
```

This operation defines a named signal within an interface. Its type is
specified in an attribute, and currently supports IntegerTypes.

Example:

```
sv.interface.signal @data : i32
```

Traits: `HasParent<InterfaceOp>`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `type` | ::mlir::TypeAttr | Any SV/HW type |

### `sv.interface.signal.assign` (::circt::sv::AssignInterfaceSignalOp)

*Assign an interfaces signal to some other signal.*

Syntax:

```
operation ::= `sv.interface.signal.assign` $iface `(` custom<IfaceTypeAndSignal>(type($iface), $signalName) `)`
              `=` $rhs attr-dict `:` qualified(type($rhs))
```

Use this to continuously assign a signal inside an interface to a
value or other signal.

```
  sv.interface.signal.assign %iface(@handshake_example::@data)
    = %zero32 : i32
```

Would result in the following SystemVerilog:

```
  assign iface.data = zero32;
```

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `signalName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `iface` | SystemVerilog interface type pointing to an InterfaceOp |
| `rhs` | any type |

### `sv.interface.signal.read` (::circt::sv::ReadInterfaceSignalOp)

*Access the data in an interface’s signal.*

Syntax:

```
operation ::= `sv.interface.signal.read` $iface `(` custom<IfaceTypeAndSignal>(type($iface), $signalName) `)`
              attr-dict `:` qualified(type($signalData))
```

This is an expression to access a signal inside of an interface.

```
  %ifaceData = sv.interface.signal.read %iface
      (@handshake_example::@data) : i32
```

Could result in the following SystemVerilog:

```
  wire [31:0] ifaceData = iface.data;
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `signalName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `iface` | SystemVerilog interface type pointing to an InterfaceOp |

#### Results:

| Result | Description |
| --- | --- |
| `signalData` | any type |

### `sv.localparam` (::circt::sv::LocalParamOp)

*Declare a localparam*

Syntax:

```
operation ::= `sv.localparam` `` custom<ImplicitSSAName>($name) attr-dict `:` qualified(type($result))
```

The localparam operation produces a `localparam` declaration. See SV spec
6.20.4 p125.

Traits: `AlwaysSpeculatableImplTrait`, `FirstAttrDerivedResultType`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::Attribute | any attribute |
| `name` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a known primitive element |

### `sv.logic` (::circt::sv::LogicOp)

*Define a logic*

Syntax:

```
operation ::= `sv.logic` (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) attr-dict
              `:` qualified(type($result))
```

Declare a SystemVerilog Variable Declaration of ’logic’ type.
See SV Spec 6.8, pp100.

Interfaces: `InnerSymbolOpInterface`, `OpAsmOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |

#### Results:

| Result | Description |
| --- | --- |
| `result` | InOutType |

### `sv.macro.decl` (::circt::sv::MacroDeclOp)

*System verilog macro declaration*

Syntax:

```
operation ::= `sv.macro.decl` $sym_name (`[` $verilogName^ `]`)? (`(` $args^ `)`)?  attr-dict
```

The `sv.macro.def` declares a macro in System Verilog. This is a
declaration; the body of the macro, which produces a verilog macro
definition is created with a `macro.def` operation.

Lacking args will be a macro without “()”. An empty args will be an empty “()”.

The verilog name is the spelling of the macro when emitting verilog.

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `args` | ::mlir::ArrayAttr | string array attribute |
| `verilogName` | ::mlir::StringAttr | string attribute |

### `sv.macro.def` (::circt::sv::MacroDefOp)

*System verilog macro definition*

Syntax:

```
operation ::= `sv.macro.def` $macroName $format_string (`(` $symbols^ `)`)? attr-dict
```

The `sv.macro.def` defines a macro in System Verilog which optionally takes
a body.

This is modeled similarly to verbatim in that the contents of the macro are
opaque (plain string). Given the general power of macros, this op does not
try to capture a return type.

This operation produces a definition for the macro declaration referenced by
`sym_name`. Argument lists are picked up from that operation.

sv.macro.def allows operand substitutions with {{0}} syntax.

Interfaces: `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `macroName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `format_string` | ::mlir::StringAttr | string attribute |
| `symbols` | ::mlir::ArrayAttr | name reference array attribute |

### `sv.macro.error` (::circt::sv::MacroErrorOp)

*Produce a compilation error using the preprocessor*

Syntax:

```
operation ::= `sv.macro.error` ($message^)? attr-dict
```

The `sv.macro.error` op represents a static error in System Verilog. Since
System Verilog lacks an error preprocessor directive, this is emitted as a
reference to an undefined macro definition.

We reserve the macro prefix \_ERROR for emitting static assertions.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `message` | ::mlir::StringAttr | string attribute |

### `sv.macro.ref` (::circt::sv::MacroRefOp)

*Statement to refer to a SystemVerilog macro*

Syntax:

```
operation ::= `sv.macro.ref` $macroName (`(` $inputs^ `)` `:` type($inputs))? attr-dict
```

This operation represent a statement by referencing a named macro.

Interfaces: `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `macroName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

### `sv.macro.ref.expr` (::circt::sv::MacroRefExprOp)

*Expression to refer to a SystemVerilog macro*

Syntax:

```
operation ::= `sv.macro.ref.expr` $macroName `(` $inputs `)` attr-dict `:` functional-type($inputs, $result)
```

This operation produces a value by referencing a named macro.

Presently, it is assumed that the referenced macro is a constant with no
side effects. This expression is subject to CSE. It can be duplicated
and emitted inline by the Verilog emitter.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `macroName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a known primitive element or InOutType |

### `sv.macro.ref.expr.se` (::circt::sv::MacroRefExprSEOp)

*Expression to refer to a SystemVerilog macro*

Syntax:

```
operation ::= `sv.macro.ref.expr.se` $macroName `(` $inputs `)` attr-dict `:` functional-type($inputs, $result)
```

This operation produces a value by referencing a named macro.

Presently, it is assumed that the referenced macro is not constant and has
side effects. This expression is not subject to CSE. It can not be
duplicated, but can be emitted inline by the Verilog emitter.

Interfaces: `HasCustomSSAName`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `macroName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a known primitive element or InOutType |

### `sv.modport.get` (::circt::sv::GetModportOp)

*Get a modport out of an interface instance*

Syntax:

```
operation ::= `sv.modport.get` $iface $field attr-dict `:` qualified(type($iface)) `->` qualified(type($result))
```

Use this to extract a modport view to an instantiated interface. For
example, to get the ‘dataflow\_in’ modport on the ‘handshake\_example’
interface:

```
%ifaceModport = sv.modport.get @dataflow_in %iface :
  !sv.interface<@handshake_example> ->
  !sv.modport<@handshake_example::@dataflow_in>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `field` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `iface` | SystemVerilog interface type pointing to an InterfaceOp |

#### Results:

| Result | Description |
| --- | --- |
| `result` | SystemVerilog type pointing to an InterfaceModportOp |

### `sv.nonstandard.deposit` (::circt::sv::DepositOp)

*`$deposit` system task*

Syntax:

```
operation ::= `sv.nonstandard.deposit` $dest `,` $src  attr-dict `:` qualified(type($src))
```

This system task sets the value of a net or variable, but doesn’t hold it.
This is a common simulation vendor extension.

Traits: `ProceduralOp`, `VendorExtension`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | InOutType |
| `src` | a valid inout element |

### `sv.ordered` (::circt::sv::OrderedOutputOp)

*A sub-graph region which guarantees to output statements in-order*

Syntax:

```
operation ::= `sv.ordered` $body attr-dict
```

This operation groups operations into a region whose purpose is to force
verilog emission to be statement-by-statement, in-order. This allows
side-effecting operations, or macro expansions which applie to subsequent
operations to be properly sequenced.
This operation is for non-procedural regions and its body is non-procedural.

Traits: `NoRegionArguments`, `NoTerminator`, `NonProceduralOp`, `SingleBlock`

### `sv.passign` (::circt::sv::PAssignOp)

*Nonblocking procedural assignment*

Syntax:

```
operation ::= `sv.passign` $dest `,` $src  attr-dict `:` qualified(type($src))
```

A SystemVerilog nonblocking procedural assignment statement ‘x <= y;’.
These occur in initial, always, task, and function blocks. The statement
can be scheduled without blocking procedural flow. See SV Spec 10.4.2.

Traits: `ProceduralOp`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | InOutType |
| `src` | a valid inout element |

### `sv.read_inout` (::circt::sv::ReadInOutOp)

*Get the value of from something of inout type (e.g. a wire or inout port) as the value itself.*

Syntax:

```
operation ::= `sv.read_inout` $input attr-dict `:` qualified(type($input))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | InOutType |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a known primitive element |

### `sv.readmem` (::circt::sv::ReadMemOp)

*Load a memory from a file in either binary or hex format*

Syntax:

```
operation ::= `sv.readmem` $dest `,` $filename `,` $base attr-dict `:` qualified(type($dest))
```

Load a memory from a file using either `$readmemh` or `$readmemb` based on
an attribute.

See Section 21.4 of IEEE 1800-2017 for more information.

Traits: `ProceduralOp`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `filename` | ::mlir::StringAttr | string attribute |
| `base` | ::MemBaseTypeAttrAttr | the numeric base of a memory file |

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | InOutType |

### `sv.reg` (::circt::sv::RegOp)

*Define a new `reg` in SystemVerilog*

Syntax:

```
operation ::= `sv.reg` (`init` $init^)? (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) attr-dict
              `:` qualified(type($result))
              custom<ImplicitInitType>(ref(type($result)),ref($init), type($init))
```

Declare a SystemVerilog Variable Declaration of ‘reg’ type.
See SV Spec 6.8, pp100.

Interfaces: `InnerSymbolOpInterface`, `OpAsmOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |

#### Operands:

| Operand | Description |
| --- | --- |
| `init` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | InOutType |

### `sv.release` (::circt::sv::ReleaseOp)

*Release procedural statement*

Syntax:

```
operation ::= `sv.release` $dest attr-dict `:` qualified(type($dest))
```

Release is used in conjunction with force. When released,
then if the variable does not currently have an active assign
procedural continuous assignment, the variable shall not immediately
change value. The variable shall maintain its current value until
the next procedural assignment or procedural continuous assignment
to the variable. Releasing a variable that currently has an
active assign procedural continuous assignment shall immediately
reestablish that assignment. See SV Spec 10.6.2.

Traits: `ProceduralOp`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | InOutType |

### `sv.reserve_names` (::circt::sv::ReserveNamesOp)

*Disallow a set of names to be used during emission*

Syntax:

```
operation ::= `sv.reserve_names` $reservedNames attr-dict
```

Traits: `HasParent<mlir::ModuleOp>`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `reservedNames` | ::mlir::ArrayAttr | string array attribute |

### `sv.return` (::circt::sv::ReturnOp)

*Function return operation*

Syntax:

```
operation ::= `sv.return` attr-dict ($operands^ `:` type($operands))?
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<FuncOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `operands` | variadic of any type |

### `sv.sformatf` (::circt::sv::SFormatFOp)

*’$sformatf’ task*

Syntax:

```
operation ::= `sv.sformatf` $format_string attr-dict (`(` $substitutions^ `)` `:` qualified(type($substitutions)))?
```

This operations represents `$sformatf` task that produces a string from
a format string and substitutions.

See section 21.3.3 of 1800-2023 for more details.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `format_string` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `substitutions` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a HW string |

### `sv.stop` (::circt::sv::StopOp)

*`$stop` system task*

Syntax:

```
operation ::= `sv.stop` $verbosity attr-dict
```

Causes the simulation to be suspended. Does not terminate the simulator.

The optional `verbosity` parameter controls how much diagnostic information
is printed when the system task is executed (see section 20.2 of IEEE
1800-2017):

* `0`: Prints nothing
* `1`: Prints simulation time and location (default)
* `2`: Prints simulation time, location, and statistics about the memory and
  CPU time used in simulation

Traits: `ProceduralOp`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `verbosity` | ::mlir::IntegerAttr | 8-bit signless integer attribute whose minimum value is 0 whose maximum value is 2 |

### `sv.struct_field_inout` (::circt::sv::StructFieldInOutOp)

*Create an subfield inout memory to produce an inout element.*

Syntax:

```
operation ::= `sv.struct_field_inout` $input `[` $field `]` attr-dict `:` qualified(type($input))
```

See SV Spec 7.2.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `field` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an inout type with struct field |

#### Results:

| Result | Description |
| --- | --- |
| `result` | InOutType |

### `sv.system` (::circt::sv::SystemFunctionOp)

*Simple SV System Function calls*

Syntax:

```
operation ::= `sv.system` $fnName `(` $args `)` attr-dict `:` functional-type($args, $out)
```

This operation calls the indicated system verilog system function. This
supports functions which take normal expression arguments.

See section 20 of the 2012 SV spec.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fnName` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `args` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `out` | any type |

### `sv.system.sampled` (::circt::sv::SampledOp)

*`$sampled` system function to sample a value*

Syntax:

```
operation ::= `sv.system.sampled` $expression attr-dict `:` qualified(type($expression))
```

Sample a value using System Verilog sampling semantics (see Section 16.5.1
of the SV 2017 specification for more information).

A use of `$sampled` is to safely read the value of a net/variable in a
concurrent assertion action block such that the value will be the same as
the value used when the assertion is triggered. See Section 16.9.3 of the
SV 2017 specification for more information.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `expression` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `sampledValue` | any type |

### `sv.system.stime` (::circt::sv::STimeOp)

*`$time` system function*

Syntax:

```
operation ::= `sv.system.stime` attr-dict `:` type($result)
```

Return a 32-bit integer value representing the time at which the function
was called.

Interfaces: `InferTypeOpInterface`

#### Results:

| Result | Description |
| --- | --- |
| `result` | 32-bit signless integer |

### `sv.system.time` (::circt::sv::TimeOp)

*`$time` system function*

Syntax:

```
operation ::= `sv.system.time` attr-dict `:` type($result)
```

Return a 64-bit integer value representing the time at which the function
was called.

Interfaces: `InferTypeOpInterface`

#### Results:

| Result | Description |
| --- | --- |
| `result` | 64-bit signless integer |

### `sv.unpacked_array_create` (::circt::sv::UnpackedArrayCreateOp)

*Create an unpacked array*

Syntax:

```
operation ::= `sv.unpacked_array_create` $inputs attr-dict `:` functional-type($inputs, $result)
```

Creates an unpacked array from a variable set of values. One or more values must be
listed.

See the HW-SV rationale document for details on operand ordering.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of a type without inout |

#### Results:

| Result | Description |
| --- | --- |
| `result` | an Unpacked ArrayType |

### `sv.unpacked_open_array_cast` (::circt::sv::UnpackedOpenArrayCastOp)

*Cast an unpacked array into an unpacked open array*

Syntax:

```
operation ::= `sv.unpacked_open_array_cast` $input attr-dict `:` functional-type($input, $result)
```

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an Unpacked ArrayType |

#### Results:

| Result | Description |
| --- | --- |
| `result` | an Unpacked Open ArrayType |

### `sv.verbatim` (::circt::sv::VerbatimOp)

*Verbatim opaque text emitted inline.*

Syntax:

```
operation ::= `sv.verbatim` $format_string (`(` $substitutions^ `)` `:`
              qualified(type($substitutions)))? attr-dict
```

This operation produces opaque text inline in the SystemVerilog output.

sv.verbatim allows operand substitutions with {{0}} syntax.

Interfaces: `InnerRefUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `format_string` | ::mlir::StringAttr | string attribute |
| `symbols` | ::mlir::ArrayAttr | name reference array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `substitutions` | variadic of any type |

### `sv.verbatim.expr` (::circt::sv::VerbatimExprOp)

*Expression that expands to a value given SystemVerilog text*

Syntax:

```
operation ::= `sv.verbatim.expr` $format_string (`(` $substitutions^ `)`)?
              `:` functional-type($substitutions, $result) attr-dict
```

This operation produces a typed value expressed by a string of
SystemVerilog. This can be used to access macros and other values that are
only sensible as Verilog text.

The text string is expected to have the highest precedence, so you should
include parentheses in the string if it isn’t a single token. This is also
assumed to not have side effects (use sv.verbatim.expr.se) if you need them.

sv.verbatim.expr allows operand substitutions with {{0}} syntax.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InnerRefUserOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `format_string` | ::mlir::StringAttr | string attribute |
| `symbols` | ::mlir::ArrayAttr | name reference array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `substitutions` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a known primitive element or InOutType |

### `sv.verbatim.expr.se` (::circt::sv::VerbatimExprSEOp)

*Expression that expands to a value given SystemVerilog text*

Syntax:

```
operation ::= `sv.verbatim.expr.se` $format_string (`(` $substitutions^ `)`)?
              `:` functional-type($substitutions, $result) attr-dict
```

This operation produces a typed value expressed by a string of
SystemVerilog. This can be used to access macros and other values that are
only sensible as Verilog text.

The text string is expected to have the highest precedence, so you should
include parentheses in the string if it isn’t a single token. This is
allowed to have side effects.

sv.verbatim.se.expr allows operand substitutions with {{0}} syntax.

Interfaces: `HasCustomSSAName`, `InnerRefUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `format_string` | ::mlir::StringAttr | string attribute |
| `symbols` | ::mlir::ArrayAttr | name reference array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `substitutions` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a known primitive element or InOutType |

### `sv.verbatim.module` (::circt::sv::SVVerbatimModuleOp)

*Verbatim verilog module*

The “sv.verbatim.module” operation represents a the module interface of a
verbatim verilog module. It references an “sv.verbatim.source” operation
that contains the actual verilog source code.

Traits: `HasParent<mlir::ModuleOp>`

Interfaces: `HWModuleLike`, `InstanceGraphModuleOpInterface`, `PortList`, `SymbolUserOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `module_type` | ::mlir::TypeAttr | type attribute of a module |
| `per_port_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `port_locs` | ::mlir::ArrayAttr | location array attribute |
| `parameters` | ::mlir::ArrayAttr | parameter array attribute |
| `source` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `verilogName` | ::mlir::StringAttr | string attribute |

### `sv.verbatim.source` (::circt::sv::SVVerbatimSourceOp)

*Verbatim verilog source*

The “sv.verbatim.source” operation represents verbatim verilog definition for
a module which may have parameters. Concrete parametrizations and their ports
are represented by a `hw.module.extern`.

Traits: `HasParent<mlir::ModuleOp>`

Interfaces: `SymbolUserOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `content` | ::mlir::StringAttr | string attribute |
| `output_file` | ::circt::hw::OutputFileAttr | Output file attribute |
| `parameters` | ::mlir::ArrayAttr | parameter array attribute |
| `additional_files` | ::mlir::ArrayAttr | symbol ref array attribute |
| `verilogName` | ::mlir::StringAttr | string attribute |

### `sv.warning` (::circt::sv::WarningOp)

*`$warning` severity message task*

Syntax:

```
operation ::= `sv.warning` ($message^ (`(` $substitutions^ `)` `:` qualified(type($substitutions)))?)?
              attr-dict
```

This system task indicates a run-time warning.

If present, the optional message is printed with any additional operands
interpolated into the message string.

Traits: `ProceduralOp`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `message` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `substitutions` | variadic of any type |

### `sv.wire` (::circt::sv::WireOp)

*Define a new wire*

Syntax:

```
operation ::= `sv.wire` (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) attr-dict
              `:` qualified(type($result))
```

Declare a SystemVerilog Net Declaration of ‘wire’ type.
See SV Spec 6.7, pp97.

Traits: `NonProceduralOp`

Interfaces: `InnerSymbolOpInterface`, `OpAsmOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |

#### Results:

| Result | Description |
| --- | --- |
| `result` | InOutType |

### `sv.xmr` (::circt::sv::XMROp)

*Encode a reference to a non-local net.*

Syntax:

```
operation ::= `sv.xmr` (`isRooted` $isRooted^)? custom<XMRPath>($path, $terminal) attr-dict `:` qualified(type($result))
```

This represents a non-local hierarchical name to a net, sometimes called a
cross-module reference. A hierarchical name may be absolute, when prefixed
with ‘$root’, in which case it is resolved from the set of top-level modules
(any non-instantiated modules). Non-absolute paths are resolved by
attempting resolution of the path locally, then recursively up the instance
graph. See SV Spec 23.6, pp721.

It is impossible to completely resolve a hierarchical name without making a
closed-world assumption in the compiler. We therefore don’t try to link
hierarchical names to what they resolve to at compile time. A frontend
generating this op should ensure that any instance or object in the intended
path has public visibility so paths are not invalidated.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `isRooted` | ::mlir::UnitAttr | unit attribute |
| `path` | ::mlir::ArrayAttr | string array attribute |
| `terminal` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | InOutType |

### `sv.xmr.ref` (::circt::sv::XMRRefOp)

*Encode a reference to something with a hw.hierpath.*

Syntax:

```
operation ::= `sv.xmr.ref` $ref ( $verbatimSuffix^ )? attr-dict `:` qualified(type($result))
```

This represents a hierarchical path, but using something which the compiler
can understand. In contrast to the XMROp (which models pure Verilog
hierarchical paths which may not map to anything knowable in the circuit),
this op uses a `hw.hierpath` to refer to something which exists in the
circuit.

Generally, this operation is always preferred for situations where
hierarchical paths cannot be known statically and may change.

`verbatimSuffix` should only be populated when the final operation on the
path is an instance of an external module.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `ref` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `verbatimSuffix` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | InOutType |

Attributes
----------

### CaseExprPatternAttr

*Represents a case expression pattern*

Syntax: `#sv.expr`

### IncludeStyleAttr

*Double-quoted local include, or angle bracketed system include*

Syntax:

```
#sv.include_style<
  ::IncludeStyle   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `::IncludeStyle` | an enum of type IncludeStyle |

### MacroIdentAttr

*Macro identifier*

Syntax:

```
#sv.macro.ident<
  ::mlir::FlatSymbolRefAttr   # ident
>
```

Represents a reference to a macro identifier.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| ident | `::mlir::FlatSymbolRefAttr` |  |

### ModportDirectionAttr

*Defines direction in a modport*

Syntax:

```
#sv.modport_direction<
  ::circt::sv::ModportDirection   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `::circt::sv::ModportDirection` | an enum of type ModportDirection |

### ModportStructAttr

Syntax:

```
#sv.mod_port<
  ::circt::sv::ModportDirectionAttr,   # direction
  mlir::FlatSymbolRefAttr   # signal
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| direction | `::circt::sv::ModportDirectionAttr` |  |
| signal | `mlir::FlatSymbolRefAttr` |  |

### SVAttributeAttr

*A Verilog Attribute*

This attribute is used to encode a Verilog *attribute*. A Verilog attribute
(not to be confused with an LLVM or MLIR attribute) is a syntactic mechanism
for adding metadata to specific declarations, statements, and expressions in
the Verilog language. *There are no “standard” attributes*. Specific tools
define and handle their own attributes.

Verilog attributes have a mandatory name and an optional constant
expression. This is encoded as a key (name) value (expression) pair.
Multiple attributes may be specified, either with multiple separate
attributes or by comman-separating name–expression pairs.

Currently, SV attributes don’t block most optimizations; therefore, users should
not expect that sv attributes always appear in the output verilog.
However, we must block optimizations that updating ops in-place
since it is mostly invalid to transfer SV attributes one to another.

For more information, refer to Section 5.12 of the SystemVerilog (1800-2017)
specification.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| name | `::mlir::StringAttr` |  |
| expression | `::mlir::StringAttr` |  |
| emitAsComment | `mlir::BoolAttr` |  |

### ValidationQualifierTypeEnumAttr

*Validation qualifier type*

Syntax:

```
#sv.validation_qualifier<
  ::circt::sv::ValidationQualifierTypeEnum   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `::circt::sv::ValidationQualifierTypeEnum` | an enum of type ValidationQualifierTypeEnum |

Type constraints
----------------

### an Unpacked Open ArrayType

Types
-----

### InterfaceType

*SystemVerilog interface type pointing to an InterfaceOp*

Syntax:

```
!sv.interface<
  ::mlir::FlatSymbolRefAttr   # interface
>
```

A MLIR type for the SV dialect’s `InterfaceOp` to allow instances in any
dialect with an open type system. Points at the InterfaceOp which defines
the SystemVerilog interface.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| interface | `::mlir::FlatSymbolRefAttr` |  |

### ModportType

*SystemVerilog type pointing to an InterfaceModportOp*

Syntax:

```
!sv.modport<
  ::mlir::SymbolRefAttr   # modport
>
```

A MLIR type for the SV dialect’s `InterfaceModportOp` to allow
interactions with any open type system dialect. Points at the
InterfaceModportOp which defines the SystemVerilog interface’s modport.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| modport | `::mlir::SymbolRefAttr` |  |

### UnpackedOpenArrayType

*SystemVerilog unpacked ‘open’ array*

Syntax:

```
!sv.open_uarray<
  ::mlir::Type   # elementType
>
```

This type represents unpacked ‘open’ array. Open array is a special type that can be
used as formal arguments of DPI import statements.

See SystemVerilog Spec 35.5.6.1.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `::mlir::Type` |  |

Enums
-----

### CaseStmtType

*Case type*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| CaseStmt | `0` | case |
| CaseXStmt | `1` | casex |
| CaseZStmt | `2` | casez |

### DeferAssert

*Assertion deferring mode*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| Immediate | `0` | immediate |
| Observed | `1` | observed |
| Final | `2` | final |

### EventControl

*Edge control trigger*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| AtPosEdge | `0` | posedge |
| AtNegEdge | `1` | negedge |
| AtEdge | `2` | edge |

### IncludeStyle

*Double-quoted local include, or angle bracketed system include*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| Local | `0` | local |
| System | `1` | system |

### MemBaseTypeAttr

*The numeric base of a memory file*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| MemBaseBin | `0` | MemBaseBin |
| MemBaseHex | `1` | MemBaseHex |

### ModportDirection

*Defines direction in a modport*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| input | `0` | input |
| output | `1` | output |
| inout | `2` | inout |

### ResetType

*Reset type*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| NoReset | `0` | noreset |
| SyncReset | `1` | syncreset |
| AsyncReset | `2` | asyncreset |

### ValidationQualifierTypeEnum

*Validation qualifier type*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| ValidationQualifierPlain | `0` | plain |
| ValidationQualifierUnique | `1` | unique |
| ValidationQualifierUnique0 | `2` | unique0 |
| ValidationQualifierPriority | `3` | priority |

'sv' Dialect Docs
-----------------

* [SV Dialect Rationale](https://circt.llvm.org/docs/Dialects/SV/RationaleSV/)

 [Prev - SSP Dialect Rationale](https://circt.llvm.org/docs/Dialects/SSP/RationaleSSP/ "SSP Dialect Rationale")
[Next - SV Dialect Rationale](https://circt.llvm.org/docs/Dialects/SV/RationaleSV/ "SV Dialect Rationale") 

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