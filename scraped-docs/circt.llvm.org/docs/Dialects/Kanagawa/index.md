'kanagawa' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'kanagawa' Dialect
==================

*Types and operations for Kanagawa dialect*

The `kanagawa` dialect is intended to support porting and eventual open sourcing
of an internal hardware development language.

* [Operations](#operations)
  + [`kanagawa.call` (::circt::kanagawa::CallOp)](#kanagawacall-circtkanagawacallop)
  + [`kanagawa.class` (::circt::kanagawa::ClassOp)](#kanagawaclass-circtkanagawaclassop)
  + [`kanagawa.container` (::circt::kanagawa::ContainerOp)](#kanagawacontainer-circtkanagawacontainerop)
  + [`kanagawa.container.instance` (::circt::kanagawa::ContainerInstanceOp)](#kanagawacontainerinstance-circtkanagawacontainerinstanceop)
  + [`kanagawa.design` (::circt::kanagawa::DesignOp)](#kanagawadesign-circtkanagawadesignop)
  + [`kanagawa.get_port` (::circt::kanagawa::GetPortOp)](#kanagawaget_port-circtkanagawagetportop)
  + [`kanagawa.get_var` (::circt::kanagawa::GetVarOp)](#kanagawaget_var-circtkanagawagetvarop)
  + [`kanagawa.instance` (::circt::kanagawa::InstanceOp)](#kanagawainstance-circtkanagawainstanceop)
  + [`kanagawa.method` (::circt::kanagawa::MethodOp)](#kanagawamethod-circtkanagawamethodop)
  + [`kanagawa.method.df` (::circt::kanagawa::DataflowMethodOp)](#kanagawamethoddf-circtkanagawadataflowmethodop)
  + [`kanagawa.path` (::circt::kanagawa::PathOp)](#kanagawapath-circtkanagawapathop)
  + [`kanagawa.pipeline.header` (::circt::kanagawa::PipelineHeaderOp)](#kanagawapipelineheader-circtkanagawapipelineheaderop)
  + [`kanagawa.port.input` (::circt::kanagawa::InputPortOp)](#kanagawaportinput-circtkanagawainputportop)
  + [`kanagawa.port.output` (::circt::kanagawa::OutputPortOp)](#kanagawaportoutput-circtkanagawaoutputportop)
  + [`kanagawa.port.read` (::circt::kanagawa::PortReadOp)](#kanagawaportread-circtkanagawaportreadop)
  + [`kanagawa.port.write` (::circt::kanagawa::PortWriteOp)](#kanagawaportwrite-circtkanagawaportwriteop)
  + [`kanagawa.return` (::circt::kanagawa::ReturnOp)](#kanagawareturn-circtkanagawareturnop)
  + [`kanagawa.sblock` (::circt::kanagawa::StaticBlockOp)](#kanagawasblock-circtkanagawastaticblockop)
  + [`kanagawa.sblock.dc` (::circt::kanagawa::DCBlockOp)](#kanagawasblockdc-circtkanagawadcblockop)
  + [`kanagawa.sblock.inline.begin` (::circt::kanagawa::InlineStaticBlockBeginOp)](#kanagawasblockinlinebegin-circtkanagawainlinestaticblockbeginop)
  + [`kanagawa.sblock.inline.end` (::circt::kanagawa::InlineStaticBlockEndOp)](#kanagawasblockinlineend-circtkanagawainlinestaticblockendop)
  + [`kanagawa.sblock.isolated` (::circt::kanagawa::IsolatedStaticBlockOp)](#kanagawasblockisolated-circtkanagawaisolatedstaticblockop)
  + [`kanagawa.sblock.return` (::circt::kanagawa::BlockReturnOp)](#kanagawasblockreturn-circtkanagawablockreturnop)
  + [`kanagawa.var` (::circt::kanagawa::VarOp)](#kanagawavar-circtkanagawavarop)
  + [`kanagawa.wire.input` (::circt::kanagawa::InputWireOp)](#kanagawawireinput-circtkanagawainputwireop)
  + [`kanagawa.wire.output` (::circt::kanagawa::OutputWireOp)](#kanagawawireoutput-circtkanagawaoutputwireop)
* [Attributes](#attributes-19)
  + [PathStepAttr](#pathstepattr)
* [Types](#types)
  + [PortRefType](#portreftype)
  + [ScopeRefType](#scopereftype)
* [Enums](#enums)
  + [Direction](#direction)
  + [PathDirection](#pathdirection)

Operations
----------

### `kanagawa.call` (::circt::kanagawa::CallOp)

*Kanagawa method call*

Syntax:

```
operation ::= `kanagawa.call` $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
```

Dispatch a call to an Kanagawa method.

Interfaces: `ArgAndResultAttrsOpInterface`, `CallOpInterface`, `InnerRefUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `callee` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |

#### Operands:

| Operand | Description |
| --- | --- |
| `operands` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| «unnamed» | variadic of any type |

### `kanagawa.class` (::circt::kanagawa::ClassOp)

*Kanagawa class*

Syntax:

```
operation ::= `kanagawa.class` ($name^)? `sym` $inner_sym attr-dict-with-keyword $body
```

Kanagawa has the notion of a class which can contain methods and member
variables.

In the low-level Kanagawa representation, the ClassOp becomes a container for
`kanagawa.port`s, `kanagawa.container`s, and contain logic for member variables.

Traits: `HasParent<DesignOp>`, `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`

Interfaces: `InnerSymbol`, `InstanceGraphModuleOpInterface`, `NamedInnerSymbol`, `RegionKindInterface`, `ScopeOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `name` | ::mlir::StringAttr | string attribute |

### `kanagawa.container` (::circt::kanagawa::ContainerOp)

*Kanagawa container*

Syntax:

```
operation ::= `kanagawa.container` ($name^)? `sym` $inner_sym (`top_level` $isTopLevel^)? attr-dict-with-keyword $body
```

An kanagawa container describes a collection of logic nested within an Kanagawa class.

Traits: `HasParent<DesignOp, ClassOp>`, `IsolatedFromAbove`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

Interfaces: `InnerSymbol`, `InstanceGraphModuleOpInterface`, `NamedInnerSymbolInterface`, `NamedInnerSymbol`, `RegionKindInterface`, `ScopeOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `isTopLevel` | ::mlir::UnitAttr | unit attribute |
| `name` | ::mlir::StringAttr | string attribute |

### `kanagawa.container.instance` (::circt::kanagawa::ContainerInstanceOp)

*Kanagawa container instance*

Syntax:

```
operation ::= `kanagawa.container.instance` $inner_sym `,` $targetName attr-dict
              custom<ScopeRefFromName>(type($scopeRef), ref($targetName))
```

Instantiates an Kanagawa container.

Interfaces: `HasCustomSSAName`, `InnerRefUserOpInterface`, `InnerSymbolOpInterface`, `InstanceGraphInstanceOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `targetName` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |

#### Results:

| Result | Description |
| --- | --- |
| `scopeRef` |  |

### `kanagawa.design` (::circt::kanagawa::DesignOp)

*All Kanagawa containers must be inside this op*

Syntax:

```
operation ::= `kanagawa.design` $sym_name attr-dict-with-keyword $body
```

Traits: `InnerSymbolTable`, `IsolatedFromAbove`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `kanagawa.get_port` (::circt::kanagawa::GetPortOp)

*Kanagawa get port*

Syntax:

```
operation ::= `kanagawa.get_port` $instance `,` $portSymbol `:` qualified(type($instance)) `->`
              qualified(type($port)) attr-dict
```

Given an Kanagawa class reference, returns a port of said class. The port
is specified by the symbol name of the port in the referenced class.

Importantly, the user must specify how they intend to use the op, by
specifying the direction of the portref type that this op is generated with.
If the request port is to be read from, the type must be `!kanagawa.portref<out T>`
and if the port is to be written to, the type must be `!kanagawa.portref<in T>`.
This is to ensure that the usage is reflected in the get\_port type which in
turn is used by the tunneling passes to create the proper ports through the
hierarchy.

This implies that the portref direction of the get\_port op is independent of
the actual direction of the target port, and only the inner portref type
must match.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InnerRefUserOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `portSymbol` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `instance` |  |

#### Results:

| Result | Description |
| --- | --- |
| `port` |  |

### `kanagawa.get_var` (::circt::kanagawa::GetVarOp)

*Dereferences an kanagawa member variable through a scoperef*

Syntax:

```
operation ::= `kanagawa.get_var` $instance `,` $varName attr-dict `:` qualified(type($instance)) `->` qualified(type($var))
```

Interfaces: `HasCustomSSAName`, `InnerRefUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `varName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `instance` |  |

#### Results:

| Result | Description |
| --- | --- |
| `var` | memref of any type values |

### `kanagawa.instance` (::circt::kanagawa::InstanceOp)

*Kanagawa class instance*

Syntax:

```
operation ::= `kanagawa.instance` $inner_sym `,` $targetName attr-dict
              custom<ScopeRefFromName>(type($scopeRef), ref($targetName))
```

Instantiates an Kanagawa class.

Interfaces: `HasCustomSSAName`, `InnerRefUserOpInterface`, `InnerSymbolOpInterface`, `InstanceGraphInstanceOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `targetName` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |

#### Results:

| Result | Description |
| --- | --- |
| `scopeRef` |  |

### `kanagawa.method` (::circt::kanagawa::MethodOp)

*Kanagawa method*

Kanagawa methods are a lot like software functions: a list of named arguments
and unnamed return values with imperative control flow.

Traits: `AutomaticAllocationScope`, `HasParent<ClassOp>`, `IsolatedFromAbove`

Interfaces: `InnerSymbolOpInterface`, `MethodLikeOpInterface`, `OpAsmOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `function_type` | ::mlir::TypeAttr | type attribute of function type |
| `argNames` | ::mlir::ArrayAttr | array attribute |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |

### `kanagawa.method.df` (::circt::kanagawa::DataflowMethodOp)

*Kanagawa dataflow method*

Kanagawa dataflow methods share the same interface as an `kanagawa.method` but
without imperative CFG-based control flow. Instead, this method implements a
graph region, and control flow is expected to be defined by dataflow operations.

Traits: `HasParent<ClassOp>`, `IsolatedFromAbove`, `SingleBlockImplicitTerminator<kanagawa::ReturnOp>`, `SingleBlock`

Interfaces: `FineGrainedDataflowRegionOpInterface`, `InnerSymbolOpInterface`, `MethodLikeOpInterface`, `RegionKindInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `function_type` | ::mlir::TypeAttr | type attribute of function type |
| `argNames` | ::mlir::ArrayAttr | array attribute |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |

### `kanagawa.path` (::circt::kanagawa::PathOp)

*Kanagawa path*

Syntax:

```
operation ::= `kanagawa.path` $path attr-dict
```

The `kanagawa.path` operation describes an instance hierarchy path relative to
the current scope. The path is specified by a list of either parent or
child identifiers (navigating up or down the hierarchy, respectively).

Scopes along the path are optionally typed, however, An `kanagawa.path` must
lways terminate in a fully typed specifier, i.e. never an `!kanagawa.scoperef<>`.

The operation returns a single `!kanagawa.scoperef`-typed value representing
the scope at the end of the path.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `InnerRefUserOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `path` | ::mlir::ArrayAttr | Path step array attribute |

#### Results:

| Result | Description |
| --- | --- |
| `instance` |  |

### `kanagawa.pipeline.header` (::circt::kanagawa::PipelineHeaderOp)

*Kanagawa pipeline header operation*

Syntax:

```
operation ::= `kanagawa.pipeline.header` attr-dict
```

This operation defines the hardware-like values used to drive a pipeline,
such as clock and reset.
This is an intermediate operation, meaning that it’s strictly used to
facilitate progressive lowering of kanagawa static blocks to scheduled pipelines.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `clock` | A type for clock-carrying wires |
| `reset` | 1-bit signless integer |
| `go` | 1-bit signless integer |
| `stall` | 1-bit signless integer |

### `kanagawa.port.input` (::circt::kanagawa::InputPortOp)

*Kanagawa input port*

Syntax:

```
operation ::= `kanagawa.port.input` ($name^)? `sym` $inner_sym `:` $type attr-dict
```

An kanagawa port has an attached ’name’ attribute. This is a name-hint used
to generate the final port name. The port name and port symbol are not
related, and all references to a port is done through the port symbol.

Traits: `HasParent<ClassOp, ContainerOp>`

Interfaces: `HasCustomSSAName`, `InferTypeOpInterface`, `InnerSymbol`, `NamedInnerSymbol`, `PortOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `type` | ::mlir::TypeAttr | type attribute of any type |
| `name` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `port` |  |

### `kanagawa.port.output` (::circt::kanagawa::OutputPortOp)

*Kanagawa output port*

Syntax:

```
operation ::= `kanagawa.port.output` ($name^)? `sym` $inner_sym `:` $type attr-dict
```

An kanagawa port has an attached ’name’ attribute. This is a name-hint used
to generate the final port name. The port name and port symbol are not
related, and all references to a port is done through the port symbol.

Traits: `HasParent<ClassOp, ContainerOp>`

Interfaces: `HasCustomSSAName`, `InferTypeOpInterface`, `InnerSymbol`, `NamedInnerSymbol`, `PortOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `type` | ::mlir::TypeAttr | type attribute of any type |
| `name` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `port` |  |

### `kanagawa.port.read` (::circt::kanagawa::PortReadOp)

*Kanagawa port read*

Syntax:

```
operation ::= `kanagawa.port.read` $port attr-dict `:` qualified(type($port))
```

Read the value of a port reference.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `port` |  |

#### Results:

| Result | Description |
| --- | --- |
| `output` | any type |

### `kanagawa.port.write` (::circt::kanagawa::PortWriteOp)

*Kanagawa port write*

Syntax:

```
operation ::= `kanagawa.port.write` $port `,` $value attr-dict `:` qualified(type($port))
```

Write a value to a port reference.

#### Operands:

| Operand | Description |
| --- | --- |
| `port` |  |
| `value` | any type |

### `kanagawa.return` (::circt::kanagawa::ReturnOp)

*Kanagawa method terminator*

Syntax:

```
operation ::= `kanagawa.return` ($retValues^)? attr-dict (`:` type($retValues)^)?
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<MethodOp, DataflowMethodOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `retValues` | variadic of any type |

### `kanagawa.sblock` (::circt::kanagawa::StaticBlockOp)

*Kanagawa block*

The `kanagawa.sblock` operation defines a block wherein a group of operations
are expected to be statically scheduleable.
The operation is not isolated from above to facilitate ease of construction.
However, once a program has been constructed and lowered to a sufficient
level, the user may run `--kanagawa-argify-blocks` to effectively isolate the
block from above, by converting SSA values referenced through dominanes into
arguments of the block

The block may contain additional attributes to specify constraints on
the block further down the compilation pipeline.

Traits: `AutomaticAllocationScope`, `SingleBlockImplicitTerminator<BlockReturnOp>`, `SingleBlock`

Interfaces: `BlockOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `maxThreads` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `kanagawa.sblock.dc` (::circt::kanagawa::DCBlockOp)

*DC-interfaced Kanagawa block*

The `kanagawa.sblock.dc` operation is like an `kanagawa.sblock` operation with
a few differences, being:

1. The operation is DC-interfaced, meaning that all arguments and results
   are dc-value typed.
2. The operation is IsolatedFromAbove.

Traits: `AutomaticAllocationScope`, `IsolatedFromAbove`, `SingleBlockImplicitTerminator<BlockReturnOp>`, `SingleBlock`

Interfaces: `BlockOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `maxThreads` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `kanagawa.sblock.inline.begin` (::circt::kanagawa::InlineStaticBlockBeginOp)

*Kanagawa inline static block begin marker*

Syntax:

```
operation ::= `kanagawa.sblock.inline.begin` attr-dict
```

The `kanagawa.sblock.inline.begin` operation is a marker that indicates the
begin of an inline static block.
The operation is used to maintain `kanagawa.sblocks` while in the Kanagawa inline
phase (to facilitate e.g. mem2reg).

The operation:

1. denotes the begin of the sblock
2. carries whatever attributes that the source `kanagawa.sblock` carried.
3. is considered side-effectfull.

Traits: `HasParent<MethodOp>`

### `kanagawa.sblock.inline.end` (::circt::kanagawa::InlineStaticBlockEndOp)

*Kanagawa inline static block end marker*

Syntax:

```
operation ::= `kanagawa.sblock.inline.end` attr-dict
```

The `kanagawa.sblock.inline.end` operation is a marker that indicates the
end of an inline static block.
The operation is used to maintain `kanagawa.sblocks` while in the Kanagawa inline
phase (to facilitate e.g. mem2reg).

Traits: `HasParent<MethodOp>`

### `kanagawa.sblock.isolated` (::circt::kanagawa::IsolatedStaticBlockOp)

*Kanagawa isolated block*

The `kanagawa.sblock.isolated` operation is like an `kanagawa.sblock` operation
but with an IsolatedFromAbove condition, meaning that all arguments and
results are passed through the block as arguments and results.

Traits: `AutomaticAllocationScope`, `IsolatedFromAbove`, `SingleBlockImplicitTerminator<BlockReturnOp>`, `SingleBlock`

Interfaces: `BlockOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `maxThreads` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `kanagawa.sblock.return` (::circt::kanagawa::BlockReturnOp)

*Kanagawa static block terminator*

Syntax:

```
operation ::= `kanagawa.sblock.return` ($retValues^)? attr-dict (`:` type($retValues)^)?
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<StaticBlockOp, IsolatedStaticBlockOp, DCBlockOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `retValues` | variadic of any type |

### `kanagawa.var` (::circt::kanagawa::VarOp)

*Kanagawa variable definition*

Syntax:

```
operation ::= `kanagawa.var` $inner_sym `:` $type attr-dict
```

Defines an Kanagawa class member variable. The variable is typed with a
`memref.memref` type, and may define either a singleton or uni-dimensional
array of values.
`kanagawa.var` defines a symbol within the encompassing class scope which can
be dereferenced through a `!kanagawa.scoperef` value of the parent class.

Interfaces: `InnerSymbolOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `type` | ::mlir::TypeAttr | any memref type |

### `kanagawa.wire.input` (::circt::kanagawa::InputWireOp)

*Kanagawa input wire*

Syntax:

```
operation ::= `kanagawa.wire.input` $inner_sym `:` qualified(type($output)) attr-dict
```

An input wire defines an `kanagawa.portref<in T>` port alongside a value
of type `T` which represents the value to-be-written to the wire.

Traits: `HasParent<ClassOp, ContainerOp>`

Interfaces: `HasCustomSSAName`, `InnerSymbolOpInterface`, `InnerSymbol`, `NamedInnerSymbol`, `PortOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `name` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `port` |  |
| `output` | any type |

### `kanagawa.wire.output` (::circt::kanagawa::OutputWireOp)

*Kanagawa output wire*

Syntax:

```
operation ::= `kanagawa.wire.output` $inner_sym `,` $input `:` qualified(type($input)) attr-dict
```

An output wire defines an `kanagawa.portref<out T>` port that can be read.
The operation takes an input value of type `T` which represents the value
on the output portref.

Traits: `HasParent<ClassOp, ContainerOp>`

Interfaces: `HasCustomSSAName`, `InferTypeOpInterface`, `InnerSymbolOpInterface`, `InnerSymbol`, `NamedInnerSymbol`, `PortOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `port` |  |

Attributes
----------

### PathStepAttr

Syntax:

```
#kanagawa.step<
  PathDirection,   # direction
  ::mlir::Type,   # type
  mlir::FlatSymbolRefAttr   # child
>
```

Used to describe a single step in a path

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| direction | `PathDirection` |  |
| type | `::mlir::Type` |  |
| child | `mlir::FlatSymbolRefAttr` |  |

Types
-----

### PortRefType

Syntax:

```
!kanagawa.portref<
  TypeAttr,   # portTypeAttr
  kanagawa::Direction   # direction
>
```

A reference to an Kanagawa port.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| portTypeAttr | `TypeAttr` |  |
| direction | `kanagawa::Direction` |  |

### ScopeRefType

A reference to an Kanagawa scope. May be either a reference to a specific
scope (given a `$scopeName` argument) or an opaque reference.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| scopeRef | `::circt::hw::InnerRefAttr` |  |

Enums
-----

### Direction

*Kanagawa port direction*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| Input | `0` | in |
| Output | `1` | out |

### PathDirection

*Path direction*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| Parent | `0` | parent |
| Child | `1` | child |

'kanagawa' Dialect Docs
-----------------------

* [`kanagawa` Dialect Rationale](https://circt.llvm.org/docs/Dialects/Kanagawa/RationaleKanagawa/)

 [Prev - HW Arith Dialect Rationale](https://circt.llvm.org/docs/Dialects/HWArith/RationaleHWArith/ "HW Arith Dialect Rationale")
[Next - `kanagawa` Dialect Rationale](https://circt.llvm.org/docs/Dialects/Kanagawa/RationaleKanagawa/ "`kanagawa` Dialect Rationale") 

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