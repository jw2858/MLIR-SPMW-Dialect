'hw' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'hw' Dialect
============

This dialect defines the `hw` dialect, which is intended to be a generic
representation of HW outside of a particular use-case.

* [Operation Definitions – Structure](#operation-definitions----structure)
  + [`hw.generator.schema` (::circt::hw::HWGeneratorSchemaOp)](#hwgeneratorschema-circthwhwgeneratorschemaop)
  + [`hw.module.extern` (::circt::hw::HWModuleExternOp)](#hwmoduleextern-circthwhwmoduleexternop)
  + [`hw.module.generated` (::circt::hw::HWModuleGeneratedOp)](#hwmodulegenerated-circthwhwmodulegeneratedop)
  + [`hw.module` (::circt::hw::HWModuleOp)](#hwmodule-circthwhwmoduleop)
  + [`hw.hierpath` (::circt::hw::HierPathOp)](#hwhierpath-circthwhierpathop)
  + [`hw.instance_choice` (::circt::hw::InstanceChoiceOp)](#hwinstance_choice-circthwinstancechoiceop)
  + [`hw.instance` (::circt::hw::InstanceOp)](#hwinstance-circthwinstanceop)
  + [`hw.output` (::circt::hw::OutputOp)](#hwoutput-circthwoutputop)
  + [`hw.triggered` (::circt::hw::TriggeredOp)](#hwtriggered-circthwtriggeredop)
* [Operation Definitions – Miscellaneous](#operation-definitions----miscellaneous)
  + [`hw.bitcast` (::circt::hw::BitcastOp)](#hwbitcast-circthwbitcastop)
  + [`hw.constant` (::circt::hw::ConstantOp)](#hwconstant-circthwconstantop)
  + [`hw.enum.cmp` (::circt::hw::EnumCmpOp)](#hwenumcmp-circthwenumcmpop)
  + [`hw.enum.constant` (::circt::hw::EnumConstantOp)](#hwenumconstant-circthwenumconstantop)
  + [`hw.param.value` (::circt::hw::ParamValueOp)](#hwparamvalue-circthwparamvalueop)
  + [`hw.wire` (::circt::hw::WireOp)](#hwwire-circthwwireop)
* [Operation Definitions – Aggregates](#operation-definitions----aggregates)
  + [`hw.aggregate_constant` (::circt::hw::AggregateConstantOp)](#hwaggregate_constant-circthwaggregateconstantop)
  + [`hw.array_concat` (::circt::hw::ArrayConcatOp)](#hwarray_concat-circthwarrayconcatop)
  + [`hw.array_create` (::circt::hw::ArrayCreateOp)](#hwarray_create-circthwarraycreateop)
  + [`hw.array_get` (::circt::hw::ArrayGetOp)](#hwarray_get-circthwarraygetop)
  + [`hw.array_inject` (::circt::hw::ArrayInjectOp)](#hwarray_inject-circthwarrayinjectop)
  + [`hw.array_slice` (::circt::hw::ArraySliceOp)](#hwarray_slice-circthwarraysliceop)
  + [`hw.struct_create` (::circt::hw::StructCreateOp)](#hwstruct_create-circthwstructcreateop)
  + [`hw.struct_explode` (::circt::hw::StructExplodeOp)](#hwstruct_explode-circthwstructexplodeop)
  + [`hw.struct_extract` (::circt::hw::StructExtractOp)](#hwstruct_extract-circthwstructextractop)
  + [`hw.struct_inject` (::circt::hw::StructInjectOp)](#hwstruct_inject-circthwstructinjectop)
  + [`hw.union_create` (::circt::hw::UnionCreateOp)](#hwunion_create-circthwunioncreateop)
  + [`hw.union_extract` (::circt::hw::UnionExtractOp)](#hwunion_extract-circthwunionextractop)
* [Operation Definitions – Type Declarations](#operation-definitions----type-declarations)
  + [`hw.type_scope` (::circt::hw::TypeScopeOp)](#hwtype_scope-circthwtypescopeop)
  + [`hw.typedecl` (::circt::hw::TypedeclOp)](#hwtypedecl-circthwtypedeclop)
* [Attribute Definitions](#attribute-definitions)
  + [EnumFieldAttr](#enumfieldattr)
  + [OutputFileAttr](#outputfileattr)
  + [ParamDeclAttr](#paramdeclattr)
  + [ParamDeclRefAttr](#paramdeclrefattr)
  + [ParamExprAttr](#paramexprattr)
  + [ParamVerbatimAttr](#paramverbatimattr)
  + [InnerRefAttr](#innerrefattr)
* [Type Definitions](#type-definitions)
  + [`hw.type_scope` (::circt::hw::TypeScopeOp)](#hwtype_scope-circthwtypescopeop-1)
  + [`hw.typedecl` (::circt::hw::TypedeclOp)](#hwtypedecl-circthwtypedeclop-1)
  + [ArrayType](#arraytype)
  + [EnumType](#enumtype)
  + [StringType](#stringtype)
  + [InOutType](#inouttype)
  + [IntType](#inttype)
  + [ModuleType](#moduletype)
  + [StructType](#structtype)
  + [TypeAliasType](#typealiastype)
  + [UnionType](#uniontype)
  + [UnpackedArrayType](#unpackedarraytype)
* [CombDataFlow (`CombDataflow`)](#combdataflow-combdataflow)
  + [Methods:](#methods)
* [HWEmittableModuleLike (`HWEmittableModuleLike`)](#hwemittablemodulelike-hwemittablemodulelike)
  + [Methods:](#methods-1)
* [HWInstanceLike (`HWInstanceLike`)](#hwinstancelike-hwinstancelike)
  + [Methods:](#methods-2)
* [HWModuleLike (`HWModuleLike`)](#hwmodulelike-hwmodulelike)
  + [Methods:](#methods-3)
* [HWMutableModuleLike (`HWMutableModuleLike`)](#hwmutablemodulelike-hwmutablemodulelike)
  + [Methods:](#methods-4)
* [InnerRefUserOpInterface (`InnerRefUserOpInterface`)](#innerrefuseropinterface-innerrefuseropinterface)
  + [Methods:](#methods-5)
* [InnerSymbolOpInterface (`InnerSymbol`)](#innersymbolopinterface-innersymbol)
  + [Methods:](#methods-6)
* [PortList (`PortList`)](#portlist-portlist)
  + [Methods:](#methods-7)
* [BitWidthTypeInterface (`BitWidthTypeInterface`)](#bitwidthtypeinterface-bitwidthtypeinterface)
  + [Methods:](#methods-8)
* [FieldIDTypeInterface (`FieldIDTypeInterface`)](#fieldidtypeinterface-fieldidtypeinterface)
  + [Methods:](#methods-9)

Operation Definitions – Structure
---------------------------------

### `hw.generator.schema` (::circt::hw::HWGeneratorSchemaOp)

*HW Generator Schema declaration*

Syntax:

```
operation ::= `hw.generator.schema` $sym_name `,` $descriptor `,` $requiredAttrs attr-dict
```

The “hw.generator.schema” operation declares a kind of generated module by
declaring the schema of meta-data required.
A generated module instance of a schema is independent of the external
method of producing it. It is assumed that for well known schema instances,
multiple external tools might exist which can process it. Generator nodes
list attributes required by hw.module.generated instances.

Example:

```
generator.schema @MEMORY, "Simple-Memory", ["ports", "write_latency", "read_latency"]
module.generated @mymem, @MEMORY(ports)
  -> (ports) {write_latency=1, read_latency=1, ports=["read","write"]}
```

Traits: `HasParent<mlir::ModuleOp>`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `descriptor` | ::mlir::StringAttr | string attribute |
| `requiredAttrs` | ::mlir::ArrayAttr | string array attribute |

### `hw.module.extern` (::circt::hw::HWModuleExternOp)

*HW external Module*

The “hw.module.extern” operation represents an external reference to a
Verilog module, including a given name and a list of ports.

The ‘verilogName’ attribute (when present) specifies the spelling of the
module name in Verilog we can use. TODO: This is a hack because we don’t
have proper parameterization in the hw.dialect. We need a way to represent
parameterized types instead of just concrete types.

Traits: `HasParent<mlir::ModuleOp>`, `InnerSymbolTable`

Interfaces: `HWModuleLike`, `HWMutableModuleLike`, `InstanceGraphModuleOpInterface`, `OpAsmOpInterface`, `PortList`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `module_type` | ::mlir::TypeAttr | type attribute of a module |
| `per_port_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `port_locs` | ::mlir::ArrayAttr | location array attribute |
| `parameters` | ::mlir::ArrayAttr | parameter array attribute |
| `verilogName` | ::mlir::StringAttr | string attribute |

### `hw.module.generated` (::circt::hw::HWModuleGeneratedOp)

*HW Generated Module*

The “hw.module.generated” operation represents a reference to an external
module that will be produced by some external process.
This represents the name and list of ports to be generated.

The ‘verilogName’ attribute (when present) specifies the spelling of the
module name in Verilog we can use. See hw.module for an explanation.

Traits: `HasParent<mlir::ModuleOp>`, `InnerSymbolTable`, `IsolatedFromAbove`

Interfaces: `HWModuleLike`, `HWMutableModuleLike`, `InstanceGraphModuleOpInterface`, `OpAsmOpInterface`, `PortList`, `SymbolUserOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `generatorKind` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `module_type` | ::mlir::TypeAttr | type attribute of a module |
| `per_port_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `port_locs` | ::mlir::ArrayAttr | location array attribute |
| `parameters` | ::mlir::ArrayAttr | parameter array attribute |
| `verilogName` | ::mlir::StringAttr | string attribute |

### `hw.module` (::circt::hw::HWModuleOp)

*HW Module*

The “hw.module” operation represents a Verilog module, including a given
name, a list of ports, a list of parameters, and a body that represents the
connections within the module.

Traits: `HasParent<mlir::ModuleOp>`, `InnerSymbolTable`, `IsolatedFromAbove`, `SingleBlockImplicitTerminator<OutputOp>`, `SingleBlock`

Interfaces: `Emittable`, `HWEmittableModuleLike`, `HWModuleLike`, `HWMutableModuleLike`, `InstanceGraphModuleOpInterface`, `OpAsmOpInterface`, `PortList`, `RegionKindInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `module_type` | ::mlir::TypeAttr | type attribute of a module |
| `per_port_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `result_locs` | ::mlir::ArrayAttr | location array attribute |
| `parameters` | ::mlir::ArrayAttr | parameter array attribute |
| `comment` | ::mlir::StringAttr | string attribute |

### `hw.hierpath` (::circt::hw::HierPathOp)

*Hierarchical path specification*

The “hw.hierpath” operation represents a path through the hierarchy.
This is used to specify namable things for use in other operations, for
example in verbatim substitution. Non-local annotations also use these.

Traits: `IsolatedFromAbove`

Interfaces: `InnerRefUserOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `namepath` | ::mlir::ArrayAttr | name reference array attribute |

### `hw.instance_choice` (::circt::hw::InstanceChoiceOp)

*Represents an instance with a target-specific reference*

This represents an instance to a module which is determined based on the
target through the ABI. Besides a default implementation, other targets can
be associated with a string, which will later determined which reference
is chosen.

For the purposes of analyses and transformations, it is assumed that any of
the targets is a possibility.

Example:

```
%b = hw.instance_choice "inst" sym
    @TargetDefault or
    @TargetA if "A" or
    @TargetB if "B"
    (a: %a: i32) -> (b: i32)
```

Interfaces: `HWInstanceLike`, `InnerSymbolOpInterface`, `InstanceGraphInstanceOpInterface`, `InstanceOpInterface`, `OpAsmOpInterface`, `PortList`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `instanceName` | ::mlir::StringAttr | string attribute |
| `moduleNames` | ::mlir::ArrayAttr | flat symbol ref array attribute |
| `optionName` | ::mlir::StringAttr | string attribute |
| `caseNames` | ::mlir::ArrayAttr | string array attribute |
| `argNames` | ::mlir::ArrayAttr | string array attribute |
| `resultNames` | ::mlir::ArrayAttr | string array attribute |
| `parameters` | ::mlir::ArrayAttr | parameter array attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `doNotPrint` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `hw.instance` (::circt::hw::InstanceOp)

*Create an instance of a module*

This represents an instance of a module. The inputs and outputs are
the referenced module’s inputs and outputs. The `argNames` and
`resultNames` attributes must match the referenced module.

Any parameters in the “old” format (slated to be removed) are stored in the
`oldParameters` dictionary.

Interfaces: `HWInstanceLike`, `InnerSymbolOpInterface`, `InstanceGraphInstanceOpInterface`, `OpAsmOpInterface`, `PortList`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `instanceName` | ::mlir::StringAttr | string attribute |
| `moduleName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `argNames` | ::mlir::ArrayAttr | string array attribute |
| `resultNames` | ::mlir::ArrayAttr | string array attribute |
| `parameters` | ::mlir::ArrayAttr | parameter array attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `doNotPrint` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `hw.output` (::circt::hw::OutputOp)

*HW termination operation*

Syntax:

```
operation ::= `hw.output` attr-dict ($outputs^ `:` qualified(type($outputs)))?
```

“hw.output” marks the end of a region in the HW dialect and the values
to put on the output ports.

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<HWModuleOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `hw.triggered` (::circt::hw::TriggeredOp)

*A procedural region with a trigger condition*

Syntax:

```
operation ::= `hw.triggered` $event $trigger  (`(` $inputs^ `)` `:` type($inputs))? $body attr-dict
```

A procedural region that can be triggered by an event. The trigger
condition is a 1-bit value that is activated based on some event control
attribute.
The operation is `IsolatedFromAbove`, and thus requires values passed into
the trigger region to be explicitly passed in through the `inputs` list.

Traits: `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `event` | circt::hw::EventControlAttr | edge control trigger |

#### Operands:

| Operand | Description |
| --- | --- |
| `trigger` | 1-bit signless integer |
| `inputs` | variadic of any type |

Operation Definitions – Miscellaneous
-------------------------------------

### `hw.bitcast` (::circt::hw::BitcastOp)

*Reinterpret one value to another value of the same size and
potentially different type. See the `hw` dialect rationale document for
more details.*

Syntax:

```
operation ::= `hw.bitcast` $input attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | Type wherein the bitwidth in hardware is known |

#### Results:

| Result | Description |
| --- | --- |
| `result` | Type wherein the bitwidth in hardware is known |

### `hw.constant` (::circt::hw::ConstantOp)

*Produce a constant value*

The constant operation produces a constant value of standard integer type
without a sign.

```
  %result = hw.constant 42 : t1
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`, `FirstAttrDerivedResultType`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::IntegerAttr | arbitrary integer attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `hw.enum.cmp` (::circt::hw::EnumCmpOp)

*Compare two values of an enumeration*

Syntax:

```
operation ::= `hw.enum.cmp` $lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs))
```

This operation compares two values with the same canonical enumeration
type, returning 0 if they are different, and 1 if they are the same.

Example:

```
  %enumcmp = hw.enum.cmp %A, %B : !hw.enum<A, B, C>, !hw.enum<A, B, C>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | a EnumType |
| `rhs` | a EnumType |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit signless integer |

### `hw.enum.constant` (::circt::hw::EnumConstantOp)

*Produce a constant enumeration value.*

The enum.constant operation produces an enumeration value of the specified
enum value attribute.

```
  %0 = hw.enum.constant A : !hw.enum<A, B, C>
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `field` | ::circt::hw::EnumFieldAttr | Enumeration field attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a EnumType |

### `hw.param.value` (::circt::hw::ParamValueOp)

*Return the value of a parameter expression as an SSA value that may be used
by other ops.*

Syntax:

```
operation ::= `hw.param.value` custom<ParamValue>($value, qualified(type($result))) attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`, `FirstAttrDerivedResultType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::Attribute | any attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a known primitive element |

### `hw.wire` (::circt::hw::WireOp)

*Assign a name or symbol to an SSA edge*

Syntax:

```
operation ::= `hw.wire` $input (`sym` $inner_sym^)? custom<ImplicitSSAName>($name) attr-dict
              `:` qualified(type($input))
```

An `hw.wire` is used to assign a human-readable name or a symbol for remote
references to an SSA edge. It takes a single operand and returns its value
unchanged as a result. The operation guarantees the following:

* If the wire has a symbol, the value of its operand remains observable
  under that symbol within the IR.
* If the wire has a name, the name is treated as a hint. If the wire
  persists until code generation the resulting wire will have this name,
  with a potential suffix to ensure uniqueness. If the wire is canonicalized
  away, its name is propagated to its input operand as a name hint.
* The users of its result will always observe the operand through the
  operation itself, meaning that optimizations cannot bypass the wire. This
  ensures that if the wire’s value is *forced*, for example through a
  Verilog force statement, the forced value will affect all users of the
  wire in the output.

Example:

```
%1 = hw.wire %0 : i42
%2 = hw.wire %0 sym @mySym : i42
%3 = hw.wire %0 name "myWire" : i42
%myWire = hw.wire %0 : i42
```

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`, `InnerSymbolOpInterface`, `OpAsmOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

Operation Definitions – Aggregates
----------------------------------

### `hw.aggregate_constant` (::circt::hw::AggregateConstantOp)

*Produce a constant aggregate value*

Syntax:

```
operation ::= `hw.aggregate_constant` $fields attr-dict `:` type($result)
```

This operation produces a constant value of an aggregate type. Clock and
reset values are supported. For nested aggregates, embedded arrays are
used.

Examples:

```
  %result = hw.aggregate_constant [1 : i1, 2 : i2, 3 : i2] : !hw.struct<a: i8, b: i8, c: i8>
  %result = hw.aggregate_constant [1 : i1, [2 : i2, 3 : i2]] : !hw.struct<a: i8, b: vector<i8, 2>>
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fields` | ::mlir::ArrayAttr | array attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | an ArrayType or StructType |

### `hw.array_concat` (::circt::hw::ArrayConcatOp)

*Concatenate some arrays*

Syntax:

```
operation ::= `hw.array_concat` $inputs attr-dict `:` custom<ArrayConcatTypes>(type($inputs), qualified(type($result)))
```

Creates an array by concatenating a variable set of arrays. One or more
values must be listed.

```
// %a, %b, %c are hw arrays of i4 with sizes 2, 5, and 4 respectively.
%array = hw.array_concat %a, %b, %c : (2, 5, 4 x i4)
// %array is !hw.array<11 x i4>
```

See the HW-SV rationale document for details on operand ordering.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of an ArrayType |

#### Results:

| Result | Description |
| --- | --- |
| `result` | an ArrayType |

### `hw.array_create` (::circt::hw::ArrayCreateOp)

*Create an array from values*

Creates an array from a variable set of values. One or more values must be
listed.

```
// %a, %b, %c are all i4
%array = hw.array_create %a, %b, %c : i4
```

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
| `result` | an ArrayType |

### `hw.array_get` (::circt::hw::ArrayGetOp)

*Extract an element from an array*

Syntax:

```
operation ::= `hw.array_get` $input `[` $index `]`
              attr-dict `:` type($input) `,` type($index)
```

Extracts the element at `index` from the given `input` array. The index must
be exactly `ceil(log2(length(input)))` bits wide.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an ArrayType |
| `index` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a type without inout |

### `hw.array_inject` (::circt::hw::ArrayInjectOp)

*Inject an element into an array*

Syntax:

```
operation ::= `hw.array_inject` $input `[` $index `]` `,` $element
              attr-dict `:` type($input) `,` type($index)
```

Takes an `input` array, changes the element at `index` to the given
`element` value, and returns the updated array value as a result. The index
must be exactly `ceil(log2(length(input)))` bits wide. The element type
must match the input array’s element type.

If the `index` is out of bounds, the result is undefined.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an ArrayType |
| `index` | a signless integer bitvector |
| `element` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | an ArrayType |

### `hw.array_slice` (::circt::hw::ArraySliceOp)

*Get a range of values from an array*

Syntax:

```
operation ::= `hw.array_slice` $input`[`$lowIndex`]` attr-dict `:`
              `(` custom<SliceTypes>(type($input), qualified(type($lowIndex))) `)` `->` qualified(type($dst))
```

Extracts a sub-range from an array. The range is from `lowIndex` to
`lowIndex` + the number of elements in the return type, non-inclusive on
the high end. For instance,

```
// Slices 16 elements starting at '%offset'.
%subArray = hw.slice %largerArray at %offset :
    (!hw.array<1024xi8>) -> !hw.array<16xi8>
```

Width of ‘idx’ is defined to be the precise number of bits required to
index the ‘input’ array. More precisely: for an input array of size M,
the width of ‘idx’ is ceil(log2(M)). Lower and upper bound indexes which
are larger than the size of the ‘input’ array results in undefined
behavior.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an ArrayType |
| `lowIndex` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `dst` | an ArrayType |

### `hw.struct_create` (::circt::hw::StructCreateOp)

*Create a struct from constituent parts.*

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | variadic of a type without inout |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a StructType |

### `hw.struct_explode` (::circt::hw::StructExplodeOp)

*Expand a struct into its constituent parts.*

```
%result:2 = hw.struct_explode %input : !hw.struct<foo: i19, bar: i7>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a StructType |

#### Results:

| Result | Description |
| --- | --- |
| `result` | variadic of a type without inout |

### `hw.struct_extract` (::circt::hw::StructExtractOp)

*Extract a named field from a struct.*

```
%result = hw.struct_extract %input["field"] : !hw.struct<field: type>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a StructType |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a type without inout |

### `hw.struct_inject` (::circt::hw::StructInjectOp)

*Inject a value into a named field of a struct.*

```
%result = hw.struct_inject %input["field"], %newValue
    : !hw.struct<field: type>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a StructType |
| `newValue` | a type without inout |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a StructType |

### `hw.union_create` (::circt::hw::UnionCreateOp)

*Create a union with the specified value.*

Create a union with the value ‘input’, which can then be accessed via the
specified field.

```
  %x = hw.constant 0 : i3
  %z = hw.union_create "bar", %x : !hw.union<bar: i3, baz: i8>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a type without inout |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a UnionType |

### `hw.union_extract` (::circt::hw::UnionExtractOp)

*Get a union member.*

Get the value of a union, interpreting it as the type of the specified
member field. Extracting a value belonging to a different field than the
union was initially created will result in undefined behavior.

```
  %u = ...
  %v = hw.union_extract %u["foo"] : !hw.union<foo: i3, bar: i16>
  // %v is of type 'i3'
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a UnionType |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a type without inout |

Operation Definitions – Type Declarations
-----------------------------------------

### `hw.type_scope` (::circt::hw::TypeScopeOp)

*Type declaration wrapper.*

Syntax:

```
operation ::= `hw.type_scope` $sym_name $body attr-dict
```

An operation whose one body block contains type declarations. This op
provides a scope for type declarations at the top level of an MLIR module.
It is a symbol that may be looked up within the module, as well as a symbol
table itself, so type declarations may be looked up.

Traits: `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `Emittable`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `hw.typedecl` (::circt::hw::TypedeclOp)

*Type declaration.*

Syntax:

```
operation ::= `hw.typedecl` $sym_name (`,` $verilogName^)? `:` $type attr-dict
```

Associate a symbolic name with a type.

Traits: `HasParent<TypeScopeOp>`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `type` | ::mlir::TypeAttr | any type attribute |
| `verilogName` | ::mlir::StringAttr | string attribute |

Attribute Definitions
---------------------

### EnumFieldAttr

*Enumeration field attribute*

This attribute represents a field of an enumeration.

Examples:

```
  #hw.enum.value<A, !hw.enum<A, B, C>>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| field | `::mlir::StringAttr` |  |
| type | `::mlir::TypeAttr` |  |

### OutputFileAttr

*Output file attribute*

This attribute represents an output file for something which will be
printed. The `filename` string is the file to be output to. If `filename`
ends in a `/` it is considered an output directory.

When ExportVerilog runs, one of the files produced is a list of all other
files which are produced. The flag `excludeFromFileList` controls if this
file should be included in this list. If any `OutputFileAttr` referring to
the same file sets this to `true`, it will be included in the file list.
This option defaults to `false`.

For each file emitted by the verilog emitter, certain prelude output will
be included before the main content. The flag `includeReplicatedOps` can
be used to disable the addition of the prelude text. All `OutputFileAttr`s
referring to the same file must use a consistent setting for this value.
This option defaults to `true`.

Examples:

```
  #hw.ouput_file<"/home/tester/t.sv">
  #hw.ouput_file<"t.sv", excludeFromFileList, includeReplicatedOps>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| filename | `::mlir::StringAttr` |  |
| excludeFromFilelist | `::mlir::BoolAttr` |  |
| includeReplicatedOps | `::mlir::BoolAttr` |  |

### ParamDeclAttr

*Module or instance parameter definition*

An attribute describing a module parameter, or instance parameter
specification.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| name | `::mlir::StringAttr` |  |
| type | `::mlir::Type` |  |
| value | `::mlir::Attribute` |  |

### ParamDeclRefAttr

*Is a reference to a parameter value.*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| name | `::mlir::StringAttr` |  |
| type | `::mlir::Type` |  |

### ParamExprAttr

*Parameter expression combining operands*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| opcode | `PEO` |  |
| operands | `::llvm::ArrayRef<::mlir::TypedAttr>` |  |
| type | `::mlir::Type` |  |

### ParamVerbatimAttr

*Represents text to emit directly to SystemVerilog for a parameter*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `::mlir::StringAttr` |  |
| type | `::mlir::Type` |  |

### InnerRefAttr

*Refer to a name inside a module*

This works like a symbol reference, but to a name inside a module.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| moduleRef | `::mlir::FlatSymbolRefAttr` |  |
| name | `::mlir::StringAttr` |  |

Type Definitions
----------------

### `hw.type_scope` (::circt::hw::TypeScopeOp)

*Type declaration wrapper.*

Syntax:

```
operation ::= `hw.type_scope` $sym_name $body attr-dict
```

An operation whose one body block contains type declarations. This op
provides a scope for type declarations at the top level of an MLIR module.
It is a symbol that may be looked up within the module, as well as a symbol
table itself, so type declarations may be looked up.

Traits: `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `Emittable`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `hw.typedecl` (::circt::hw::TypedeclOp)

*Type declaration.*

Syntax:

```
operation ::= `hw.typedecl` $sym_name (`,` $verilogName^)? `:` $type attr-dict
```

Associate a symbolic name with a type.

Traits: `HasParent<TypeScopeOp>`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `type` | ::mlir::TypeAttr | any type attribute |
| `verilogName` | ::mlir::StringAttr | string attribute |

### ArrayType

*Fixed-sized array*

Syntax:

```
!hw.array<
  ::mlir::Type,   # elementType
  ::mlir::Attribute   # sizeAttr
>
```

Fixed sized HW arrays are roughly similar to C arrays. On the wire (vs.
in a memory), arrays are always packed. Memory layout is not defined as
it does not need to be since in silicon there is not implicit memory
sharing.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `::mlir::Type` |  |
| sizeAttr | `::mlir::Attribute` |  |

### EnumType

*HW Enum type*

Represents an enumeration of values. Enums are interpreted as integers with
a synthesis-defined encoding.

```
!hw.enum<field1, field2>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| fields | `mlir::ArrayAttr` |  |

### StringType

*String type*

Syntax: `!hw.string`

Defines a string type for the hw-centric dialects

### InOutType

*Inout type*

Syntax:

```
!hw.inout<
  ::mlir::Type   # elementType
>
```

InOut type is used for model operations and values that have “connection”
semantics, instead of typical dataflow behavior. This is used for wires
and inout ports in Verilog.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `::mlir::Type` |  |

### IntType

*Parameterized-width integer*

Parameterized integer types are equivalent to the MLIR standard integer
type: it is signless, and may be any width integer. This type represents
the case when the width is a parameter in the HW dialect sense.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| width | `::mlir::TypedAttr` |  |

### ModuleType

*Module Type*

Module types have ports.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| ports | `::llvm::ArrayRef<::circt::hw::ModulePort>` | port list |

### StructType

*HW struct type*

Represents a structure of name, value pairs.

```
!hw.struct<fieldName1: Type1, fieldName2: Type2>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elements | `::llvm::ArrayRef<::circt::hw::StructType::FieldInfo>` | struct fields |

### TypeAliasType

*An symbolic reference to a type declaration*

A TypeAlias is parameterized by a SymbolRefAttr, which points to a
TypedeclOp. The root reference should refer to a TypeScope within the same
outer ModuleOp, and the leaf reference should refer to a type within that
TypeScope. A TypeAlias is further parameterized by the inner type, which is
needed to be known at the time the type is parsed.

Upon construction, a TypeAlias stores the symbol reference and type, and
canonicalizes the type to resolve any nested type aliases. The canonical
type is also cached to avoid recomputing it when needed.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| ref | `mlir::SymbolRefAttr` |  |
| innerType | `mlir::Type` |  |
| canonicalType | `mlir::Type` |  |

### UnionType

*An untagged union of types*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elements | `::llvm::ArrayRef<::circt::hw::UnionType::FieldInfo>` | union fields |

### UnpackedArrayType

*SystemVerilog ‘unpacked’ fixed-sized array*

Syntax:

```
!hw.uarray<
  ::mlir::Type,   # elementType
  ::mlir::Attribute   # sizeAttr
>
```

Unpacked arrays are a more flexible array representation than packed arrays,
and are typically used to model memories. See SystemVerilog Spec 7.4.2.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `::mlir::Type` |  |
| sizeAttr | `::mlir::Attribute` |  |

CombDataFlow (`CombDataflow`)
-----------------------------

This interface is used for specifying the combinational dataflow that exists
in the results and operands of an operation.
Any operation that doesn’t implement this interface is assumed to have a
combinational dependence from each operand to each result.

### Methods:

#### `computeDataFlow`

```
llvm::SmallVector<std::pair<circt::FieldRef, circt::FieldRef>> computeDataFlow();
```

Get the combinational dataflow relations between the operands and the results.
This returns a pair of ground type fieldrefs. The first element is the destination
and the second is the source of the dependence. The default implementation returns
an empty list, which implies that the operation is not combinational.

NOTE: This method *must* be implemented by the user.

HWEmittableModuleLike (`HWEmittableModuleLike`)
-----------------------------------------------

This interface indicates that the module like op is emittable in SV and
requires SV legalization on its body.

### Methods:

HWInstanceLike (`HWInstanceLike`)
---------------------------------

Provide common instance information.

### Methods:

#### `getInputName`

```
::mlir::StringAttr getInputName(size_t idx);
```

Return the name of the specified input port or null if it cannot be
determined.

NOTE: This method *must* be implemented by the user.

#### `getOutputName`

```
::mlir::StringAttr getOutputName(size_t idx);
```

Return the name of the specified result or null if it cannot be
determined.

NOTE: This method *must* be implemented by the user.

#### `setInputName`

```
void setInputName(size_t idx, ::mlir::StringAttr name);
```

Change the name of the specified input port.

NOTE: This method *must* be implemented by the user.

#### `setOutputName`

```
void setOutputName(size_t idx, ::mlir::StringAttr name);
```

Change the name of the specified output port.

NOTE: This method *must* be implemented by the user.

#### `getInputNames`

```
::mlir::ArrayAttr getInputNames();
```

Return the names of all input ports. If the instance operation stores the
names in an ArrayAttr this can avoid attribute constructions.

NOTE: This method *must* be implemented by the user.

#### `getOutputNames`

```
::mlir::ArrayAttr getOutputNames();
```

Return the name of all ouput ports. If the instance operation stores the
names in an ArrayAttr this can avoid attribute constructions.

NOTE: This method *must* be implemented by the user.

#### `setInputNames`

```
void setInputNames(::mlir::ArrayAttr names);
```

Change the names of all input ports. If all names have to be changed, this
can avoid repeated intermediate attribute constructions.

NOTE: This method *must* be implemented by the user.

#### `setOutputNames`

```
void setOutputNames(::mlir::ArrayAttr names);
```

Change the names of all output ports. If all names have to be changed, this
can avoid repeated intermediate attribute constructions.

NOTE: This method *must* be implemented by the user.

#### `getDoNotPrint`

```
bool getDoNotPrint();
```

True if this instance is a phony placeholder

NOTE: This method *must* be implemented by the user.

HWModuleLike (`HWModuleLike`)
-----------------------------

Provide common module information.

### Methods:

#### `getHWModuleType`

```
::circt::hw::ModuleType getHWModuleType();
```

Get the module type

NOTE: This method *must* be implemented by the user.

#### `getAllPortAttrs`

```
ArrayRef<Attribute> getAllPortAttrs();
```

Get the port Attributes. This will return either an empty array or an array of size numPorts.

NOTE: This method *must* be implemented by the user.

#### `setAllPortAttrs`

```
void setAllPortAttrs(ArrayRef<Attribute> attrs);
```

Set the port Attributes

NOTE: This method *must* be implemented by the user.

#### `removeAllPortAttrs`

```
void removeAllPortAttrs();
```

Remove the port Attributes

NOTE: This method *must* be implemented by the user.

#### `getAllPortLocs`

```
SmallVector<Location> getAllPortLocs();
```

Get the port Locations

NOTE: This method *must* be implemented by the user.

#### `setAllPortLocsAttrs`

```
void setAllPortLocsAttrs(ArrayRef<Attribute> locs);
```

Set the port Locations

NOTE: This method *must* be implemented by the user.

#### `setHWModuleType`

```
void setHWModuleType(::circt::hw::ModuleType type);
```

Set the module type (and port names)

NOTE: This method *must* be implemented by the user.

#### `setAllPortNames`

```
void setAllPortNames(ArrayRef<Attribute> names);
```

Set the port names

NOTE: This method *must* be implemented by the user.

HWMutableModuleLike (`HWMutableModuleLike`)
-------------------------------------------

Provide methods to mutate a module.

### Methods:

#### `getPortLookupInfo`

```
::circt::hw::ModulePortLookupInfo getPortLookupInfo();
```

Get a handle to a utility class which provides by-name lookup of port indices. The returned object does *not* update if the module is mutated.

NOTE: This method *must* be implemented by the user.

#### `modifyPorts`

```
void modifyPorts(ArrayRef<std::pair<unsigned, circt::hw::PortInfo>> insertInputs, ArrayRef<std::pair<unsigned, circt::hw::PortInfo>> insertOutputs, ArrayRef<unsigned> eraseInputs, ArrayRef<unsigned> eraseOutputs);
```

Insert and remove input and output ports

#### `insertPorts`

```
void insertPorts(ArrayRef<std::pair<unsigned, circt::hw::PortInfo>> insertInputs, ArrayRef<std::pair<unsigned, circt::hw::PortInfo>> insertOutputs);
```

Insert ports into this module

NOTE: This method *must* be implemented by the user.

#### `erasePorts`

```
void erasePorts(ArrayRef<unsigned> eraseInputs, ArrayRef<unsigned> eraseOutputs);
```

Erase ports from this module

NOTE: This method *must* be implemented by the user.

#### `appendOutputs`

```
void appendOutputs(ArrayRef<std::pair<StringAttr, Value>> outputs);
```

Append output values to this module

NOTE: This method *must* be implemented by the user.

InnerRefUserOpInterface (`InnerRefUserOpInterface`)
---------------------------------------------------

This interface describes an operation that may use a `InnerRef`. This
interface allows for users of inner symbols to hook into verification and
other inner symbol related utilities that are either costly or otherwise
disallowed within a traditional operation.

### Methods:

#### `verifyInnerRefs`

```
::mlir::LogicalResult verifyInnerRefs(::circt::hw::InnerRefNamespace&ns);
```

Verify the inner ref uses held by this operation.

NOTE: This method *must* be implemented by the user.

InnerSymbolOpInterface (`InnerSymbol`)
--------------------------------------

This interface describes an operation that may define an
`inner_sym`. An `inner_sym` operation resides
in arbitrarily-nested regions of a region that defines a
`InnerSymbolTable`.
Inner Symbols are different from normal symbols due to
MLIR symbol table resolution rules. Specifically normal
symbols are resolved by first going up to the closest
parent symbol table and resolving from there (recursing
down for complex symbol paths). In HW and SV, modules
define a symbol in a circuit or std.module symbol table.
For instances to be able to resolve the modules they
instantiate, the symbol use in an instance must resolve
in the top-level symbol table. If a module were a
symbol table, instances resolving a symbol would start from
their own module, never seeing other modules (since
resolution would start in the parent module of the
instance and be unable to go to the global scope).
The second problem arises from nesting. Symbols defining
ops must be immediate children of a symbol table. HW
and SV operations which define a inner\_sym are grandchildren,
at least, of a symbol table and may be much further nested.
Lastly, ports need to define inner\_sym, something not allowed
by normal symbols.

Any operation implementing an InnerSymbol may have the inner symbol be
optional and all methods should be robuse to the attribute not being
defined.

### Methods:

#### `getInnerNameAttr`

```
::mlir::StringAttr getInnerNameAttr();
```

Returns the name of the top-level inner symbol defined by this operation, if present.

NOTE: This method *must* be implemented by the user.

#### `getInnerName`

```
::std::optional<::mlir::StringRef> getInnerName();
```

Returns the name of the top-level inner symbol defined by this operation, if present.

NOTE: This method *must* be implemented by the user.

#### `setInnerSymbol`

```
void setInnerSymbol(::mlir::StringAttr name);
```

Sets the name of the top-level inner symbol defined by this operation to the specified string, dropping any symbols on fields.

NOTE: This method *must* be implemented by the user.

#### `setInnerSymbolAttr`

```
void setInnerSymbolAttr(::circt::hw::InnerSymAttr sym);
```

Sets the inner symbols defined by this operation.

NOTE: This method *must* be implemented by the user.

#### `getInnerRef`

```
::circt::hw::InnerRefAttr getInnerRef();
```

Returns an InnerRef to this operation’s top-level inner symbol, which must be present.

NOTE: This method *must* be implemented by the user.

#### `getInnerSymAttr`

```
::circt::hw::InnerSymAttr getInnerSymAttr();
```

Returns the InnerSymAttr representing all inner symbols defined by this operation.

NOTE: This method *must* be implemented by the user.

#### `supportsPerFieldSymbols`

```
static bool supportsPerFieldSymbols();
```

Returns whether per-field symbols are supported for this operation type.

NOTE: This method *must* be implemented by the user.

#### `getTargetResultIndex`

```
static std::optional<size_t> getTargetResultIndex();
```

Returns the index of the result the innner symbol targets, if applicable. Per-field symbols are resolved into this.

NOTE: This method *must* be implemented by the user.

#### `getTargetResult`

```
OpResult getTargetResult();
```

Returns the result the innner symbol targets, if applicable. Per-field symbols are resolved into this.

NOTE: This method *must* be implemented by the user.

PortList (`PortList`)
---------------------

Operations which produce a unified port list representation

### Methods:

#### `getPortList`

```
SmallVector<::circt::hw::PortInfo> getPortList();
```

Get port list

NOTE: This method *must* be implemented by the user.

#### `getPort`

```
::circt::hw::PortInfo getPort(size_t idx);
```

Get port list

NOTE: This method *must* be implemented by the user.

#### `getPortIdForInputId`

```
size_t getPortIdForInputId(size_t idx);
```

Get the port a specific input

NOTE: This method *must* be implemented by the user.

#### `getPortIdForOutputId`

```
size_t getPortIdForOutputId(size_t idx);
```

Get the port a specific output

NOTE: This method *must* be implemented by the user.

#### `getNumPorts`

```
size_t getNumPorts();
```

Get the number of ports

NOTE: This method *must* be implemented by the user.

#### `getNumInputPorts`

```
size_t getNumInputPorts();
```

Get the number of input ports

NOTE: This method *must* be implemented by the user.

#### `getNumOutputPorts`

```
size_t getNumOutputPorts();
```

Get the number of output ports

NOTE: This method *must* be implemented by the user.

BitWidthTypeInterface (`BitWidthTypeInterface`)
-----------------------------------------------

Type interface for types that have a statically computable bit width.
This allows types from different dialects to participate in bit width
calculations, enabling operations like hw.bitcast to work with non-HW types.

### Methods:

#### `getBitWidth`

```
std::optional<int64_t> getBitWidth();
```

Return the hardware bit width of the type. Does not reflect any encoding,
padding, or storage scheme, just the bit (and wire width) of a
statically-sized type. Reflects the number of wires needed to transmit a
value of this type. Returns std::nullopt if the type’s bit width cannot
be statically computed.

NOTE: This method *must* be implemented by the user.

FieldIDTypeInterface (`FieldIDTypeInterface`)
---------------------------------------------

Common methods for types which can be indexed by a FieldID.
FieldID is a depth-first numbering of the elements of a type. For example:

```
struct a  /* 0 */ {
  int b; /* 1 */
  struct c /* 2 */ {
    int d; /* 3 */
  }
}

int e; /* 0 */
```

### Methods:

#### `getMaxFieldID`

```
uint64_t getMaxFieldID();
```

Get the maximum field ID for this type

NOTE: This method *must* be implemented by the user.

#### `getSubTypeByFieldID`

```
std::pair<::mlir::Type, uint64_t> getSubTypeByFieldID(uint64_t fieldID);
```

Get the sub-type of a type for a field ID, and the subfield’s ID. Strip
off a single layer of this type and return the sub-type and a field ID
targeting the same field, but rebased on the sub-type.

The resultant type *may* not be a FieldIDTypeInterface if the resulting
fieldID is zero. This means that leaf types may be ground without
implementing an interface. An empty aggregate will also appear as a
zero.

NOTE: This method *must* be implemented by the user.

#### `projectToChildFieldID`

```
std::pair<uint64_t, bool> projectToChildFieldID(uint64_t fieldID, uint64_t index);
```

Returns the effective field id when treating the index field as the
root of the type. Essentially maps a fieldID to a fieldID after a
subfield op. Returns the new id and whether the id is in the given
child.

NOTE: This method *must* be implemented by the user.

#### `getIndexForFieldID`

```
uint64_t getIndexForFieldID(uint64_t fieldID);
```

Returns the index (e.g. struct or vector element) for a given FieldID.
This returns the containing index in the case that the fieldID points to a
child field of a field.

NOTE: This method *must* be implemented by the user.

#### `getFieldID`

```
uint64_t getFieldID(uint64_t index);
```

Return the fieldID of a given index (e.g. struct or vector element).
Field IDs start at 1, and are assigned
to each field in a recursive depth-first walk of all
elements. A field ID of 0 is used to reference the type itself.

NOTE: This method *must* be implemented by the user.

#### `getIndexAndSubfieldID`

```
std::pair<uint64_t, uint64_t> getIndexAndSubfieldID(uint64_t fieldID);
```

Find the index of the element that contains the given fieldID.
As well, rebase the fieldID to the element.

NOTE: This method *must* be implemented by the user.

'hw' Dialect Docs
-----------------

* [HW Dialect Rationale](https://circt.llvm.org/docs/Dialects/HW/RationaleHW/)

 [Prev - Handshake Dialect Rationale](https://circt.llvm.org/docs/Dialects/Handshake/RationaleHandshake/ "Handshake Dialect Rationale")
[Next - HW Dialect Rationale](https://circt.llvm.org/docs/Dialects/HW/RationaleHW/ "HW Dialect Rationale") 

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