'om' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'om' Dialect
============

*The Object Model dialect captures design intent from various domains in an
object model.*

For more information about the Object Model dialect, see the
[Object Model Dialect Rationale](/docs/Dialects/OM/RationaleOM/).

* [Operations](#operations)
  + [`om.any_cast` (::circt::om::AnyCastOp)](#omany_cast-circtomanycastop)
  + [`om.basepath_create` (::circt::om::BasePathCreateOp)](#ombasepath_create-circtombasepathcreateop)
  + [`om.class` (::circt::om::ClassOp)](#omclass-circtomclassop)
  + [`om.class.extern` (::circt::om::ClassExternOp)](#omclassextern-circtomclassexternop)
  + [`om.class.fields` (::circt::om::ClassFieldsOp)](#omclassfields-circtomclassfieldsop)
  + [`om.constant` (::circt::om::ConstantOp)](#omconstant-circtomconstantop)
  + [`om.frozenbasepath_create` (::circt::om::FrozenBasePathCreateOp)](#omfrozenbasepath_create-circtomfrozenbasepathcreateop)
  + [`om.frozenpath_create` (::circt::om::FrozenPathCreateOp)](#omfrozenpath_create-circtomfrozenpathcreateop)
  + [`om.frozenpath_empty` (::circt::om::FrozenEmptyPathOp)](#omfrozenpath_empty-circtomfrozenemptypathop)
  + [`om.integer.add` (::circt::om::IntegerAddOp)](#omintegeradd-circtomintegeraddop)
  + [`om.integer.mul` (::circt::om::IntegerMulOp)](#omintegermul-circtomintegermulop)
  + [`om.integer.shl` (::circt::om::IntegerShlOp)](#omintegershl-circtomintegershlop)
  + [`om.integer.shr` (::circt::om::IntegerShrOp)](#omintegershr-circtomintegershrop)
  + [`om.list_concat` (::circt::om::ListConcatOp)](#omlist_concat-circtomlistconcatop)
  + [`om.list_create` (::circt::om::ListCreateOp)](#omlist_create-circtomlistcreateop)
  + [`om.object` (::circt::om::ObjectOp)](#omobject-circtomobjectop)
  + [`om.object.field` (::circt::om::ObjectFieldOp)](#omobjectfield-circtomobjectfieldop)
  + [`om.path_create` (::circt::om::PathCreateOp)](#ompath_create-circtompathcreateop)
  + [`om.path_empty` (::circt::om::EmptyPathOp)](#ompath_empty-circtomemptypathop)
  + [`om.unknown` (::circt::om::UnknownValueOp)](#omunknown-circtomunknownvalueop)
* [Attributes](#attributes-10)
  + [IntegerAttr](#integerattr)
  + [ListAttr](#listattr)
  + [PathAttr](#pathattr)
  + [SymbolRefAttr](#symbolrefattr)
  + [ReferenceAttr](#referenceattr)
* [Types](#types)
  + [BasePathType](#basepathtype)
  + [ClassType](#classtype)
  + [FrozenBasePathType](#frozenbasepathtype)
  + [FrozenPathType](#frozenpathtype)
  + [ListType](#listtype)
  + [AnyType](#anytype)
  + [OMIntegerType](#omintegertype)
  + [PathType](#pathtype)
  + [ReferenceType](#referencetype)
  + [StringType](#stringtype)
  + [SymbolRefType](#symbolreftype)
* [Enums](#enums)
  + [TargetKind](#targetkind)

Operations
----------

### `om.any_cast` (::circt::om::AnyCastOp)

*Cast any value to any type.*

Syntax:

```
operation ::= `om.any_cast` $input attr-dict `:` functional-type($input, $result)
```

Casts any value to AnyType. This is useful for situations where a value of
AnyType is needed, but a value of some concrete type is known.

In the evaluator, this is a noop, and the value of concrete type is used.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A type that represents any valid OM type. |

### `om.basepath_create` (::circt::om::BasePathCreateOp)

*Produce a base path value*

Syntax:

```
operation ::= `om.basepath_create` $basePath $target attr-dict
```

Produces a value which represents a fragment of a hierarchical path to a
target. Given a base path, extend it with the name of a module instance, to
produce a new base path. The instance is identified via an NLA. Once the
final verilog name of the instance is known, this op can be converted into
a FrozenBasePathOp.

Example:

```
hw.module @Foo() -> () {
  hw.inst "bar" sym @bar @Bar() -> ()
}
hw.hierpath @Path [@Foo::@bar]
om.class @OM(%basepath: !om.basepath) {
  %0 = om.basepath_create %base @Path
}
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `target` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `basePath` | A fragment of a path to a hardware component |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A fragment of a path to a hardware component |

### `om.class` (::circt::om::ClassOp)

Traits: `IsolatedFromAbove`, `SingleBlockImplicitTerminator<ClassFieldsOp>`, `SingleBlock`

Interfaces: `ClassLike`, `OpAsmOpInterface`, `RegionKindInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `formalParamNames` | ::mlir::ArrayAttr | string array attribute |
| `fieldNames` | ::mlir::ArrayAttr | array attribute |
| `fieldTypes` | ::mlir::DictionaryAttr | dictionary of named attribute values |

### `om.class.extern` (::circt::om::ClassExternOp)

Traits: `IsolatedFromAbove`, `NoTerminator`

Interfaces: `ClassLike`, `OpAsmOpInterface`, `RegionKindInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `formalParamNames` | ::mlir::ArrayAttr | string array attribute |
| `fieldNames` | ::mlir::ArrayAttr | array attribute |
| `fieldTypes` | ::mlir::DictionaryAttr | dictionary of named attribute values |

### `om.class.fields` (::circt::om::ClassFieldsOp)

Syntax:

```
operation ::= `om.class.fields` attr-dict ($fields^ `:` qualified(type($fields)))?
              custom<FieldLocs>($field_locs)
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<ClassOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `field_locs` | ::mlir::ArrayAttr | location array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `fields` | variadic of any type |

### `om.constant` (::circt::om::ConstantOp)

Syntax:

```
operation ::= `om.constant` $value attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::TypedAttr | TypedAttr instance |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `om.frozenbasepath_create` (::circt::om::FrozenBasePathCreateOp)

*Produce a frozen base path value*

Syntax:

```
operation ::= `om.frozenbasepath_create` $basePath custom<BasePathString>($path) attr-dict
```

Produces a value which represents a fragment of a hierarchical path to a
target.

Example:

```
om.class @OM(%basepath: !om.basepath)
  %0 = om.frozenbasepath_create %basepath "Foo/bar:Bar/baz"
}
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `path` | ::circt::om::PathAttr | An attribute that represents an instance path |

#### Operands:

| Operand | Description |
| --- | --- |
| `basePath` | A frozen fragment of a path to a hardware component |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A frozen fragment of a path to a hardware component |

### `om.frozenpath_create` (::circt::om::FrozenPathCreateOp)

*Produce a frozen path value*

Syntax:

```
operation ::= `om.frozenpath_create` $targetKind $basePath custom<PathString>($path, $module, $ref, $field)
              attr-dict
```

Produces a value which represents a hierarchical path to a hardware
component from a base path to a target.

Example:

```
om.class @OM(%basepath: !om.basepath)
  %0 = om.frozenpath_create reference %base "Foo/bar:Bar>w.a"
}
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `targetKind` | ::circt::om::TargetKindAttr | object model target kind |
| `path` | ::circt::om::PathAttr | An attribute that represents an instance path |
| `module` | ::mlir::StringAttr | string attribute |
| `ref` | ::mlir::StringAttr | string attribute |
| `field` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `basePath` | A frozen fragment of a path to a hardware component |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A frozen path to a hardware component |

### `om.frozenpath_empty` (::circt::om::FrozenEmptyPathOp)

*Produce a frozen path value to nothing*

Syntax:

```
operation ::= `om.frozenpath_empty` attr-dict
```

Produces a value which represents a hierarchical path to nothing.

Example:

```
om.class @OM()
  %0 = om.frozenpath_empty
}
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `result` | A frozen path to a hardware component |

### `om.integer.add` (::circt::om::IntegerAddOp)

*Add two OMIntegerType values*

Syntax:

```
operation ::= `om.integer.add` $lhs `,` $rhs attr-dict `:` type($result)
```

Perform arbitrary precision signed integer addition of two OMIntegerType
values.

Example:

```
%2 = om.integer.add %0, %1 : !om.integer
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `IntegerBinaryArithmeticOp`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | A type that represents an arbitrary width integer. |
| `rhs` | A type that represents an arbitrary width integer. |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A type that represents an arbitrary width integer. |

### `om.integer.mul` (::circt::om::IntegerMulOp)

*Multiply two OMIntegerType values*

Syntax:

```
operation ::= `om.integer.mul` $lhs `,` $rhs attr-dict `:` type($result)
```

Perform arbitrary prevision signed integer multiplication of two
OMIntegerType values.

Example:

```
%2 = om.integer.mul %0, %1 : !om.integer
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `IntegerBinaryArithmeticOp`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | A type that represents an arbitrary width integer. |
| `rhs` | A type that represents an arbitrary width integer. |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A type that represents an arbitrary width integer. |

### `om.integer.shl` (::circt::om::IntegerShlOp)

*Shift an OMIntegerType value left by an OMIntegerType value*

Syntax:

```
operation ::= `om.integer.shl` $lhs `,` $rhs attr-dict `:` type($result)
```

Perform arbitrary precision signed integer arithmetic shift left of the lhs
OMIntegerType value by the rhs OMIntegerType value. The rhs value must be
non-negative.

Example:

```
%2 = om.integer.shl %0, %1 : !om.integer
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `IntegerBinaryArithmeticOp`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | A type that represents an arbitrary width integer. |
| `rhs` | A type that represents an arbitrary width integer. |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A type that represents an arbitrary width integer. |

### `om.integer.shr` (::circt::om::IntegerShrOp)

*Shift an OMIntegerType value right by an OMIntegerType value*

Syntax:

```
operation ::= `om.integer.shr` $lhs `,` $rhs attr-dict `:` type($result)
```

Perform arbitrary precision signed integer arithmetic shift right of the lhs
OMIntegerType value by the rhs OMIntegerType value. The rhs value must be
non-negative.

Example:

```
%2 = om.integer.shr %0, %1 : !om.integer
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `IntegerBinaryArithmeticOp`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | A type that represents an arbitrary width integer. |
| `rhs` | A type that represents an arbitrary width integer. |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A type that represents an arbitrary width integer. |

### `om.list_concat` (::circt::om::ListConcatOp)

*Concatenate multiple lists to produce a new list*

Syntax:

```
operation ::= `om.list_concat` $subLists attr-dict `:` type($result)
```

Produces a value of list type by concatenating the provided lists.

Example:

```
%3 = om.list_concat %0, %1, %2 : !om.list<string>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `subLists` | variadic of A type that represents a list. |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A type that represents a list. |

### `om.list_create` (::circt::om::ListCreateOp)

*Create a list of values*

Creates a list from a sequence of inputs.

```
%list = om.list_create %a, %b, %c : !om.ref
```

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A type that represents a list. |

### `om.object` (::circt::om::ObjectOp)

Syntax:

```
operation ::= `om.object` $className `(` $actualParams `)`  attr-dict `:`
              functional-type($actualParams, $result)
```

Interfaces: `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `className` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `actualParams` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A type that represents a reference to a Class. |

### `om.object.field` (::circt::om::ObjectFieldOp)

Syntax:

```
operation ::= `om.object.field` $object `,` $fieldPath attr-dict `:` functional-type($object, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldPath` | ::mlir::ArrayAttr | flat symbol ref array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `object` | A type that represents a reference to a Class. |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `om.path_create` (::circt::om::PathCreateOp)

*Produce a path value*

Syntax:

```
operation ::= `om.path_create` $targetKind $basePath $target attr-dict
```

Produces a value which represents a hierarchical path to a hardware
target.
from a base path to a target.

Example:

```
hw.module @Foo() -> () {
  %wire = hw.wire sym @w: !i1
}
hw.hierpath @Path [@Foo::@w]
om.class @OM(%basepath: !om.basepath)
  %0 = om.path_create reference %basepath @Path
}
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `targetKind` | ::circt::om::TargetKindAttr | object model target kind |
| `target` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `basePath` | A fragment of a path to a hardware component |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A path to a hardware component |

### `om.path_empty` (::circt::om::EmptyPathOp)

*Produce a path value to nothing*

Syntax:

```
operation ::= `om.path_empty` attr-dict
```

Produces a value which represents a hierarchical path to nothing.

Example:

```
om.class @OM()
  %0 = om.path_empty
}
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `result` | A path to a hardware component |

### `om.unknown` (::circt::om::UnknownValueOp)

*An operation with an unknown value used to tie-off connections*

Syntax:

```
operation ::= `om.unknown` attr-dict `:` type($result)
```

An operation that is used to tie-off connections to objects for which we do
not care about the value being used.

This is used in situations where a class exists and is instantiated, but the
instantiator has no reasonable value that it can or should drive.

Interfaces: `SymbolUserOpInterface`

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

Attributes
----------

### IntegerAttr

*An attribute that represents an arbitrary integer*

Syntax:

```
#om.integer<
  mlir::IntegerAttr   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `mlir::IntegerAttr` |  |

### ListAttr

*An attribute that represents a list*

Syntax:

```
#om.list<
  mlir::Type,   # elementType
  mlir::ArrayAttr   # elements
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `mlir::Type` |  |
| elements | `mlir::ArrayAttr` |  |

### PathAttr

*An attribute that represents an instance path*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| path | `::llvm::ArrayRef<::circt::om::PathElement>` |  |

### SymbolRefAttr

*An attribute that wraps a FlatSymbolRefAttr type*

Syntax:

```
#om.sym_ref<
  mlir::FlatSymbolRefAttr   # ref
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| ref | `mlir::FlatSymbolRefAttr` |  |

### ReferenceAttr

*An attribute that wraps a #hw.innerNameRef with !om.ref type*

Syntax:

```
#om.ref<
  circt::hw::InnerRefAttr   # innerRef
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| innerRef | `circt::hw::InnerRefAttr` |  |

Types
-----

### BasePathType

*A fragment of a path to a hardware component*

Syntax: `!om.basepath`

### ClassType

*A type that represents a reference to a Class.*

Syntax:

```
!om.class.type<
  mlir::FlatSymbolRefAttr   # className
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| className | `mlir::FlatSymbolRefAttr` |  |

### FrozenBasePathType

*A frozen fragment of a path to a hardware component*

Syntax: `!om.frozenbasepath`

### FrozenPathType

*A frozen path to a hardware component*

Syntax: `!om.frozenpath`

### ListType

*A type that represents a list.*

Syntax:

```
!om.list<
  mlir::Type   # elementType
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `mlir::Type` |  |

### AnyType

*A type that represents any valid OM type.*

Syntax: `!om.any`

### OMIntegerType

*A type that represents an arbitrary width integer.*

Syntax: `!om.integer`

### PathType

*A path to a hardware component*

Syntax: `!om.path`

### ReferenceType

*A type that represents a reference to a hardware entity.*

Syntax: `!om.ref`

### StringType

*A type that represents a string.*

Syntax: `!om.string`

### SymbolRefType

*A type that represents a reference to a flat symbol reference.*

Syntax: `!om.sym_ref`

Enums
-----

### TargetKind

*Object model target kind*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| DontTouch | `0` | dont\_touch |
| Instance | `1` | instance |
| MemberInstance | `2` | member\_instance |
| MemberReference | `3` | member\_reference |
| Reference | `4` | reference |

'om' Dialect Docs
-----------------

* [Object Model Dialect Rationale](https://circt.llvm.org/docs/Dialects/OM/RationaleOM/)

 [Prev - 'msft' Dialect](https://circt.llvm.org/docs/Dialects/MSFT/ "'msft' Dialect")
[Next - Object Model Dialect Rationale](https://circt.llvm.org/docs/Dialects/OM/RationaleOM/ "Object Model Dialect Rationale") 

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