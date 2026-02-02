'comb' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'comb' Dialect
==============

*Types and operations for comb dialect*

This dialect defines the `comb` dialect, which is intended to be a generic
representation of combinational logic outside of a particular use-case.

* [Operations](#operations)
  + [`comb.add` (::circt::comb::AddOp)](#combadd-circtcombaddop)
  + [`comb.and` (::circt::comb::AndOp)](#comband-circtcombandop)
  + [`comb.concat` (::circt::comb::ConcatOp)](#combconcat-circtcombconcatop)
  + [`comb.divs` (::circt::comb::DivSOp)](#combdivs-circtcombdivsop)
  + [`comb.divu` (::circt::comb::DivUOp)](#combdivu-circtcombdivuop)
  + [`comb.extract` (::circt::comb::ExtractOp)](#combextract-circtcombextractop)
  + [`comb.icmp` (::circt::comb::ICmpOp)](#combicmp-circtcombicmpop)
  + [`comb.mods` (::circt::comb::ModSOp)](#combmods-circtcombmodsop)
  + [`comb.modu` (::circt::comb::ModUOp)](#combmodu-circtcombmoduop)
  + [`comb.mul` (::circt::comb::MulOp)](#combmul-circtcombmulop)
  + [`comb.mux` (::circt::comb::MuxOp)](#combmux-circtcombmuxop)
  + [`comb.or` (::circt::comb::OrOp)](#combor-circtcomborop)
  + [`comb.parity` (::circt::comb::ParityOp)](#combparity-circtcombparityop)
  + [`comb.replicate` (::circt::comb::ReplicateOp)](#combreplicate-circtcombreplicateop)
  + [`comb.reverse` (::circt::comb::ReverseOp)](#combreverse-circtcombreverseop)
  + [`comb.shl` (::circt::comb::ShlOp)](#combshl-circtcombshlop)
  + [`comb.shrs` (::circt::comb::ShrSOp)](#combshrs-circtcombshrsop)
  + [`comb.shru` (::circt::comb::ShrUOp)](#combshru-circtcombshruop)
  + [`comb.sub` (::circt::comb::SubOp)](#combsub-circtcombsubop)
  + [`comb.truth_table` (::circt::comb::TruthTableOp)](#combtruth_table-circtcombtruthtableop)
  + [`comb.xor` (::circt::comb::XorOp)](#combxor-circtcombxorop)
* [Enums](#enums)
  + [ICmpPredicate](#icmppredicate)

Operations
----------

### `comb.add` (::circt::comb::AddOp)

Syntax:

```
operation ::= `comb.add` (`bin` $twoState^)? $inputs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.and` (::circt::comb::AndOp)

Syntax:

```
operation ::= `comb.and` (`bin` $twoState^)? $inputs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.concat` (::circt::comb::ConcatOp)

*Concatenate a variadic list of operands together.*

See the comb rationale document for details on operand ordering.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.divs` (::circt::comb::DivSOp)

Syntax:

```
operation ::= `comb.divs` (`bin` $twoState^)? $lhs `,` $rhs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | a signless integer bitvector |
| `rhs` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.divu` (::circt::comb::DivUOp)

Syntax:

```
operation ::= `comb.divu` (`bin` $twoState^)? $lhs `,` $rhs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | a signless integer bitvector |
| `rhs` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.extract` (::circt::comb::ExtractOp)

*Extract a range of bits into a smaller value, lowBit specifies the lowest bit included.*

Syntax:

```
operation ::= `comb.extract` $input `from` $lowBit attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `lowBit` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.icmp` (::circt::comb::ICmpOp)

*Compare two integer values*

Syntax:

```
operation ::= `comb.icmp` (`bin` $twoState^)? $predicate $lhs `,` $rhs attr-dict `:` qualified(type($lhs))
```

This operation compares two integers using a predicate. If the predicate is
true, returns 1, otherwise returns 0. This operation always returns a one
bit wide result.

```
    %r = comb.icmp eq %a, %b : i4
```

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `predicate` | circt::comb::ICmpPredicateAttr | hw.icmp comparison predicate |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | a signless integer bitvector |
| `rhs` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit signless integer |

### `comb.mods` (::circt::comb::ModSOp)

Syntax:

```
operation ::= `comb.mods` (`bin` $twoState^)? $lhs `,` $rhs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | a signless integer bitvector |
| `rhs` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.modu` (::circt::comb::ModUOp)

Syntax:

```
operation ::= `comb.modu` (`bin` $twoState^)? $lhs `,` $rhs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | a signless integer bitvector |
| `rhs` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.mul` (::circt::comb::MulOp)

Syntax:

```
operation ::= `comb.mul` (`bin` $twoState^)? $inputs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.mux` (::circt::comb::MuxOp)

*Return one or the other operand depending on a selector bit*

Syntax:

```
operation ::= `comb.mux` (`bin` $twoState^)? $cond `,` $trueValue `,` $falseValue  attr-dict `:` qualified(type($result))
```

```
  %0 = mux %pred, %tvalue, %fvalue : i4
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `cond` | 1-bit signless integer |
| `trueValue` | any type |
| `falseValue` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `comb.or` (::circt::comb::OrOp)

Syntax:

```
operation ::= `comb.or` (`bin` $twoState^)? $inputs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.parity` (::circt::comb::ParityOp)

Syntax:

```
operation ::= `comb.parity` (`bin` $twoState^)? $input attr-dict `:` qualified(type($input))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit signless integer |

### `comb.replicate` (::circt::comb::ReplicateOp)

*Concatenate the operand a constant number of times*

Syntax:

```
operation ::= `comb.replicate` $input attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.reverse` (::circt::comb::ReverseOp)

*Reverses the bit order of an integer value*

Syntax:

```
operation ::= `comb.reverse` $input attr-dict `:` type($input)
```

Reverses the bit ordering of a value. The LSB becomes the MSB and vice versa.

Example:

```
  %out = comb.reverse %in : i4
```

If %in is 4’b1101, then %out is 4’b1011.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.shl` (::circt::comb::ShlOp)

Syntax:

```
operation ::= `comb.shl` (`bin` $twoState^)? $lhs `,` $rhs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | a signless integer bitvector |
| `rhs` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.shrs` (::circt::comb::ShrSOp)

Syntax:

```
operation ::= `comb.shrs` (`bin` $twoState^)? $lhs `,` $rhs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | a signless integer bitvector |
| `rhs` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.shru` (::circt::comb::ShrUOp)

Syntax:

```
operation ::= `comb.shru` (`bin` $twoState^)? $lhs `,` $rhs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | a signless integer bitvector |
| `rhs` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.sub` (::circt::comb::SubOp)

Syntax:

```
operation ::= `comb.sub` (`bin` $twoState^)? $lhs `,` $rhs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | a signless integer bitvector |
| `rhs` | a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `comb.truth_table` (::circt::comb::TruthTableOp)

*Return a true/false based on a lookup table*

Syntax:

```
operation ::= `comb.truth_table` $inputs `->` $lookupTable attr-dict
```

```
  %a = ... : i1
  %b = ... : i1
  %0 = comb.truth_table %a, %b -> [false, true, true, false]
```

This operation assumes a fully elaborated table – 2^n entries. Inputs are
sorted MSB -> LSB from left to right and the offset into `lookupTable` is
computed from them. The table is sorted from 0 -> (2^n - 1) from left to
right.

No difference from array\_get into an array of constants except for xprop
behavior. If one of the inputs is unknown, but said input doesn’t make a
difference in the output (based on the lookup table) the result should not
be ‘x’ – it should be the well-known result.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `lookupTable` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit signless integer |

### `comb.xor` (::circt::comb::XorOp)

Syntax:

```
operation ::= `comb.xor` (`bin` $twoState^)? $inputs attr-dict `:` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `twoState` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of a signless integer bitvector |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

Enums
-----

### ICmpPredicate

*Hw.icmp comparison predicate*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| eq | `0` | eq |
| ne | `1` | ne |
| slt | `2` | slt |
| sle | `3` | sle |
| sgt | `4` | sgt |
| sge | `5` | sge |
| ult | `6` | ult |
| ule | `7` | ule |
| ugt | `8` | ugt |
| uge | `9` | uge |
| ceq | `10` | ceq |
| cne | `11` | cne |
| weq | `12` | weq |
| wne | `13` | wne |

'comb' Dialect Docs
-------------------

* [`comb` Dialect Rationale](https://circt.llvm.org/docs/Dialects/Comb/RationaleComb/)

 [Prev - 'chirrtl' Dialect](https://circt.llvm.org/docs/Dialects/CHIRRTL/ "'chirrtl' Dialect")
[Next - `comb` Dialect Rationale](https://circt.llvm.org/docs/Dialects/Comb/RationaleComb/ "`comb` Dialect Rationale") 

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