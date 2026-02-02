Simulation Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

Simulation Dialect
==================

This dialect provides a high-level representation for simulator-specific
operations. The purpose of the dialect is to provide a high-level representation
for constructs which interact with simulators (Verilator, VCS, Arc, …) that
are easy to analyze and transform in the compiler.

* [Rationale](#rationale)
  + [Plusargs](#plusargs)
* [Types](#types)
  + [DynamicStringType](#dynamicstringtype)
  + [FormatStringType](#formatstringtype)
* [Operations](#operations)
  + [`sim.clocked_pause` (::circt::sim::ClockedPauseOp)](#simclocked_pause-circtsimclockedpauseop)
  + [`sim.clocked_terminate` (::circt::sim::ClockedTerminateOp)](#simclocked_terminate-circtsimclockedterminateop)
  + [`sim.func.dpi.call` (::circt::sim::DPICallOp)](#simfuncdpicall-circtsimdpicallop)
  + [`sim.func.dpi` (::circt::sim::DPIFuncOp)](#simfuncdpi-circtsimdpifuncop)
  + [`sim.fmt.bin` (::circt::sim::FormatBinOp)](#simfmtbin-circtsimformatbinop)
  + [`sim.fmt.char` (::circt::sim::FormatCharOp)](#simfmtchar-circtsimformatcharop)
  + [`sim.fmt.dec` (::circt::sim::FormatDecOp)](#simfmtdec-circtsimformatdecop)
  + [`sim.fmt.flt` (::circt::sim::FormatFloatOp)](#simfmtflt-circtsimformatfloatop)
  + [`sim.fmt.gen` (::circt::sim::FormatGeneralOp)](#simfmtgen-circtsimformatgeneralop)
  + [`sim.fmt.hex` (::circt::sim::FormatHexOp)](#simfmthex-circtsimformathexop)
  + [`sim.fmt.literal` (::circt::sim::FormatLiteralOp)](#simfmtliteral-circtsimformatliteralop)
  + [`sim.fmt.oct` (::circt::sim::FormatOctOp)](#simfmtoct-circtsimformatoctop)
  + [`sim.fmt.exp` (::circt::sim::FormatScientificOp)](#simfmtexp-circtsimformatscientificop)
  + [`sim.fmt.concat` (::circt::sim::FormatStringConcatOp)](#simfmtconcat-circtsimformatstringconcatop)
  + [`sim.string.int_to_string` (::circt::sim::IntToStringOp)](#simstringint_to_string-circtsiminttostringop)
  + [`sim.pause` (::circt::sim::PauseOp)](#simpause-circtsimpauseop)
  + [`sim.plusargs.test` (::circt::sim::PlusArgsTestOp)](#simplusargstest-circtsimplusargstestop)
  + [`sim.plusargs.value` (::circt::sim::PlusArgsValueOp)](#simplusargsvalue-circtsimplusargsvalueop)
  + [`sim.print` (::circt::sim::PrintFormattedOp)](#simprint-circtsimprintformattedop)
  + [`sim.proc.print` (::circt::sim::PrintFormattedProcOp)](#simprocprint-circtsimprintformattedprocop)
  + [`sim.string.concat` (::circt::sim::StringConcatOp)](#simstringconcat-circtsimstringconcatop)
  + [`sim.string.literal` (::circt::sim::StringConstantOp)](#simstringliteral-circtsimstringconstantop)
  + [`sim.string.length` (::circt::sim::StringLengthOp)](#simstringlength-circtsimstringlengthop)
  + [`sim.terminate` (::circt::sim::TerminateOp)](#simterminate-circtsimterminateop)

Rationale [¶](#rationale)
-------------------------

### Plusargs [¶](#plusargs)

The `sim.plusarg_test` and `sim.plusarg_value` operations are wrappers around
the SystemVerilog built-ins which access command-line arguments.
They are cleaner from a data-flow perspective, as they package the wires
and if-statements involved into compact operations that can be trivially
handled by analyses.

Types [¶](#types)
-----------------

### DynamicStringType [¶](#dynamicstringtype)

*String type with variable length*

Syntax: `!sim.dstring`

The dynamic string type represents a string with a variable length that is
only known at runtime. The contents of the string are held as an array of
`i8` bytes. The interpretation of these bytes as a concrete character
encoding is undefined, but treating the bytes as a UTF-8 string is
recommended. A string’s length must fit into an `i64`.

Even though practical lowerings of this type likely have to hold the string
contents in dynamically-allocated heap memory, *this type has value
semantics*. If the same string SSA value is passed to multiple users, any
mutation performed by one user is not visible to other users.

### FormatStringType [¶](#formatstringtype)

*Format string type*

Syntax: `!sim.fstring`

A format string type represents either a single formatting fragment or the
concatenation of an arbitrary but finite number of fragments.
A formatting fragment is either a static string literal or the association
of a dynamic hardware value with a format specifier.

Operations [¶](#operations)
---------------------------

### `sim.clocked_pause` (::circt::sim::ClockedPauseOp) [¶](#simclocked_pause-circtsimclockedpauseop)

*Pause the simulation*

Syntax:

```
operation ::= `sim.clocked_pause` $clock `,` $condition `,`
              custom<KeywordBool>($verbose, "\"verbose\"" , "\"quiet\"") attr-dict
```

Implements the semantics of `sim.pause` if the given condition is true on
the rising edge of the clock operand.

#### Attributes: [¶](#attributes)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `verbose` | ::mlir::BoolAttr | bool attribute |

#### Operands: [¶](#operands)

| Operand | Description |
| --- | --- |
| `clock` | A type for clock-carrying wires |
| `condition` | 1-bit signless integer |

### `sim.clocked_terminate` (::circt::sim::ClockedTerminateOp) [¶](#simclocked_terminate-circtsimclockedterminateop)

*Terminate the simulation*

Syntax:

```
operation ::= `sim.clocked_terminate` $clock `,` $condition `,`
              custom<KeywordBool>($success, "\"success\"", "\"failure\"") `,`
              custom<KeywordBool>($verbose, "\"verbose\"" , "\"quiet\"") attr-dict
```

Implements the semantics of `sim.terminate` if the given condition is true
on the rising edge of the clock operand.

#### Attributes: [¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `success` | ::mlir::BoolAttr | bool attribute |
| `verbose` | ::mlir::BoolAttr | bool attribute |

#### Operands: [¶](#operands-1)

| Operand | Description |
| --- | --- |
| `clock` | A type for clock-carrying wires |
| `condition` | 1-bit signless integer |

### `sim.func.dpi.call` (::circt::sim::DPICallOp) [¶](#simfuncdpicall-circtsimdpicallop)

*A call option for DPI function with an optional clock and enable*

Syntax:

```
operation ::= `sim.func.dpi.call` $callee `(` $inputs `)` (`clock` $clock^)? (`enable` $enable^)?
              attr-dict `:` functional-type($inputs, results)
```

`sim.func.dpi.call` represents SystemVerilog DPI function call. There are two
optional operands `clock` and `enable`.

If `clock` is not provided, the callee is invoked when input values are changed.
If provided, the DPI function is called at clock’s posedge. The result values behave
like registers and the DPI function is used as a state transfer function of them.

`enable` operand is used to conditionally call the DPI since DPI call could be quite
more expensive than native constructs. When `enable` is low, results of unclocked
calls are undefined and in SV results they are lowered into `X`. Users are expected
to gate result values by another `enable` to model a default value of results.

For clocked calls, a low enable means that its register state transfer function is
not called. Hence their values will not be modify in that clock.

Traits: `AttrSizedOperandSegments`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallOpInterface`, `SymbolUserOpInterface`

#### Attributes: [¶](#attributes-2)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `callee` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands: [¶](#operands-2)

| Operand | Description |
| --- | --- |
| `clock` | A type for clock-carrying wires |
| `enable` | 1-bit signless integer |
| `inputs` | variadic of any type |

#### Results: [¶](#results)

| Result | Description |
| --- | --- |
| «unnamed» | variadic of any type |

### `sim.func.dpi` (::circt::sim::DPIFuncOp) [¶](#simfuncdpi-circtsimdpifuncop)

*A System Verilog function*

`sim.func.dpi` models an external function in a core dialect.

Traits: `IsolatedFromAbove`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallableOpInterface`, `FunctionOpInterface`, `OpAsmOpInterface`, `Symbol`

#### Attributes: [¶](#attributes-3)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `module_type` | ::mlir::TypeAttr | type attribute of a module |
| `per_argument_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `argument_locs` | ::mlir::ArrayAttr | location array attribute |
| `verilogName` | ::mlir::StringAttr | string attribute |

### `sim.fmt.bin` (::circt::sim::FormatBinOp) [¶](#simfmtbin-circtsimformatbinop)

*Binary format specifier*

Syntax:

```
operation ::= `sim.fmt.bin` $value
              (`isLeftAligned` $isLeftAligned^)?
              (`paddingChar` $paddingChar^)?
              (`specifierWidth` $specifierWidth^)?
              attr-dict `:` qualified(type($value))
```

Format the given integer value as binary (base two) string.

The printed value will be left-padded with ‘0’ up to the number
of bits of the argument’s type. Zero width values will produce
the empty string. No further prefix will be added.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ValueFormatter`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-4)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `isLeftAligned` | ::mlir::BoolAttr | bool attribute |
| `paddingChar` | ::mlir::IntegerAttr | 8-bit signless integer attribute |
| `specifierWidth` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-3)

| Operand | Description |
| --- | --- |
| `value` | integer |

#### Results: [¶](#results-1)

| Result | Description |
| --- | --- |
| `result` | Format string type |

### `sim.fmt.char` (::circt::sim::FormatCharOp) [¶](#simfmtchar-circtsimformatcharop)

*Character format specifier*

Syntax:

```
operation ::= `sim.fmt.char` $value attr-dict `:` qualified(type($value))
```

Format the given integer value as a single character.

For integer values up to 127, ASCII compatible encoding is assumed.
For larger values, the encoding is unspecified.

If the argument’s type width is less than eight bits, the value is
zero extended.
If the width is greater than eight bits, the resulting formatted string
is undefined.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ValueFormatter`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-4)

| Operand | Description |
| --- | --- |
| `value` | integer |

#### Results: [¶](#results-2)

| Result | Description |
| --- | --- |
| `result` | Format string type |

### `sim.fmt.dec` (::circt::sim::FormatDecOp) [¶](#simfmtdec-circtsimformatdecop)

*Decimal format specifier*

Syntax:

```
operation ::= `sim.fmt.dec` $value
              (`isLeftAligned` $isLeftAligned^)?
              (`paddingChar` $paddingChar^)?
              (`specifierWidth` $specifierWidth^)?
              (`signed` $isSigned^)?
              attr-dict `:` qualified(type($value))
```

Format the given integer value as signed or unsigned decimal string.

Leading zeros are omitted. Non-negative or unsigned values will
*not* be prefixed with a ‘+’.

For unsigned formatting, the printed value will
be left-padded with spaces up to *at least* the length required to print
the maximum unsigned value of the argument’s type.
For signed formatting, the printed value will be
left-padded with spaces up to *at least* the length required
to print the minimum signed value of the argument’s type
including the ‘-’ character.
E.g., a zero value of type `i1` requires no padding for unsigned
formatting and one leading space for signed formatting.
Format specifiers of same argument type and signedness must be
padded to the same width. Zero width values will produce
a single ‘0’.

Backends are recommended to not exceed the required amount of padding.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ValueFormatter`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-5)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `isLeftAligned` | ::mlir::BoolAttr | bool attribute |
| `paddingChar` | ::mlir::IntegerAttr | 8-bit signless integer attribute |
| `specifierWidth` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `isSigned` | ::mlir::UnitAttr | unit attribute |

#### Operands: [¶](#operands-5)

| Operand | Description |
| --- | --- |
| `value` | integer |

#### Results: [¶](#results-3)

| Result | Description |
| --- | --- |
| `result` | Format string type |

### `sim.fmt.flt` (::circt::sim::FormatFloatOp) [¶](#simfmtflt-circtsimformatfloatop)

*Real format specifier*

Syntax:

```
operation ::= `sim.fmt.flt` $value
              (`isLeftAligned` $isLeftAligned^)?
              (`fieldWidth` $fieldWidth^)?
              (`fracDigits` $fracDigits^)?
              attr-dict `:` qualified(type($value))
```

Format the given real value in floating point (decimal) notation

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ValueFormatter`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-6)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `isLeftAligned` | ::mlir::BoolAttr | bool attribute |
| `fieldWidth` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fracDigits` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-6)

| Operand | Description |
| --- | --- |
| `value` | floating-point |

#### Results: [¶](#results-4)

| Result | Description |
| --- | --- |
| `result` | Format string type |

### `sim.fmt.gen` (::circt::sim::FormatGeneralOp) [¶](#simfmtgen-circtsimformatgeneralop)

*Real format specifier*

Syntax:

```
operation ::= `sim.fmt.gen` $value
              (`isLeftAligned` $isLeftAligned^)?
              (`fieldWidth` $fieldWidth^)?
              (`fracDigits` $fracDigits^)?
              attr-dict `:` qualified(type($value))
```

Format floating-point numbers using either decimal or scientific notation
based on the exponent and precision parameter fracDigits (default: 6)

For exp ≥ 0:
Uses decimal notation if exp + 1 ≤ max(1, fracDigits), otherwise switches
to scientific notation with max(0, fracDigits - 1) fractional digits

For exp < 0:
If exp ≥ -4: Uses decimal notation with max(b - 1, 0) + |exp| fractional digits
If exp < -4: Uses scientific notation with max(b - 1, 0) fractional digits

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ValueFormatter`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-7)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `isLeftAligned` | ::mlir::BoolAttr | bool attribute |
| `fieldWidth` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fracDigits` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-7)

| Operand | Description |
| --- | --- |
| `value` | floating-point |

#### Results: [¶](#results-5)

| Result | Description |
| --- | --- |
| `result` | Format string type |

### `sim.fmt.hex` (::circt::sim::FormatHexOp) [¶](#simfmthex-circtsimformathexop)

*Hexadecimal format specifier*

Syntax:

```
operation ::= `sim.fmt.hex` $value `,`
              `isUpper`  $isHexUppercase
              (`isLeftAligned` $isLeftAligned^)?
              (`paddingChar` $paddingChar^)?
              (`specifierWidth` $specifierWidth^)?
              attr-dict `:` qualified(type($value))
```

Format the given integer value as lower-case hexadecimal string.

The printed value will be left-padded with ‘0’ up to the
length required to print the maximum value of the argument’s
type. Zero width values will produce the empty string.
No further prefix will be added.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ValueFormatter`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-8)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `isHexUppercase` | ::mlir::BoolAttr | bool attribute |
| `isLeftAligned` | ::mlir::BoolAttr | bool attribute |
| `paddingChar` | ::mlir::IntegerAttr | 8-bit signless integer attribute |
| `specifierWidth` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-8)

| Operand | Description |
| --- | --- |
| `value` | integer |

#### Results: [¶](#results-6)

| Result | Description |
| --- | --- |
| `result` | Format string type |

### `sim.fmt.literal` (::circt::sim::FormatLiteralOp) [¶](#simfmtliteral-circtsimformatliteralop)

*Literal string fragment*

Syntax:

```
operation ::= `sim.fmt.literal` $literal attr-dict
```

Creates a constant, raw ASCII string literal for formatted printing.
The given string attribute will be outputted as is,
including non-printable characters. The literal may be empty or contain
null characters (’\0’) which must not be interpreted as string
terminators by backends.

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-9)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `literal` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-7)

| Result | Description |
| --- | --- |
| `result` | Format string type |

### `sim.fmt.oct` (::circt::sim::FormatOctOp) [¶](#simfmtoct-circtsimformatoctop)

*Octal format specifier*

Syntax:

```
operation ::= `sim.fmt.oct` $value
              (`isLeftAligned` $isLeftAligned^)?
              (`paddingChar` $paddingChar^)?
              (`specifierWidth` $specifierWidth^)?
              attr-dict `:` qualified(type($value))
```

Format the given integer value as lower-case octal (base eight) string.

The printed value will be left-padded with ‘0’ up to the
length required to print the maximum value of the argument’s
type. Zero width values will produce the empty string.
No further prefix will be added.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ValueFormatter`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-10)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `isLeftAligned` | ::mlir::BoolAttr | bool attribute |
| `paddingChar` | ::mlir::IntegerAttr | 8-bit signless integer attribute |
| `specifierWidth` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-9)

| Operand | Description |
| --- | --- |
| `value` | integer |

#### Results: [¶](#results-8)

| Result | Description |
| --- | --- |
| `result` | Format string type |

### `sim.fmt.exp` (::circt::sim::FormatScientificOp) [¶](#simfmtexp-circtsimformatscientificop)

*Real format specifier*

Syntax:

```
operation ::= `sim.fmt.exp` $value
              (`isLeftAligned` $isLeftAligned^)?
              (`fieldWidth` $fieldWidth^)?
              (`fracDigits` $fracDigits^)?
              attr-dict `:` qualified(type($value))
```

Format the given real value in scientific (exponential) notation

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ValueFormatter`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-11)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `isLeftAligned` | ::mlir::BoolAttr | bool attribute |
| `fieldWidth` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fracDigits` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-10)

| Operand | Description |
| --- | --- |
| `value` | floating-point |

#### Results: [¶](#results-9)

| Result | Description |
| --- | --- |
| `result` | Format string type |

### `sim.fmt.concat` (::circt::sim::FormatStringConcatOp) [¶](#simfmtconcat-circtsimformatstringconcatop)

*Concatenate format strings*

Syntax:

```
operation ::= `sim.fmt.concat` ` ` `(` $inputs `)` attr-dict
```

Concatenates an arbitrary number of format strings from
left to right. If the argument list is empty, the empty string
literal is produced.

Concatenations must not be recursive. I.e., a concatenated string should
not contain itself directly or indirectly.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-11)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of Format string type |

#### Results: [¶](#results-10)

| Result | Description |
| --- | --- |
| `result` | Format string type |

### `sim.string.int_to_string` (::circt::sim::IntToStringOp) [¶](#simstringint_to_string-circtsiminttostringop)

*Convert an integer into a string*

Syntax:

```
operation ::= `sim.string.int_to_string` $input attr-dict `:` qualified(type($input))
```

Converts an integer value to a string following Verilog semantics, where
string literals are represented as right-aligned bitvectors.

The operation reads the integer byte-by-byte, interpreting each non-zero
byte as an ASCII character. Zero-valued bytes are skipped and
not included in the output string. See Sec 6.16 of the standard saying
“A string variable shall not contain the special character “\0”.
Assigning the value 0 to a string character shall be ignored”

Example:
Consider the hex value 0x00\_00\_00\_48\_00\_00\_6C\_6F
Reading from MSB to LSB and filtering out null bytes:
- 0x48 = ‘H’
- 0x6C = ’l’
- 0x6F = ‘o’
This produces the string “Hlo”, not “\0\0\0H\0\0lo”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-12)

| Operand | Description |
| --- | --- |
| `input` | integer |

#### Results: [¶](#results-11)

| Result | Description |
| --- | --- |
| `result` | String type with variable length |

### `sim.pause` (::circt::sim::PauseOp) [¶](#simpause-circtsimpauseop)

*Pause the simulation*

Syntax:

```
operation ::= `sim.pause` custom<KeywordBool>($verbose, "\"verbose\"" , "\"quiet\"") attr-dict
```

Interrupt the simulation and give control back to the user in case of an
interactive simulation. Non-interactive simulations may choose to terminate
instead. Depending on the `verbose` operand, simulators may print additional
information about the current simulation time and hierarchical location of
the op.

This op corresponds to the following SystemVerilog constructs:

| Operation | SystemVerilog |
| --- | --- |
| `sim.pause quiet` | `$stop(0)` |
| `sim.pause verbose` | `$stop` or `$stop(1)` |

#### Attributes: [¶](#attributes-12)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `verbose` | ::mlir::BoolAttr | bool attribute |

### `sim.plusargs.test` (::circt::sim::PlusArgsTestOp) [¶](#simplusargstest-circtsimplusargstestop)

*SystemVerilog `$test$plusargs` call*

Syntax:

```
operation ::= `sim.plusargs.test` $formatString attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-13)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `formatString` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-12)

| Result | Description |
| --- | --- |
| `found` | 1-bit signless integer |

### `sim.plusargs.value` (::circt::sim::PlusArgsValueOp) [¶](#simplusargsvalue-circtsimplusargsvalueop)

*SystemVerilog `$value$plusargs` call*

Syntax:

```
operation ::= `sim.plusargs.value` $formatString attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-14)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `formatString` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-13)

| Result | Description |
| --- | --- |
| `found` | 1-bit signless integer |
| `result` | any type |

### `sim.print` (::circt::sim::PrintFormattedOp) [¶](#simprint-circtsimprintformattedop)

*Print a formatted string on a given clock and condition*

Syntax:

```
operation ::= `sim.print` $input `on` $clock `if` $condition attr-dict
```

Evaluate a format string and print it to the simulation console on the
rising edge of the given clock, if, and only if, the condition argument
is ’true’.

Multiple print operations in the same module and on the same clock edge
are performed according to their order of occurence in the IR. The order
of printing for operations in different modules, instances or on different
clocks is undefined.

#### Operands: [¶](#operands-13)

| Operand | Description |
| --- | --- |
| `input` | Format string type |
| `clock` | A type for clock-carrying wires |
| `condition` | 1-bit signless integer |

### `sim.proc.print` (::circt::sim::PrintFormattedProcOp) [¶](#simprocprint-circtsimprintformattedprocop)

*Print a formatted string within a procedural region*

Syntax:

```
operation ::= `sim.proc.print` $input attr-dict
```

Evaluate a format string and print it to the simulation console.

This operation must be within a procedural region.

#### Operands: [¶](#operands-14)

| Operand | Description |
| --- | --- |
| `input` | Format string type |

### `sim.string.concat` (::circt::sim::StringConcatOp) [¶](#simstringconcat-circtsimstringconcatop)

*Concatenate strings*

Syntax:

```
operation ::= `sim.string.concat` ` ` `(` $inputs `)` attr-dict
```

Concatenates multiple strings into a single, longer string. The first
operand appears first in the output string. The last character of the first
operand is immediately followed by the first character of the second
operand. If the operation has a single operand, its result is equal to that
operand. If the operation has zero operands, it returns an empty string.
The length of the result is the sum of each operand’s length.

Example:

```
%0 = sim.string.literal "abc"
%1 = sim.string.literal "def"
%2 = sim.string.literal "ghj"
%3 = sim.string.concat (%0, %1, %2)  // %3 is "abcdefghj"
%4 = sim.string.concat (%0)          // %4 is "abc"
%5 = sim.string.concat ()            // %5 is ""
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-15)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of String type with variable length |

#### Results: [¶](#results-14)

| Result | Description |
| --- | --- |
| `result` | String type with variable length |

### `sim.string.literal` (::circt::sim::StringConstantOp) [¶](#simstringliteral-circtsimstringconstantop)

*Literal string fragment*

Syntax:

```
operation ::= `sim.string.literal` $literal attr-dict
```

Creates a dynamic string from a constant string literal.

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-15)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `literal` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-15)

| Result | Description |
| --- | --- |
| `result` | String type with variable length |

### `sim.string.length` (::circt::sim::StringLengthOp) [¶](#simstringlength-circtsimstringlengthop)

*Length of a string*

Syntax:

```
operation ::= `sim.string.length` $input attr-dict
```

Returns the length of a dynamic string as an `i64` value.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-16)

| Operand | Description |
| --- | --- |
| `input` | String type with variable length |

#### Results: [¶](#results-16)

| Result | Description |
| --- | --- |
| `result` | 64-bit signless integer |

### `sim.terminate` (::circt::sim::TerminateOp) [¶](#simterminate-circtsimterminateop)

*Terminate the simulation*

Syntax:

```
operation ::= `sim.terminate` custom<KeywordBool>($success, "\"success\"", "\"failure\"") `,`
              custom<KeywordBool>($verbose, "\"verbose\"" , "\"quiet\"") attr-dict
```

Terminate the simulation with a success or failure exit code. Depending on
the `verbose` operand, simulators may print additional information about the
current simulation time and hierarchical location of the op.

This op correpsonds to the following SystemVerilog constructs:

| Operation | SystemVerilog |
| --- | --- |
| `sim.terminate success, quiet` | `$finish(0)` |
| `sim.terminate success, verbose` | `$finish` or `$finish(1)` |
| `sim.terminate failure, quiet` | `$fatal(0)` |
| `sim.terminate failure, verbose` | `$fatal` or `$fatal(1)` |

Note that this op does not match the behavior of the `$exit` system task in
SystemVerilog, which blocks execution of the calling process until all
`program` instances have terminated, and then calls `$finish`.

#### Attributes: [¶](#attributes-16)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `success` | ::mlir::BoolAttr | bool attribute |
| `verbose` | ::mlir::BoolAttr | bool attribute |

 [Prev - Random Test Generation (RTG) Rationale](https://circt.llvm.org/docs/Dialects/RTG/ "Random Test Generation (RTG) Rationale")
[Next - SMT Dialect](https://circt.llvm.org/docs/Dialects/SMT/ "SMT Dialect") 

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