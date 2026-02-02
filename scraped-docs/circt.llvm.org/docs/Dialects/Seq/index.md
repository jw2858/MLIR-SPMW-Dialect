'seq' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'seq' Dialect
=============

*Types and operations for seq dialect*

The `seq` dialect is intended to model digital sequential logic.

* [Operations](#operations)
  + [`seq.clock_div` (::circt::seq::ClockDividerOp)](#seqclock_div-circtseqclockdividerop)
  + [`seq.clock_gate` (::circt::seq::ClockGateOp)](#seqclock_gate-circtseqclockgateop)
  + [`seq.clock_inv` (::circt::seq::ClockInverterOp)](#seqclock_inv-circtseqclockinverterop)
  + [`seq.clock_mux` (::circt::seq::ClockMuxOp)](#seqclock_mux-circtseqclockmuxop)
  + [`seq.compreg` (::circt::seq::CompRegOp)](#seqcompreg-circtseqcompregop)
  + [`seq.compreg.ce` (::circt::seq::CompRegClockEnabledOp)](#seqcompregce-circtseqcompregclockenabledop)
  + [`seq.const_clock` (::circt::seq::ConstClockOp)](#seqconst_clock-circtseqconstclockop)
  + [`seq.fifo` (::circt::seq::FIFOOp)](#seqfifo-circtseqfifoop)
  + [`seq.firmem` (::circt::seq::FirMemOp)](#seqfirmem-circtseqfirmemop)
  + [`seq.firmem.read_port` (::circt::seq::FirMemReadOp)](#seqfirmemread_port-circtseqfirmemreadop)
  + [`seq.firmem.read_write_port` (::circt::seq::FirMemReadWriteOp)](#seqfirmemread_write_port-circtseqfirmemreadwriteop)
  + [`seq.firmem.write_port` (::circt::seq::FirMemWriteOp)](#seqfirmemwrite_port-circtseqfirmemwriteop)
  + [`seq.firreg` (::circt::seq::FirRegOp)](#seqfirreg-circtseqfirregop)
  + [`seq.from_clock` (::circt::seq::FromClockOp)](#seqfrom_clock-circtseqfromclockop)
  + [`seq.from_immutable` (::circt::seq::FromImmutableOp)](#seqfrom_immutable-circtseqfromimmutableop)
  + [`seq.hlmem` (::circt::seq::HLMemOp)](#seqhlmem-circtseqhlmemop)
  + [`seq.initial` (::circt::seq::InitialOp)](#seqinitial-circtseqinitialop)
  + [`seq.read` (::circt::seq::ReadPortOp)](#seqread-circtseqreadportop)
  + [`seq.shiftreg` (::circt::seq::ShiftRegOp)](#seqshiftreg-circtseqshiftregop)
  + [`seq.to_clock` (::circt::seq::ToClockOp)](#seqto_clock-circtseqtoclockop)
  + [`seq.write` (::circt::seq::WritePortOp)](#seqwrite-circtseqwriteportop)
  + [`seq.yield` (::circt::seq::YieldOp)](#seqyield-circtseqyieldop)
* [Attributes](#attributes-12)
  + [ClockConstAttr](#clockconstattr)
  + [FirMemInitAttr](#firmeminitattr)
* [Type constraints](#type-constraints)
  + [an ImmutableType](#an-immutabletype)
* [Types](#types)
  + [ClockType](#clocktype)
  + [FirMemType](#firmemtype)
  + [HLMemType](#hlmemtype)
  + [ImmutableType](#immutabletype)
* [Enums](#enums)
  + [ClockConst](#clockconst)
  + [RUW](#ruw)
  + [WUW](#wuw)

Operations
----------

### `seq.clock_div` (::circt::seq::ClockDividerOp)

*Produces a clock divided by a power of two*

Syntax:

```
operation ::= `seq.clock_div` $input `by` $pow2 attr-dict
```

The output clock is phase-aligned to the input clock.

```
%div_clock = seq.clock_div %clock by 1
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `pow2` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | A type for clock-carrying wires |

#### Results:

| Result | Description |
| --- | --- |
| `output` | A type for clock-carrying wires |

### `seq.clock_gate` (::circt::seq::ClockGateOp)

*Safely gates a clock with an enable signal*

Syntax:

```
operation ::= `seq.clock_gate` $input `,` $enable (`,` $test_enable^)? (`sym` $inner_sym^)? attr-dict
```

The `seq.clock_gate` enables and disables a clock safely, without glitches,
based on a boolean enable value. If the enable operand is 1, the output
clock produced by the clock gate is identical to the input clock. If the
enable operand is 0, the output clock is a constant zero.

The `enable` operand is sampled at the rising edge of the input clock; any
changes on the enable before or after that edge are ignored and do not
affect the output clock.

The `test_enable` operand is optional and if present is OR’d together with
the `enable` operand to determine whether the output clock is gated or not.

The op can be referred to using an inner symbol. Upon translation, the
symbol will target the instance to the external module it lowers to.

```
%gatedClock = seq.clock_gate %clock, %enable
%gatedClock = seq.clock_gate %clock, %enable, %test_enable
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `InnerSymbolOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | A type for clock-carrying wires |
| `enable` | 1-bit signless integer |
| `test_enable` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `output` | A type for clock-carrying wires |

### `seq.clock_inv` (::circt::seq::ClockInverterOp)

*Inverts the clock signal*

Syntax:

```
operation ::= `seq.clock_inv` $input attr-dict
```

Note that the compiler can optimize inverters away, preventing their
use as part of explicit clock buffers.

```
%inv_clock = seq.clock_inv %clock
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | A type for clock-carrying wires |

#### Results:

| Result | Description |
| --- | --- |
| `output` | A type for clock-carrying wires |

### `seq.clock_mux` (::circt::seq::ClockMuxOp)

*Safely selects a clock based on a condition*

Syntax:

```
operation ::= `seq.clock_mux` $cond `,` $trueClock `,` $falseClock attr-dict
```

The `seq.clock_mux` op selects a clock from two options. If `cond` is
true, the first clock operand is selected to drive downstream logic.
Otherwise, the second clock is used.

```
%clock = seq.clock_mux %cond, %trueClock, %falseClock
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `cond` | 1-bit signless integer |
| `trueClock` | A type for clock-carrying wires |
| `falseClock` | A type for clock-carrying wires |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A type for clock-carrying wires |

### `seq.compreg` (::circt::seq::CompRegOp)

*Register a value, storing it for one cycle*

Syntax:

```
operation ::= `seq.compreg` (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) $input `,` $clk
              (`reset` $reset^ `,` $resetValue)?
              (`initial` $initialValue^)? attr-dict `:` type($data)
              custom<OptionalTypeMatch>(ref(type($data)), ref($resetValue), type($resetValue))
              custom<OptionalImmutableTypeMatch>(ref(type($data)), ref($initialValue), type($initialValue))
```

See the Seq dialect rationale for a longer description

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`

Interfaces: `Clocked`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `InnerSymbolOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `Resettable`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | any type |
| `clk` | A type for clock-carrying wires |
| `reset` | 1-bit signless integer |
| `resetValue` | any type |
| `initialValue` | an ImmutableType |

#### Results:

| Result | Description |
| --- | --- |
| `data` | any type |

### `seq.compreg.ce` (::circt::seq::CompRegClockEnabledOp)

*When enabled, register a value*

Syntax:

```
operation ::= `seq.compreg.ce` (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) $input `,` $clk `,` $clockEnable
              (`reset` $reset^ `,` $resetValue)?
              (`initial` $initialValue^)? attr-dict `:` type($data)
              custom<OptionalTypeMatch>(ref(type($data)), ref($resetValue), type($resetValue))
              custom<OptionalImmutableTypeMatch>(ref(type($data)), ref($initialValue), type($initialValue))
```

See the Seq dialect rationale for a longer description

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`

Interfaces: `Clocked`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `InnerSymbolOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `Resettable`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | any type |
| `clk` | A type for clock-carrying wires |
| `clockEnable` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `resetValue` | any type |
| `initialValue` | an ImmutableType |

#### Results:

| Result | Description |
| --- | --- |
| `data` | any type |

### `seq.const_clock` (::circt::seq::ConstClockOp)

*Produce constant clock value*

Syntax:

```
operation ::= `seq.const_clock` $value attr-dict
```

The constant operation produces a constant clock value.

```
  %clock = seq.const_clock low
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::circt::seq::ClockConstAttr | clock constant |

#### Results:

| Result | Description |
| --- | --- |
| `result` | A type for clock-carrying wires |

### `seq.fifo` (::circt::seq::FIFOOp)

*A high-level FIFO operation*

Syntax:

```
operation ::= `seq.fifo` `depth` $depth
              (`rd_latency` $rdLatency^)?
              custom<FIFOAFThreshold>($almostFullThreshold, type($almostFull))
              custom<FIFOAEThreshold>($almostEmptyThreshold, type($almostEmpty))
              `in` $input `rdEn` $rdEn `wrEn` $wrEn `clk` $clk `rst` $rst attr-dict `:` type($input)
```

This operation represents a high-level abstraction of a FIFO. Access to the
FIFO is structural, and thus may be composed with other core RTL dialect
operations.
The fifo operation is configurable with the following parameters:

1. Depth (cycles)
2. Read latency (cycles) is the number of cycles it takes for a read to
   return data after the read enable signal is asserted.
3. Almost full/empty thresholds (optional). If not provided, these will
   be asserted when the FIFO is full/empty.

Like `seq.hlmem` there are no guarantees that all possible fifo configuration
are able to be lowered. Available lowering passes will pattern match on the
requested fifo configuration and attempt to provide a legal lowering.

Traits: `AttrSizedResultSegments`

Interfaces: `OpAsmOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `depth` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |
| `rdLatency` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 0 |
| `almostFullThreshold` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 0 |
| `almostEmptyThreshold` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 0 |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | any type |
| `rdEn` | 1-bit signless integer |
| `wrEn` | 1-bit signless integer |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `output` | any type |
| `full` | 1-bit signless integer |
| `empty` | 1-bit signless integer |
| `almostFull` | 1-bit signless integer |
| `almostEmpty` | 1-bit signless integer |

### `seq.firmem` (::circt::seq::FirMemOp)

*A FIRRTL-flavored memory*

Syntax:

```
operation ::= `seq.firmem` (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name)
              $readLatency `,` $writeLatency `,` $ruw `,` $wuw
              attr-dict `:` type($memory)
```

The `seq.firmem` op represents memories lowered from the FIRRTL dialect. It
is used to capture some of the peculiarities of what FIRRTL expects from
memories, while still representing them at the HW dialect level.

A `seq.firmem` declares the memory and captures the memory-level parameters
such as width and depth or how read/write collisions are resolved. The read,
write, and read-write ports are expressed as separate operations that take
the declared memory as an operand.

Interfaces: `InnerSymbolOpInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{MemoryEffects::Allocate on ::mlir::SideEffects::DefaultResource}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `readLatency` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `writeLatency` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `ruw` | circt::seq::RUWAttr | Read-Under-Write Behavior |
| `wuw` | circt::seq::WUWAttr | Write-Under-Write Behavior |
| `name` | ::mlir::StringAttr | string attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `init` | ::circt::seq::FirMemInitAttr | Memory initialization information |
| `prefix` | ::mlir::StringAttr | string attribute |
| `output_file` | ::mlir::Attribute | any attribute |

#### Results:

| Result | Description |
| --- | --- |
| `memory` | A FIRRTL-flavored memory |

### `seq.firmem.read_port` (::circt::seq::FirMemReadOp)

*A memory read port*

Syntax:

```
operation ::= `seq.firmem.read_port` $memory `[` $address `]` `,` `clock` $clk
              (`enable` $enable^)?
              attr-dict `:` type($memory)
```

The `seq.firmem.read_port` op represents a read port on a `seq.firmem`
memory. It takes the memory as an operand, together with the address to
be read, the clock on which the read is synchronized, and an optional
enable. Omitting the enable operand has the same effect as passing a
constant `true` to it.

Interfaces: `Clocked`, `InferTypeOpInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource}`

#### Operands:

| Operand | Description |
| --- | --- |
| `memory` | A FIRRTL-flavored memory |
| `address` | signless integer |
| `clk` | A type for clock-carrying wires |
| `enable` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `data` | signless integer |

### `seq.firmem.read_write_port` (::circt::seq::FirMemReadWriteOp)

*A memory read-write port*

Syntax:

```
operation ::= `seq.firmem.read_write_port` $memory `[` $address `]` `=` $writeData `if` $mode `,` `clock` $clk
              (`enable` $enable^)? (`mask` $mask^)?
              attr-dict `:` type($memory) (`,` type($mask)^)?
```

The `seq.firmem.read_write_port` op represents a read-write port on a
`seq.firmem` memory. It takes the memory as an operand, together with the
address and data to be written, a mode operand indicating whether the port
should perform a read (`mode=0`) or a write (`mode=1`), the clock on which
the read and write is synchronized, an optional enable, and and optional
write mask. Omitting the enable operand has the same effect as passing a
constant `true` to it. Omitting the write mask operand has the same effect
as passing an all-ones value to it. A write mask operand can only be present
if the `seq.firmem` specifies a mask width; otherwise it must be omitted.

Traits: `AttrSizedOperandSegments`

Interfaces: `Clocked`, `InferTypeOpInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource, MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### Operands:

| Operand | Description |
| --- | --- |
| `memory` | A FIRRTL-flavored memory |
| `address` | signless integer |
| `clk` | A type for clock-carrying wires |
| `enable` | 1-bit signless integer |
| `writeData` | signless integer |
| `mode` | 1-bit signless integer |
| `mask` | signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `readData` | signless integer |

### `seq.firmem.write_port` (::circt::seq::FirMemWriteOp)

*A memory write port*

Syntax:

```
operation ::= `seq.firmem.write_port` $memory `[` $address `]` `=` $data `,` `clock` $clk
              (`enable` $enable^)? (`mask` $mask^)?
              attr-dict `:` type($memory) (`,` type($mask)^)?
```

The `seq.firmem.write_port` op represents a write port on a `seq.firmem`
memory. It takes the memory as an operand, together with the address and
data to be written, the clock on which the write is synchronized, an
optional enable, and and optional write mask. Omitting the enable operand
has the same effect as passing a constant `true` to it. Omitting the write
mask operand has the same effect as passing an all-ones value to it. A write
mask operand can only be present if the `seq.firmem` specifies a mask width;
otherwise it must be omitted.

Traits: `AttrSizedOperandSegments`

Interfaces: `Clocked`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### Operands:

| Operand | Description |
| --- | --- |
| `memory` | A FIRRTL-flavored memory |
| `address` | signless integer |
| `clk` | A type for clock-carrying wires |
| `enable` | 1-bit signless integer |
| `data` | signless integer |
| `mask` | signless integer |

### `seq.firreg` (::circt::seq::FirRegOp)

*Register with preset and sync or async reset*

`firreg` represents registers originating from FIRRTL after the lowering
of the IR to HW. The register is used as an intermediary in the process
of lowering to SystemVerilog to facilitate optimisation at the HW level,
compactly representing a register with a single operation instead of
composing it from register definitions, always blocks and if statements.

The `data` output of the register accesses the value it stores. On the
rising edge of the `clk` input, the register takes a new value provided
by the `next` signal. Optionally, the register can also be provided with
a synchronous or an asynchronous `reset` signal and `resetValue`, as shown
in the example below.

```
%name = seq.firreg %next clock %clk [ sym @sym ]
    [ reset (sync|async) %reset, %value ]
    [ preset value ] : type
```

Implicitly, all registers are pre-set to a randomized value.

A register implementing a counter starting at 0 from reset can be defined
as follows:

```
%zero = hw.constant 0 : i32
%reg = seq.firreg %next clock %clk reset sync %reset, %zero : i32
%one = hw.constant 1 : i32
%next = comb.add %reg, %one : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `SameVariadicOperandSize`

Interfaces: `Clocked`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `InnerSymbolOpInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `Resettable`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource, MemoryEffects::Read on ::mlir::SideEffects::DefaultResource, MemoryEffects::Allocate on ::mlir::SideEffects::DefaultResource}`, `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `preset` | ::mlir::IntegerAttr | arbitrary integer attribute |
| `isAsync` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `next` | any type |
| `clk` | A type for clock-carrying wires |
| `reset` | 1-bit signless integer |
| `resetValue` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `data` | any type |

### `seq.from_clock` (::circt::seq::FromClockOp)

*Cast from a clock type to a wire type*

Syntax:

```
operation ::= `seq.from_clock` $input attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | A type for clock-carrying wires |

#### Results:

| Result | Description |
| --- | --- |
| `output` | 1-bit signless integer |

### `seq.from_immutable` (::circt::seq::FromImmutableOp)

*Cast from an immutable type to a wire type*

Syntax:

```
operation ::= `seq.from_immutable` $input attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an ImmutableType |

#### Results:

| Result | Description |
| --- | --- |
| `output` | any type |

### `seq.hlmem` (::circt::seq::HLMemOp)

*Instantiate a high-level memory.*

Syntax:

```
operation ::= `seq.hlmem` $name $clk `,` $rst attr-dict `:` type($handle)
```

See the Seq dialect rationale for a longer description

Interfaces: `Clocked`, `OpAsmOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `handle` | Multi-dimensional memory type |

### `seq.initial` (::circt::seq::InitialOp)

*Operation that produces values for initialization*

Syntax:

```
operation ::= `seq.initial` `(` $inputs `)` $body attr-dict `:` functional-type($inputs, results)
```

`seq.initial` op creates values wrapped types with !seq.immutable.
See the Seq dialect rationale for a longer description.

Traits: `IsolatedFromAbove`, `RecursiveMemoryEffects`, `RecursivelySpeculatableImplTrait`, `SingleBlockImplicitTerminator<YieldOp>`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of an ImmutableType |

#### Results:

| Result | Description |
| --- | --- |
| «unnamed» | variadic of an ImmutableType |

### `seq.read` (::circt::seq::ReadPortOp)

*Structural read access to a seq.hlmem, with an optional read enable signal.*

Traits: `AttrSizedOperandSegments`

Interfaces: `OpAsmOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `latency` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `memory` | Multi-dimensional memory type |
| `addresses` | variadic of a signless integer bitvector |
| `rdEn` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `readData` | any type |

### `seq.shiftreg` (::circt::seq::ShiftRegOp)

*Shift register*

Syntax:

```
operation ::= `seq.shiftreg` `[` $numElements `]`
              (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) $input `,` $clk `,` $clockEnable
              (`reset` $reset^ `,` $resetValue)?
              (`powerOn` $powerOnValue^)? attr-dict `:` type($data)
              custom<OptionalTypeMatch>(ref(type($data)), ref($resetValue), type($resetValue))
              custom<OptionalTypeMatch>(ref(type($data)), ref($powerOnValue), type($powerOnValue))
```

The `seq.shiftreg` op represents a shift register. It takes the input
value and shifts it every cycle when `clockEnable` is asserted.
The `reset` and `resetValue` operands are optional and if present, every
entry in the shift register will be initialized to `resetValue` upon
assertion of the reset signal. Exact reset behavior (sync/async) is
implementation defined.

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`

Interfaces: `Clocked`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `InnerSymbolOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `Resettable`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `numElements` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |
| `name` | ::mlir::StringAttr | string attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | any type |
| `clk` | A type for clock-carrying wires |
| `clockEnable` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `resetValue` | any type |
| `powerOnValue` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `data` | any type |

### `seq.to_clock` (::circt::seq::ToClockOp)

*Cast from a wire type to a clock type*

Syntax:

```
operation ::= `seq.to_clock` $input attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `output` | A type for clock-carrying wires |

### `seq.write` (::circt::seq::WritePortOp)

*Structural write access to a seq.hlmem*

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `latency` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `memory` | Multi-dimensional memory type |
| `addresses` | variadic of a signless integer bitvector |
| `inData` | any type |
| `wrEn` | 1-bit signless integer |

### `seq.yield` (::circt::seq::YieldOp)

*Yield values*

Syntax:

```
operation ::= `seq.yield` attr-dict ($operands^ `:` type($operands))?
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<InitialOp>`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `operands` | variadic of any type |

Attributes
----------

### ClockConstAttr

*Clock constant*

Syntax:

```
#seq.clock_constant<
  circt::seq::ClockConst   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `circt::seq::ClockConst` | an enum of type ClockConst |

### FirMemInitAttr

*Memory initialization information*

Syntax:

```
#seq.firmem.init<
  mlir::StringAttr,   # filename
  bool,   # isBinary
  bool   # isInline
>
```

This attribute captures what the initial contents of a memory should be.
At the moment this is modeled primarily with simulation in mind, where the
memory contents are pre-loaded from a file at simulation startup.

The `filename` specifies a file on disk that contains the initial contents
for this memory. If `isBinary` is set, the file is interpreted as a binary
file, otherwise it is treated as hexadecimal. This is modeled after the
`$readmemh` and `$readmemb` SystemVerilog functions. If `isInline` is set,
the initialization is emitted directly in the memory model; otherwise it is
split out into a separate module that can be bound in.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| filename | `mlir::StringAttr` |  |
| isBinary | `bool` |  |
| isInline | `bool` |  |

Type constraints
----------------

### an ImmutableType

Types
-----

### ClockType

*A type for clock-carrying wires*

Syntax: `!seq.clock`

The `!seq.clock` type represents signals which can be used to drive the
clock input of sequential operations.

### FirMemType

*A FIRRTL-flavored memory*

Syntax:

```
!seq.firmem<
  uint64_t,   # depth
  uint32_t,   # width
  std::optional<uint32_t>   # maskWidth
>
```

The `!seq.firmem` type represents a FIRRTL-flavored memory declared by a
`seq.firmem` operation. It captures the parameters of the memory that are
relevant to the read, write, and read-write ports, such as width and depth.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| depth | `uint64_t` |  |
| width | `uint32_t` |  |
| maskWidth | `std::optional<uint32_t>` |  |

### HLMemType

*Multi-dimensional memory type*

Syntax:

```
hlmem-type ::== `hlmem` `<` dim-list element-type `>`
```

The HLMemType represents the type of an addressable memory structure. The
type is inherently multidimensional. Dimensions must be known integer values.

Note: unidimensional memories are represented as <1x{element type}> -
<{element type}> is illegal.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| shape | `::llvm::ArrayRef<int64_t>` |  |
| elementType | `Type` |  |

### ImmutableType

*Value type that is immutable after initialization*

Syntax:

```
!seq.immutable<
  ::mlir::Type   # innerType
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| innerType | `::mlir::Type` |  |

Enums
-----

### ClockConst

*Clock constant*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| Low | `0` | low |
| High | `1` | high |

### RUW

*Read-Under-Write Behavior*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| Undefined | `0` | undefined |
| Old | `1` | old |
| New | `2` | new |

### WUW

*Write-Under-Write Behavior*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| Undefined | `0` | undefined |
| PortOrder | `1` | port\_order |

'seq' Dialect Docs
------------------

* [Seq(uential) Dialect Rationale](https://circt.llvm.org/docs/Dialects/Seq/RationaleSeq/)

 [Prev - 'rtgtest' Dialect](https://circt.llvm.org/docs/Dialects/RTGTest/ "'rtgtest' Dialect")
[Next - Seq(uential) Dialect Rationale](https://circt.llvm.org/docs/Dialects/Seq/RationaleSeq/ "Seq(uential) Dialect Rationale") 

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