LLHD Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

LLHD Dialect
============

This dialect provides operations and types to interact with an event queue in an event-based simulation.
It describes how signals change over time in reaction to changes in other signals and physical time advancing.
Established hardware description languages such as SystemVerilog and VHDL use an event queue as their programming model to describe combinational and sequential logic, as well as test harnesses and test benches.

* [Rationale](#rationale)
  + [Register Reset Values](#register-reset-values)
* [Types](#types)
  + [TimeType](#timetype)
  + [RefType](#reftype)
* [Attributes](#attributes)
  + [InnerSymAttr](#innersymattr)
  + [InnerSymPropertiesAttr](#innersympropertiesattr)
  + [TimeAttr](#timeattr)
* [Operations](#operations)
  + [`llhd.combinational` (::circt::llhd::CombinationalOp)](#llhdcombinational-circtllhdcombinationalop)
  + [`llhd.constant_time` (::circt::llhd::ConstantTimeOp)](#llhdconstant_time-circtllhdconstanttimeop)
  + [`llhd.current_time` (::circt::llhd::CurrentTimeOp)](#llhdcurrent_time-circtllhdcurrenttimeop)
  + [`llhd.delay` (::circt::llhd::DelayOp)](#llhddelay-circtllhddelayop)
  + [`llhd.drv` (::circt::llhd::DriveOp)](#llhddrv-circtllhddriveop)
  + [`llhd.final` (::circt::llhd::FinalOp)](#llhdfinal-circtllhdfinalop)
  + [`llhd.get_global_signal` (::circt::llhd::GetGlobalSignalOp)](#llhdget_global_signal-circtllhdgetglobalsignalop)
  + [`llhd.global_signal` (::circt::llhd::GlobalSignalOp)](#llhdglobal_signal-circtllhdglobalsignalop)
  + [`llhd.halt` (::circt::llhd::HaltOp)](#llhdhalt-circtllhdhaltop)
  + [`llhd.int_to_time` (::circt::llhd::IntToTimeOp)](#llhdint_to_time-circtllhdinttotimeop)
  + [`llhd.output` (::circt::llhd::OutputOp)](#llhdoutput-circtllhdoutputop)
  + [`llhd.prb` (::circt::llhd::ProbeOp)](#llhdprb-circtllhdprobeop)
  + [`llhd.process` (::circt::llhd::ProcessOp)](#llhdprocess-circtllhdprocessop)
  + [`llhd.sig.array_get` (::circt::llhd::SigArrayGetOp)](#llhdsigarray_get-circtllhdsigarraygetop)
  + [`llhd.sig.array_slice` (::circt::llhd::SigArraySliceOp)](#llhdsigarray_slice-circtllhdsigarraysliceop)
  + [`llhd.sig.extract` (::circt::llhd::SigExtractOp)](#llhdsigextract-circtllhdsigextractop)
  + [`llhd.sig.struct_extract` (::circt::llhd::SigStructExtractOp)](#llhdsigstruct_extract-circtllhdsigstructextractop)
  + [`llhd.sig` (::circt::llhd::SignalOp)](#llhdsig-circtllhdsignalop)
  + [`llhd.time_to_int` (::circt::llhd::TimeToIntOp)](#llhdtime_to_int-circtllhdtimetointop)
  + [`llhd.wait` (::circt::llhd::WaitOp)](#llhdwait-circtllhdwaitop)
  + [`llhd.yield` (::circt::llhd::YieldOp)](#llhdyield-circtllhdyieldop)
* [Passes](#passes)
  + [`-llhd-combine-drives`](#-llhd-combine-drives)
  + [`-llhd-deseq`](#-llhd-deseq)
  + [`-llhd-hoist-signals`](#-llhd-hoist-signals)
  + [`-llhd-inline-calls`](#-llhd-inline-calls)
  + [`-llhd-lower-processes`](#-llhd-lower-processes)
  + [`-llhd-mem2reg`](#-llhd-mem2reg)
  + [`-llhd-remove-control-flow`](#-llhd-remove-control-flow)
  + [`-llhd-sig2reg`](#-llhd-sig2reg)
  + [`-llhd-unroll-loops`](#-llhd-unroll-loops)
  + [`-llhd-wrap-procedural-ops`](#-llhd-wrap-procedural-ops)

Rationale [¶](#rationale)
-------------------------

### Register Reset Values [¶](#register-reset-values)

Resets are problematic since Verilog forces designers to describe them as edge-sensitive triggers.
This does *not* match the async resets found on almost all standard cell flip-flops, which are level-sensitive.
Therefore a pass lowering from Verilog-style processes to a structural register such as `seq.compreg` would have to verify that the reset value is a constant in order to make the mapping from edge-sensitive Verilog description to level-sensitive standard cell valid.

In practice, designers commonly implement registers encapsulated in a Verilog module, with the reset value being provided as a module input
port.
This makes determining whether the input is a constant much more difficult.
Most commercial tools relax this constraint and simply map the edge-sensitive reset to a level-sensitive one.
This does not preserve the semantics of the input, which is bad.
Most synthesis tools will then go ahead and fail during synthesis if a register’s reset value does not end up being a constant value.

Therefore the Deseq pass does not verify that a register’s reset value is a constant.
Instead, it applies the same transform from edge-sensitive to level-sensitive reset as most other tools.

Types [¶](#types)
-----------------

### TimeType [¶](#timetype)

*Time type*

Syntax: `!llhd.time`

Represents a simulation time value as a combination of a real time value in
seconds (or any smaller SI time unit), a delta value representing
infinitesimal time steps, and an epsilon value representing an absolute time
slot within a delta step (used to model SystemVerilog scheduling regions).

### RefType [¶](#reftype)

*A reference to a signal or variable*

Syntax:

```
!llhd.ref<
  Type   # nestedType
>
```

Represents a reference to a value. Signals return a reference to the
underlying value, which allows them to be probed or driven.

#### Parameters: [¶](#parameters)

| Parameter | C++ type | Description |
| --- | --- | --- |
| nestedType | `Type` |  |

Attributes [¶](#attributes)
---------------------------

### InnerSymAttr [¶](#innersymattr)

*Inner symbol definition*

Defines the properties of an inner\_sym attribute. It specifies the symbol
name and symbol visibility for each field ID. For any ground types,
there are no subfields and the field ID is 0. For aggregate types, a
unique field ID is assigned to each field by visiting them in a
depth-first pre-order. The custom assembly format ensures that for ground
types, only `@<sym_name>` is printed.

#### Parameters: [¶](#parameters-1)

| Parameter | C++ type | Description |
| --- | --- | --- |
| props | `::llvm::ArrayRef<InnerSymPropertiesAttr>` |  |

### InnerSymPropertiesAttr [¶](#innersympropertiesattr)

#### Parameters: [¶](#parameters-2)

| Parameter | C++ type | Description |
| --- | --- | --- |
| name | `::mlir::StringAttr` |  |
| fieldID | `uint64_t` |  |
| sym\_visibility | `::mlir::StringAttr` |  |

### TimeAttr [¶](#timeattr)

*Time attribute*

Represents a value of the LLHD time type.

Example: `#llhd.time<0ns, 1d, 0e>`

#### Parameters: [¶](#parameters-3)

| Parameter | C++ type | Description |
| --- | --- | --- |
| type | `llhd::TimeType` |  |
| time | `uint64_t` |  |
| timeUnit | `::llvm::StringRef` | SI time unit |
| delta | `unsigned` |  |
| epsilon | `unsigned` |  |

Operations [¶](#operations)
---------------------------

### `llhd.combinational` (::circt::llhd::CombinationalOp) [¶](#llhdcombinational-circtllhdcombinationalop)

*A process that runs when any of its operand values change*

Syntax:

```
operation ::= `llhd.combinational` (`->` type($results)^)?
              attr-dict-with-keyword $body
```

An `llhd.combinational` op encapsulates a region of IR that executes once at
the beginning of the simulation, and subsequently whenever any of the values
used in its body change. Control flow must eventually end in an `llhd.yield`
terminator. The process may have results, in which case the `llhd.yield`
terminators must provide a list of values to yield for the process results.
Whenever any of the values used in the body change, the process reexecutes
in order to compute updated results.

This op is commonly used to embed a control-flow description of some
combinational logic inside the surrounding module’s graph region.

Example:

```
hw.module @Foo() {
  %0, %1 = llhd.combinational -> i42, i9001 {
    cf.cond_br %2, ^bb1(%3, %4 : i42, i9001), ^bb1(%5, %6 : i42, i9001)
  ^bb1(%7: i42, %8: i9001):
    llhd.yield %7, %8 : i42, i9001
  }
}
```

Traits: `HasParent<hw::HWModuleOp>`, `NoRegionArguments`, `ProceduralRegion`, `RecursiveMemoryEffects`

#### Results: [¶](#results)

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `llhd.constant_time` (::circt::llhd::ConstantTimeOp) [¶](#llhdconstant_time-circtllhdconstanttimeop)

*Introduce a new time constant.*

Syntax:

```
operation ::= `llhd.constant_time` $value attr-dict
```

The `llhd.constant_time` instruction introduces a new constant time value as
an SSA-operator.

Example:

```
%1 = llhd.constant_time #llhd.time<1ns, 2d, 3d>
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | llhd::TimeAttr | time attribute |

#### Results: [¶](#results-1)

| Result | Description |
| --- | --- |
| `result` | time type |

### `llhd.current_time` (::circt::llhd::CurrentTimeOp) [¶](#llhdcurrent_time-circtllhdcurrenttimeop)

*Get the current simulation time*

Syntax:

```
operation ::= `llhd.current_time` attr-dict
```

Materializes the current simulation time as an SSA value. This is equivalent
to the `$time`, `$stime`, and `$realtime` system tasks in SystemVerilog, and
the `now` keyword in VHDL.

This operation has a memory read side effect to avoid motion and CSE across
`llhd.wait` operations, and other operations that may suspend execution.

Interfaces: `InferTypeOpInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource}`

#### Results: [¶](#results-2)

| Result | Description |
| --- | --- |
| `result` | time type |

### `llhd.delay` (::circt::llhd::DelayOp) [¶](#llhddelay-circtllhddelayop)

*Specifies value propagation delay*

Syntax:

```
operation ::= `llhd.delay` $input `by` $delay attr-dict `:` type($result)
```

This operation propagates all value changes of the input to the output after
the specified time delay.
Reference values are not supported (e.g., pointers, inout, etc.)
since the store-like operation used for those types should encode a delayed
store.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-2)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `delay` | llhd::TimeAttr | time attribute |

#### Operands: [¶](#operands)

| Operand | Description |
| --- | --- |
| `input` | a type without inout |

#### Results: [¶](#results-3)

| Result | Description |
| --- | --- |
| `result` | a type without inout |

### `llhd.drv` (::circt::llhd::DriveOp) [¶](#llhddrv-circtllhddriveop)

*Drive a value into a signal.*

Syntax:

```
operation ::= `llhd.drv` $signal `,` $value `after` $time ( `if` $enable^ )? attr-dict `:`
              type($value)
```

The `llhd.drv` operation drives a new value onto a signal. A time
operand also has to be passed, which specifies the frequency at which
the drive will be performed. An optional enable value can be passed as
last argument. In this case the drive will only be performed if the
value is 1. In case no enable signal is passed the drive will always be
performed. This operation does not define any new SSA operands.

Example:

```
%true = hw.constant true
%false = hw.constant false
%time = llhd.constant_time <1ns, 0d, 0e>
%sig = llhd.sig %true : i1

llhd.drv %sig, %false after %time : i1
llhd.drv %sig, %false after %time if %true : i1
```

Interfaces: `DestructurableAccessorOpInterface`, `SafeMemorySlotAccessOpInterface`

#### Operands: [¶](#operands-1)

| Operand | Description |
| --- | --- |
| `signal` | a reference to a signal or variable |
| `value` | any type |
| `time` | time type |
| `enable` | 1-bit signless integer |

### `llhd.final` (::circt::llhd::FinalOp) [¶](#llhdfinal-circtllhdfinalop)

*A process that runs at the end of simulation*

Syntax:

```
operation ::= `llhd.final` attr-dict-with-keyword $body
```

An `llhd.final` op encapsulates a region of IR that is to be executed after
the last time step of a simulation has completed. This can be used to
implement various forms of state cleanup and tear-down. Some verifications
ops may also want to check that certain final conditions hold at the end of
a simulation run.

The `llhd.wait` terminator is not allowed in `llhd.final` processes since
there is no later time slot for the execution to resume. Control flow must
eventually end in an `llhd.halt` terminator.

Execution order between multiple `llhd.final` ops is undefined.

Example:

```
hw.module @Foo() {
  llhd.final {
    func.call @printSimulationStatistics() : () -> ()
    llhd.halt
  }
}
```

Traits: `HasParent<hw::HWModuleOp>`, `NoRegionArguments`, `ProceduralRegion`, `RecursiveMemoryEffects`

### `llhd.get_global_signal` (::circt::llhd::GetGlobalSignalOp) [¶](#llhdget_global_signal-circtllhdgetglobalsignalop)

*Get a reference to a global signal*

Syntax:

```
operation ::= `llhd.get_global_signal` $global_name attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-3)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `global_name` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Results: [¶](#results-4)

| Result | Description |
| --- | --- |
| `result` | a reference to a signal or variable |

### `llhd.global_signal` (::circt::llhd::GlobalSignalOp) [¶](#llhdglobal_signal-circtllhdglobalsignalop)

*A global signal declaration*

Syntax:

```
operation ::= `llhd.global_signal` $sym_name attr-dict `:` $type (`init` $initRegion^)?
```

Define a global signal identified by a symbol name. The corresponding
`llhd.get_global_signal` operation can be used to get a `!llhd.ref` to it.
Global signals behave just like normal signals.

Traits: `IsolatedFromAbove`, `NoRegionArguments`, `SingleBlock`

Interfaces: `Symbol`

#### Attributes: [¶](#attributes-4)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `type` | ::mlir::TypeAttr | type attribute of any type |

### `llhd.halt` (::circt::llhd::HaltOp) [¶](#llhdhalt-circtllhdhaltop)

*Terminate execution of a process*

Syntax:

```
operation ::= `llhd.halt` ($yieldOperands^ `:` type($yieldOperands))?
              attr-dict
```

The `llhd.halt` terminator suspends execution of the parent process forever,
effectively terminating it. The `yieldOperands` are yielded as the result
values of the parent process.

Example:

```
llhd.halt
llhd.halt %0, %1 : i42, i9001
```

Traits: `HasParent<ProcessOp, FinalOp>`, `Terminator`

#### Operands: [¶](#operands-2)

| Operand | Description |
| --- | --- |
| `yieldOperands` | variadic of any type |

### `llhd.int_to_time` (::circt::llhd::IntToTimeOp) [¶](#llhdint_to_time-circtllhdinttotimeop)

*Convert an integer number of femtoseconds to a time*

Syntax:

```
operation ::= `llhd.int_to_time` $input attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-3)

| Operand | Description |
| --- | --- |
| `input` | 64-bit signless integer |

#### Results: [¶](#results-5)

| Result | Description |
| --- | --- |
| `result` | time type |

### `llhd.output` (::circt::llhd::OutputOp) [¶](#llhdoutput-circtllhdoutputop)

*Introduce a new signal and drive a value onto it.*

Syntax:

```
operation ::= `llhd.output` ( $name^ )? $value `after` $time attr-dict `:` type($value)
```

The `llhd.output` operation introduces a new signal and continuously
drives a the given value onto it after a given time-delay. The same
value is used to initialize the signal in the same way as the ‘init’
value in `llhd.sig`. An optional name can be given to the created signal.
This shows up, e.g., in the simulation trace.

Example:

```
%value = hw.constant true
%time = llhd.constant_time <1ns, 0d, 0e>
%sig = llhd.output "sigName" %value after %time : i1

// is equivalent to

%value = hw.constant true
%time = llhd.constant_time <1ns, 0d, 0e>
%sig = llhd.sig "sigName" %value : i1
llhd.drv %sig, %value after %time : i1
```

Interfaces: `InferTypeOpInterface`

#### Attributes: [¶](#attributes-5)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-4)

| Operand | Description |
| --- | --- |
| `value` | any type |
| `time` | time type |

#### Results: [¶](#results-6)

| Result | Description |
| --- | --- |
| `result` | a reference to a signal or variable |

### `llhd.prb` (::circt::llhd::ProbeOp) [¶](#llhdprb-circtllhdprobeop)

*Probe a signal.*

Syntax:

```
operation ::= `llhd.prb` $signal attr-dict `:` type($result)
```

This operation probes a signal and returns the value it
currently carries as a new SSA operand. The result type is always
the type carried by the signal. In SSACFG regions, the operation has a read
side effect on the signal operand. In graph regions, the operation is
memory-effect free.

Example:

```
%true = hw.constant true
%sig_i1 = llhd.sig %true : i1
%prbd = llhd.prb %sig_i1 : i1
```

Interfaces: `DestructurableAccessorOpInterface`, `InferTypeOpInterface`, `MemoryEffectOpInterface`, `SafeMemorySlotAccessOpInterface`

#### Operands: [¶](#operands-5)

| Operand | Description |
| --- | --- |
| `signal` | a reference to a signal or variable |

#### Results: [¶](#results-7)

| Result | Description |
| --- | --- |
| `result` | any type |

### `llhd.process` (::circt::llhd::ProcessOp) [¶](#llhdprocess-circtllhdprocessop)

*A process that runs concurrently during simulation*

Syntax:

```
operation ::= `llhd.process` (`->` type($results)^)?
              attr-dict-with-keyword $body
```

An `llhd.process` op encapsulates a region of IR that executes concurrently
during simulation. Execution can be suspended using the `llhd.wait`
terminator, which also includes a list of values that will cause the process
execution to resume whenever they change. The `llhd.halt` terminator can be
used to suspend execution forever. The process may have results, in which
case any `llhd.wait` or `llhd.halt` terminators must provide a list of
values to yield for the process results whenever execution is suspended. The
process holds these result values until it is resumed and new result values
are yielded.

Example:

```
hw.module @top() {
  %0, %1 = llhd.process -> i42, i9001 {
    llhd.wait yield (%2, %3 : i42, i9001), ^bb1
  ^bb1:
    llhd.halt %4, %5 : i42, i9001
  }
}
```

Traits: `HasParent<hw::HWModuleOp>`, `NoRegionArguments`, `ProceduralRegion`, `RecursiveMemoryEffects`

#### Results: [¶](#results-8)

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `llhd.sig.array_get` (::circt::llhd::SigArrayGetOp) [¶](#llhdsigarray_get-circtllhdsigarraygetop)

*Extract an element from a signal of an array.*

Syntax:

```
operation ::= `llhd.sig.array_get` $input `[` $index `]` attr-dict `:` type($input)
```

The `llhd.sig.array_get` operation allows to access the element of the
`$input` operand at position `$index`. A new subsignal aliasing the element
will be returned.

Example:

```
// Returns a !llhd.ref<i8>
%0 = llhd.sig.array_get %arr[%index] : !llhd.ref<!hw.array<4xi8>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `DestructurableAccessorOpInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SafeMemorySlotAccessOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-6)

| Operand | Description |
| --- | --- |
| `input` | ref of an ArrayType |
| `index` | a signless integer bitvector |

#### Results: [¶](#results-9)

| Result | Description |
| --- | --- |
| `result` | a reference to a signal or variable |

### `llhd.sig.array_slice` (::circt::llhd::SigArraySliceOp) [¶](#llhdsigarray_slice-circtllhdsigarraysliceop)

*Get a range of consecutive values from a signal of an array*

Syntax:

```
operation ::= `llhd.sig.array_slice` $input `at` $lowIndex attr-dict `:` type($input) `->` type($result)
```

The `llhd.sig.array_slice` operation allows to access a sub-range of the
`$input` operand, starting at the index given by the `$lowIndex`
operand. The resulting slice length is defined by the result type.
Returns a signal aliasing the elements of the slice.

Width of ’lowIndex’ is defined to be the precise number of bits required to
index the ‘input’ array. More precisely: for an input array of size M,
the width of ’lowIndex’ is ceil(log2(M)). Lower and upper bound indexes
which are larger than the size of the ‘input’ array results in undefined
behavior.

Example:

```
%3 = llhd.sig.array_slice %input at %lowIndex :
  (!llhd.ref<!hw.array<4xi8>>) -> !llhd.ref<!hw.array<2xi8>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-7)

| Operand | Description |
| --- | --- |
| `input` | ref of an ArrayType |
| `lowIndex` | a signless integer bitvector |

#### Results: [¶](#results-10)

| Result | Description |
| --- | --- |
| `result` | ref of an ArrayType |

### `llhd.sig.extract` (::circt::llhd::SigExtractOp) [¶](#llhdsigextract-circtllhdsigextractop)

*Extract a range of bits from an integer signal*

Syntax:

```
operation ::= `llhd.sig.extract` $input `from` $lowBit attr-dict `:` type($input) `->` type($result)
```

The `llhd.sig.extract` operation allows to access a range of bits
of the `$input` operand, starting at the index given by the `$lowBit`
operand. The result length is defined by the result type.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `DestructurableAccessorOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SafeMemorySlotAccessOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-8)

| Operand | Description |
| --- | --- |
| `input` | ref of a signless integer bitvector |
| `lowBit` | a signless integer bitvector |

#### Results: [¶](#results-11)

| Result | Description |
| --- | --- |
| `result` | ref of a signless integer bitvector |

### `llhd.sig.struct_extract` (::circt::llhd::SigStructExtractOp) [¶](#llhdsigstruct_extract-circtllhdsigstructextractop)

*Extract a field from a signal of a struct.*

Syntax:

```
operation ::= `llhd.sig.struct_extract` $input `[` $field `]` attr-dict `:` type($input)
```

The `llhd.sig.struct_extract` operation allows access to the field of the
`$input` operand given by its name via the `$field` attribute.
A new subsignal aliasing the field will be returned.

Example:

```
// Returns a !llhd.ref<i8>
%0 = llhd.sig.struct_extract %struct["foo"]
  : !llhd.ref<!hw.struct<foo: i8, bar: i16>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `DestructurableAccessorOpInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SafeMemorySlotAccessOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-6)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `field` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-9)

| Operand | Description |
| --- | --- |
| `input` | ref of a StructType |

#### Results: [¶](#results-12)

| Result | Description |
| --- | --- |
| `result` | ref of any type |

### `llhd.sig` (::circt::llhd::SignalOp) [¶](#llhdsig-circtllhdsignalop)

*Create a signal.*

Syntax:

```
operation ::= `llhd.sig` `` custom<ImplicitSSAName>($name) $init attr-dict
              `:` type($init)
```

The `llhd.sig` instruction introduces a new signal in the IR. The input
operand determines the initial value carried by the signal, while the
result type will always be a signal carrying the type of the init operand.
A signal defines a unique name within the entity it resides in.

Example:

```
%c123_i64 = hw.constant 123 : i64
%foo = llhd.sig %c123_i64 : i64
%0 = llhd.sig name "foo" %c123_i64 : i64
```

This example creates a new signal named “foo”, carrying an `i64` type with
initial value of 123.

Interfaces: `DestructurableAllocationOpInterface`, `InferTypeOpInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-7)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-10)

| Operand | Description |
| --- | --- |
| `init` | any type |

#### Results: [¶](#results-13)

| Result | Description |
| --- | --- |
| `result` | a reference to a signal or variable |

### `llhd.time_to_int` (::circt::llhd::TimeToIntOp) [¶](#llhdtime_to_int-circtllhdtimetointop)

*Convert a time to an integer number of femtoseconds*

Syntax:

```
operation ::= `llhd.time_to_int` $input attr-dict
```

If the time value converted to femtoseconds does not fit into an `i64`, the
result is undefined.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-11)

| Operand | Description |
| --- | --- |
| `input` | time type |

#### Results: [¶](#results-14)

| Result | Description |
| --- | --- |
| `result` | 64-bit signless integer |

### `llhd.wait` (::circt::llhd::WaitOp) [¶](#llhdwait-circtllhdwaitop)

*Suspend execution of a process*

Syntax:

```
operation ::= `llhd.wait` (`yield` ` ` `(` $yieldOperands^ `:` type($yieldOperands) `)` `,`)?
              (`delay` $delay^ `,`)?
              (`(`$observed^ `:` qualified(type($observed))`)` `,`)?
              $dest (`(` $destOperands^ `:` qualified(type($destOperands)) `)`)?
              attr-dict
```

The `llhd.wait` terminator suspends execution of the parent process until
any of the `observed` values change or a fixed `delay` has passed. Execution
resumes at the `dest` block with the `destOperands` arguments. The
`yieldOperands` are yielded as the result values of the parent process.

Example:

```
llhd.wait ^bb1(%0, %1 : i42, i9001)
llhd.wait yield (%0, %1 : i42, i9001), ^bb1
llhd.wait delay %time, ^bb1
llhd.wait (%0, %1 : i42, i9001), ^bb1
```

Traits: `AttrSizedOperandSegments`, `HasParent<ProcessOp>`, `Terminator`

#### Operands: [¶](#operands-12)

| Operand | Description |
| --- | --- |
| `yieldOperands` | variadic of any type |
| `delay` | time type |
| `observed` | variadic of a known primitive element |
| `destOperands` | variadic of any type |

#### Successors: [¶](#successors)

| Successor | Description |
| --- | --- |
| `dest` | any successor |

### `llhd.yield` (::circt::llhd::YieldOp) [¶](#llhdyield-circtllhdyieldop)

*Yield results back from a combinational process*

Syntax:

```
operation ::= `llhd.yield` ($yieldOperands^ `:` type($yieldOperands))?
              attr-dict
```

The `llhd.yield` terminator terminates control flow in the parent process
and yields the `yieldOperands` as the result values of the process.

Example:

```
llhd.combinational {
  llhd.yield
}
%2:2 = llhd.combinational -> i42, i9001 {
  llhd.yield %0, %1 : i42, i9001
}
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<CombinationalOp, GlobalSignalOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-13)

| Operand | Description |
| --- | --- |
| `yieldOperands` | variadic of any type |

Passes [¶](#passes)
-------------------

### `-llhd-combine-drives` [¶](#-llhd-combine-drives)

*Combine scalar drives into aggregate drives*

If individual drives cover all of an aggregate signal’s fields, merge them
into a single drive of the whole aggregate value.

### `-llhd-deseq` [¶](#-llhd-deseq)

*Convert sequential processes to registers*

### `-llhd-hoist-signals` [¶](#-llhd-hoist-signals)

*Hoist probes and promote drives to process results*

### `-llhd-inline-calls` [¶](#-llhd-inline-calls)

*Inline all function calls in HW modules*

Inlines all `func.call` operations nested within `llhd.combinational`,
`llhd.process`, and `llhd.final` ops within `hw.module`s. The `func.func`
definitions are left untouched. After the inlining, MLIR’s symbol DCE pass
may be used to eliminate the function definitions where possible.

This pass expects the `llhd-wrap-procedural-ops` pass to have already run.
Otherwise call ops immediately in the module body, a graph region, cannot be
inlined at all since the graph region cannot accommodate the function’s
control flow.

#### Statistics [¶](#statistics)

```
calls-inlined : Number of call ops that were inlined
```

### `-llhd-lower-processes` [¶](#-llhd-lower-processes)

*Convert process ops to combinational ops where possible*

### `-llhd-mem2reg` [¶](#-llhd-mem2reg)

*Promotes memory and signal slots into values.*

### `-llhd-remove-control-flow` [¶](#-llhd-remove-control-flow)

*Remove acyclic control flow and replace block args with muxes*

Remove the control flow in `llhd.combinational` operations by merging all
blocks into the entry block and replacing block arguments with multiplexers.
This requires the control flow to be acyclic, for example by unrolling all
loops beforehand. Additionally, since this moves operations from
conditionally executed blocks into the unconditionally executed entry block,
all operations must be side-effect free.

### `-llhd-sig2reg` [¶](#-llhd-sig2reg)

*Promote LLHD signals to SSA values*

### `-llhd-unroll-loops` [¶](#-llhd-unroll-loops)

*Unroll control flow loops with static bounds*

Unroll loops in `llhd.combinational` operations by replicating the loop body
and replacing induction variables with constants. The loop bounds must be
known at compile time.

### `-llhd-wrap-procedural-ops` [¶](#-llhd-wrap-procedural-ops)

*Wrap procedural ops in modules to make them inlinable*

Operations such as `func.call` or `scf.if` may appear in an `hw.module` body
directly. They cannot be inlined into the module though, since the inlined
function or result of converting the SCF ops to the CF dialect may create
control-flow operations with multiple blocks. This pass wraps such
operations in `llhd.combinational` ops to give them an SSACFG region to
inline into.

#### Statistics [¶](#statistics-1)

```
ops-wrapped : Number of procedural ops wrapped
```

 [Prev - Interoperability Dialect Rationale](https://circt.llvm.org/docs/Dialects/Interop/RationaleInterop/ "Interoperability Dialect Rationale")
[Next - LTL Dialect](https://circt.llvm.org/docs/Dialects/LTL/ "LTL Dialect") 

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