'handshake' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'handshake' Dialect
===================

*Types and operations for the handshake dialect*

This dialect defined the `handshake` dialect, modeling dataflow circuits.
Handshake/dataflow IR is describes independent, unsynchronized processes
communicating data through First-in First-out (FIFO) communication channels.

* [Operations](#operations)
  + [`handshake.br` (::circt::handshake::BranchOp)](#handshakebr-circthandshakebranchop)
  + [`handshake.buffer` (::circt::handshake::BufferOp)](#handshakebuffer-circthandshakebufferop)
  + [`handshake.cond_br` (::circt::handshake::ConditionalBranchOp)](#handshakecond_br-circthandshakeconditionalbranchop)
  + [`handshake.constant` (::circt::handshake::ConstantOp)](#handshakeconstant-circthandshakeconstantop)
  + [`handshake.control_merge` (::circt::handshake::ControlMergeOp)](#handshakecontrol_merge-circthandshakecontrolmergeop)
  + [`handshake.esi_instance` (::circt::handshake::ESIInstanceOp)](#handshakeesi_instance-circthandshakeesiinstanceop)
  + [`handshake.extmemory` (::circt::handshake::ExternalMemoryOp)](#handshakeextmemory-circthandshakeexternalmemoryop)
  + [`handshake.fork` (::circt::handshake::ForkOp)](#handshakefork-circthandshakeforkop)
  + [`handshake.func` (::circt::handshake::FuncOp)](#handshakefunc-circthandshakefuncop)
  + [`handshake.instance` (::circt::handshake::InstanceOp)](#handshakeinstance-circthandshakeinstanceop)
  + [`handshake.join` (::circt::handshake::JoinOp)](#handshakejoin-circthandshakejoinop)
  + [`handshake.lazy_fork` (::circt::handshake::LazyForkOp)](#handshakelazy_fork-circthandshakelazyforkop)
  + [`handshake.load` (::circt::handshake::LoadOp)](#handshakeload-circthandshakeloadop)
  + [`handshake.memory` (::circt::handshake::MemoryOp)](#handshakememory-circthandshakememoryop)
  + [`handshake.merge` (::circt::handshake::MergeOp)](#handshakemerge-circthandshakemergeop)
  + [`handshake.mux` (::circt::handshake::MuxOp)](#handshakemux-circthandshakemuxop)
  + [`handshake.never` (::circt::handshake::NeverOp)](#handshakenever-circthandshakeneverop)
  + [`handshake.pack` (::circt::handshake::PackOp)](#handshakepack-circthandshakepackop)
  + [`handshake.return` (::circt::handshake::ReturnOp)](#handshakereturn-circthandshakereturnop)
  + [`handshake.sink` (::circt::handshake::SinkOp)](#handshakesink-circthandshakesinkop)
  + [`handshake.source` (::circt::handshake::SourceOp)](#handshakesource-circthandshakesourceop)
  + [`handshake.store` (::circt::handshake::StoreOp)](#handshakestore-circthandshakestoreop)
  + [`handshake.sync` (::circt::handshake::SyncOp)](#handshakesync-circthandshakesyncop)
  + [`handshake.unpack` (::circt::handshake::UnpackOp)](#handshakeunpack-circthandshakeunpackop)
* [Attributes](#attributes-7)
  + [BufferTypeEnumAttr](#buffertypeenumattr)
* [Enums](#enums)
  + [BufferTypeEnum](#buffertypeenum)

Operations
----------

### `handshake.br` (::circt::handshake::BranchOp)

*Branch operation*

The branch operation represents an unconditional
branch. The single data input is propagated to the single
successor. The input must be triggered by some predecessor to
avoid continous triggering of a successor block.

Example:

```
%1 = br %0 : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `ControlInterface`, `ExecutableOpInterface`, `GeneralOpInterface`, `InferTypeOpInterface`, `NamedIOInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SOSTInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `dataOperand` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `dataResult` | any type |

### `handshake.buffer` (::circt::handshake::BufferOp)

*Buffer operation*

The buffer operation represents a buffer operation. $slots
must be an unsigned integer larger than 0. $bufferType=BufferTypeEnum::seq indicates a
nontransparent buffer, while $bufferType=BufferTypeEnum::fifo indicates a transparent
buffer.

An ‘initValues’ attribute containing a list of integer values may be provided.
The list must be of the same length as the number of slots. This will
initialize the buffer with the given values upon reset.
For now, only sequential buffers are allowed to have initial values.
@todo: How to support different init types? these have to be stored (and
retrieved) as attributes, hence they must be of a known type.

Traits: `AlwaysSpeculatableImplTrait`, `HasClock`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `ControlInterface`, `ExecutableOpInterface`, `GeneralOpInterface`, `InferTypeOpInterface`, `NamedIOInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SOSTInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `slots` | ::mlir::IntegerAttr | 32-bit signless integer attribute whose minimum value is 1 |
| `bufferType` | ::circt::handshake::BufferTypeEnumAttr | BufferOp seq or fifo |
| `initValues` | ::mlir::ArrayAttr | 64-bit integer array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `operand` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `handshake.cond_br` (::circt::handshake::ConditionalBranchOp)

*Conditional branch operation*

The cbranch operation represents a conditional
branch. The data input is propagated to one of the two outputs
based on the condition input.

Example:

```
%true, %false = conditional_branch %cond, %data : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ConditionallySpeculatable`, `ControlInterface`, `ExecutableOpInterface`, `InferTypeOpInterface`, `NamedIOInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `conditionOperand` | 1-bit signless integer |
| `dataOperand` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `trueResult` | any type |
| `falseResult` | any type |

### `handshake.constant` (::circt::handshake::ConstantOp)

*Constant operation*

Syntax:

```
operation ::= `handshake.constant` $ctrl attr-dict `:` qualified(type($result))
```

The const has a constant value. When triggered by its
single `ctrl` input, it sends the constant value to its single
successor.

Example:

```
%0 = constant %ctrl {value = 42 : i32} : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ConditionallySpeculatable`, `ControlInterface`, `ExecutableOpInterface`, `GeneralOpInterface`, `NamedIOInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::TypedAttr | TypedAttr instance |

#### Operands:

| Operand | Description |
| --- | --- |
| `ctrl` | none type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `handshake.control_merge` (::circt::handshake::ControlMergeOp)

*Control merge operation*

The control\_merge operation represents a
(nondeterministic) control merge. Any input is propagated to the
first output and the index of the propagated input is sent to the
second output. The number of inputs corresponds to the number of
predecessor blocks.

Example:

```
%0, %idx = control_merge %a, %b, %c {attributes} : i32, index
```

Traits: `AlwaysSpeculatableImplTrait`, `HasClock`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ConditionallySpeculatable`, `ControlInterface`, `ExecutableOpInterface`, `MergeLikeOpInterface`, `NamedIOInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SOSTInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `dataOperands` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |
| `index` | any type |

### `handshake.esi_instance` (::circt::handshake::ESIInstanceOp)

*Instantiate a Handshake circuit*

Syntax:

```
operation ::= `handshake.esi_instance` $module $instName `clk` $clk `rst` $rst
              `(` $opOperands `)` attr-dict `:` functional-type($opOperands, results)
```

Instantiate (call) a Handshake function in a non-Handshake design using ESI
channels as the outside connections.

Traits: `HasClock`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `module` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `instName` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `opOperands` | variadic of an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| «unnamed» | variadic of an ESI channel |

### `handshake.extmemory` (::circt::handshake::ExternalMemoryOp)

*External memory*

Syntax:

```
operation ::= `handshake.extmemory` `[` `ld` `=` $ldCount `,` `st` `=`  $stCount `]` `(` $memref `:` qualified(type($memref)) `)` `(` $inputs `)` attr-dict `:` functional-type($inputs, $outputs)
```

An ExternalMemoryOp represents a wrapper around a memref input to a
handshake function. The semantics of the load/store operands are identical
to what is decribed for MemoryOp. The only difference is that the first
operand to this operand is a `memref` value.
Upon lowering to FIRRTL, a handshake interface will be created in the
top-level component for each load- and store which connected to this memory.

Example:

```
handshake.func @main(%i: index, %v: i32, %mem : memref<10xi32>, %ctrl: none) -> none {
  %stCtrl = extmemory[ld = 0, st = 1](%mem : memref<10xi32>)(%vout, %addr) {id = 0 : i32} : (i32, index) -> (none)
  %vout, %addr = store(%v, %i, %ctrl) : (i32, index, none) -> (i32, index)
  ...
}
```

Traits: `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ControlInterface`, `ExecutableOpInterface`, `NamedIOInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `ldCount` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `stCount` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `id` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `memref` | memref of any type values |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `handshake.fork` (::circt::handshake::ForkOp)

*Fork operation*

The fork operation represents a fork operation. A
single input is replicated to N outputs and distributed to each
output as soon as the corresponding successor is available.

Example:

```
%1:2 = fork [2] %0 : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `HasClock`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `ControlInterface`, `ExecutableOpInterface`, `GeneralOpInterface`, `NamedIOInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SOSTInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `operand` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | variadic of any type |

### `handshake.func` (::circt::handshake::FuncOp)

*Handshake dialect function.*

The func operation represents a handshaked function.
This is almost exactly like a standard FuncOp, except that it has
some extra verification conditions. In particular, each Value must
only have a single use.

Traits: `HasClock`, `IsolatedFromAbove`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallableOpInterface`, `FineGrainedDataflowRegionOpInterface`, `FunctionOpInterface`, `OpAsmOpInterface`, `RegionKindInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `function_type` | ::mlir::TypeAttr | type attribute of function type |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |

### `handshake.instance` (::circt::handshake::InstanceOp)

*Module instantiate operation*

Syntax:

```
operation ::= `handshake.instance` $module `(` $opOperands `)` attr-dict `:` functional-type($opOperands, results)
```

The `instance` operation represents the instantiation of a module. This
is similar to a function call, except that different instances of the
same module are guaranteed to have their own distinct state.
The instantiated module is encoded as a symbol reference attribute named
“module”. An instance operation takes a control input as its last argument
and returns a control output as its last result.

Example:

```
%2:2 = handshake.instance @my_add(%0, %1, %ctrl) : (f32, f32, none) -> (f32, none)
```

Traits: `HasClock`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallOpInterface`, `ControlInterface`, `NamedIOInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `module` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `opOperands` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| «unnamed» | variadic of any type |

### `handshake.join` (::circt::handshake::JoinOp)

*Join operation*

A control-only synchronizer. Produces a valid output when all
inputs become available.

Example:

```
%0 = join %a, %b, %c : i32, i1, none
```

Traits: `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ControlInterface`, `ExecutableOpInterface`, `GeneralOpInterface`, `InferTypeOpInterface`, `NamedIOInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `data` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | none type |

### `handshake.lazy_fork` (::circt::handshake::LazyForkOp)

*Lazy fork operation*

The lazy\_fork operation represents a lazy fork operation.
A single input is replicated to N outputs and distributed to each
output when all successors are available.

Example:

```
%1:2 = lazy_fork [2] %0 : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `ControlInterface`, `NamedIOInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SOSTInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `operand` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | variadic of any type |

### `handshake.load` (::circt::handshake::LoadOp)

*Load operation*

Load memory port, sends load requests to MemoryOp. From dataflow
predecessor, receives address indices and a control-only value
which signals completion of all previous memory accesses which
target the same memory. When all inputs are received, the load
sends the address indices to MemoryOp. When the MemoryOp returns
a piece of data, the load sends it to its dataflow successor.

Operands: address indices (from predecessor), data (from MemoryOp), control-only input.
Results: data (to successor), address indices (to MemoryOp).

Example:

```
%dataToSucc, %addr1ToMem, %addr2ToMem = load [%addr1, %addr2] %dataFromMem, %ctrl : i8, i16, index
```

Traits: `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ControlInterface`, `ExecutableOpInterface`, `NamedIOInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `addresses` | variadic of any type |
| `data` | any type |
| `ctrl` | none type |

#### Results:

| Result | Description |
| --- | --- |
| `dataResult` | any type |
| `addressResults` | variadic of any type |

### `handshake.memory` (::circt::handshake::MemoryOp)

*Memory*

Syntax:

```
operation ::= `handshake.memory` `[` `ld` `=` $ldCount `,` `st` `=`  $stCount `]` `(` $inputs `)` attr-dict `:` $memRefType `,` functional-type($inputs, $outputs)
```

Each MemoryOp represents an independent memory or memory region (BRAM or external memory).
It receives memory access requests from load and store operations. For every request,
it returns data (for load) and a data-less token indicating completion.
The memory op represents a flat, unidimensional memory.
Operands: all stores (stdata1, staddr1, stdata2, staddr2, …), then all loads (ldaddr1, ldaddr2,…)
Outputs: all load outputs, ordered the same as
load data (lddata1, lddata2, …), followed by all none outputs,
ordered as operands (stnone1, stnone2,…ldnone1, ldnone2,…)

Traits: `HasClock`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ControlInterface`, `ExecutableOpInterface`, `MemoryOpInterface`, `NamedIOInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `ldCount` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `stCount` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `id` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `memRefType` | ::mlir::TypeAttr | memref type attribute |
| `lsq` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `handshake.merge` (::circt::handshake::MergeOp)

*Merge operation*

The merge operation represents a (nondeterministic)
merge operation. Any input is propagated to the single output. The
number of inputs corresponds to the number of predecessor
blocks.

Example:

```
%0 = merge %a, %b, %c : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `ControlInterface`, `ExecutableOpInterface`, `MergeLikeOpInterface`, `NamedIOInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SOSTInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `dataOperands` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `handshake.mux` (::circt::handshake::MuxOp)

*Mux operation*

The mux operation represents a(deterministic) merge operation.
Operands: select, data0, data1, data2, …

The ‘select’ operand is received from ControlMerge of the same
block and it represents the index of the data operand that the mux
should propagate to its single output. The number of data inputs
corresponds to the number of predecessor blocks.

The mux operation is intended solely for control+dataflow selection.
For purely dataflow selection, use the ‘select’ operation instead.

Example:

```
%0 = mux %select [%data0, %data1, %data2] {attributes}: index, i32
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ConditionallySpeculatable`, `ControlInterface`, `ExecutableOpInterface`, `InferTypeOpInterface`, `MergeLikeOpInterface`, `NamedIOInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `selectOperand` | any type |
| `dataOperands` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `handshake.never` (::circt::handshake::NeverOp)

*Never operation*

Syntax:

```
operation ::= `handshake.never` attr-dict `:` qualified(type($result))
```

The never operation represents disconnected data
source. The source never sets any ‘valid’ signal which will
never trigger the successor at any point in time.

Traits: `AlwaysSpeculatableImplTrait`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ConditionallySpeculatable`, `ControlInterface`, `NamedIOInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `handshake.pack` (::circt::handshake::PackOp)

*Packs a tuple*

The `pack` operation constructs a tuple from separate values.
The number of operands corresponds to the number of tuple elements.
Similar to `join`, the output is ready when all inputs are ready.

Example:

```
%tuple = handshake.pack %a, %b {attributes} : tuple<i32, i64>
```

Traits: `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ControlInterface`, `ExecutableOpInterface`, `GeneralOpInterface`, `NamedIOInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | Fixed-sized collection of other types |

### `handshake.return` (::circt::handshake::ReturnOp)

*Handshake dialect return.*

Syntax:

```
operation ::= `handshake.return` attr-dict ($opOperands^ `:` type($opOperands))?
```

The return operation represents a handshaked
function. This is almost exactly like a standard ReturnOp, except
that it exists in a handshake.func. It has the same operands as
standard ReturnOp which it replaces and an additional control -
only operand(exit point of control - only network).

Traits: `HasParentInterface<FineGrainedDataflowRegionOpInterface>`, `Terminator`

Interfaces: `ControlInterface`, `NamedIOInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `opOperands` | variadic of any type |

### `handshake.sink` (::circt::handshake::SinkOp)

*Sink operation*

The sink operation discards any data that arrives at its
input.The sink has no successors and it can continuously consume data.

Example:

```
sink %data : i32
```

Traits: `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ControlInterface`, `ExecutableOpInterface`, `NamedIOInterface`, `SOSTInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `operand` | any type |

### `handshake.source` (::circt::handshake::SourceOp)

*Source operation*

The source operation represents continuous token
source. The source continously sets a ‘valid’ signal which the
successor can consume at any point in time.

Traits: `AlwaysSpeculatableImplTrait`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ConditionallySpeculatable`, `ControlInterface`, `NamedIOInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SOSTInterface`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `handshake.store` (::circt::handshake::StoreOp)

*Store operation*

Store memory port, sends store requests to MemoryOp. From dataflow
predecessors, receives address indices, data, and a control-only
value which signals completion of all previous memory accesses
which target the same memory. When all inputs are received, the
store sends the address and data to MemoryOp.

Operands: address indices, data, control-only input.
Results: data and address indices (sent to MemoryOp).
Types: data type followed by address type.

Example:

```
%dataToMem, %addrToMem = store [%addr1, %addr2] %dataFromPred , %ctrl : i8, i16, index
```

Traits: `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ControlInterface`, `ExecutableOpInterface`, `GeneralOpInterface`, `NamedIOInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `addresses` | variadic of any type |
| `data` | any type |
| `ctrl` | none type |

#### Results:

| Result | Description |
| --- | --- |
| `dataResult` | any type |
| `addressResult` | variadic of any type |

### `handshake.sync` (::circt::handshake::SyncOp)

*Sync operation*

Syntax:

```
operation ::= `handshake.sync` $operands attr-dict `:` type($operands)
```

Synchronizes an arbitrary set of inputs. Synchronization implies applying
join semantics in between all in- and output ports.

Example:

```
%aSynced, %bSynced, %cSynced = sync %a, %b, %c : i32, i1, none
```

Traits: `HasClock`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ControlInterface`, `ExecutableOpInterface`, `GeneralOpInterface`, `NamedIOInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `operands` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `handshake.unpack` (::circt::handshake::UnpackOp)

*Unpacks a tuple*

The `unpack` operation assigns each value of a tuple to a separate
value for further processing. The number of results corresponds
to the number of tuple elements.
Similar to `fork`, each output is distributed as soon as the corresponding
successor is ready.

Example:

```
%a, %b = handshake.unpack %tuple {attributes} : tuple<i32, i64>
```

Traits: `HasClock`, `HasParentInterface<FineGrainedDataflowRegionOpInterface>`

Interfaces: `ControlInterface`, `ExecutableOpInterface`, `GeneralOpInterface`, `NamedIOInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | Fixed-sized collection of other types |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

Attributes
----------

### BufferTypeEnumAttr

*BufferOp seq or fifo*

Syntax:

```
#handshake.buffer_type_enum<
  ::BufferTypeEnum   # value
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| value | `::BufferTypeEnum` | an enum of type BufferTypeEnum |

Enums
-----

### BufferTypeEnum

*BufferOp seq or fifo*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| seq | `0` | seq |
| fifo | `1` | fifo |

'handshake' Dialect Docs
------------------------

* [Handshake Dialect Rationale](https://circt.llvm.org/docs/Dialects/Handshake/RationaleHandshake/)

 [Prev - FSM Dialect Rationale](https://circt.llvm.org/docs/Dialects/FSM/RationaleFSM/ "FSM Dialect Rationale")
[Next - Handshake Dialect Rationale](https://circt.llvm.org/docs/Dialects/Handshake/RationaleHandshake/ "Handshake Dialect Rationale") 

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