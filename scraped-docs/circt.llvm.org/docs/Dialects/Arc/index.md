'arc' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'arc' Dialect
=============

*Canonical representation of state transfer in a circuit*

This is the `arc` dialect, useful for representing state transfer functions
in a circuit.

* [Operations](#operations)
  + [`arc.alloc_memory` (circt::arc::AllocMemoryOp)](#arcalloc_memory-circtarcallocmemoryop)
  + [`arc.alloc_state` (circt::arc::AllocStateOp)](#arcalloc_state-circtarcallocstateop)
  + [`arc.alloc_storage` (circt::arc::AllocStorageOp)](#arcalloc_storage-circtarcallocstorageop)
  + [`arc.call` (circt::arc::CallOp)](#arccall-circtarccallop)
  + [`arc.clock_domain` (circt::arc::ClockDomainOp)](#arcclock_domain-circtarcclockdomainop)
  + [`arc.define` (circt::arc::DefineOp)](#arcdefine-circtarcdefineop)
  + [`arc.execute` (circt::arc::ExecuteOp)](#arcexecute-circtarcexecuteop)
  + [`arc.final` (circt::arc::FinalOp)](#arcfinal-circtarcfinalop)
  + [`arc.initial` (circt::arc::InitialOp)](#arcinitial-circtarcinitialop)
  + [`arc.lut` (circt::arc::LutOp)](#arclut-circtarclutop)
  + [`arc.memory` (circt::arc::MemoryOp)](#arcmemory-circtarcmemoryop)
  + [`arc.memory_read` (circt::arc::MemoryReadOp)](#arcmemory_read-circtarcmemoryreadop)
  + [`arc.memory_read_port` (circt::arc::MemoryReadPortOp)](#arcmemory_read_port-circtarcmemoryreadportop)
  + [`arc.memory_write` (circt::arc::MemoryWriteOp)](#arcmemory_write-circtarcmemorywriteop)
  + [`arc.memory_write_port` (circt::arc::MemoryWritePortOp)](#arcmemory_write_port-circtarcmemorywriteportop)
  + [`arc.model` (circt::arc::ModelOp)](#arcmodel-circtarcmodelop)
  + [`arc.output` (circt::arc::OutputOp)](#arcoutput-circtarcoutputop)
  + [`arc.root_input` (circt::arc::RootInputOp)](#arcroot_input-circtarcrootinputop)
  + [`arc.root_output` (circt::arc::RootOutputOp)](#arcroot_output-circtarcrootoutputop)
  + [`arc.runtime.model` (circt::arc::RuntimeModelOp)](#arcruntimemodel-circtarcruntimemodelop)
  + [`arc.sim.emit` (circt::arc::SimEmitValueOp)](#arcsimemit-circtarcsimemitvalueop)
  + [`arc.sim.get_port` (circt::arc::SimGetPortOp)](#arcsimget_port-circtarcsimgetportop)
  + [`arc.sim.instantiate` (circt::arc::SimInstantiateOp)](#arcsiminstantiate-circtarcsiminstantiateop)
  + [`arc.sim.set_input` (circt::arc::SimSetInputOp)](#arcsimset_input-circtarcsimsetinputop)
  + [`arc.sim.step` (circt::arc::SimStepOp)](#arcsimstep-circtarcsimstepop)
  + [`arc.state` (circt::arc::StateOp)](#arcstate-circtarcstateop)
  + [`arc.state_read` (circt::arc::StateReadOp)](#arcstate_read-circtarcstatereadop)
  + [`arc.state_write` (circt::arc::StateWriteOp)](#arcstate_write-circtarcstatewriteop)
  + [`arc.storage.get` (circt::arc::StorageGetOp)](#arcstorageget-circtarcstoragegetop)
  + [`arc.tap` (circt::arc::TapOp)](#arctap-circtarctapop)
  + [`arc.vectorize` (circt::arc::VectorizeOp)](#arcvectorize-circtarcvectorizeop)
  + [`arc.vectorize.return` (circt::arc::VectorizeReturnOp)](#arcvectorizereturn-circtarcvectorizereturnop)
  + [`arc.zero_count` (circt::arc::ZeroCountOp)](#arczero_count-circtarczerocountop)
* [Attributes](#attributes-19)
  + [TraceTapAttr](#tracetapattr)
* [Types](#types)
  + [MemoryType](#memorytype)
  + [SimModelInstanceType](#simmodelinstancetype)
  + [StateType](#statetype)
  + [StorageType](#storagetype)
* [Enums](#enums)
  + [ZeroCountPredicate](#zerocountpredicate)

Operations [¶](#operations)
---------------------------

### `arc.alloc_memory` (circt::arc::AllocMemoryOp) [¶](#arcalloc_memory-circtarcallocmemoryop)

*Allocate a memory*

Syntax:

```
operation ::= `arc.alloc_memory` $storage attr-dict `:` functional-type($storage, $memory)
```

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Allocate on ::mlir::SideEffects::DefaultResource}`

#### Operands: [¶](#operands)

| Operand | Description |
| --- | --- |
| `storage` |  |

#### Results: [¶](#results)

| Result | Description |
| --- | --- |
| `memory` |  |

### `arc.alloc_state` (circt::arc::AllocStateOp) [¶](#arcalloc_state-circtarcallocstateop)

*Allocate internal state*

Syntax:

```
operation ::= `arc.alloc_state` $storage (`tap` $tap^)? attr-dict `:` functional-type($storage, $state)
```

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Allocate on ::mlir::SideEffects::DefaultResource}`

#### Attributes: [¶](#attributes)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `tap` | ::mlir::UnitAttr | unit attribute |

#### Operands: [¶](#operands-1)

| Operand | Description |
| --- | --- |
| `storage` |  |

#### Results: [¶](#results-1)

| Result | Description |
| --- | --- |
| `state` |  |

### `arc.alloc_storage` (circt::arc::AllocStorageOp) [¶](#arcalloc_storage-circtarcallocstorageop)

*Allocate contiguous storage space from a larger storage space*

Syntax:

```
operation ::= `arc.alloc_storage` $input (`[` $offset^ `]`)? attr-dict `:` functional-type($input, $output)
```

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Allocate on ::mlir::SideEffects::DefaultResource}`

#### Attributes: [¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `offset` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-2)

| Operand | Description |
| --- | --- |
| `input` |  |

#### Results: [¶](#results-2)

| Result | Description |
| --- | --- |
| `output` |  |

### `arc.call` (circt::arc::CallOp) [¶](#arccall-circtarccallop)

*Calls an arc*

Syntax:

```
operation ::= `arc.call` $arc `(` $inputs `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `MemRefsNormalizable`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallOpInterface`, `ClockedOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-2)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `arc` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands: [¶](#operands-3)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results: [¶](#results-3)

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `arc.clock_domain` (circt::arc::ClockDomainOp) [¶](#arcclock_domain-circtarcclockdomainop)

*A clock domain*

Syntax:

```
operation ::= `arc.clock_domain` ` ` `(` $inputs `)` `clock` $clock attr-dict `:`
              functional-type($inputs, results) $body
```

Traits: `IsolatedFromAbove`, `RecursiveMemoryEffects`, `SingleBlockImplicitTerminator<arc::OutputOp>`, `SingleBlock`

Interfaces: `RegionKindInterface`

#### Operands: [¶](#operands-4)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |
| `clock` | A type for clock-carrying wires |

#### Results: [¶](#results-4)

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `arc.define` (circt::arc::DefineOp) [¶](#arcdefine-circtarcdefineop)

*State transfer arc definition*

Traits: `HasParent<mlir::ModuleOp>`, `IsolatedFromAbove`, `SingleBlockImplicitTerminator<arc::OutputOp>`, `SingleBlock`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallableOpInterface`, `FunctionOpInterface`, `RegionKindInterface`, `Symbol`

#### Attributes: [¶](#attributes-3)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `function_type` | ::mlir::TypeAttr | type attribute of function type |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |

### `arc.execute` (circt::arc::ExecuteOp) [¶](#arcexecute-circtarcexecuteop)

*Execute an SSACFG region*

Syntax:

```
operation ::= `arc.execute` (` ` `(` $inputs^ `:` type($inputs) `)`)?
              (`->` `(` type($results)^ `)`)?
              attr-dict-with-keyword $body
```

The `arc.execute` op allows an SSACFG region to be embedded in a parent
graph region, or another SSACFG region. Whenever execution reaches this op,
its body region is executed and the results yielded from the body are
produced as the `arc.execute` op’s results. The op is isolated from above.
Any SSA values defined outside the op that are used inside the body have to
be captured as operands and then referred to as entry block arguments in the
body.

Traits: `IsolatedFromAbove`, `RecursiveMemoryEffects`

#### Operands: [¶](#operands-5)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results: [¶](#results-5)

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `arc.final` (circt::arc::FinalOp) [¶](#arcfinal-circtarcfinalop)

*Region to be executed at the end of simulation*

Syntax:

```
operation ::= `arc.final` attr-dict-with-keyword $body
```

Traits: `HasParent<ModelOp>`, `NoRegionArguments`, `NoTerminator`, `RecursiveMemoryEffects`, `SingleBlock`

### `arc.initial` (circt::arc::InitialOp) [¶](#arcinitial-circtarcinitialop)

*Region to be executed at the start of simulation*

Syntax:

```
operation ::= `arc.initial` attr-dict-with-keyword $body
```

Traits: `HasParent<ModelOp>`, `NoRegionArguments`, `NoTerminator`, `RecursiveMemoryEffects`, `SingleBlock`

### `arc.lut` (circt::arc::LutOp) [¶](#arclut-circtarclutop)

*A lookup-table.*

Syntax:

```
operation ::= `arc.lut` `(` $inputs `)` `:` functional-type($inputs, $output)
              attr-dict-with-keyword $body
```

Represents a lookup-table as one operation. The operations that map the
lookup/input values to the corresponding table-entry are collected inside
the body of this operation.
Note that the operation is marked to be isolated from above to guarantee
that all input values have to be passed as an operand. This allows for
simpler analyses and canonicalizations of the LUT as well as lowering.
Only combinational operations are allowed inside the LUT, i.e., no
side-effects, state, time delays, etc.

Traits: `AlwaysSpeculatableImplTrait`, `IsolatedFromAbove`, `SingleBlockImplicitTerminator<arc::OutputOp>`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-6)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of signless integer |

#### Results: [¶](#results-6)

| Result | Description |
| --- | --- |
| `output` | signless integer |

### `arc.memory` (circt::arc::MemoryOp) [¶](#arcmemory-circtarcmemoryop)

*Memory*

Syntax:

```
operation ::= `arc.memory` type($memory) attr-dict
```

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Allocate on ::mlir::SideEffects::DefaultResource}`

#### Results: [¶](#results-7)

| Result | Description |
| --- | --- |
| `memory` |  |

### `arc.memory_read` (circt::arc::MemoryReadOp) [¶](#arcmemory_read-circtarcmemoryreadop)

*Read a word from a memory*

Syntax:

```
operation ::= `arc.memory_read` $memory `[` $address `]` attr-dict `:` type($memory)
```

Interfaces: `InferTypeOpInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource}`

#### Operands: [¶](#operands-7)

| Operand | Description |
| --- | --- |
| `memory` |  |
| `address` | integer |

#### Results: [¶](#results-8)

| Result | Description |
| --- | --- |
| `data` | integer |

### `arc.memory_read_port` (circt::arc::MemoryReadPortOp) [¶](#arcmemory_read_port-circtarcmemoryreadportop)

*Read port from a memory*

Syntax:

```
operation ::= `arc.memory_read_port` $memory `[` $address `]` attr-dict `:` type($memory)
```

Represents a combinatorial memory read port. No memory read side-effect
trait is necessary because at the stage of the Arc lowering where this
operation is legal to be present, it is guaranteed that all reads from the
same address produce the same output. This is because all writes are
reordered to happen at the end of the cycle in LegalizeStateUpdates (or
alternatively produce the necessary temporaries).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-8)

| Operand | Description |
| --- | --- |
| `memory` |  |
| `address` | integer |

#### Results: [¶](#results-9)

| Result | Description |
| --- | --- |
| `data` | integer |

### `arc.memory_write` (circt::arc::MemoryWriteOp) [¶](#arcmemory_write-circtarcmemorywriteop)

*Write a word to a memory*

Syntax:

```
operation ::= `arc.memory_write` $memory `[` $address `]` `,` $data (`if` $enable^)?
              attr-dict `:` type($memory)
```

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### Operands: [¶](#operands-9)

| Operand | Description |
| --- | --- |
| `memory` |  |
| `address` | integer |
| `enable` | 1-bit signless integer |
| `data` | integer |

### `arc.memory_write_port` (circt::arc::MemoryWritePortOp) [¶](#arcmemory_write_port-circtarcmemorywriteportop)

*Write port to a memory*

Syntax:

```
operation ::= `arc.memory_write_port` $memory `,` $arc  `(` $inputs `)` (`clock` $clock^)?  (`enable` $enable^)?
              (`mask` $mask^)? `latency` $latency attr-dict `:`
              type($memory) `,` type($inputs)
```

Traits: `AttrSizedOperandSegments`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallOpInterface`, `ClockedOpInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### Attributes: [¶](#attributes-4)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `arc` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `enable` | ::mlir::UnitAttr | unit attribute |
| `mask` | ::mlir::UnitAttr | unit attribute |
| `latency` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-10)

| Operand | Description |
| --- | --- |
| `memory` |  |
| `inputs` | variadic of any type |
| `clock` | A type for clock-carrying wires |

### `arc.model` (circt::arc::ModelOp) [¶](#arcmodel-circtarcmodelop)

*A model with stratified clocks*

Syntax:

```
operation ::= `arc.model` $sym_name `io` $io
              (`initializer` $initialFn^)?
              (`finalizer` $finalFn^)?
              (`traceTaps` $traceTaps^)?
              attr-dict-with-keyword $body
```

A model with stratified clocks. The `io` optional attribute
specifies the I/O of the module associated to this model.

Traits: `IsolatedFromAbove`, `NoTerminator`

Interfaces: `SymbolUserOpInterface`, `Symbol`

#### Attributes: [¶](#attributes-5)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `io` | ::mlir::TypeAttr | type attribute of a module |
| `initialFn` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `finalFn` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `traceTaps` | ::mlir::ArrayAttr | Array of trace metadata |

### `arc.output` (circt::arc::OutputOp) [¶](#arcoutput-circtarcoutputop)

*Arc terminator*

Syntax:

```
operation ::= `arc.output` attr-dict ($outputs^ `:` qualified(type($outputs)))?
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<DefineOp, LutOp, ClockDomainOp, ExecuteOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-11)

| Operand | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `arc.root_input` (circt::arc::RootInputOp) [¶](#arcroot_input-circtarcrootinputop)

*A root input*

Syntax:

```
operation ::= `arc.root_input` $name `,` $storage attr-dict `:` functional-type($storage, $state)
```

Interfaces: `OpAsmOpInterface`

#### Attributes: [¶](#attributes-6)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-12)

| Operand | Description |
| --- | --- |
| `storage` |  |

#### Results: [¶](#results-10)

| Result | Description |
| --- | --- |
| `state` |  |

### `arc.root_output` (circt::arc::RootOutputOp) [¶](#arcroot_output-circtarcrootoutputop)

*A root output*

Syntax:

```
operation ::= `arc.root_output` $name `,` $storage attr-dict `:` functional-type($storage, $state)
```

Interfaces: `OpAsmOpInterface`

#### Attributes: [¶](#attributes-7)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-13)

| Operand | Description |
| --- | --- |
| `storage` |  |

#### Results: [¶](#results-11)

| Result | Description |
| --- | --- |
| `state` |  |

### `arc.runtime.model` (circt::arc::RuntimeModelOp) [¶](#arcruntimemodel-circtarcruntimemodelop)

*Provides static metadata of an Arc model used by the runtime*

Syntax:

```
operation ::= `arc.runtime.model` $sym_name $name `numStateBytes` $numStateBytes
              (`traceTaps` $traceTaps^)? attr-dict
```

Collection of static metadata for a specific Arc model accessed by the
arcilator runtime library:

* name: Name of the model
* numStateBytes: Number of bytes required to store the model’s internal
  state
* traceTaps: Traced signal metadata

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<::mlir::ModuleOp>`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `Symbol`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-8)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `name` | ::mlir::StringAttr | string attribute |
| `numStateBytes` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `traceTaps` | ::mlir::ArrayAttr | Array of trace metadata |

### `arc.sim.emit` (circt::arc::SimEmitValueOp) [¶](#arcsimemit-circtarcsimemitvalueop)

*Sends a value to the simulation driver*

Syntax:

```
operation ::= `arc.sim.emit` $valueName `,` $value attr-dict `:` type($value)
```

Sends a named value to the simulation driver. This is notably useful
for printing values during simulation.

#### Attributes: [¶](#attributes-9)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `valueName` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-14)

| Operand | Description |
| --- | --- |
| `value` | any type |

### `arc.sim.get_port` (circt::arc::SimGetPortOp) [¶](#arcsimget_port-circtarcsimgetportop)

*Gets the value of a port of the model instance*

Syntax:

```
operation ::= `arc.sim.get_port` $instance `,` $port attr-dict
              `:` type($value) `,` qualified(type($instance))
```

Gets the value of the given port in a specific instance of a model. The
provided port must be of the type of the expected value.

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource}`

#### Attributes: [¶](#attributes-10)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `port` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-15)

| Operand | Description |
| --- | --- |
| `instance` |  |

#### Results: [¶](#results-12)

| Result | Description |
| --- | --- |
| `value` | any type |

### `arc.sim.instantiate` (circt::arc::SimInstantiateOp) [¶](#arcsiminstantiate-circtarcsiminstantiateop)

*Instantiates an Arc model for simulation*

Creates an instance of an Arc model in scope, in order to simulate it.
The model can be used from within the associated region, modelling its
lifetime.

Traits: `NoTerminator`

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{MemoryEffects::Allocate on ::mlir::SideEffects::DefaultResource}`

#### Attributes: [¶](#attributes-11)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `runtimeModel` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `runtimeArgs` | ::mlir::StringAttr | string attribute |

### `arc.sim.set_input` (circt::arc::SimSetInputOp) [¶](#arcsimset_input-circtarcsimsetinputop)

*Sets the value of an input of the model instance*

Syntax:

```
operation ::= `arc.sim.set_input` $instance `,` $input `=` $value attr-dict
              `:` type($value) `,` qualified(type($instance))
```

Sets the value of an input port in a specific instance of a model. The
provided input port must be of input type on the model and its type must
match the type of the value operand.

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### Attributes: [¶](#attributes-12)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `input` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-16)

| Operand | Description |
| --- | --- |
| `instance` |  |
| `value` | any type |

### `arc.sim.step` (circt::arc::SimStepOp) [¶](#arcsimstep-circtarcsimstepop)

*Evaluates one step of the simulation for the provided model instance*

Syntax:

```
operation ::= `arc.sim.step` $instance attr-dict `:` qualified(type($instance))
```

Evaluates one step of the simulation for the provided model instance,
updating ports accordingly.

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource, MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### Operands: [¶](#operands-17)

| Operand | Description |
| --- | --- |
| `instance` |  |

### `arc.state` (circt::arc::StateOp) [¶](#arcstate-circtarcstateop)

*Instantiates a state element with input from a transfer arc*

Syntax:

```
operation ::= `arc.state` $arc `(` $inputs `)` (`clock` $clock^)? (`enable` $enable^)?
              (`reset` $reset^)?
              ( `initial` ` ` `(` $initials^ `:` type($initials) `)`)?
              `latency` $latency attr-dict `:` functional-type($inputs, results)
```

Traits: `AttrSizedOperandSegments`, `MemRefsNormalizable`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallOpInterface`, `ClockedOpInterface`, `SymbolUserOpInterface`

#### Attributes: [¶](#attributes-13)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `arc` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `latency` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-18)

| Operand | Description |
| --- | --- |
| `clock` | A type for clock-carrying wires |
| `enable` | 1-bit signless integer |
| `reset` | 1-bit signless integer |
| `inputs` | variadic of any type |
| `initials` | variadic of any type |

#### Results: [¶](#results-13)

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `arc.state_read` (circt::arc::StateReadOp) [¶](#arcstate_read-circtarcstatereadop)

*Read a state’s value*

Syntax:

```
operation ::= `arc.state_read` $state attr-dict `:` type($state)
```

Interfaces: `InferTypeOpInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource}`

#### Operands: [¶](#operands-19)

| Operand | Description |
| --- | --- |
| `state` |  |

#### Results: [¶](#results-14)

| Result | Description |
| --- | --- |
| `value` | any type |

### `arc.state_write` (circt::arc::StateWriteOp) [¶](#arcstate_write-circtarcstatewriteop)

*Update a state’s value*

Syntax:

```
operation ::= `arc.state_write` $state `=` $value (`if` $condition^)?
              (`tap` $traceTapModel`[`$traceTapIndex^`]` )? attr-dict `:` type($state)
```

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### Attributes: [¶](#attributes-14)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `traceTapModel` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `traceTapIndex` | ::mlir::IntegerAttr | 64-bit unsigned integer attribute |

#### Operands: [¶](#operands-20)

| Operand | Description |
| --- | --- |
| `state` |  |
| `value` | any type |
| `condition` | 1-bit signless integer |

### `arc.storage.get` (circt::arc::StorageGetOp) [¶](#arcstorageget-circtarcstoragegetop)

*Access an allocated state, memory, or storage slice*

Syntax:

```
operation ::= `arc.storage.get` $storage `[` $offset `]` attr-dict
              `:` qualified(type($storage)) `->` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-15)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `offset` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-21)

| Operand | Description |
| --- | --- |
| `storage` |  |

#### Results: [¶](#results-15)

| Result | Description |
| --- | --- |
| `result` | or or |

### `arc.tap` (circt::arc::TapOp) [¶](#arctap-circtarctapop)

*A tracker op to observe a value under one or more given names*

Syntax:

```
operation ::= `arc.tap` $value attr-dict `:` type($value)
```

#### Attributes: [¶](#attributes-16)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `names` | ::mlir::ArrayAttr | string array attribute |

#### Operands: [¶](#operands-22)

| Operand | Description |
| --- | --- |
| `value` | any type |

### `arc.vectorize` (circt::arc::VectorizeOp) [¶](#arcvectorize-circtarcvectorizeop)

*Isolated subgraph of operations to be vectorized*

Syntax:

```
operation ::= `arc.vectorize` $inputs attr-dict `:` functional-type($inputs, $results) $body
```

This operation represents a vectorized computation DAG. It places a
convenient boundary between the subgraph to be vectorized and the
surrounding non-vectorizable parts of the original graph.

This allows us to split the vectorization transformations into multiple
parts/passes:

* Finding an initial set of operations to be vectorized
* Optimizing this set by pulling in more operations into the nested block,
  splitting it such that the vector width does not exceed a given limit,
  applying a cost model and potentially reverting the decision to
  vectorize this subgraph (e.g., because not enough ops could be pulled
  in)
* Performing the actual vectorization by lowering this operation. This
  operation allows to perform the lowering of the boundary and the body
  separately and either via 1D `vector` types for SIMD vectorization or
  plain integers for manual vectorization within a scalar register.

For each block argument of the nested block, there is a list of operands
that represent the elements of the vector. If the boundary is already
vectorized each list will only contain a single SSA value of either vector
type or an integer representing the concatenation of all original operands
of that vector.

Example:

Given the following two AND operations in the IR

```
%0 = arith.and %in0, %in1 : i1
%1 = arith.and %in2, %in2 : i1
```

they could be vectorized by putting one such AND operation in the body block
of the `arc.vectorize` operation and forwarding the operands accordingly.

```
%0:2 = arc.vectorize (%in0, %in1), (%in2, %in2) :
  (i1, i1, i1, i1) -> (i1, i1) {
^bb0(%arg0: i1, %arg1: i1):
  %1 = arith.and %arg0, %arg1 : i1
  arc.output %1 : i1
}
```

In a next step, the boundary could be lowered/vectorized. This can happen
in terms of integers for vectorization within scalar registers:

```
%0 = comb.concat %in0, %in1 : i1, i1
%1 = comb.replicate %in2 : (i1) -> i2
%2 = arc.vectorize (%0), (%1) : (i2, i2) -> (i2) {
^bb0(%arg0: i1, %arg1: i1):
  %1 = arith.and %arg0, %arg1 : i1
  arc.output %1 : i1
}
%3 = comb.extract %2 from 1 : (i2) -> i1
%4 = comb.extract %2 from 0 : (i2) -> i1
```

Or via `vector` types for SIMD vectorization:

```
%cst = arith.constant dense<0> : vector<2xi1>
%0 = vector.insert %in0, %cst[0] : i1 into vector<2xi1>
%1 = vector.insert %in1, %0[1] : i1 into vector<2xi1>
%2 = vector.broadcast %in2 : i1 to vector<2xi1>
%3 = arc.vectorize (%1), (%2) :
  (vector<2xi1>, vector<2xi1>) -> (vector<2xi1>) {
^bb0(%arg0: i1, %arg1: i1):
  %1 = arith.and %arg0, %arg1 : i1
  arc.output %1 : i1
}
%4 = vector.extract %2[0] : vector<2xi1>
%5 = vector.extract %2[1] : vector<2xi1>
```

Alternatively, the body could be vectorized first. Again, as integers

```
%0:2 = arc.vectorize (%in0, %in1), (%in2, %in2) :
  (i1, i1, i1, i1) -> (i1, i1) {
^bb0(%arg0: i2, %arg1: i2):
  %1 = arith.and %arg0, %arg1 : i2
  arc.output %1 : i2
}
```

or SIMD vectors.

```
%0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) :
  (i1, i1, i1, i1) -> (i1, i1) {
^bb0(%arg0: vector<2xi1>, %arg1: vector<2xi1>):
  %1 = arith.and %arg0, %arg1 : vector<2xi1>
  arc.output %1 : vector<2xi1>
}
```

Once both sides are lowered, the `arc.vectorize` op simply becomes a
passthrough for the operands and can be removed by inlining the nested
block. The integer based vectorization would then look like the following:

```
%0 = comb.concat %in0, %in1 : i1, i1
%1 = comb.replicate %in2 : (i1) -> i2
%2 = arith.and %0, %1 : i2
%3 = comb.extract %2 from 1 : (i2) -> i1
%4 = comb.extract %2 from 0 : (i2) -> i1
```

The SIMD vector based lowering would result in the following IR:

```
%cst = arith.constant dense<0> : vector<2xi1>
%0 = vector.insert %in0, %cst[0] : i1 into vector<2xi1>
%1 = vector.insert %in1, %0[1] : i1 into vector<2xi1>
%2 = vector.broadcast %in2 : i1 to vector<2xi1>
%3 = arith.and %1, %2 : vector<2xi1>
%4 = vector.extract %3[0] : vector<2xi1>
%5 = vector.extract %3[1] : vector<2xi1>
```

Traits: `IsolatedFromAbove`, `RecursiveMemoryEffects`

#### Attributes: [¶](#attributes-17)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inputOperandSegments` | ::mlir::DenseI32ArrayAttr | i32 dense array attribute |

#### Operands: [¶](#operands-23)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results: [¶](#results-16)

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `arc.vectorize.return` (circt::arc::VectorizeReturnOp) [¶](#arcvectorizereturn-circtarcvectorizereturnop)

*Arc.vectorized terminator*

Syntax:

```
operation ::= `arc.vectorize.return` operands attr-dict `:` qualified(type(operands))
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<VectorizeOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-24)

| Operand | Description |
| --- | --- |
| `value` | any type |

### `arc.zero_count` (circt::arc::ZeroCountOp) [¶](#arczero_count-circtarczerocountop)

*Leading/trailing zero count operation*

Syntax:

```
operation ::= `arc.zero_count` $predicate $input attr-dict `:` type($input)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-18)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `predicate` | circt::arc::ZeroCountPredicateAttr | arc.zero\_count predicate |

#### Operands: [¶](#operands-25)

| Operand | Description |
| --- | --- |
| `input` | signless integer |

#### Results: [¶](#results-17)

| Result | Description |
| --- | --- |
| `output` | signless integer |

Attributes [¶](#attributes-19)
------------------------------

### TraceTapAttr [¶](#tracetapattr)

*Metadata of a signal with trace instrumentation*

Syntax:

```
#arc.trace_tap<
  mlir::TypeAttr,   # sigType
  uint64_t,   # stateOffset
  mlir::ArrayAttr   # names
>
```

#### Parameters: [¶](#parameters)

| Parameter | C++ type | Description |
| --- | --- | --- |
| sigType | `mlir::TypeAttr` |  |
| stateOffset | `uint64_t` |  |
| names | `mlir::ArrayAttr` |  |

Types [¶](#types)
-----------------

### MemoryType [¶](#memorytype)

Syntax:

```
!arc.memory<
  unsigned,   # numWords
  ::mlir::IntegerType,   # wordType
  ::mlir::IntegerType   # addressType
>
```

#### Parameters: [¶](#parameters-1)

| Parameter | C++ type | Description |
| --- | --- | --- |
| numWords | `unsigned` |  |
| wordType | `::mlir::IntegerType` |  |
| addressType | `::mlir::IntegerType` |  |

### SimModelInstanceType [¶](#simmodelinstancetype)

Syntax:

```
!arc.sim.instance<
  mlir::FlatSymbolRefAttr   # model
>
```

#### Parameters: [¶](#parameters-2)

| Parameter | C++ type | Description |
| --- | --- | --- |
| model | `mlir::FlatSymbolRefAttr` |  |

### StateType [¶](#statetype)

Syntax:

```
!arc.state<
  ::mlir::Type   # type
>
```

#### Parameters: [¶](#parameters-3)

| Parameter | C++ type | Description |
| --- | --- | --- |
| type | `::mlir::Type` |  |

### StorageType [¶](#storagetype)

Syntax:

```
!arc.storage<
  unsigned   # size
>
```

#### Parameters: [¶](#parameters-4)

| Parameter | C++ type | Description |
| --- | --- | --- |
| size | `unsigned` |  |

Enums [¶](#enums)
-----------------

### ZeroCountPredicate [¶](#zerocountpredicate)

*Arc.zero\_count predicate*

#### Cases: [¶](#cases)

| Symbol | Value | String |
| --- | --- | --- |
| leading | `0` | leading |
| trailing | `1` | trailing |

 [Prev -](https://circt.llvm.org/docs/Dialects/ESIAppID/)
[Next - 'calyx' Dialect](https://circt.llvm.org/docs/Dialects/Calyx/ "'calyx' Dialect") 

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