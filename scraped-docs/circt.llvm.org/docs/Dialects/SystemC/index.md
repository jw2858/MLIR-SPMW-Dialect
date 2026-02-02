'systemc' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'systemc' Dialect
=================

*Types and operations for the SystemC dialect*

This dialect defines the `systemc` dialect, which represents various
constructs of the SystemC library (IEEE 1666-2011) useful for emission.

* [Operations](#operations)
  + [`systemc.convert` (::circt::systemc::ConvertOp)](#systemcconvert-circtsystemcconvertop)
  + [`systemc.cpp.assign` (::circt::systemc::AssignOp)](#systemccppassign-circtsystemcassignop)
  + [`systemc.cpp.call` (::circt::systemc::CallOp)](#systemccppcall-circtsystemccallop)
  + [`systemc.cpp.call_indirect` (::circt::systemc::CallIndirectOp)](#systemccppcall_indirect-circtsystemccallindirectop)
  + [`systemc.cpp.delete` (::circt::systemc::DeleteOp)](#systemccppdelete-circtsystemcdeleteop)
  + [`systemc.cpp.destructor` (::circt::systemc::DestructorOp)](#systemccppdestructor-circtsystemcdestructorop)
  + [`systemc.cpp.func` (::circt::systemc::FuncOp)](#systemccppfunc-circtsystemcfuncop)
  + [`systemc.cpp.member_access` (::circt::systemc::MemberAccessOp)](#systemccppmember_access-circtsystemcmemberaccessop)
  + [`systemc.cpp.new` (::circt::systemc::NewOp)](#systemccppnew-circtsystemcnewop)
  + [`systemc.cpp.return` (::circt::systemc::ReturnOp)](#systemccppreturn-circtsystemcreturnop)
  + [`systemc.cpp.variable` (::circt::systemc::VariableOp)](#systemccppvariable-circtsystemcvariableop)
  + [`systemc.ctor` (::circt::systemc::CtorOp)](#systemcctor-circtsystemcctorop)
  + [`systemc.func` (::circt::systemc::SCFuncOp)](#systemcfunc-circtsystemcscfuncop)
  + [`systemc.instance.bind_port` (::circt::systemc::BindPortOp)](#systemcinstancebind_port-circtsystemcbindportop)
  + [`systemc.instance.decl` (::circt::systemc::InstanceDeclOp)](#systemcinstancedecl-circtsystemcinstancedeclop)
  + [`systemc.interop.verilated` (::circt::systemc::InteropVerilatedOp)](#systemcinteropverilated-circtsystemcinteropverilatedop)
  + [`systemc.method` (::circt::systemc::MethodOp)](#systemcmethod-circtsystemcmethodop)
  + [`systemc.module` (::circt::systemc::SCModuleOp)](#systemcmodule-circtsystemcscmoduleop)
  + [`systemc.sensitive` (::circt::systemc::SensitiveOp)](#systemcsensitive-circtsystemcsensitiveop)
  + [`systemc.signal` (::circt::systemc::SignalOp)](#systemcsignal-circtsystemcsignalop)
  + [`systemc.signal.read` (::circt::systemc::SignalReadOp)](#systemcsignalread-circtsystemcsignalreadop)
  + [`systemc.signal.write` (::circt::systemc::SignalWriteOp)](#systemcsignalwrite-circtsystemcsignalwriteop)
  + [`systemc.thread` (::circt::systemc::ThreadOp)](#systemcthread-circtsystemcthreadop)
* [Type constraints](#type-constraints)
  + [a SystemC sc\_bigint type](#a-systemc-sc_bigintw-type)
  + [a SystemC sc\_biguint type](#a-systemc-sc_biguintw-type)
  + [a SystemC sc\_bv\_base type](#a-systemc-sc_bv_base-type)
  + [a SystemC sc\_bv type](#a-systemc-sc_bvw-type)
  + [a SystemC sc\_int\_base type](#a-systemc-sc_int_base-type)
  + [a SystemC sc\_int type](#a-systemc-sc_intw-type)
  + [a SystemC sc\_lv\_base type](#a-systemc-sc_lv_base-type)
  + [a SystemC sc\_lv type](#a-systemc-sc_lvw-type)
  + [FunctionType with no inputs and results](#functiontype-with-no-inputs-and-results)
  + [a SystemC sc\_signed type](#a-systemc-sc_signed-type)
  + [a SystemC sc\_uint\_base type](#a-systemc-sc_uint_base-type)
  + [a SystemC sc\_uint type](#a-systemc-sc_uintw-type)
  + [a SystemC sc\_unsigned type](#a-systemc-sc_unsigned-type)
  + [a SystemC sc\_value\_base type](#a-systemc-sc_value_base-type)
* [Types](#types)
  + [InOutType](#inouttype)
  + [InputType](#inputtype)
  + [LogicType](#logictype)
  + [ModuleType](#moduletype)
  + [OutputType](#outputtype)
  + [SignalType](#signaltype)
* [Enums](#enums)
  + [MemberAccessKind](#memberaccesskind)

Operations
----------

### `systemc.convert` (::circt::systemc::ConvertOp)

*Converts between various integer and bit vector types.*

Syntax:

```
operation ::= `systemc.convert` $input attr-dict `:` functional-type($input, $result)
```

Allows conversions between the various integer and bit vector types in
SystemC, including MLIRs signless integers that are used to represent the
primitive C integer types, according to the explicit and implicit
constructors, implicit access operators and explicit conversion member
functions defined in the respective data type class in the SystemC spec
(refer to the description of the supported types and the spec chapters
listed there for more information).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | integer or a SystemC sc\_value\_base type or signless integer or a SystemC sc\_bv\_base type or a SystemC sc\_bv type or a SystemC sc\_lv\_base type or a SystemC sc\_lv type or a SystemC sc\_logic type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | integer or a SystemC sc\_value\_base type or signless integer or a SystemC sc\_bv\_base type or a SystemC sc\_bv type or a SystemC sc\_lv\_base type or a SystemC sc\_lv type or a SystemC sc\_logic type |

### `systemc.cpp.assign` (::circt::systemc::AssignOp)

*A C++ assignment.*

Syntax:

```
operation ::= `systemc.cpp.assign` $dest `=` $source attr-dict `:` type($dest)
```

Assigns one SSA value to another. Note that there is no notion of lvalues
and rvalues. This means that one can assign to a value that is the result
of, e.g., an addition, which is not allowed in C++. It is the responsibility
of the user and the implementor of a lowering pass that creates this
operation to make sure that the operands are valid according to C++
semantics. Rationale for that: implementing these constraints would add
quite some complexity, but still does not guarantee that the assignment
is valid because we also make use of non-verifyable verbatim types, etc.

Traits: `SameTypeOperands`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | any type |
| `source` | any type |

### `systemc.cpp.call` (::circt::systemc::CallOp)

*Call operation*

Syntax:

```
operation ::= `systemc.cpp.call` $callee `(` $callee_operands `)` attr-dict `:`
              functional-type($callee_operands, $results)
```

The `systemc.cpp.call` operation represents a direct call to a function that
is within the same symbol scope as the call. The operands and result types
of the call must match the specified function type. The callee is encoded as
a symbol reference attribute named “callee”.

Example:

```
%2 = systemc.cpp.call @my_add(%0, %1) : (i32, i32) -> i32
```

Interfaces: `ArgAndResultAttrsOpInterface`, `CallOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `callee` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `callee_operands` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `systemc.cpp.call_indirect` (::circt::systemc::CallIndirectOp)

*Indirect call operation*

Syntax:

```
operation ::= `systemc.cpp.call_indirect` $callee `(` $callee_operands `)` attr-dict `:` type($callee)
```

The `systemc.cpp.call_indirect` operation represents an indirect call to a
value of function type. The operands and result types of the call must match
the specified function type.

Example:

```
%func = systemc.cpp.member_access %object dot "func" : () -> i32
%result = systemc.cpp.call_indirect %func() : () -> i32
```

Interfaces: `ArgAndResultAttrsOpInterface`, `CallOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `callee` | function type |
| `callee_operands` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `systemc.cpp.delete` (::circt::systemc::DeleteOp)

*A C++ delete expression.*

Syntax:

```
operation ::= `systemc.cpp.delete` $pointer attr-dict `:` qualified(type($pointer))
```

Destroys objects previously allocated by the new expression and releases
the allocated memory.

#### Operands:

| Operand | Description |
| --- | --- |
| `pointer` | EmitC pointer type |

### `systemc.cpp.destructor` (::circt::systemc::DestructorOp)

*A C++ destructor definition.*

Syntax:

```
operation ::= `systemc.cpp.destructor` attr-dict-with-keyword $body
```

This operation models a C++ destructor of a class or struct. It is not an
operation modelling some abstract SystemC construct, but still required to
support more complex functionality such as having a pointer to an external
object inside a SystemC module, e.g., for interoperability purposes.

Traits: `HasParent<SCModuleOp>`, `NoTerminator`, `SingleBlock`

### `systemc.cpp.func` (::circt::systemc::FuncOp)

*An operation with a name containing a single `SSACFG` region*

Operations within the function cannot implicitly capture values defined
outside of the function, i.e. Functions are `IsolatedFromAbove`. All
external references must use function arguments or attributes that establish
a symbolic connection (e.g. symbols referenced by name via a string
attribute like SymbolRefAttr). An external function declaration (used when
referring to a function declared in some other module) has no body. While
the MLIR textual form provides a nice inline syntax for function arguments,
they are internally represented as “block arguments” to the first block in
the region.

Argument names are stored in a ‘argNames’ attribute, but used directly as
the SSA value’s names. They are verified to be unique and can be used to
print them, e.g., as C function argument names.

Only dialect attribute names may be specified in the attribute dictionaries
for function arguments, results, or the function itself.

Example:

```
// External function definitions.
systemc.cpp.func @abort()
systemc.cpp.func externC @scribble(i32, i64) -> i64

// A function that returns its argument twice:
systemc.cpp.func @count(%argumentName: i64) -> (i64, i64) {
  return %argumentName, %argumentName: i64, i64
}

// A function with an attribute
systemc.cpp.func @exampleFnAttr() attributes {dialectName.attrName = false}
```

Traits: `AutomaticAllocationScope`, `IsolatedFromAbove`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallableOpInterface`, `FunctionOpInterface`, `OpAsmOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `function_type` | ::mlir::TypeAttr | type attribute of function type |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `argNames` | ::mlir::ArrayAttr | string array attribute |
| `externC` | ::mlir::UnitAttr | unit attribute |
| `sym_visibility` | ::mlir::StringAttr | string attribute |

### `systemc.cpp.member_access` (::circt::systemc::MemberAccessOp)

*A C++ member access expression.*

Syntax:

```
operation ::= `systemc.cpp.member_access` $object $accessKind $memberName attr-dict `:`
              functional-type($object, $result)
```

Represents the C++ member access operators `.` and `->`.
The member name is passed as a plain string and is not checked for validity.
Additional qualifications (`A::E1`) or template disambiguation
(`template E1`) can be manually added to this plain member name string for
emission.
The type of the object and result cannot be verified as it is allowed to use
varbatim types such as `emitc.opaque`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `memberName` | ::mlir::StringAttr | string attribute |
| `accessKind` | ::MemberAccessKindAttr | C++ member access kind |

#### Operands:

| Operand | Description |
| --- | --- |
| `object` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `systemc.cpp.new` (::circt::systemc::NewOp)

*A C++ ’new’ expression.*

Syntax:

```
operation ::= `systemc.cpp.new` `(` $args `)` attr-dict `:` functional-type($args, $result)
```

Creates and initializes C++ objects using dynamic storage.
Note that the types of the constructor arguments are not verified in any way
w.r.t. the result’s type because it is allowed to use an opaque type for the
pointee.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `args` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | EmitC pointer type |

### `systemc.cpp.return` (::circt::systemc::ReturnOp)

*Function return operation*

Syntax:

```
operation ::= `systemc.cpp.return` attr-dict ($returnValues^ `:` type($returnValues))?
```

The `systemc.cpp.return` operation represents a return operation within a
function. The operand number and types must match the signature of the
function that contains the operation.

Example:

```
systemc.cpp.func @foo() : i32 {
  ...
  systemc.cpp.return %0 : i32
}
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<systemc::FuncOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `returnValues` | variadic of any type |

### `systemc.cpp.variable` (::circt::systemc::VariableOp)

*Declare a C++ variable with optional initialization value.*

Declares a variable according to C++ semantics. If an initialization value
is present, the variable will be assigned that value at the declaration,
e.g, `int varname = 0;`.

Interfaces: `HasCustomSSAName`, `SystemCNameDeclOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `init` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `variable` | any type |

### `systemc.ctor` (::circt::systemc::CtorOp)

*A constructor definition.*

Syntax:

```
operation ::= `systemc.ctor` attr-dict-with-keyword $body
```

Represents the SC\_CTOR macro as described in IEEE 1666-2011 §5.2.7.
The name of the module being constructed does not have to be passed
to this operation, but is automatically added during emission.

Traits: `HasParent<SCModuleOp>`, `NoTerminator`, `SingleBlock`

### `systemc.func` (::circt::systemc::SCFuncOp)

*A (void)->void member function of a SC\_MODULE.*

Syntax:

```
operation ::= `systemc.func` `` custom<ImplicitSSAName>($name) attr-dict-with-keyword $body
```

This operation does not represent a specific SystemC construct, but a
regular C++ member function with no arguments and a void return type.
These are used to implement module-internal logic and are registered to the
module using the SC\_METHOD, SC\_THREAD, and SC\_CTHREAD macros.

Traits: `HasParent<SCModuleOp>`, `NoTerminator`, `SingleBlock`

Interfaces: `HasCustomSSAName`, `InferTypeOpInterface`, `SystemCNameDeclOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `handle` | FunctionType with no inputs and results |

### `systemc.instance.bind_port` (::circt::systemc::BindPortOp)

*Binds a port of a module instance to a channel.*

The ports of a submodule have to be bound to channels in a module further
up in the instance hierarchy (as opposed to `sc_export` where the channel
has to reside in the same module or a submodule). Therefore, a port can be
bound to either a signal declared by the `systemc.signal` operation or to
a port with matching direction (and thus bound to a channel further up in
the hierarchy).
More information on ports can be found in IEEE 1666-2011 §5.12.,
in particular IEEE 1666-2011 §5.12.7. is about port binding.
More information on predefined channels can be found in IEEE 1666-2011 §6.

Traits: `HasParent<CtorOp>`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `portId` | ::mlir::IntegerAttr | index attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `instance` | a SystemC module type |
| `channel` | a SystemC sc\_in type or a SystemC sc\_inout type or a SystemC sc\_out type or a SystemC sc\_signal type |

### `systemc.instance.decl` (::circt::systemc::InstanceDeclOp)

*Declares a SystemC module instance.*

Syntax:

```
operation ::= `systemc.instance.decl` custom<ImplicitSSAName>($name) $moduleName attr-dict
              `:` qualified(type($instanceHandle))
```

Declares an instantiation of a SystemC module inside another SystemC module
by value. The instance handle returned by this operation can then be used
to initialize it in the constructor and to access its fields.
More information can be found in IEEE 1666-2011 §4.1.1.

Traits: `HasParent<SCModuleOp>`

Interfaces: `HasCustomSSAName`, `InstanceOpInterface`, `SymbolUserOpInterface`, `SystemCNameDeclOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `moduleName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Results:

| Result | Description |
| --- | --- |
| `instanceHandle` | a SystemC module type |

### `systemc.interop.verilated` (::circt::systemc::InteropVerilatedOp)

*Instantiates a verilated module.*

Syntax:

```
operation ::= `systemc.interop.verilated` $instanceName $moduleName
              custom<InputPortList>($inputs, type($inputs), $inputNames) `->`
              custom<OutputPortList>(type($results), $resultNames) attr-dict
```

Instantiates a verilated module represented by a hw.module operation
(usually the extern variant).

This operation also encodes the interoparability layer to connect its
context (i.e. the surrounding operation, input values, result values, and
types) to the C++ code of the verilated module.
When residing in a context that understands C++ (e.g., inside a SystemC
module), this refers to the instantiation of the class, assignment of the
input ports, the call to the eval() function and reading the output ports.

Additionally, properties of the verilated module can be specified in
a config attribute which influences the interop layer code generation
(not yet implemented).

Interfaces: `OpAsmOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `instanceName` | ::mlir::StringAttr | string attribute |
| `moduleName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `inputNames` | ::mlir::ArrayAttr | string array attribute |
| `resultNames` | ::mlir::ArrayAttr | string array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `systemc.method` (::circt::systemc::MethodOp)

*Represents the SystemC SC\_METHOD macro.*

Syntax:

```
operation ::= `systemc.method` $funcHandle attr-dict
```

Represents the SC\_METHOD macro as described in IEEE 1666-2011 §5.2.9.

#### Operands:

| Operand | Description |
| --- | --- |
| `funcHandle` | FunctionType with no inputs and results |

### `systemc.module` (::circt::systemc::SCModuleOp)

*Define a SystemC SC\_MODULE.*

Represents the SC\_MODULE macro as described in IEEE 1666-2011 §5.2.5.
Models input, output and inout ports as module arguments (as opposed to
`sc_signal`s which are modeled by a separate `systemc.signal` operation),
but are nonetheless emitted as regular struct fields.

Traits: `HasParent<mlir::ModuleOp>`, `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallableOpInterface`, `FunctionOpInterface`, `ModuleOpInterface`, `OpAsmOpInterface`, `RegionKindInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `function_type` | ::mlir::TypeAttr | type attribute of function type |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `portNames` | ::mlir::ArrayAttr | string array attribute |

### `systemc.sensitive` (::circt::systemc::SensitiveOp)

*Describes the static sensitivity of an unspawned process.*

Syntax:

```
operation ::= `systemc.sensitive` $sensitivities attr-dict ( `:` qualified(type($sensitivities))^ )?
```

This operation allows to specify the static sensitivity of unspawned
processes. An unspawned process is one created by the `SC_METHOD`,
`SC_THREAD`, or `SC_CTHREAD` macro. The operands to this operation are
registered as sensitivities to the process last created (in control-flow
order).
Each `SC_MODULE` contains an instance of the `sc_sensitive` class to do the
registration. For a description of the `sc_sensitive` class refer to
IEEE 1666-2011 §5.4. For a description of the `sensitive` data member of
`SC_MODULE` refer to IEEE 1666-2011 §5.2.14.

Traits: `HasParent<CtorOp>`

#### Operands:

| Operand | Description |
| --- | --- |
| `sensitivities` | variadic of a SystemC sc\_in type or a SystemC sc\_inout type or a SystemC sc\_out type or a SystemC sc\_signal type |

### `systemc.signal` (::circt::systemc::SignalOp)

*Declares a SystemC `sc_signal<T>`.*

Syntax:

```
operation ::= `systemc.signal` custom<ImplicitSSAName>($name) ( `named` $named^ )? attr-dict
              `:` qualified(type($signal))
```

Represents the `sc_signal` template as described in IEEE 1666-2011 §6.4.
Adding the ’named’ attribute will lead to the signal being emitted with the
`SC_NAMED` convenience macro. Note that this macro is not part of
IEEE 1666-2011, but was added in version 2.3.3.

Traits: `HasParent<SCModuleOp>`

Interfaces: `HasCustomSSAName`, `SystemCNameDeclOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `named` | ::mlir::UnitAttr | unit attribute |

#### Results:

| Result | Description |
| --- | --- |
| `signal` | a SystemC sc\_signal type |

### `systemc.signal.read` (::circt::systemc::SignalReadOp)

*Returns the current value of a signal or port.*

Syntax:

```
operation ::= `systemc.signal.read` $input attr-dict `:` qualified(type($input))
```

Represents the member function `const T& read() const;` and operator
`operator const T& () const;` of class `sc_signal` as described in
IEEE 1666-2011 §6.4.7., of class `sc_in` as described in §6.8.3., and
of class `sc_inout` as decribed in §6.10.3. It shall return the current
value of the signal/port.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a SystemC sc\_in type or a SystemC sc\_inout type or a SystemC sc\_signal type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `systemc.signal.write` (::circt::systemc::SignalWriteOp)

*Writes a value to a signal or port.*

Syntax:

```
operation ::= `systemc.signal.write` $dest `,` $src attr-dict `:` qualified(type($dest))
```

Represents the member function `void write(const T&);` and several variants
of the operator `operator=` of class `sc_signal` as described in
IEEE 1666-2011 §6.4.8., of class `sc_inout` as described in §6.10.3., and of
class `sc_out` as decribed in §6.12.3. It shall modify the value of the
signal/port such that it appears to have the new value (as observed using
the `sytemc.signal.read` operation) in the next delta cycle but not before
then.

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | a SystemC sc\_out type or a SystemC sc\_inout type or a SystemC sc\_signal type |
| `src` | any type |

### `systemc.thread` (::circt::systemc::ThreadOp)

*Represents the SystemC SC\_THREAD macro.*

Syntax:

```
operation ::= `systemc.thread` $funcHandle attr-dict
```

Represents the SC\_THREAD macro as described in IEEE 1666-2011 §5.2.9.

#### Operands:

| Operand | Description |
| --- | --- |
| `funcHandle` | FunctionType with no inputs and results |

Type constraints
----------------

### a SystemC sc\_bigint type

### a SystemC sc\_biguint type

### a SystemC sc\_bv\_base type

### a SystemC sc\_bv type

### a SystemC sc\_int\_base type

### a SystemC sc\_int type

### a SystemC sc\_lv\_base type

### a SystemC sc\_lv type

### FunctionType with no inputs and results

### a SystemC sc\_signed type

### a SystemC sc\_uint\_base type

### a SystemC sc\_uint type

### a SystemC sc\_unsigned type

### a SystemC sc\_value\_base type

Types
-----

### InOutType

*A SystemC sc\_inout type*

Syntax:

```
!systemc.inout<
  ::mlir::Type   # baseType
>
```

Represents the specialized SystemC port class sc\_inout as described in
IEEE 1666-2011 §6.10.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| baseType | `::mlir::Type` |  |

### InputType

*A SystemC sc\_in type*

Syntax:

```
!systemc.in<
  ::mlir::Type   # baseType
>
```

Represents the specialized SystemC port class sc\_in as described in
IEEE 1666-2011 §6.8.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| baseType | `::mlir::Type` |  |

### LogicType

*A SystemC sc\_logic type*

Syntax: `!systemc.logic`

Represents a single bit with with a value corresponding to one of the four
logic states ‘0’, ‘1’, ‘Z’, and ‘X’ in SystemC as described in
IEEE 1666-2011 §7.9.2.
A value of this type can be created using the ’emitc.constant’ operation
with a string attribute containing “0”, “1”, “X”, “Z”, “x”, or “z” or an
i1 attribute representing “true” or “false”. Any other value will be
interpreted as ‘X’ (unknown state).

### ModuleType

*A SystemC module type*

Represents a SystemC module instantiation. Example:
`!systemc.module<moduleName(portName1: type1, portName2: type2)>`

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| moduleName | `mlir::StringAttr` |  |
| ports | `::llvm::ArrayRef<::circt::systemc::ModuleType::PortInfo>` | module ports |

### OutputType

*A SystemC sc\_out type*

Syntax:

```
!systemc.out<
  ::mlir::Type   # baseType
>
```

Represents the specialized SystemC port class sc\_out as described in
IEEE 1666-2011 §6.12.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| baseType | `::mlir::Type` |  |

### SignalType

*A SystemC sc\_signal type*

Syntax:

```
!systemc.signal<
  ::mlir::Type   # baseType
>
```

Represents the predefined primitive channel sc\_signal as described in
IEEE 1666-2011 §6.4.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| baseType | `::mlir::Type` |  |

Enums
-----

### MemberAccessKind

*C++ member access kind*

#### Cases:

| Symbol | Value | String |
| --- | --- | --- |
| Dot | `0` | dot |
| Arrow | `1` | arrow |

'systemc' Dialect Docs
----------------------

* [SystemC Dialect Rationale](https://circt.llvm.org/docs/Dialects/SystemC/RationaleSystemC/)

 [Prev - Synth Longest Path Analysis](https://circt.llvm.org/docs/Dialects/Synth/LongestPathAnalysis/ "Synth Longest Path Analysis")
[Next - SystemC Dialect Rationale](https://circt.llvm.org/docs/Dialects/SystemC/RationaleSystemC/ "SystemC Dialect Rationale") 

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