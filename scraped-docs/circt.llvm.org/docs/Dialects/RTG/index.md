Random Test Generation (RTG) Rationale - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

Random Test Generation (RTG) Rationale
======================================

This dialect provides types, operations, and interfaces modeling randomization
constructs for random test generation. Furthermore, it contains passes to
perform and analyze the represented randomization.

* [Rationale](#rationale)
* [Interfacing with the RTG dialect](#interfacing-with-the-rtg-dialect)
* [Randomization](#randomization)
* [Example](#example)
* [Use-case-specific constructs](#use-case-specific-constructs)
  + [ISA Tests](#isa-tests)
* [Frontends](#frontends)
* [Backends](#backends)
* [Operations](#operations)
  + [`rtg.array_create` (::circt::rtg::ArrayCreateOp)](#rtgarray_create-circtrtgarraycreateop)
  + [`rtg.array_extract` (::circt::rtg::ArrayExtractOp)](#rtgarray_extract-circtrtgarrayextractop)
  + [`rtg.array_inject` (::circt::rtg::ArrayInjectOp)](#rtgarray_inject-circtrtgarrayinjectop)
  + [`rtg.array_size` (::circt::rtg::ArraySizeOp)](#rtgarray_size-circtrtgarraysizeop)
  + [`rtg.bag_convert_to_set` (::circt::rtg::BagConvertToSetOp)](#rtgbag_convert_to_set-circtrtgbagconverttosetop)
  + [`rtg.bag_create` (::circt::rtg::BagCreateOp)](#rtgbag_create-circtrtgbagcreateop)
  + [`rtg.bag_difference` (::circt::rtg::BagDifferenceOp)](#rtgbag_difference-circtrtgbagdifferenceop)
  + [`rtg.bag_select_random` (::circt::rtg::BagSelectRandomOp)](#rtgbag_select_random-circtrtgbagselectrandomop)
  + [`rtg.bag_union` (::circt::rtg::BagUnionOp)](#rtgbag_union-circtrtgbagunionop)
  + [`rtg.bag_unique_size` (::circt::rtg::BagUniqueSizeOp)](#rtgbag_unique_size-circtrtgbaguniquesizeop)
  + [`rtg.comment` (::circt::rtg::CommentOp)](#rtgcomment-circtrtgcommentop)
  + [`rtg.isa.concat_immediate` (::circt::rtg::ConcatImmediateOp)](#rtgisaconcat_immediate-circtrtgconcatimmediateop)
  + [`rtg.constant` (::circt::rtg::ConstantOp)](#rtgconstant-circtrtgconstantop)
  + [`rtg.constraint` (::circt::rtg::ConstraintOp)](#rtgconstraint-circtrtgconstraintop)
  + [`rtg.context_switch` (::circt::rtg::ContextSwitchOp)](#rtgcontext_switch-circtrtgcontextswitchop)
  + [`rtg.embed_sequence` (::circt::rtg::EmbedSequenceOp)](#rtgembed_sequence-circtrtgembedsequenceop)
  + [`rtg.get_sequence` (::circt::rtg::GetSequenceOp)](#rtgget_sequence-circtrtggetsequenceop)
  + [`rtg.isa.int_to_immediate` (::circt::rtg::IntToImmediateOp)](#rtgisaint_to_immediate-circtrtginttoimmediateop)
  + [`rtg.interleave_sequences` (::circt::rtg::InterleaveSequencesOp)](#rtginterleave_sequences-circtrtginterleavesequencesop)
  + [`rtg.label_decl` (::circt::rtg::LabelDeclOp)](#rtglabel_decl-circtrtglabeldeclop)
  + [`rtg.label` (::circt::rtg::LabelOp)](#rtglabel-circtrtglabelop)
  + [`rtg.label_unique_decl` (::circt::rtg::LabelUniqueDeclOp)](#rtglabel_unique_decl-circtrtglabeluniquedeclop)
  + [`rtg.isa.memory_alloc` (::circt::rtg::MemoryAllocOp)](#rtgisamemory_alloc-circtrtgmemoryallocop)
  + [`rtg.isa.memory_base_address` (::circt::rtg::MemoryBaseAddressOp)](#rtgisamemory_base_address-circtrtgmemorybaseaddressop)
  + [`rtg.isa.memory_block_declare` (::circt::rtg::MemoryBlockDeclareOp)](#rtgisamemory_block_declare-circtrtgmemoryblockdeclareop)
  + [`rtg.isa.memory_size` (::circt::rtg::MemorySizeOp)](#rtgisamemory_size-circtrtgmemorysizeop)
  + [`rtg.on_context` (::circt::rtg::OnContextOp)](#rtgon_context-circtrtgoncontextop)
  + [`rtg.random_number_in_range` (::circt::rtg::RandomNumberInRangeOp)](#rtgrandom_number_in_range-circtrtgrandomnumberinrangeop)
  + [`rtg.randomize_sequence` (::circt::rtg::RandomizeSequenceOp)](#rtgrandomize_sequence-circtrtgrandomizesequenceop)
  + [`rtg.sequence` (::circt::rtg::SequenceOp)](#rtgsequence-circtrtgsequenceop)
  + [`rtg.set_cartesian_product` (::circt::rtg::SetCartesianProductOp)](#rtgset_cartesian_product-circtrtgsetcartesianproductop)
  + [`rtg.set_convert_to_bag` (::circt::rtg::SetConvertToBagOp)](#rtgset_convert_to_bag-circtrtgsetconverttobagop)
  + [`rtg.set_create` (::circt::rtg::SetCreateOp)](#rtgset_create-circtrtgsetcreateop)
  + [`rtg.set_difference` (::circt::rtg::SetDifferenceOp)](#rtgset_difference-circtrtgsetdifferenceop)
  + [`rtg.set_select_random` (::circt::rtg::SetSelectRandomOp)](#rtgset_select_random-circtrtgsetselectrandomop)
  + [`rtg.set_size` (::circt::rtg::SetSizeOp)](#rtgset_size-circtrtgsetsizeop)
  + [`rtg.set_union` (::circt::rtg::SetUnionOp)](#rtgset_union-circtrtgsetunionop)
  + [`rtg.isa.slice_immediate` (::circt::rtg::SliceImmediateOp)](#rtgisaslice_immediate-circtrtgsliceimmediateop)
  + [`rtg.isa.space` (::circt::rtg::SpaceOp)](#rtgisaspace-circtrtgspaceop)
  + [`rtg.isa.string_data` (::circt::rtg::StringDataOp)](#rtgisastring_data-circtrtgstringdataop)
  + [`rtg.substitute_sequence` (::circt::rtg::SubstituteSequenceOp)](#rtgsubstitute_sequence-circtrtgsubstitutesequenceop)
  + [`rtg.target` (::circt::rtg::TargetOp)](#rtgtarget-circtrtgtargetop)
  + [`rtg.test.failure` (::circt::rtg::TestFailureOp)](#rtgtestfailure-circtrtgtestfailureop)
  + [`rtg.test` (::circt::rtg::TestOp)](#rtgtest-circtrtgtestop)
  + [`rtg.test.success` (::circt::rtg::TestSuccessOp)](#rtgtestsuccess-circtrtgtestsuccessop)
  + [`rtg.tuple_create` (::circt::rtg::TupleCreateOp)](#rtgtuple_create-circtrtgtuplecreateop)
  + [`rtg.tuple_extract` (::circt::rtg::TupleExtractOp)](#rtgtuple_extract-circtrtgtupleextractop)
  + [`rtg.validate` (::circt::rtg::ValidateOp)](#rtgvalidate-circtrtgvalidateop)
  + [`rtg.virtual_reg` (::circt::rtg::VirtualRegisterOp)](#rtgvirtual_reg-circtrtgvirtualregisterop)
  + [`rtg.yield` (::circt::rtg::YieldOp)](#rtgyield-circtrtgyieldop)
* [Types](#types)
  + [ArrayType](#arraytype)
  + [BagType](#bagtype)
  + [BFloat16Type](#bfloat16type)
  + [ComplexType](#complextype)
  + [Float4E2M1FNType](#float4e2m1fntype)
  + [Float6E2M3FNType](#float6e2m3fntype)
  + [Float6E3M2FNType](#float6e3m2fntype)
  + [Float8E3M4Type](#float8e3m4type)
  + [Float8E4M3Type](#float8e4m3type)
  + [Float8E4M3B11FNUZType](#float8e4m3b11fnuztype)
  + [Float8E4M3FNType](#float8e4m3fntype)
  + [Float8E4M3FNUZType](#float8e4m3fnuztype)
  + [Float8E5M2Type](#float8e5m2type)
  + [Float8E5M2FNUZType](#float8e5m2fnuztype)
  + [Float8E8M0FNUType](#float8e8m0fnutype)
  + [Float16Type](#float16type)
  + [Float32Type](#float32type)
  + [Float64Type](#float64type)
  + [Float80Type](#float80type)
  + [Float128Type](#float128type)
  + [FloatTF32Type](#floattf32type)
  + [FunctionType](#functiontype)
  + [GraphType](#graphtype)
  + [IndexType](#indextype)
  + [IntegerType](#integertype)
  + [MemRefType](#memreftype)
  + [NoneType](#nonetype)
  + [OpaqueType](#opaquetype)
  + [RankedTensorType](#rankedtensortype)
  + [TupleType](#tupletype)
  + [UnrankedMemRefType](#unrankedmemreftype)
  + [UnrankedTensorType](#unrankedtensortype)
  + [VectorType](#vectortype)
  + [DictType](#dicttype)
  + [ImmediateType](#immediatetype)
  + [LabelType](#labeltype)
  + [MemoryBlockType](#memoryblocktype)
  + [MemoryType](#memorytype)
  + [RandomizedSequenceType](#randomizedsequencetype)
  + [SequenceType](#sequencetype)
  + [SetType](#settype)
  + [TupleType](#tupletype-1)
* [Passes](#passes)
  + [`-rtg-elaborate`](#-rtg-elaborate)
  + [`-rtg-embed-validation-values`](#-rtg-embed-validation-values)
  + [`-rtg-emit-isa-assembly`](#-rtg-emit-isa-assembly)
  + [`-rtg-inline-sequences`](#-rtg-inline-sequences)
  + [`-rtg-insert-test-to-file-mapping`](#-rtg-insert-test-to-file-mapping)
  + [`-rtg-linear-scan-register-allocation`](#-rtg-linear-scan-register-allocation)
  + [`-rtg-lower-unique-labels`](#-rtg-lower-unique-labels)
  + [`-rtg-lower-validate-to-labels`](#-rtg-lower-validate-to-labels)
  + [`-rtg-materialize-constraints`](#-rtg-materialize-constraints)
  + [`-rtg-memory-allocation`](#-rtg-memory-allocation)
  + [`-rtg-print-test-names`](#-rtg-print-test-names)
  + [`-rtg-simple-test-inliner`](#-rtg-simple-test-inliner)
  + [`-rtg-unique-validate`](#-rtg-unique-validate)

Rationale [¶](#rationale)
-------------------------

This dialect aims to provide a unified representation for randomized tests, more
precisely the parts of the tests that encode the randomization constructs (e.g.,
picking a random resource from a set of resources).
This means, this dialect is only useful in combination with at least one other
dialect that represents the actual test constructs, i.e., the parts that get
randomized. After all the randomization constructs are fully elaborated, the
resulting test will only consist of those dialects.

Examples for tests that can be randomized, and thus candidates for companion
dialects, are instruction sequences of any ISA (Instruction Set Architecture),
transaction sequences over a latency insensitive (ready-valid) channel, input
sequences to an FSM (Finite State Machine), transaction sequences for
potentially any other protocol. Albeit, the initial motivation is ISA tests and
will thus be the best supported, at least initially.

While it should be valid to add constructs to this dialect that are special to
any of the above mentioned use-cases, the dialect should generally be designed
such that all of them can be supported.

Interfacing with the RTG dialect [¶](#interfacing-with-the-rtg-dialect)
-----------------------------------------------------------------------

The RTG dialect is only useful in combination with at least one other dialect
representing the test. Such a dialect has to be aware of the RTG dialect and
has to implement various interfaces (depending on which parts of the RTG
infrastructure it intends to use).

A test can be declared with the `rtg.test` operation which takes an
`!rtg.target` type attribute to specify requirements for the test to be able to
be generated and executed. In addition to the target having to provide an
`!rtg.target` typed value that is a refinement of the required target type, the
test can specify additional requirements using the `rtg.require` operation.

Targets can be defined using the `rtg.target` operation. Certain operations
are only allowed inside the target operation but not the test operation to
guarantee that certain resources (specifically contexts) cannot be generated
on-the-fly.

The RTG framework will match all tests with all targets that fulfill their
requirements. The user can specify which targets and tests should be included
in the test generation process and how many tests should be generated per
target or in total.

Currently, there are three concepts in the RTG dialect the companion dialects
can map their constructs to:

* *Instructions*: instructions/operations intended to be performed in series
  by the test, such as ISA instructions, or protocol transactions
* *Resources*: resources the instructions operate on, such as registers or
  memories in ISA tests, or channels in hardware transaction tests
* *Contexts*: the context or environment the instuctions are performed in,
  e.g., on which CPU and in which mode for ISA test.

To express their mapping, the companion dialects must implement the the
interfaces as follows.

**Instructions**

They implement `InstructionOpInterface` on all operations that represent an
instruction, transaction, or similar. Those operations are intended to be
statements and not define SSA values.

**Resources**

Operations that create, declare, define, or produce a handle to a resource
instance must implement the `ResourceOpInterface` interface and define at least
one SSA value of a type that implements the `ResourceTypeInterface` interface.
(TODO: maybe we don’t actually need the `ResourceTypeInterface`?)

The `RegisterOpInterface` can be implemented in addition to the
`ResourceOpInterface` if the resource is a register to become supported by the
register allocation and assembly emission pass.

**Contexts**

The `ContextResourceType` is used to represent contexts. Instructions can be
placed in specific context using the `rtg.on_context` operation (Note: we might
promote this to a type interface in the future).
Operations that define contexts (i.e., create new contexts) must implement the
`ContextResourceOpInterface` interface.
Operations implementing the `ContextResourceOpInterface` interface are only
allowed inside the `rtg.target` operation.

Randomization [¶](#randomization)
---------------------------------

This dialect aims to provide utilities that allow users to generate tests
anywhere on the spectrum from directed tests to (almost) fully random tests
(refer to *fuzzing*).

* *Constraints*: allow the user to specify constraints to avoid generating
  illegal or useless tests, e.g., by
  + allowing to specify a sequence of instructions that always have to be picked
    in exactly that order and form (see `rtg.sequence` operation)
  + dependencies between resources or resource usages
  + etc.
* *Probabilities and Biases*: allow certain tests (or parts of a test) to be
  picked more likely than others (can be seen as ‘soft-constraints’) (see
  `!rtg.bag` type and associated operations)
* *Enumerations*: allow to enumerate all tests that can be produced by the
  current randomness constraints, possibly in a way that places the more likely
  tests to occur earlier in the enumeration

(TODO: expand here once these things are built out)

Main IR constructs to introduce randomness:

* *Sets*: a set of elements of the same type; the usual set operations apply as
  well as uniformly at random picking one element of the set
* *Bags/Biased Sets*: a generalization of *sets* that allows one element to
  occur multiple times and thus make it more likely to be picked (i.e., models
  non-uniform distributions)

Example [¶](#example)
---------------------

This section provides an (almost) E2E walkthrough of a simple example starting
at a Python implmentation using the library wrapping around the python bindings,
showing the generated RTG IR, and the fully elaborated IR. This compilation
process can be performed in one go using the `rtgtool.py` driver script.

```
# Define a test target (essentially a design/machine with 4 CPUs)
@rtg.target([('cpus', rtg.set_of(rtg.context_resource()))])
def example_target():
  # return a set containing 4 CPUs which the test can schedule instruction
  # sequences on
  return [rtg.Set.create([rv64.core(0), rv64.core(1), rv64.core(2), rv64.core(3)])]

# Define a sequence (for simplicity it only contains one instruction)
# Note: not adding the sequence decorator is also valid in this example  but
# means the function is fully inlined at python execution time. It is not valid,
# however, if the sequence is added to a set or bag to be selected at random.
@rtg.sequence
def seq(register):
  # ADD Immediate instruction adding 4 to to the given register
  rtgtest.addi(register, rtgtest.imm(4, 12))

# Define a test that requires a target with CPUs to schedule instruction
# sequences on
@rtg.test([('cpus', rtg.set_of(rtg.context_resource()))])
def example(cpus):
  # Pick a random CPU and schedule the ADD operation on it
  with rtg.context(cpus.get_random_and_exclude()):
    rtg.label('label0')
    seq(rtgtest.sp())

  # Pick a random CPU that was not already picked above and zero out the stack
  # pointer register on it
  with rtg.context(cpus.get_random()):
    ra = rtgtest.sp()
    rtgtest.xor(ra, ra)
```

The driver script will elaborate this python file and produce the following RTG
IR as an intermediate step:

```
rtg.target @example_target : !rtg.target<cpus: !rtg.set<!rtg.context_resource>> {
  %0 = rtgtest.coreid 0
  %1 = rtgtest.coreid 1
  %2 = rtgtest.coreid 2
  %3 = rtgtest.coreid 3
  %4 = rtg.set_create %0, %1, %2, %3 : !rtg.context_resource
  rtg.yield %4 : !rtg.set<!rtg.context_resource>
}
rtg.sequence @seq {
^bb0(%reg: !rtgtest.reg):
  %c4_i12 = arith.constant 4 : i12
  rtgtest.addi %reg, %c4_i12
}
rtg.sequence @context0 {
  // Labels are declared before being placed in the instruction stream such that
  // we can insert jump instructions before the jump target.
  %0 = rtg.label.decl "label0" -> index
  rtg.label %0 : index
  %sp = rtgtest.sp
  // Construct a closure such that it can be easily passed around, e.g.,
  // inserted into a set with other sequence closures to be selected at random.
  %1 = rtg.sequence_closure @seq(%sp) : !rtgtest.reg
  // Place the sequence here (i.e., inline it here with the arguments passed to
  // the closure).
  // This is essentially the same as an `rtg.on_context` with the context
  // operand matching the one of the parent `on_context`.
  rtg.sequence_invoke %1
}
rtg.sequence @context1 {
  %0 = rtgtest.sp
  rtgtest.xor %sp, %sp
}
rtg.test @example : !rtg.target<cpus: !rtg.set<!rtg.context_resource>> {
^bb0(%arg0: !rtg.set<!rtg.context_resource>):
  // Select an element from the set uniformly at random
  %0 = rtg.set_select_random %arg0 : !rtg.set<!rtg.context_resource>
  %3 = rtg.sequence_closure @context0
  // Place the sequence closure on the given context. In this example, there
  // will be guards inserted that make sure the inlined sequence is only
  // executed by the CPU specified by the selected coreid.
  rtg.on_context %0, %3 : !rtg.context_resource
  // Construct a new set that doesn't contain the selected element (RTG sets are
  // immutable) and select another element randomly from this new set.
  %1 = rtg.set_create %0 : !rtg.context_resource
  %2 = rtg.set_difference %arg0, %1 : !rtg.set<!rtg.context_resource>
  %7 = rtg.set_select_random %2 : !rtg.set<!rtg.context_resource>
  %8 = rtg.sequence_closure @context1
  rtg.on_context %7, %8 : !rtg.context_resource
}
```

Once all the RTG randomization passes were performed, the example looks like
this:

```
// Two regions, the first one to be executed on CPU with coreid 0 and the second
// one on CPU with coreid 2
rtg.rendered_context [0,2] {
  %0 = rtg.label.decl "label0" -> index
  rtg.label %0 : index
  %reg = rtgtest.sp
  %c4 = arith.constant 4 : i12
  rtgtest.addi %sp, %c4
  // Is emitted to assembly looking something like:
  // label0:
  // addi sp, 4
}, {
  %sp = rtgtest.sp
  rtgtest.xor %sp, %sp
}
```

The last step to run this test is to print it in assembly format and invoke the
assembler. This also includes the insertion of a considerable amount of
boilerplate setup code to run the above instructions on the right CPUs which is
omitted in this example for clarity.

Use-case-specific constructs [¶](#use-case-specific-constructs)
---------------------------------------------------------------

This section provides an overview of operations/types/interfaces added with the
intend to be only used for one (or a few) specific use-cases/test-targets.

### ISA Tests [¶](#isa-tests)

* *Labels*: Handling of labels is added to the RTG dialect because they are common
  across ISAs. Leaving them to the ISA specific companion dialects would likely
  lead to frequent code duplication (once for each ISA).
* *Register Allocation Pass*
* *Assembly Emission Pass*

Frontends [¶](#frontends)
-------------------------

Any dialect or entry point is allowed to generate valid RTG IR with a companion
dialect. The dialect already comes with an extensive CAPI, Python Bindings, and
a small Python library that simplify usage over the pure Python Bindings.

Backends [¶](#backends)
-----------------------

The RTG dialect does not have a backend itself. It is fully lowered by its
dialect transformation passes that perform the randomization. The result will be
IR consisting purely of the companion dialect and thus it is up to this
companion dialect to define any backends.

Operations [¶](#operations)
---------------------------

### `rtg.array_create` (::circt::rtg::ArrayCreateOp) [¶](#rtgarray_create-circtrtgarraycreateop)

*Create an array with an initial list of elements*

This operation creates an array from a list of values. The element on the
left-most position in the MLIR assembly format ends up at index 0.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands)

| Operand | Description |
| --- | --- |
| `elements` | variadic of any type |

#### Results: [¶](#results)

| Result | Description |
| --- | --- |
| `result` | an array type with dynamic size |

### `rtg.array_extract` (::circt::rtg::ArrayExtractOp) [¶](#rtgarray_extract-circtrtgarrayextractop)

*Get an element from an array*

Syntax:

```
operation ::= `rtg.array_extract` $array `[` $index `]` `:` qualified(type($array)) attr-dict
```

This operation returns the element at the given index of the array.
Accessing out-of-bounds indices is (immediate) UB.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-1)

| Operand | Description |
| --- | --- |
| `array` | an array type with dynamic size |
| `index` | index |

#### Results: [¶](#results-1)

| Result | Description |
| --- | --- |
| `result` | any type |

### `rtg.array_inject` (::circt::rtg::ArrayInjectOp) [¶](#rtgarray_inject-circtrtgarrayinjectop)

*Set an element in an array*

Syntax:

```
operation ::= `rtg.array_inject` $array `[` $index `]` `,` $value `:` qualified(type($array)) attr-dict
```

This operation produces a new array of the same type as the input array and
sets the element at the given index to the given value. All other values
remain the same. An OOB access is (immediate) undefined behavior.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-2)

| Operand | Description |
| --- | --- |
| `array` | an array type with dynamic size |
| `index` | index |
| `value` | any type |

#### Results: [¶](#results-2)

| Result | Description |
| --- | --- |
| `result` | an array type with dynamic size |

### `rtg.array_size` (::circt::rtg::ArraySizeOp) [¶](#rtgarray_size-circtrtgarraysizeop)

*Return the size of an array*

Syntax:

```
operation ::= `rtg.array_size` $array `:` qualified(type($array)) attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-3)

| Operand | Description |
| --- | --- |
| `array` | an array type with dynamic size |

#### Results: [¶](#results-3)

| Result | Description |
| --- | --- |
| `result` | index |

### `rtg.bag_convert_to_set` (::circt::rtg::BagConvertToSetOp) [¶](#rtgbag_convert_to_set-circtrtgbagconverttosetop)

*Convert a bag to a set*

Syntax:

```
operation ::= `rtg.bag_convert_to_set` $input `:` qualified(type($input)) attr-dict
```

This operation converts a bag to a set by dropping all duplicate elements.
For example, the bag `{a, a, b}` is converted to `{a, b}`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-4)

| Operand | Description |
| --- | --- |
| `input` | a bag of values |

#### Results: [¶](#results-4)

| Result | Description |
| --- | --- |
| `result` | a set of values |

### `rtg.bag_create` (::circt::rtg::BagCreateOp) [¶](#rtgbag_create-circtrtgbagcreateop)

*Constructs a bag*

This operation constructs a bag with the provided values and associated
multiples. This means the bag constructed in the following example contains
two of each `%arg0` and `%arg0` (`{%arg0, %arg0, %arg1, %arg1}`).

```
%0 = arith.constant 2 : index
%1 = rtg.bag_create (%0 x %arg0, %0 x %arg1) : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `SameVariadicOperandSize`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-5)

| Operand | Description |
| --- | --- |
| `elements` | variadic of any type |
| `multiples` | variadic of index |

#### Results: [¶](#results-5)

| Result | Description |
| --- | --- |
| `bag` | a bag of values |

### `rtg.bag_difference` (::circt::rtg::BagDifferenceOp) [¶](#rtgbag_difference-circtrtgbagdifferenceop)

*Computes the difference of two bags*

Syntax:

```
operation ::= `rtg.bag_difference` $original `,` $diff (`inf` $inf^)? `:` qualified(type($output)) attr-dict
```

For each element the resulting bag will have as many fewer than the
‘original’ bag as there are in the ‘diff’ bag. However, if the ‘inf’
attribute is attached, all elements of that kind will be removed (i.e., it
is assumed the ‘diff’ bag has infinitely many copies of each element).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inf` | ::mlir::UnitAttr | unit attribute |

#### Operands: [¶](#operands-6)

| Operand | Description |
| --- | --- |
| `original` | a bag of values |
| `diff` | a bag of values |

#### Results: [¶](#results-6)

| Result | Description |
| --- | --- |
| `output` | a bag of values |

### `rtg.bag_select_random` (::circt::rtg::BagSelectRandomOp) [¶](#rtgbag_select_random-circtrtgbagselectrandomop)

*Select a random element from the bag*

Syntax:

```
operation ::= `rtg.bag_select_random` $bag `:` qualified(type($bag)) attr-dict
```

This operation returns an element from the bag selected uniformely at
random. Therefore, the number of duplicates of each element can be used to
bias the distribution.
If the bag does not contain any elements, the behavior of this operation is
undefined.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-7)

| Operand | Description |
| --- | --- |
| `bag` | a bag of values |

#### Results: [¶](#results-7)

| Result | Description |
| --- | --- |
| `output` | any type |

### `rtg.bag_union` (::circt::rtg::BagUnionOp) [¶](#rtgbag_union-circtrtgbagunionop)

*Computes the union of bags*

Syntax:

```
operation ::= `rtg.bag_union` $bags `:` qualified(type($result)) attr-dict
```

Computes the union of the given bags. The list of sets must contain at
least one element.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-8)

| Operand | Description |
| --- | --- |
| `bags` | variadic of a bag of values |

#### Results: [¶](#results-8)

| Result | Description |
| --- | --- |
| `result` | a bag of values |

### `rtg.bag_unique_size` (::circt::rtg::BagUniqueSizeOp) [¶](#rtgbag_unique_size-circtrtgbaguniquesizeop)

*Returns the number of unique elements in the bag*

Syntax:

```
operation ::= `rtg.bag_unique_size` $bag `:` qualified(type($bag)) attr-dict
```

This operation returns the number of unique elements in the bag, i.e., for
the bag `{a, a, b, c, c}` it returns 3.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-9)

| Operand | Description |
| --- | --- |
| `bag` | a bag of values |

#### Results: [¶](#results-9)

| Result | Description |
| --- | --- |
| `result` | index |

### `rtg.comment` (::circt::rtg::CommentOp) [¶](#rtgcomment-circtrtgcommentop)

*Emit a comment in instruction stream*

Syntax:

```
operation ::= `rtg.comment` $comment attr-dict
```

#### Attributes: [¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `comment` | ::mlir::StringAttr | string attribute |

### `rtg.isa.concat_immediate` (::circt::rtg::ConcatImmediateOp) [¶](#rtgisaconcat_immediate-circtrtgconcatimmediateop)

*Concatenate immediates*

Syntax:

```
operation ::= `rtg.isa.concat_immediate` $operands `:` qualified(type($operands)) attr-dict
```

This operation concatenates a variadic number of immediates into a single
immediate. The operands are concatenated in order, with the first operand
becoming the most significant bits of the result.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-10)

| Operand | Description |
| --- | --- |
| `operands` | variadic of an ISA immediate |

#### Results: [¶](#results-10)

| Result | Description |
| --- | --- |
| `result` | an ISA immediate |

### `rtg.constant` (::circt::rtg::ConstantOp) [¶](#rtgconstant-circtrtgconstantop)

*Create an SSA value from an attribute*

Syntax:

```
operation ::= `rtg.constant` $value attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-2)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::TypedAttr | TypedAttr instance |

#### Results: [¶](#results-11)

| Result | Description |
| --- | --- |
| `result` | any type |

### `rtg.constraint` (::circt::rtg::ConstraintOp) [¶](#rtgconstraint-circtrtgconstraintop)

*Enforce a constraint*

Syntax:

```
operation ::= `rtg.constraint` $condition attr-dict
```

This operation enforces a constraint. This allows to specify additional
constraints on values after their construction. It should be avoided when
possible as these backward constraints are computationally more expensive.

Constraints should be tried to be solved such that all of them evaluate to
’true’. If the constraint system turns out to be unsolvable, the compiler
should error out gracefully as early as possible (it might also error out
when the system is too expensive to solve).

#### Operands: [¶](#operands-11)

| Operand | Description |
| --- | --- |
| `condition` | 1-bit signless integer |

### `rtg.context_switch` (::circt::rtg::ContextSwitchOp) [¶](#rtgcontext_switch-circtrtgcontextswitchop)

*A specification of how to switch contexts*

Syntax:

```
operation ::= `rtg.context_switch` $from `->` $to `,` $sequence `:` qualified(type($sequence)) attr-dict
```

This operation allows the user to specify a sequence of instructions to
switch from context ‘from’ to context ’to’, randomize and embed a provided
sequence, and switch back from context ’to’ to context ‘from’. This
sequence of instructions should be provided as the ‘sequence’ operand which
is a sequence of the type ‘!rtg.sequence<context-type-interface,
context-type-interface, !rtg.sequence>’. The first parameter is the ‘from’
context, the second one the ’to’ context, and the third is the sequence to
randomize and embed under the ’to’ context.

Traits: `HasParent<rtg::TargetOp>`

#### Attributes: [¶](#attributes-3)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `from` | ::circt::rtg::ContextResourceAttrInterface | ContextResourceAttrInterface instance |
| `to` | ::circt::rtg::ContextResourceAttrInterface | ContextResourceAttrInterface instance |

#### Operands: [¶](#operands-12)

| Operand | Description |
| --- | --- |
| `sequence` | handle to a sequence or sequence family |

### `rtg.embed_sequence` (::circt::rtg::EmbedSequenceOp) [¶](#rtgembed_sequence-circtrtgembedsequenceop)

*Embed a sequence of instructions into another sequence*

Syntax:

```
operation ::= `rtg.embed_sequence` $sequence attr-dict
```

This operation takes a fully randomized sequence and embeds it into another
sequence or test at the position of this operation.
In particular, this is not any kind of function call, it doesn’t set up a
stack frame, etc. It behaves as if the sequence of instructions it refers to
were directly inlined relacing this operation.

#### Operands: [¶](#operands-13)

| Operand | Description |
| --- | --- |
| `sequence` | handle to a fully randomized sequence |

### `rtg.get_sequence` (::circt::rtg::GetSequenceOp) [¶](#rtgget_sequence-circtrtggetsequenceop)

*Create a sequence value*

Syntax:

```
operation ::= `rtg.get_sequence` $sequence `:` qualified(type($ref)) attr-dict
```

This operation creates a sequence value referring to the provided sequence
by symbol. It allows sequences to be passed around as an SSA value. For
example, it can be inserted into a set and selected at random which is one
of the main ways to do randomization.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-4)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sequence` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Results: [¶](#results-12)

| Result | Description |
| --- | --- |
| `ref` | handle to a sequence or sequence family |

### `rtg.isa.int_to_immediate` (::circt::rtg::IntToImmediateOp) [¶](#rtgisaint_to_immediate-circtrtginttoimmediateop)

*Construct an immediate from an integer*

Syntax:

```
operation ::= `rtg.isa.int_to_immediate` $input `:` qualified(type($result)) attr-dict
```

Create an immediate of static bit-width from the provided integer. If the
integer does not fit in the specified bit-width, an error shall be emitted
when executing this operation.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-14)

| Operand | Description |
| --- | --- |
| `input` | index |

#### Results: [¶](#results-13)

| Result | Description |
| --- | --- |
| `result` | an ISA immediate |

### `rtg.interleave_sequences` (::circt::rtg::InterleaveSequencesOp) [¶](#rtginterleave_sequences-circtrtginterleavesequencesop)

*Interleave a list of sequences*

Syntax:

```
operation ::= `rtg.interleave_sequences` $sequences (`batch` $batchSize^)? attr-dict
```

This operation takes a list of (at least one) fully randomized sequences and
interleaves them by taking the next `batchSize` number of operations
implementing the `InstructionOpInterface` of each sequence round-robin.

Therefore, if only one sequence is in the list, this operation returns that
sequence unchanged.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-5)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `batchSize` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-15)

| Operand | Description |
| --- | --- |
| `sequences` | variadic of handle to a fully randomized sequence |

#### Results: [¶](#results-14)

| Result | Description |
| --- | --- |
| `interleavedSequence` | handle to a fully randomized sequence |

### `rtg.label_decl` (::circt::rtg::LabelDeclOp) [¶](#rtglabel_decl-circtrtglabeldeclop)

*Declares a label for an instruction sequence*

Syntax:

```
operation ::= `rtg.label_decl` $formatString (`,` $args^)? attr-dict
```

Declares a label that can then be placed by an `rtg.label` operation in an
instruction sequence, passed on to sequences via their arguments, and used
by instructions (e.g., as jump targets) by allowing ISA dialects to use them
directly as an operand of an instruction or by casting it to a value
representing an immediate.

The format string may contain placeholders of the form `{i}` where `i`
refers to the i-th element in `args`.
The declared label is uniqued by the compiler to no collide with any other
label declarations.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-6)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `formatString` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-16)

| Operand | Description |
| --- | --- |
| `args` | variadic of index |

#### Results: [¶](#results-15)

| Result | Description |
| --- | --- |
| `label` | a reference to a label |

### `rtg.label` (::circt::rtg::LabelOp) [¶](#rtglabel-circtrtglabelop)

*Places a label in an instruction sequence*

Syntax:

```
operation ::= `rtg.label` $visibility $label attr-dict
```

Any declared label must only be placed at most once in any fully elaborated
instruction sequence.

#### Attributes: [¶](#attributes-7)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `visibility` | ::circt::rtg::LabelVisibilityAttr | visibility specifiers for labels |

#### Operands: [¶](#operands-17)

| Operand | Description |
| --- | --- |
| `label` | a reference to a label |

### `rtg.label_unique_decl` (::circt::rtg::LabelUniqueDeclOp) [¶](#rtglabel_unique_decl-circtrtglabeluniquedeclop)

*Declares a unique label for an instruction sequence*

Syntax:

```
operation ::= `rtg.label_unique_decl` $formatString (`,` $args^)? attr-dict
```

Declares a label that can then be placed by an `rtg.label` operation in an
instruction sequence, passed on to sequences via their arguments, and used
by instructions (e.g., as jump targets) by allowing ISA dialects to use them
directly as an operand of an instruction or by casting it to a value
representing an immediate.

The format string may contain placeholders of the form `{i}` where `i`
refers to the i-th element in `args`.
The declared label is uniqued by the compiler to no collide with any other
label declarations.

Interfaces: `InferTypeOpInterface`

#### Attributes: [¶](#attributes-8)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `formatString` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-18)

| Operand | Description |
| --- | --- |
| `args` | variadic of index |

#### Results: [¶](#results-16)

| Result | Description |
| --- | --- |
| `label` | a reference to a label |

### `rtg.isa.memory_alloc` (::circt::rtg::MemoryAllocOp) [¶](#rtgisamemory_alloc-circtrtgmemoryallocop)

*Allocate a memory with the provided properties*

Syntax:

```
operation ::= `rtg.isa.memory_alloc` $memoryBlock `,` $size `,` $alignment
              `:` qualified(type($memoryBlock)) attr-dict
```

This operation declares a memory to be allocated with the provided
properties. It is only allowed to declare new memories in the `rtg.target`
operations and must be passed as argument to the `rtg.test`.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-19)

| Operand | Description |
| --- | --- |
| `memoryBlock` | handle to a memory block |
| `size` | index |
| `alignment` | index |

#### Results: [¶](#results-17)

| Result | Description |
| --- | --- |
| `result` | handle to a memory |

### `rtg.isa.memory_base_address` (::circt::rtg::MemoryBaseAddressOp) [¶](#rtgisamemory_base_address-circtrtgmemorybaseaddressop)

*Get the memory base address as an immediate*

Syntax:

```
operation ::= `rtg.isa.memory_base_address` $memory `:` qualified(type($memory)) attr-dict
```

This operation returns the base address of the given memory. The bit-width
of the returned immediate must match the address width of the given memory.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-20)

| Operand | Description |
| --- | --- |
| `memory` | handle to a memory |

#### Results: [¶](#results-18)

| Result | Description |
| --- | --- |
| `result` | an ISA immediate |

### `rtg.isa.memory_block_declare` (::circt::rtg::MemoryBlockDeclareOp) [¶](#rtgisamemory_block_declare-circtrtgmemoryblockdeclareop)

*Declare a memory block with the provided properties*

This operation declares a memory block to be allocated with the provided
properties. It is only allowed to declare new memory blocks in the
`rtg.target` operations and must be passed as argument to the `rtg.test`.
This is because the available memory blocks are specified by the hardware
design. This specification is fixed from the start and thus a test should
not be able to declare new memory blocks on-the-fly. However, tests are
allowed to allocate memory regions from these memory blocks.

The ‘baseAddress’ attribute specifies the first memory address (lowest
address representing a valid access to the memory) and the ’endAddress’
represents the last address (highest address that is valid to access the
memory).

Traits: `HasParent<rtg::TargetOp>`

#### Attributes: [¶](#attributes-9)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `baseAddress` | ::mlir::IntegerAttr | arbitrary integer attribute |
| `endAddress` | ::mlir::IntegerAttr | arbitrary integer attribute |

#### Results: [¶](#results-19)

| Result | Description |
| --- | --- |
| `result` | handle to a memory block |

### `rtg.isa.memory_size` (::circt::rtg::MemorySizeOp) [¶](#rtgisamemory_size-circtrtgmemorysizeop)

*Get the size of the memory in bytes*

Syntax:

```
operation ::= `rtg.isa.memory_size` $memory `:` qualified(type($memory)) attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-21)

| Operand | Description |
| --- | --- |
| `memory` | handle to a memory |

#### Results: [¶](#results-20)

| Result | Description |
| --- | --- |
| `result` | index |

### `rtg.on_context` (::circt::rtg::OnContextOp) [¶](#rtgon_context-circtrtgoncontextop)

*Places a sequence on a context*

Syntax:

```
operation ::= `rtg.on_context` $context `,` $sequence `:` qualified(type($context)) attr-dict
```

This operation takes a context and a fully substituted, but not yet
randomized sequence and inserts the necessary instructions to switch from
the current context to the provided context, randomizes and embeds the given
sequence under the given context, and inserts instructions to switch back to
the original context.

These instructions are provided by the ‘rtg.context\_switch’ operation. If no
‘rtg.context\_switch’ for this transition is provided, the compiler will
error out. If multiple such context switches apply, the most recently
registered one takes precedence.

#### Operands: [¶](#operands-22)

| Operand | Description |
| --- | --- |
| `context` | ContextResourceTypeInterface instance |
| `sequence` | fully substituted sequence type |

### `rtg.random_number_in_range` (::circt::rtg::RandomNumberInRangeOp) [¶](#rtgrandom_number_in_range-circtrtgrandomnumberinrangeop)

*Returns a number uniformly at random within the given range*

Syntax:

```
operation ::= `rtg.random_number_in_range` ` ` `[` $lowerBound `,` $upperBound `]` attr-dict
```

This operation computes a random number based on a uniform distribution
within the given range. Both the lower and upper bounds are inclusive. If
the range is empty, compilation will fail. This is (obviously) more
performant than inserting all legal numbers into a set and using
‘set\_select\_random’, but yields the same behavior.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-23)

| Operand | Description |
| --- | --- |
| `lowerBound` | index |
| `upperBound` | index |

#### Results: [¶](#results-21)

| Result | Description |
| --- | --- |
| `result` | index |

### `rtg.randomize_sequence` (::circt::rtg::RandomizeSequenceOp) [¶](#rtgrandomize_sequence-circtrtgrandomizesequenceop)

*Randomize the content of a sequence*

Syntax:

```
operation ::= `rtg.randomize_sequence` $sequence attr-dict
```

This operation takes a fully substituted sequence and randomizes its
content. This means, no operations the returned sequence does not contain
any randomization constructs anymore (such as random selection from sets and
bags, or other ‘randomize\_sequence’ operations).

It is useful to have this operation separate from ’embed\_sequence’ such that
the exact same sequence (i.e., with the same random choices taken) can be
embedded at multiple places.
It is also useful to have this separate from sequence substitution because
this operation is sensitive to the context, but the substitution values for
a sequence family might already be available in a parent sequence that is
placed on a different context. Thus, not having it separated would mean that
the substitution values must all be passed down as arguments to the child
sequence instead of a a single fully substituted sequence value.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-24)

| Operand | Description |
| --- | --- |
| `sequence` | fully substituted sequence type |

#### Results: [¶](#results-22)

| Result | Description |
| --- | --- |
| `randomizedSequence` | handle to a fully randomized sequence |

### `rtg.sequence` (::circt::rtg::SequenceOp) [¶](#rtgsequence-circtrtgsequenceop)

*A sequence of instructions*

This operation collects a sequence of instructions such that they can be
placed as one unit. This is effectively the way to impose a constraint on
the order and presence of some instructions.

It is allowed to contain randomization constructs and invokations on any
contexts. It is not allowed to create new context resources inside a
sequence, however.

This operation can be invoked by the `invoke` and `on_context` operations.
It is referred to by symbol and isolated from above to ease multi-threading
and it allows the `rtg.test` operation to be isolated-from-above to provide
stronger top-level isolation guarantees.

Traits: `HasParent<mlir::ModuleOp>`, `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`

Interfaces: `SymbolOpInterface`

#### Attributes: [¶](#attributes-10)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `sequenceType` | ::mlir::TypeAttr | type attribute of handle to a sequence or sequence family |

### `rtg.set_cartesian_product` (::circt::rtg::SetCartesianProductOp) [¶](#rtgset_cartesian_product-circtrtgsetcartesianproductop)

*Computes the n-ary cartesian product of sets*

Syntax:

```
operation ::= `rtg.set_cartesian_product` $inputs `:` qualified(type($inputs)) attr-dict
```

This operation computes a set of tuples from a list of input sets such that
each combination of elements from the input sets is present in the result
set. More formally, for n input sets it computes
`X_1 x ... x X_n = {(x_1, ..., x_n) | x_i \in X_i for i \in {1, ..., n}}`.
At least one input set has to be provided (i.e., `n > 0`).

For example, given two sets A and B with elements
`A = {a0, a1}, B = {b0, b1}` the result set R will be
`R = {(a0, b0), (a0, b1), (a1, b0), (a1, b1)}`.

Note that an RTG set does not provide any guarantees about the order of
elements an can thus not be iterated over or indexed into, however, a
random element can be selected and subtracted from the set until it is
empty. This procedure is determinstic and will yield the same sequence of
elements for a fixed seed and RTG version. If more guarantees about the
order of elements is necessary, use arrays instead (and compute the
cartesian product manually using nested loops).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-25)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of a set of values |

#### Results: [¶](#results-23)

| Result | Description |
| --- | --- |
| `result` | a set of values |

### `rtg.set_convert_to_bag` (::circt::rtg::SetConvertToBagOp) [¶](#rtgset_convert_to_bag-circtrtgsetconverttobagop)

*Convert a set to a bag*

Syntax:

```
operation ::= `rtg.set_convert_to_bag` $input `:` qualified(type($input)) attr-dict
```

This operation converts a set to a bag. Each element in the set occurs
exactly once in the resulting bag.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-26)

| Operand | Description |
| --- | --- |
| `input` | a set of values |

#### Results: [¶](#results-24)

| Result | Description |
| --- | --- |
| `result` | a bag of values |

### `rtg.set_create` (::circt::rtg::SetCreateOp) [¶](#rtgset_create-circtrtgsetcreateop)

*Constructs a set of the given values*

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-27)

| Operand | Description |
| --- | --- |
| `elements` | variadic of any type |

#### Results: [¶](#results-25)

| Result | Description |
| --- | --- |
| `set` | a set of values |

### `rtg.set_difference` (::circt::rtg::SetDifferenceOp) [¶](#rtgset_difference-circtrtgsetdifferenceop)

*Computes the difference of two sets*

Syntax:

```
operation ::= `rtg.set_difference` $original `,` $diff `:` qualified(type($output)) attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-28)

| Operand | Description |
| --- | --- |
| `original` | a set of values |
| `diff` | a set of values |

#### Results: [¶](#results-26)

| Result | Description |
| --- | --- |
| `output` | a set of values |

### `rtg.set_select_random` (::circt::rtg::SetSelectRandomOp) [¶](#rtgset_select_random-circtrtgsetselectrandomop)

*Selects an element uniformly at random from a set*

Syntax:

```
operation ::= `rtg.set_select_random` $set `:` qualified(type($set)) attr-dict
```

This operation returns an element from the given set uniformly at random.
Applying this operation to an empty set is undefined behavior.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-29)

| Operand | Description |
| --- | --- |
| `set` | a set of values |

#### Results: [¶](#results-27)

| Result | Description |
| --- | --- |
| `output` | any type |

### `rtg.set_size` (::circt::rtg::SetSizeOp) [¶](#rtgset_size-circtrtgsetsizeop)

*Returns the number of elements in the set*

Syntax:

```
operation ::= `rtg.set_size` $set `:` qualified(type($set)) attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-30)

| Operand | Description |
| --- | --- |
| `set` | a set of values |

#### Results: [¶](#results-28)

| Result | Description |
| --- | --- |
| `result` | index |

### `rtg.set_union` (::circt::rtg::SetUnionOp) [¶](#rtgset_union-circtrtgsetunionop)

*Computes the union of sets*

Syntax:

```
operation ::= `rtg.set_union` $sets `:` qualified(type($result)) attr-dict
```

Computes the union of the given sets. The list of sets must contain at
least one element.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-31)

| Operand | Description |
| --- | --- |
| `sets` | variadic of a set of values |

#### Results: [¶](#results-29)

| Result | Description |
| --- | --- |
| `result` | a set of values |

### `rtg.isa.slice_immediate` (::circt::rtg::SliceImmediateOp) [¶](#rtgisaslice_immediate-circtrtgsliceimmediateop)

*Extract a slice from an immediate*

Syntax:

```
operation ::= `rtg.isa.slice_immediate` $input `from` $lowBit `:`
              qualified(type($input)) `->` qualified(type($result)) attr-dict
```

This operation extracts a contiguous slice of bits from an immediate value.
The slice is specified by a low bit index (inclusive) and the width of the
slice is determined by the result type. The slice must fit within the input
immediate’s width.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-11)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `lowBit` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-32)

| Operand | Description |
| --- | --- |
| `input` | an ISA immediate |

#### Results: [¶](#results-30)

| Result | Description |
| --- | --- |
| `result` | an ISA immediate |

### `rtg.isa.space` (::circt::rtg::SpaceOp) [¶](#rtgisaspace-circtrtgspaceop)

*Reserve the given number of bytes*

Syntax:

```
operation ::= `rtg.isa.space` $size attr-dict
```

#### Operands: [¶](#operands-33)

| Operand | Description |
| --- | --- |
| `size` | index |

### `rtg.isa.string_data` (::circt::rtg::StringDataOp) [¶](#rtgisastring_data-circtrtgstringdataop)

*Reserve a string*

Syntax:

```
operation ::= `rtg.isa.string_data` $data attr-dict
```

Always appends a zero character (\00).

#### Attributes: [¶](#attributes-12)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `data` | ::mlir::StringAttr | string attribute |

### `rtg.substitute_sequence` (::circt::rtg::SubstituteSequenceOp) [¶](#rtgsubstitute_sequence-circtrtgsubstitutesequenceop)

*Partially substitute arguments of a sequence family*

This operation substitutes the first N of the M >= N arguments of the given
sequence family, where N is the size of provided argument substitution list.
A new sequence (if N == M) or sequence family with M-N will be returned.

Not having to deal with sequence arguments after randomly selecting a
sequence simplifies the problem of coming up with values to pass as
arguments, but also provides a way for the user to constrain the arguments
at the location where they are added to a set or bag.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-34)

| Operand | Description |
| --- | --- |
| `sequence` | handle to a sequence or sequence family |
| `replacements` | variadic of any type |

#### Results: [¶](#results-31)

| Result | Description |
| --- | --- |
| `result` | handle to a sequence or sequence family |

### `rtg.target` (::circt::rtg::TargetOp) [¶](#rtgtarget-circtrtgtargetop)

*Defines a test target*

Syntax:

```
operation ::= `rtg.target` $sym_name `:` $target attr-dict-with-keyword $bodyRegion
```

This operation specifies capabilities of a specific test target and can
provide additional information about it. These are added as operands to the
`yield` terminator and implicitly packed up into an `!rtg.dict` type which
is passed to tests that are matched with this target.

These capabilities can, for example, consist of the number of CPUs, supported
priviledge modes, available memories, etc.

Traits: `HasParent<mlir::ModuleOp>`, `IsolatedFromAbove`, `NoRegionArguments`, `SingleBlockImplicitTerminator<rtg::YieldOp>`, `SingleBlock`

Interfaces: `Symbol`

#### Attributes: [¶](#attributes-13)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `target` | ::mlir::TypeAttr | type attribute of a dictionary |

### `rtg.test.failure` (::circt::rtg::TestFailureOp) [¶](#rtgtestfailure-circtrtgtestfailureop)

*Exit the test and report failure*

Syntax:

```
operation ::= `rtg.test.failure` $errorMessage attr-dict
```

#### Attributes: [¶](#attributes-14)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `errorMessage` | ::mlir::StringAttr | string attribute |

### `rtg.test` (::circt::rtg::TestOp) [¶](#rtgtest-circtrtgtestop)

*The root of a test*

This operation declares the root of a randomized or directed test.
The target attribute specifies requirements of this test. These can be
refined by `rtg.require` operations inside this operation’s body. A test
can only be matched with a target if the target fulfills all the test’s
requirements. However, the target may provide more than the test requires.
For example, if the target allows execution in a user and privileged mode,
but the test only requires and runs in user mode, it can still be matched
with that target.

By default each test can be matched with all targets that fulfill its
requirements, but the user can also directly provide a target via the
’target’ attribute. In that case, the test will only be randomized against
that target.

The ’templateName’ attribute specifies the name of the original test
template (mostly for result reporting purposes). This is because a test
(template) can be matched against many targets and during this process one
test per match is created, but all of them preserve the same test template
name.

The body of this operation shall be processed the same way as an
`rtg.sequence`’s body with the exception of the block arguments.
The arguments must match the fields of the dict type in the target attribute
exactly. The test must not have any additional arguments and cannot be
referenced by an `rtg.get_sequence` operation.

If the end of the test is reached without executing an `rtg.test.success`
or `rtg.test.failure` it is as if an `rtg.test.success` is executed at the
very end.

Traits: `HasParent<mlir::ModuleOp>`, `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`

Interfaces: `Emittable`, `OpAsmOpInterface`, `SymbolUserOpInterface`, `Symbol`

#### Attributes: [¶](#attributes-15)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `templateName` | ::mlir::StringAttr | string attribute |
| `targetType` | ::mlir::TypeAttr | type attribute of a dictionary |
| `target` | ::mlir::StringAttr | string attribute |

### `rtg.test.success` (::circt::rtg::TestSuccessOp) [¶](#rtgtestsuccess-circtrtgtestsuccessop)

*Exit the test and report success*

Syntax:

```
operation ::= `rtg.test.success` attr-dict
```

### `rtg.tuple_create` (::circt::rtg::TupleCreateOp) [¶](#rtgtuple_create-circtrtgtuplecreateop)

*Create a tuple*

Syntax:

```
operation ::= `rtg.tuple_create` ($elements^ `:` qualified(type($elements)))? attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-35)

| Operand | Description |
| --- | --- |
| `elements` | variadic of any type |

#### Results: [¶](#results-32)

| Result | Description |
| --- | --- |
| `result` | a tuple of zero or more fields |

### `rtg.tuple_extract` (::circt::rtg::TupleExtractOp) [¶](#rtgtuple_extract-circtrtgtupleextractop)

*Get an element from a tuple*

Syntax:

```
operation ::= `rtg.tuple_extract` $tuple `at` $index `:` qualified(type($tuple)) attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-16)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `index` | ::mlir::IntegerAttr | index attribute |

#### Operands: [¶](#operands-36)

| Operand | Description |
| --- | --- |
| `tuple` | a tuple of zero or more fields |

#### Results: [¶](#results-33)

| Result | Description |
| --- | --- |
| `result` | any type |

### `rtg.validate` (::circt::rtg::ValidateOp) [¶](#rtgvalidate-circtrtgvalidateop)

*Validate the value in the given resource*

Syntax:

```
operation ::= `rtg.validate` $ref `,` $defaultValue (`,` $id^)?
              (` ``(` $defaultUsedValues^ `else` $elseValues `:`
              qualified(type($defaultUsedValues)) `)`)? `:`
              qualified(type($ref)) `->` qualified(type($defaultValue)) attr-dict
```

Validates the content of a reference-style value from the payload dialect
at the position of this operation. This validation may happen in a single
lowering step, e.g., a compiler pass that interprets the IR and inlines the
interpreted value directly, or a program that can be run to generate the
desired values may be generated first and in a second compilation run those
values (possibly stored in a file by the first run) may be inlined at the
position of these operations. For the latter, the ID attribute may be used
to match the values to the right operations and the ‘defaultValue’ is used
by the first run instead of the simulated value.

If the control-flow of the payload program visits this operation multiple
times, a possibly different value may be logged each time. In such
situations, the lowering should fail as no single value can be determined
that can be hardcoded/inlined in its place.

The value returned by this operation is not known during elaboration and
is thus treated like a value with identity (even though it might just be
a simple integer). Therefore, it is strongly recommended to not use the
result value of this operation in situations that expect structural
equivalence checks such as adding it to sets or bags.

The ‘defaultUsedValues’ are forwarded to the ‘values’ results without any
modification whenever the ‘defaultValue’ is used as replacement for ‘value’.
Otherwise, the ’elseValues’ are forwarded. This can be used to conditionally
execute code based on whether the default value was used or a proper value
was used as replacement. Note that this is not the most light-weight
implementation as, in principle, a single ‘i1’ result could achieve the same
in combination with an ‘scf.if’ or ‘select’ operation. However, these
operations are fully resolved during elaboration while the ‘validate’
operation remains until later in the pipeline because the repeated
compilation runs to resolve the validate operations should use the same
elaboration result which is difficult to achieve with multiple elaboration
runs even with the same seed as a different elaboration of the validate op
for the different compilation runs can lead to subtle differences in the
RNG querrying behavior.

Another alternative could be a region that is conditionally inlined or
deleted. However, this is even more heavy-weight and implies a strategy that
involves some instructions to be present in one run but not the other which
can lead to different label addresses, etc. and thus more likely to problems
with AOT co-simulation.

Traits: `AttrSizedOperandSegments`

#### Attributes: [¶](#attributes-17)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `id` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-37)

| Operand | Description |
| --- | --- |
| `ref` | ValidationTypeInterface instance |
| `defaultValue` | any type |
| `defaultUsedValues` | variadic of any type |
| `elseValues` | variadic of any type |

#### Results: [¶](#results-34)

| Result | Description |
| --- | --- |
| `value` | any type |
| `values` | variadic of any type |

### `rtg.virtual_reg` (::circt::rtg::VirtualRegisterOp) [¶](#rtgvirtual_reg-circtrtgvirtualregisterop)

*Returns a value representing a virtual register*

Syntax:

```
operation ::= `rtg.virtual_reg` $allowedRegs attr-dict
```

This operation creates a value representing a virtual register. The
‘allowedRegisters’ attribute specifies the concrete registers that may be
chosen during register allocation.

Interfaces: `InferTypeOpInterface`

#### Attributes: [¶](#attributes-18)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `allowedRegs` | ::circt::rtg::VirtualRegisterConfigAttr | an allowed register configuration for a virtual register |

#### Results: [¶](#results-35)

| Result | Description |
| --- | --- |
| `result` | RegisterTypeInterface instance |

### `rtg.yield` (::circt::rtg::YieldOp) [¶](#rtgyield-circtrtgyieldop)

*Terminates RTG operation regions*

Syntax:

```
operation ::= `rtg.yield` ($operands^ `:` type($operands))? attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-38)

| Operand | Description |
| --- | --- |
| `operands` | variadic of any type |

Types [¶](#types)
-----------------

### ArrayType [¶](#arraytype)

*An array type with dynamic size*

Syntax:

```
!rtg.array<
  ::mlir::Type   # elementType
>
```

Represents an array type with dynamic size. The array contains elements of
the specified type only.

#### Parameters: [¶](#parameters)

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `::mlir::Type` |  |

### BagType [¶](#bagtype)

*A bag of values*

Syntax:

```
!rtg.bag<
  ::mlir::Type   # elementType
>
```

This type represents a standard bag/multiset datastructure. It does not make
any assumptions about the underlying implementation.

#### Parameters: [¶](#parameters-1)

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `::mlir::Type` |  |

### BFloat16Type [¶](#bfloat16type)

*Bfloat16 floating-point type*

### ComplexType [¶](#complextype)

*Complex number with a parameterized element type*

Syntax:

```
complex-type ::= `complex` `<` type `>`
```

The value of `complex` type represents a complex number with a parameterized
element type, which is composed of a real and imaginary value of that
element type. The element must be a floating point or integer scalar type.

#### Example: [¶](#example-1)

```
complex<f32>
complex<i32>
```

#### Parameters: [¶](#parameters-2)

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `Type` |  |

### Float4E2M1FNType [¶](#float4e2m1fntype)

*4-bit floating point with 2-bit exponent and 1-bit mantissa*

An 4-bit floating point type with 1 sign bit, 2 bits exponent and 1 bit
mantissa. This is not a standard type as defined by IEEE-754, but it
follows similar conventions with the following characteristics:

* bit encoding: S1E2M1
* exponent bias: 1
* infinities: Not supported
* NaNs: Not supported
* denormals when exponent is 0

Open Compute Project (OCP) microscaling formats (MX) specification:
<https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>

### Float6E2M3FNType [¶](#float6e2m3fntype)

*6-bit floating point with 2-bit exponent and 3-bit mantissa*

An 6-bit floating point type with 1 sign bit, 2 bits exponent and 3 bits
mantissa. This is not a standard type as defined by IEEE-754, but it
follows similar conventions with the following characteristics:

* bit encoding: S1E2M3
* exponent bias: 1
* infinities: Not supported
* NaNs: Not supported
* denormals when exponent is 0

Open Compute Project (OCP) microscaling formats (MX) specification:
<https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>

### Float6E3M2FNType [¶](#float6e3m2fntype)

*6-bit floating point with 3-bit exponent and 2-bit mantissa*

An 6-bit floating point type with 1 sign bit, 3 bits exponent and 2 bits
mantissa. This is not a standard type as defined by IEEE-754, but it
follows similar conventions with the following characteristics:

* bit encoding: S1E3M2
* exponent bias: 3
* infinities: Not supported
* NaNs: Not supported
* denormals when exponent is 0

Open Compute Project (OCP) microscaling formats (MX) specification:
<https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>

### Float8E3M4Type [¶](#float8e3m4type)

*8-bit floating point with 3 bits exponent and 4 bit mantissa*

An 8-bit floating point type with 1 sign bit, 3 bits exponent and 4 bits
mantissa. This is not a standard type as defined by IEEE-754, but it
follows similar conventions with the following characteristics:

* bit encoding: S1E3M4
* exponent bias: 3
* infinities: supported with exponent set to all 1s and mantissa 0s
* NaNs: supported with exponent bits set to all 1s and mantissa values of
  {0,1}⁴ except S.111.0000
* denormals when exponent is 0

### Float8E4M3Type [¶](#float8e4m3type)

*8-bit floating point with 3 bit mantissa*

An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits
mantissa. This is not a standard type as defined by IEEE-754, but it
follows similar conventions with the following characteristics:

* bit encoding: S1E4M3
* exponent bias: 7
* infinities: supported with exponent set to all 1s and mantissa 0s
* NaNs: supported with exponent bits set to all 1s and mantissa of
  (001, 010, 011, 100, 101, 110, 111)
* denormals when exponent is 0

### Float8E4M3B11FNUZType [¶](#float8e4m3b11fnuztype)

*8-bit floating point with 3 bit mantissa*

An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits
mantissa. This is not a standard type as defined by IEEE-754, but it follows
similar conventions, with the exception that there are no infinity values,
no negative zero, and only one NaN representation. This type has the
following characteristics:

* bit encoding: S1E4M3
* exponent bias: 11
* infinities: Not supported
* NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits set to all 0s
* denormals when exponent is 0

Related to:
<https://dl.acm.org/doi/10.5555/3454287.3454728>

### Float8E4M3FNType [¶](#float8e4m3fntype)

*8-bit floating point with 3 bit mantissa*

An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits
mantissa. This is not a standard type as defined by IEEE-754, but it follows
similar conventions, with the exception that there are no infinity values
and only two NaN representations. This type has the following
characteristics:

* bit encoding: S1E4M3
* exponent bias: 7
* infinities: Not supported
* NaNs: supported with exponent bits and mantissa bits set to all 1s
* denormals when exponent is 0

Described in:
<https://arxiv.org/abs/2209.05433>

### Float8E4M3FNUZType [¶](#float8e4m3fnuztype)

*8-bit floating point with 3 bit mantissa*

An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits
mantissa. This is not a standard type as defined by IEEE-754, but it follows
similar conventions, with the exception that there are no infinity values,
no negative zero, and only one NaN representation. This type has the
following characteristics:

* bit encoding: S1E4M3
* exponent bias: 8
* infinities: Not supported
* NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits set to all 0s
* denormals when exponent is 0

Described in:
<https://arxiv.org/abs/2209.05433>

### Float8E5M2Type [¶](#float8e5m2type)

*8-bit floating point with 2 bit mantissa*

An 8-bit floating point type with 1 sign bit, 5 bits exponent and 2 bits
mantissa. This is not a standard type as defined by IEEE-754, but it
follows similar conventions with the following characteristics:

* bit encoding: S1E5M2
* exponent bias: 15
* infinities: supported with exponent set to all 1s and mantissa 0s
* NaNs: supported with exponent bits set to all 1s and mantissa of
  (01, 10, or 11)
* denormals when exponent is 0

Described in:
<https://arxiv.org/abs/2209.05433>

### Float8E5M2FNUZType [¶](#float8e5m2fnuztype)

*8-bit floating point with 2 bit mantissa*

An 8-bit floating point type with 1 sign bit, 5 bits exponent and 2 bits
mantissa. This is not a standard type as defined by IEEE-754, but it follows
similar conventions, with the exception that there are no infinity values,
no negative zero, and only one NaN representation. This type has the
following characteristics:

* bit encoding: S1E5M2
* exponent bias: 16
* infinities: Not supported
* NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits set to all 0s
* denormals when exponent is 0

Described in:
<https://arxiv.org/abs/2206.02915>

### Float8E8M0FNUType [¶](#float8e8m0fnutype)

*8-bit floating point with 8-bit exponent, no mantissa or sign*

An 8-bit floating point type with no sign bit, 8 bits exponent and no
mantissa. This is not a standard type as defined by IEEE-754; it is intended
to be used for representing scaling factors, so it cannot represent zeros
and negative numbers. The values it can represent are powers of two in the
range [-127,127] and NaN.

* bit encoding: S0E8M0
* exponent bias: 127
* infinities: Not supported
* NaNs: Supported with all bits set to 1
* denormals: Not supported

Open Compute Project (OCP) microscaling formats (MX) specification:
<https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>

### Float16Type [¶](#float16type)

*16-bit floating-point type*

### Float32Type [¶](#float32type)

*32-bit floating-point type*

### Float64Type [¶](#float64type)

*64-bit floating-point type*

### Float80Type [¶](#float80type)

*80-bit floating-point type*

### Float128Type [¶](#float128type)

*128-bit floating-point type*

### FloatTF32Type [¶](#floattf32type)

*TF32 floating-point type*

### FunctionType [¶](#functiontype)

*Map from a list of inputs to a list of results*

Syntax:

```
// Function types may have multiple results.
function-result-type ::= type-list-parens | non-function-type
function-type ::= type-list-parens `->` function-result-type
```

The function type can be thought of as a function signature. It consists of
a list of formal parameter types and a list of formal result types.

#### Example: [¶](#example-2)

```
func.func @add_one(%arg0 : i64) -> i64 {
  %c1 = arith.constant 1 : i64
  %0 = arith.addi %arg0, %c1 : i64
  return %0 : i64
}
```

#### Parameters: [¶](#parameters-3)

| Parameter | C++ type | Description |
| --- | --- | --- |
| inputs | `ArrayRef<Type>` |  |
| results | `ArrayRef<Type>` |  |

### GraphType [¶](#graphtype)

*Map from a list of inputs to a list of results*

Syntax:

```
// Function types may have multiple results.
function-result-type ::= type-list-parens | non-function-type
function-type ::= type-list-parens `->` function-result-type
```

The function type can be thought of as a function signature. It consists of
a list of formal parameter types and a list of formal result types.

#### Example: [¶](#example-3)

```
func.func @add_one(%arg0 : i64) -> i64 {
  %c1 = arith.constant 1 : i64
  %0 = arith.addi %arg0, %c1 : i64
  return %0 : i64
}
```

#### Parameters: [¶](#parameters-4)

| Parameter | C++ type | Description |
| --- | --- | --- |
| inputs | `ArrayRef<Type>` |  |
| results | `ArrayRef<Type>` |  |

### IndexType [¶](#indextype)

*Integer-like type with unknown platform-dependent bit width*

Syntax:

```
// Target word-sized integer.
index-type ::= `index`
```

The index type is a signless integer whose size is equal to the natural
machine word of the target (
[rationale](../../Rationale/Rationale/#integer-signedness-semantics) )
and is used by the affine constructs in MLIR.

**Rationale:** integers of platform-specific bit widths are practical to
express sizes, dimensionalities and subscripts.

### IntegerType [¶](#integertype)

*Integer type with arbitrary precision up to a fixed limit*

Syntax:

```
// Sized integers like i1, i4, i8, i16, i32.
signed-integer-type ::= `si` [1-9][0-9]*
unsigned-integer-type ::= `ui` [1-9][0-9]*
signless-integer-type ::= `i` [1-9][0-9]*
integer-type ::= signed-integer-type |
                 unsigned-integer-type |
                 signless-integer-type
```

Integer types have a designated bit width and may optionally have signedness
semantics.

**Rationale:** low precision integers (like `i2`, `i4` etc) are useful for
low-precision inference chips, and arbitrary precision integers are useful
for hardware synthesis (where a 13 bit multiplier is a lot cheaper/smaller
than a 16 bit one).

#### Parameters: [¶](#parameters-5)

| Parameter | C++ type | Description |
| --- | --- | --- |
| width | `unsigned` |  |
| signedness | `SignednessSemantics` |  |

### MemRefType [¶](#memreftype)

*Shaped reference to a region of memory*

Syntax:

```
layout-specification ::= attribute-value
memory-space ::= attribute-value
memref-type ::= `memref` `<` dimension-list-ranked type
                (`,` layout-specification)? (`,` memory-space)? `>`
```

A `memref` type is a reference to a region of memory (similar to a buffer
pointer, but more powerful). The buffer pointed to by a memref can be
allocated, aliased and deallocated. A memref can be used to read and write
data from/to the memory region which it references. Memref types use the
same shape specifier as tensor types. Note that `memref<f32>`,
`memref<0 x f32>`, `memref<1 x 0 x f32>`, and `memref<0 x 1 x f32>` are all
different types.

A `memref` is allowed to have an unknown rank (e.g. `memref<*xf32>`). The
purpose of unranked memrefs is to allow external library functions to
receive memref arguments of any rank without versioning the functions based
on the rank. Other uses of this type are disallowed or will have undefined
behavior.

Are accepted as elements:

* built-in integer types;
* built-in index type;
* built-in floating point types;
* built-in vector types with elements of the above types;
* another memref type;
* any other type implementing `MemRefElementTypeInterface`.

##### Layout [¶](#layout)

A memref may optionally have a layout that indicates how indices are
transformed from the multi-dimensional form into a linear address. The
layout must avoid internal aliasing, i.e., two distinct tuples of
*in-bounds* indices must be pointing to different elements in memory. The
layout is an attribute that implements `MemRefLayoutAttrInterface`. The
bulitin dialect offers two kinds of layouts: strided and affine map, each
of which is available as an attribute. Other attributes may be used to
represent the layout as long as they can be converted to a
[semi-affine map](Affine.md/#semi-affine-maps) and implement the required
interface. Users of memref are expected to fallback to the affine
representation when handling unknown memref layouts. Multi-dimensional
affine forms are interpreted in *row-major* fashion.

In absence of an explicit layout, a memref is considered to have a
multi-dimensional identity affine map layout. Identity layout maps do not
contribute to the MemRef type identification and are discarded on
construction. That is, a type with an explicit identity map is
`memref<?x?xf32, (i,j)->(i,j)>` is strictly the same as the one without a
layout, `memref<?x?xf32>`.

##### Affine Map Layout [¶](#affine-map-layout)

The layout may be represented directly as an affine map from the index space
to the storage space. For example, the following figure shows an index map
which maps a 2-dimensional index from a 2x2 index space to a 3x3 index
space, using symbols `S0` and `S1` as offsets.

![Index Map Example](/includes/img/index-map.svg)

Semi-affine maps are sufficiently flexible to represent a wide variety of
dense storage layouts, including row- and column-major and tiled:

```
// MxN matrix stored in row major layout in memory:
#layout_map_row_major = (i, j) -> (i, j)

// MxN matrix stored in column major layout in memory:
#layout_map_col_major = (i, j) -> (j, i)

// MxN matrix stored in a 2-d blocked/tiled layout with 64x64 tiles.
#layout_tiled = (i, j) -> (i floordiv 64, j floordiv 64, i mod 64, j mod 64)
```

##### Strided Layout [¶](#strided-layout)

Memref layout can be expressed using strides to encode the distance, in
number of elements, in (linear) memory between successive entries along a
particular dimension. For example, a row-major strided layout for
`memref<2x3x4xf32>` is `strided<[12, 4, 1]>`, where the last dimension is
contiguous as indicated by the unit stride and the remaining strides are
products of the sizes of faster-variying dimensions. Strided layout can also
express non-contiguity, e.g., `memref<2x3, strided<[6, 2]>>` only accesses
even elements of the dense consecutive storage along the innermost
dimension.

The strided layout supports an optional *offset* that indicates the
distance, in the number of elements, between the beginning of the memref
and the first accessed element. When omitted, the offset is considered to
be zero. That is, `memref<2, strided<[2], offset: 0>>` and
`memref<2, strided<[2]>>` are strictly the same type.

Both offsets and strides may be *dynamic*, that is, unknown at compile time.
This is represented by using a question mark (`?`) instead of the value in
the textual form of the IR.

The strided layout converts into the following canonical one-dimensional
affine form through explicit linearization:

```
affine_map<(d0, ... dN)[offset, stride0, ... strideN] ->
            (offset + d0 * stride0 + ... dN * strideN)>
```

Therefore, it is never subject to the implicit row-major layout
interpretation.

##### Codegen of Unranked Memref [¶](#codegen-of-unranked-memref)

Using unranked memref in codegen besides the case mentioned above is highly
discouraged. Codegen is concerned with generating loop nests and specialized
instructions for high-performance, unranked memref is concerned with hiding
the rank and thus, the number of enclosing loops required to iterate over
the data. However, if there is a need to code-gen unranked memref, one
possible path is to cast into a static ranked type based on the dynamic
rank. Another possible path is to emit a single while loop conditioned on a
linear index and perform delinearization of the linear index to a dynamic
array containing the (unranked) indices. While this is possible, it is
expected to not be a good idea to perform this during codegen as the cost
of the translations is expected to be prohibitive and optimizations at this
level are not expected to be worthwhile. If expressiveness is the main
concern, irrespective of performance, passing unranked memrefs to an
external C++ library and implementing rank-agnostic logic there is expected
to be significantly simpler.

Unranked memrefs may provide expressiveness gains in the future and help
bridge the gap with unranked tensors. Unranked memrefs will not be expected
to be exposed to codegen but one may query the rank of an unranked memref
(a special op will be needed for this purpose) and perform a switch and cast
to a ranked memref as a prerequisite to codegen.

Example:

```
// With static ranks, we need a function for each possible argument type
%A = alloc() : memref<16x32xf32>
%B = alloc() : memref<16x32x64xf32>
call @helper_2D(%A) : (memref<16x32xf32>)->()
call @helper_3D(%B) : (memref<16x32x64xf32>)->()

// With unknown rank, the functions can be unified under one unranked type
%A = alloc() : memref<16x32xf32>
%B = alloc() : memref<16x32x64xf32>
// Remove rank info
%A_u = memref_cast %A : memref<16x32xf32> -> memref<*xf32>
%B_u = memref_cast %B : memref<16x32x64xf32> -> memref<*xf32>
// call same function with dynamic ranks
call @helper(%A_u) : (memref<*xf32>)->()
call @helper(%B_u) : (memref<*xf32>)->()
```

The core syntax and representation of a layout specification is a
[semi-affine map](Affine.md/#semi-affine-maps). Additionally,
syntactic sugar is supported to make certain layout specifications more
intuitive to read. For the moment, a `memref` supports parsing a strided
form which is converted to a semi-affine map automatically.

The memory space of a memref is specified by a target-specific attribute.
It might be an integer value, string, dictionary or custom dialect attribute.
The empty memory space (attribute is None) is target specific.

The notionally dynamic value of a memref value includes the address of the
buffer allocated, as well as the symbols referred to by the shape, layout
map, and index maps.

Examples of memref static type

```
// Identity index/layout map
#identity = affine_map<(d0, d1) -> (d0, d1)>

// Column major layout.
#col_major = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

// A 2-d tiled layout with tiles of size 128 x 256.
#tiled_2d_128x256 = affine_map<(d0, d1) -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>

// A tiled data layout with non-constant tile sizes.
#tiled_dynamic = affine_map<(d0, d1)[s0, s1] -> (d0 floordiv s0, d1 floordiv s1,
                             d0 mod s0, d1 mod s1)>

// A layout that yields a padding on two at either end of the minor dimension.
#padded = affine_map<(d0, d1) -> (d0, (d1 + 2) floordiv 2, (d1 + 2) mod 2)>


// The dimension list "16x32" defines the following 2D index space:
//
//   { (i, j) : 0 <= i < 16, 0 <= j < 32 }
//
memref<16x32xf32, #identity>

// The dimension list "16x4x?" defines the following 3D index space:
//
//   { (i, j, k) : 0 <= i < 16, 0 <= j < 4, 0 <= k < N }
//
// where N is a symbol which represents the runtime value of the size of
// the third dimension.
//
// %N here binds to the size of the third dimension.
%A = alloc(%N) : memref<16x4x?xf32, #col_major>

// A 2-d dynamic shaped memref that also has a dynamically sized tiled
// layout. The memref index space is of size %M x %N, while %B1 and %B2
// bind to the symbols s0, s1 respectively of the layout map #tiled_dynamic.
// Data tiles of size %B1 x %B2 in the logical space will be stored
// contiguously in memory. The allocation size will be
// (%M ceildiv %B1) * %B1 * (%N ceildiv %B2) * %B2 f32 elements.
%T = alloc(%M, %N) [%B1, %B2] : memref<?x?xf32, #tiled_dynamic>

// A memref that has a two-element padding at either end. The allocation
// size will fit 16 * 64 float elements of data.
%P = alloc() : memref<16x64xf32, #padded>

// Affine map with symbol 's0' used as offset for the first dimension.
#imapS = affine_map<(d0, d1) [s0] -> (d0 + s0, d1)>
// Allocate memref and bind the following symbols:
// '%n' is bound to the dynamic second dimension of the memref type.
// '%o' is bound to the symbol 's0' in the affine map of the memref type.
%n = ...
%o = ...
%A = alloc (%n)[%o] : <16x?xf32, #imapS>
```

#### Parameters: [¶](#parameters-6)

| Parameter | C++ type | Description |
| --- | --- | --- |
| shape | `::llvm::ArrayRef<int64_t>` |  |
| elementType | `Type` |  |
| layout | `MemRefLayoutAttrInterface` |  |
| memorySpace | `Attribute` |  |

### NoneType [¶](#nonetype)

*A unit type*

Syntax:

```
none-type ::= `none`
```

NoneType is a unit type, i.e. a type with exactly one possible value, where
its value does not have a defined dynamic representation.

#### Example: [¶](#example-4)

```
func.func @none_type() {
  %none_val = "foo.unknown_op"() : () -> none
  return
}
```

### OpaqueType [¶](#opaquetype)

*Type of a non-registered dialect*

Syntax:

```
opaque-type ::= `opaque` `<` type `>`
```

Opaque types represent types of non-registered dialects. These are types
represented in their raw string form, and can only usefully be tested for
type equality.

#### Example: [¶](#example-5)

```
opaque<"llvm", "struct<(i32, float)>">
opaque<"pdl", "value">
```

#### Parameters: [¶](#parameters-7)

| Parameter | C++ type | Description |
| --- | --- | --- |
| dialectNamespace | `StringAttr` |  |
| typeData | `::llvm::StringRef` |  |

### RankedTensorType [¶](#rankedtensortype)

*Multi-dimensional array with a fixed number of dimensions*

Syntax:

```
tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
dimension-list ::= (dimension `x`)*
dimension ::= `?` | decimal-literal
encoding ::= attribute-value
```

Values with tensor type represents aggregate N-dimensional data values, and
have a known element type and a fixed rank with a list of dimensions. Each
dimension may be a static non-negative decimal constant or be dynamically
determined (indicated by `?`).

The runtime representation of the MLIR tensor type is intentionally
abstracted - you cannot control layout or get a pointer to the data. For
low level buffer access, MLIR has a
[`memref` type](#memreftype). This
abstracted runtime representation holds both the tensor data values as well
as information about the (potentially dynamic) shape of the tensor. The
[`dim` operation](MemRef.md/#memrefdim-mlirmemrefdimop) returns the size of a
dimension from a value of tensor type.

The `encoding` attribute provides additional information on the tensor.
An empty attribute denotes a straightforward tensor without any specific
structure. But particular properties, like sparsity or other specific
characteristics of the data of the tensor can be encoded through this
attribute. The semantics are defined by a type and attribute interface
and must be respected by all passes that operate on tensor types.
TODO: provide this interface, and document it further.

Note: hexadecimal integer literals are not allowed in tensor type
declarations to avoid confusion between `0xf32` and `0 x f32`. Zero sizes
are allowed in tensors and treated as other sizes, e.g.,
`tensor<0 x 1 x i32>` and `tensor<1 x 0 x i32>` are different types. Since
zero sizes are not allowed in some other types, such tensors should be
optimized away before lowering tensors to vectors.

#### Example: [¶](#example-6)

```
// Known rank but unknown dimensions.
tensor<? x ? x ? x ? x f32>

// Partially known dimensions.
tensor<? x ? x 13 x ? x f32>

// Full static shape.
tensor<17 x 4 x 13 x 4 x f32>

// Tensor with rank zero. Represents a scalar.
tensor<f32>

// Zero-element dimensions are allowed.
tensor<0 x 42 x f32>

// Zero-element tensor of f32 type (hexadecimal literals not allowed here).
tensor<0xf32>

// Tensor with an encoding attribute (where #ENCODING is a named alias).
tensor<?x?xf64, #ENCODING>
```

#### Parameters: [¶](#parameters-8)

| Parameter | C++ type | Description |
| --- | --- | --- |
| shape | `::llvm::ArrayRef<int64_t>` |  |
| elementType | `Type` |  |
| encoding | `Attribute` |  |

### TupleType [¶](#tupletype)

*Fixed-sized collection of other types*

Syntax:

```
tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
```

The value of `tuple` type represents a fixed-size collection of elements,
where each element may be of a different type.

**Rationale:** Though this type is first class in the type system, MLIR
provides no standard operations for operating on `tuple` types
(
[rationale](../../Rationale/Rationale/#tuple-types)).

#### Example: [¶](#example-7)

```
// Empty tuple.
tuple<>

// Single element
tuple<f32>

// Many elements.
tuple<i32, f32, tensor<i1>, i5>
```

#### Parameters: [¶](#parameters-9)

| Parameter | C++ type | Description |
| --- | --- | --- |
| types | `ArrayRef<Type>` |  |

### UnrankedMemRefType [¶](#unrankedmemreftype)

*Shaped reference, with unknown rank, to a region of memory*

Syntax:

```
unranked-memref-type ::= `memref` `<*x` type (`,` memory-space)? `>`
memory-space ::= attribute-value
```

A `memref` type with an unknown rank (e.g. `memref<*xf32>`). The purpose of
unranked memrefs is to allow external library functions to receive memref
arguments of any rank without versioning the functions based on the rank.
Other uses of this type are disallowed or will have undefined behavior.

See
[MemRefType](#memreftype) for more information on
memref types.

#### Examples: [¶](#examples)

```
memref<*f32>

// An unranked memref with a memory space of 10.
memref<*f32, 10>
```

#### Parameters: [¶](#parameters-10)

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `Type` |  |
| memorySpace | `Attribute` |  |

### UnrankedTensorType [¶](#unrankedtensortype)

*Multi-dimensional array with unknown dimensions*

Syntax:

```
tensor-type ::= `tensor` `<` `*` `x` type `>`
```

An unranked tensor is a type of tensor in which the set of dimensions have
unknown rank. See
[RankedTensorType](#rankedtensortype)
for more information on tensor types.

#### Examples: [¶](#examples-1)

```
tensor<*xf32>
```

#### Parameters: [¶](#parameters-11)

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `Type` |  |

### VectorType [¶](#vectortype)

*Multi-dimensional SIMD vector type*

Syntax:

```
vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
vector-element-type ::= float-type | integer-type | index-type
vector-dim-list := (static-dim-list `x`)?
static-dim-list ::= static-dim (`x` static-dim)*
static-dim ::= (decimal-literal | `[` decimal-literal `]`)
```

The vector type represents a SIMD style vector used by target-specific
operation sets like AVX or SVE. While the most common use is for 1D
vectors (e.g. vector<16 x f32>) we also support multidimensional registers
on targets that support them (like TPUs). The dimensions of a vector type
can be fixed-length, scalable, or a combination of the two. The scalable
dimensions in a vector are indicated between square brackets ([ ]).

Vector shapes must be positive decimal integers. 0D vectors are allowed by
omitting the dimension: `vector<f32>`.

Note: hexadecimal integer literals are not allowed in vector type
declarations, `vector<0x42xi32>` is invalid because it is interpreted as a
2D vector with shape `(0, 42)` and zero shapes are not allowed.

#### Examples: [¶](#examples-2)

```
// A 2D fixed-length vector of 3x42 i32 elements.
vector<3x42xi32>

// A 1D scalable-length vector that contains a multiple of 4 f32 elements.
vector<[4]xf32>

// A 2D scalable-length vector that contains a multiple of 2x8 f32 elements.
vector<[2]x[8]xf32>

// A 2D mixed fixed/scalable vector that contains 4 scalable vectors of 4 f32 elements.
vector<4x[4]xf32>

// A 3D mixed fixed/scalable vector in which only the inner dimension is
// scalable.
vector<2x[4]x8xf32>
```

#### Parameters: [¶](#parameters-12)

| Parameter | C++ type | Description |
| --- | --- | --- |
| shape | `::llvm::ArrayRef<int64_t>` |  |
| elementType | `::mlir::Type` | VectorElementTypeInterface instance |
| scalableDims | `::llvm::ArrayRef<bool>` |  |

### DictType [¶](#dicttype)

*A dictionary*

This type is a dictionary with a static set of entries. This datatype does
not make any assumptions about how the values are stored (could be a struct,
a map, etc.). Furthermore, two values of this type should be considered
equivalent if they have the same set of entry names and types and the values
match for each entry, independent of the order.

#### Parameters: [¶](#parameters-13)

| Parameter | C++ type | Description |
| --- | --- | --- |
| entries | `::llvm::ArrayRef<::circt::rtg::DictEntry>` | dict entries |

### ImmediateType [¶](#immediatetype)

*An ISA immediate*

Syntax:

```
!rtg.isa.immediate<
  uint32_t   # width
>
```

This type represents immediates of arbitrary but fixed bit-width.
The RTG dialect provides this type to avoid duplication in ISA payload
dialects.

#### Parameters: [¶](#parameters-14)

| Parameter | C++ type | Description |
| --- | --- | --- |
| width | `uint32_t` |  |

### LabelType [¶](#labeltype)

*A reference to a label*

Syntax: `!rtg.isa.label`

This type represents a label. Payload dialects can add operations to cast
from this type to an immediate type they can use as an operand to an
instruction or allow an operand of this type directly.

### MemoryBlockType [¶](#memoryblocktype)

*Handle to a memory block*

Syntax:

```
!rtg.isa.memory_block<
  uint32_t   # addressWidth
>
```

A memory block is representing a continuous region in a memory map with a
fixed size and base address. It can refer to actual memory or a memory
mapped device.

It is assumed that there is only a single address space.

#### Parameters: [¶](#parameters-15)

| Parameter | C++ type | Description |
| --- | --- | --- |
| addressWidth | `uint32_t` |  |

### MemoryType [¶](#memorytype)

*Handle to a memory*

Syntax:

```
!rtg.isa.memory<
  uint32_t   # addressWidth
>
```

This type is used to represent memory resources that are allocated from
memory blocks and can be accessed and manipulated by payload dialect
operations.

#### Parameters: [¶](#parameters-16)

| Parameter | C++ type | Description |
| --- | --- | --- |
| addressWidth | `uint32_t` |  |

### RandomizedSequenceType [¶](#randomizedsequencetype)

*Handle to a fully randomized sequence*

Syntax: `!rtg.randomized_sequence`

Sequences can contain operations to randomize their content in various ways.
A sequence of this type is guaranteed to not have any such operations
anymore (transitively).

### SequenceType [¶](#sequencetype)

*Handle to a sequence or sequence family*

Syntax:

```
!rtg.sequence<
  ::llvm::ArrayRef<mlir::Type>   # elementTypes
>
```

An SSA value of this type refers to a sequence if the list of element types
is empty or a sequence family if there are elements left to be substituted.

#### Parameters: [¶](#parameters-17)

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementTypes | `::llvm::ArrayRef<mlir::Type>` | element types |

### SetType [¶](#settype)

*A set of values*

Syntax:

```
!rtg.set<
  ::mlir::Type   # elementType
>
```

This type represents a standard set datastructure. It does not make any
assumptions about the underlying implementation. Thus a hash set, tree set,
etc. can be used in a backend.

#### Parameters: [¶](#parameters-18)

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `::mlir::Type` |  |

### TupleType [¶](#tupletype-1)

*A tuple of zero or more fields*

Syntax:

```
!rtg.tuple<
  ::llvm::ArrayRef<::mlir::Type>   # fieldTypes
>
```

This type represents a tuple of zero or more fields. The fields can be
of any type. The builtin tuple type is not used because it does not allow
zero fields.

#### Parameters: [¶](#parameters-19)

| Parameter | C++ type | Description |
| --- | --- | --- |
| fieldTypes | `::llvm::ArrayRef<::mlir::Type>` | tuple field types |

Passes [¶](#passes)
-------------------

### `-rtg-elaborate` [¶](#-rtg-elaborate)

*Elaborate the randomization parts*

This pass interprets most RTG operations to perform the represented
randomization and in the process get rid of those operations. This means,
after this pass the IR does not contain any random constructs within tests
anymore.

#### Options [¶](#options)

```
-seed                   : The seed for any RNG constructs used in the pass.
-delete-unmatched-tests : Delete tests that could not be matched with a target.
```

### `-rtg-embed-validation-values` [¶](#-rtg-embed-validation-values)

*Lower validate operations to the externally provided values*

This pass replaces ‘rtg.validate’ operations with concrete values read from
an external file. The file should contain the expected validation values
with matching IDs computed, e.g., by running the program in a simulator.

Each validate operation is matched with its corresponding value using the
unique identifier. If an identifier occurs multiple times in the file, the
pass fails. If an identifier for a ‘validate’ operation is missing in the
file, the pass will not modify that operation. Otherwise, the values are
parsed according to the implementation provided by the
‘ValidationTypeInterface’ implementation of the ‘ref’ operand type and
materialized in the IR as constants to replace the ‘validate’ operation.

This pass is typically used as part of a two-phase compilation process that
forks after the ‘rtg-unique-valiate’ pass is run:

1. Run the ‘rtg-lower-validate-ops-to-labels’ pass and the rest of the
   pipeline, then simulate the output in a reference simulator to generate
   a file with the expected values.
2. Run this pass and the rest of the pipeline to produce the final test.

#### Options [¶](#options-1)

```
-filename : The file with the validation values.
```

### `-rtg-emit-isa-assembly` [¶](#-rtg-emit-isa-assembly)

*Emits the instructions in a format understood by assemblers*

This pass expects all instructions to be inside ’emit.file’ operations with
an appropriate filename attribute. There are two special filenames:

* “-” means that the output should be emitted to stdout.
* "" means that the output should be emitted to stderr.

In order to operate on ’emit.file’ operations in parallel, the pass
requires that all ’emit.file’ operations have a unique filename (this is not
checked by the pass and violations will result in race conditions).

There are two options to specify lists of instructions that are not
supported by the assembler. For instructions in any of those lists, this
pass will emit the equivalent binary representation.

#### Options [¶](#options-2)

```
-unsupported-instructions-file : An absolute path to a file with a list of instruction names not supported by the assembler.
-unsupported-instructions      : A list of ISA instruction names not supported by the assembler.
```

### `-rtg-inline-sequences` [¶](#-rtg-inline-sequences)

*Inline and interleave sequences*

Inline all sequences into tests and remove the ‘rtg.sequence’ operations.
Also computes and materializes all interleaved sequences
(‘interleave\_sequences’ operation).

#### Statistics [¶](#statistics)

```
num-sequences-inlined     : Number of sequences inlined into another sequence or test.
num-sequences-interleaved : Number of sequences interleaved with another sequence.
```

### `-rtg-insert-test-to-file-mapping` [¶](#-rtg-insert-test-to-file-mapping)

*Insert Emit dialect ops to prepare for emission*

This pass inserts emit dialect operations to group tests to output files.
All tests can be put in a single output file, each test in its own file, or
tests can be grouped according to some properties (e.g., machine mode vs.
user mode tests) (TODO).

#### Options [¶](#options-3)

```
-split-output : If 'true' emits one file per 'rtg.test' in the IR. The name of the file matches the test name and is placed in 'path'. Otherwise, path is interpreted as the full file path including filename.
-path         : The directory or file path in which the output files should be created. If empty is is emitted to stderr (not allowed if 'split-output' is set to 'true')
```

### `-rtg-linear-scan-register-allocation` [¶](#-rtg-linear-scan-register-allocation)

*Simple linear scan register allocation for RTG*

Performs a simple version of the linear scan register allocation algorithm
based on the ‘rtg.virtual\_reg’ operations.

This pass is expected to be run after elaboration.

#### Statistics [¶](#statistics-1)

```
num-registers-spilled : Number of registers spilled to the stack.
```

### `-rtg-lower-unique-labels` [¶](#-rtg-lower-unique-labels)

*Lower label\_unique\_decl to label\_decl operations*

This pass lowers label\_unique\_decl operations to label\_decl operations by
creating a unique label string based on all the existing unique and
non-unique label declarations in the module.

#### Statistics [¶](#statistics-2)

```
num-labels-lowered : Number of unique labels lowered to regular label declarations.
```

### `-rtg-lower-validate-to-labels` [¶](#-rtg-lower-validate-to-labels)

*Lower validation operations to intrinsic labels*

Lowers the ‘rtg.validate’ operations to special intrinsic labels understood
by the target simulator to print the register contents.

### `-rtg-materialize-constraints` [¶](#-rtg-materialize-constraints)

*Materialize implicit constraints*

### `-rtg-memory-allocation` [¶](#-rtg-memory-allocation)

*Lower memories to immediates or labels*

This pass lowers ‘memory\_alloc’ and other memory handling operations to
immediates or labels by computing offsets within memory blocks according to
the memory allocation’s size and alignments.

#### Options [¶](#options-4)

```
-use-immediates : Whether the pass should lower memories to immediates instead of labels.
```

#### Statistics [¶](#statistics-3)

```
num-memories-allocated : Number of memories allocated from memory blocks.
```

### `-rtg-print-test-names` [¶](#-rtg-print-test-names)

*Print the names of all tests to the given file*

Prints the names of all tests in the module to the given file.
A CSV format is used with the first column being the properly uniqued name
of the test and the second column being the original name of the test as it
appeared in the frontend. The original name of a test may occur several
times because the test might have been duplicated for multiple targets or
because multiple elaborations of the same test/target pair were requested.

#### Options [¶](#options-5)

```
-filename : The file to print the test names to.
```

### `-rtg-simple-test-inliner` [¶](#-rtg-simple-test-inliner)

*Inline test contents*

This is a simple pass to inline test contents into ’emit.file’ operations
in which they are referenced. No “glue code” is inserted between tests
added to the same file. Thus this pass is not intended to be used in a
production pipeline but just to bring the IR into a structure understood by
the RTG ISA assembly emission pass to avoid making that pass more complex.

### `-rtg-unique-validate` [¶](#-rtg-unique-validate)

*Compute unique IDs for validate operations*

This pass visits all ‘rtg.validate’ operations without an ID attribute and
assigns a unique ID to them.

 [Prev - Moore Dialect](https://circt.llvm.org/docs/Dialects/Moore/ "Moore Dialect")
[Next - Simulation Dialect](https://circt.llvm.org/docs/Dialects/Sim/ "Simulation Dialect") 

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