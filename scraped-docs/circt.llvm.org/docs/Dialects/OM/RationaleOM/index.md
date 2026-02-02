Object Model Dialect Rationale - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

Object Model Dialect Rationale
==============================

This document describes various design points of the `om` dialect. This follows
in the spirit of other
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

Motivation [¶](#motivation)
---------------------------

The goal of the `om` dialect is to develop an IR suitable for domain modeling. It
intends to accomplish this by providing constructs for capturing an
[Object Model](https://en.wikipedia.org/wiki/Object_model) (OM). Domain modeling
in this context means capturing design intent and supporting tooling related to
the creation of CPUs and SoCs. Anything other than RTL level design entry falls
under this broad categorization, and must be intimately tied to RTL level design
entry in a stable way.

This includes modeling domains such as:

* Physical hierarchies
* Power specifications
* Clocks and clock domains
* Bus interfaces
* Software bringup
* And much more…

This dialect encompasses a very generic compiler IR capable of representing
concepts from these varied domains. This is not to say that we should never
develop domain-specific compiler IRs for such domains. To the contrary, it is
expected that over time we will continuously re-evaluate the uses of the `om`
dialect, and graduate well-understood models into domain-specific IRs through
further dialects.

Domain models should support tooling built around the design with well-defined
and type safe APIs. This includes tools generating collateral such as:

* Synthesis constraints
* IP-XACT
* DTS
* And much more…

Background [¶](#background)
---------------------------

Lots of discussions and prior art have informed the design of the `om` dialect.

### Ontologies [¶](#ontologies)

Domain modeling is intimately tied to the philosophical topic of
[ontologies](https://en.wikipedia.org/wiki/Ontology_%28information_science%29). The
`om` dialect should be able to support an ontology of interest as it evolves.
This requires data structures that derive their power from simplicity. The
dialect attempts to provide a few, orthogonal structures that can be composed to
flexibly support a variety of domain modeling needs in powerful ways.

### Domain Models APIs [¶](#domain-models-apis)

The domain models should not exist in a vacuum. They must be queried via
well-defined APIs to be of use to the various tools that require them. The design
of the data structures should simultaneously capture the domain model and support
querying it in a natural way.

### Inspiration from Modern Languages [¶](#inspiration-from-modern-languages)

While many modern languages have influenced this design, the simplest analogy
might be in terms of Java classes. The `om` dialect constructs are similar to
how Java defines classes, with constructors and public data members, but without
generic data types.

Class constructors and public data member separate how Objects are created from
how they are used. The removal of generic data types requires the domain model
to be monomorphized.

The design for Classes also draws on OCaml and C++.

Core Constructs [¶](#core-constructs)
-------------------------------------

### Classes [¶](#classes)

Class definitions have a name, a formal parameter list, and a body.

The formal parameter list provides a name and a type for each parameter that
must be supplied to instantiate the Class. In the IR, formal parameters become
block arguments, much like function formal parameters in software compilers.
After Classes are defined, they can be instantiated as Objects in the IR by
supplying actual parameters for each declared formal parameter.

The body provides a way to describe both the publicly exported Fields of the
Class, as well as its internal implementation.

The list of formal parameters describes what is needed to create a Class, and
the list of Fields describes how a Class can be used. They need not be the same,
although for simple Classes, like Scala case classes, they may be. Together, the
formal parameters and Fields describe the public API of a Class.

One or more Classes may be entrypoints into the domain model, providing
different top-level views into the graph. Such Classes have no formal
parameters, and define their entrypoint to the domain model in terms of the
Fields they expose, which may be instantiated Objects or other primitives. One
way to think of this is as an alternative to top-level Object instances that are
not inside any Class; such Objects can always be made Fields of a new top-level
Class with no actual parameters.

The internal implementation is responsible for assigning values to the Fields,
and is defined in terms of a small expression grammar including:

* Instantiations of Classes to create Objects by passing expressions as actual
  parameters
* Accessing Fields on instantiated Objects
* References to formal parameters
* Primitive values like integers, strings, and symbols
* Container values like lists
* Expressions involving primitive and container values

This modeling of Classes might be most similar to Java classes, which define
public constructors and members. The proposed modeling of Classes is restricted
to data members, and there is no such things as a method for Classes.

### Fields [¶](#fields)

Fields are how arbitrary data can be exposed by each Object, according to the
types defined in the Class the Object is instantiating. Fields are namespaced by
the class or interface they are defined in. Fields are name-value pairs, where
the name is given in the Class definition and the value is any expression in the
same small expression grammar described under Classes.

Where the expression assigned to each Field comes from is an internal detail
that cannot be accessed via Objects of a Class. Only the Field’s value can be
accessed, by name, externally.

The type system for Fields is left open and extensible: any Type representable
in MLIR is allowed. This includes:

* Builtin MLIR types like integer, float, symbol references, etc.
* Core CIRCT types like HW inner reference
* Types defined by the domain model via Classes
* Container types like lists
* Other new types defined for domain modeling

### Objects [¶](#objects)

Objects represent a specific entity from the ontology that is being captured in
the domain model. They have a very small list of required attributes, which
allows the platform to reason about all Objects in a uniform way. As such,
addition of new attributes should be reviewed and debated carefully.

The initial list of required attributes for all Objects is:

* Class definition name: the name of the Class this Object is instantiating

An Object instantiation must also supply a list of actual parameters that match
the Class’s formal parameters. These actual parameters can be any expression in
the same small expression grammar used to assign values to Fields of Classes.
Object instances are values just like other expressions, and can be assigned to
named Fields in a Class or passed as actual parameters in other Object
instantiations.

The type of an Object is governed by its Class, and a new custom type is added
to support a reference to a Class in the type system. When Objects are passed as
actual parameters to other Object instantiations, they are passed by reference.

Fields defined by Classes can be accessed from concrete Object instances. Field
accesses accept an instance of an Object, and a list of Fields to refer to
within the Object and potential children Objects. Object Field accesses are
values just like other expressions, and can be assigned to named Fields in a
Class or passed as actual parameters in Object instantiations.

### Expressions [¶](#expressions)

The small expression grammar described under Classes includes expressions
involving primitive and container values. This sections describes the rationale
for such expressions.

In order for a Class to effectively capture parts of a domain model, it may be
necessary for the Class to represent computation in terms of its formal
parameters.

For example, a Class might represent a device that is attached to a bus, and
accessible at some address. If that address is implemented as an offset relative
to some base address, the Class could have an input integer as a formal
parameter representing the base address, a Field representing the device
address, and internally add some constant offset to the base address before
assigning the resulting value into the Field.

As another example, a Class might internally instantiate Objects of some other
Classes, access Fields of those Objects, create a container holding the values
of those fields, and assign the container to a Field. Using container
construction expressions, this can be represented directly in the Class,
allowing it to abstract over the Objects it creates internally.

Alternatives Considered [¶](#alternatives-considered)
-----------------------------------------------------

### Other Libraries and Tools for Domain Modeling [¶](#other-libraries-and-tools-for-domain-modeling)

Many powerful libraries and tools exist for building domain models in a variety
of programming languages. It would be possible to implement the goals using any
such system, but this misses the opportunity to continue leveraging MLIR based
tooling. These other systems may be powerful, but so is MLIR, and integrating new
systems comes with a cost. With MLIR, the infrastructure is already in place to
achieve the goal of creating modular libraries in a common framework.

### Specific IR for Domain Modeling [¶](#specific-ir-for-domain-modeling)

MLIR has excellent support for flexibly defining IRs that model different
domains. Why not use this power to model domains directly in the IR using the
appropriate abstractions?

Operations could be defined for bus interfaces, address maps, and all the other
domain models. This would enable very precise modeling, verification,
documentation, etc. to all be captured in the IR. We could build up a library of
all the domain models we care about.

The problem with this approach is it puts the burden of domain modeling on
compiler engineers. Any new property or domain model will require new IR
definitions. This puts compiler engineers on the critical path of all domain
model related tasks, which is not scalable.

There is a potential future world where it becomes tractable to define specific
IRs without getting a compiler engineer involved. Tools like
[IRDL](https://pldi22.sigplan.org/details/pldi-2022-pldi/33/IRDL-An-IR-Definition-Language-for-SSA-Compilers)
are being built right now, which could allow developers to conveniently express
domain-specific IRs in an ergonomic way from a programming language of their
choosing. But this is still very research oriented, and may not be ready for
production in the near term.

In the meantime, defining a few core abstractions that let the compiler reason
about the domain models in a limited but useful way is sufficient to enable many
use-cases without putting compiler engineers on the critical path.

However, as mentioned in the initial motivation, we also want to be diligent
about reviewing the uses of the generic model, and promoting well-defined
modeling into domain-specific IRs once they are ready to be hardened and little
churn in the modeling is expected.

### Generic IR with Objects [¶](#generic-ir-with-objects)

The first `om` dialect proposal simply had Containers, Properties on Containers,
and references between Containers. This was a Smalltalk-style “everything is an
object” system, where the only way to interact with an object would be through
its Properties and References (as opposed to messages in Smalltalk).

Such a system is extremely simple, yet powerful. However, this does not lend
itself to all of our stated goals. This leads to a dynamically typed system,
where the compiler has little static knowledge. This is actually somewhat akin to
the current Object Model JSON design. We want to keep the simplicity and power,
but bring a more statically typed approach to bear.

### Generic IR with Objects and Protocols [¶](#generic-ir-with-objects-and-protocols)

The second `om` dialect proposal split Containers into two concepts: Objects and
Protocols. Objects had Properties, and Protocols defined a set of Properties that
an Object must have. This started to bring more static information to the system,
where multiple Objects could be reasoned about in terms of Protocols.

However, this was still insufficient. The design point of having Objects
implement Protocols was a step in the right direction, but only a step. The type
system gained slightly more static information, but was still only defined in
terms of loose collections of Objects.

### Generic IR with Objects, Structs, and Traits [¶](#generic-ir-with-objects-structs-and-traits)

The third `om` dialect proposal split Protocols into two concepts: Structs and
Traits. The name and design was inspired by Rust. Structs defined Fields that
instances of Structs would have. This separation between definition and instance
was another crucial step in the right direction. The type system was defined in
terms of Structs, inheritance, and Traits that Structs implement. Objects were
typed in terms of the Struct they were instantiating.

However, there was still a key ingredient missing. Struct definitions were a
simple list of flat Fields, and Object instantiations simply provided concrete
values for each field. The missing ingredient was a separation between what was
needed to instantiate a Struct and what was exposed as Fields of a Struct.

This addition led to the current design of the `om` dialect. Modulo some
renaming, the key difference was that Struct definitions were given a formal
parameter list, and Struct bodies were updated to admit a small grammar of
expressions to compute the Struct’s various Fields.

 [Prev - 'om' Dialect](https://circt.llvm.org/docs/Dialects/OM/ "'om' Dialect")
[Next - 'pipeline' Dialect](https://circt.llvm.org/docs/Dialects/Pipeline/ "'pipeline' Dialect") 

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
    - ['om' Dialect-](https://circt.llvm.org/docs/Dialects/OM/)
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