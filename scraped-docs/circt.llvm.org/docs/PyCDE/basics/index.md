PyCDE Basics - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

PyCDE Basics
============

You know what’s more difficult than forcing yourself to write documentation?
Maintaining it! We apologize for the inevitable inaccuracies.

Modules, Generators, and Systems [¶](#modules-generators-and-systems)
---------------------------------------------------------------------

```
from pycde import Input, Output, Module, System
from pycde import generator
from pycde.types import Bits

class OrInts(Module):
    a = Input(Bits(32))
    b = Input(Bits(32))
    c = Output(Bits(32))

    @generator
    def construct(self):
        self.c = self.a | self.b


system = System([OrInts], name="ExampleSystem", output_directory="exsys")
system.compile()
```

Hardware modules extend `pycde.Module`. They define any number of typed inputs
and outputs by setting class members.

The `pycde.generator` decorator is used to denote the module construction code.
It is called once per module at `compile` time. Unlike standard python, the body
of the generator does not have access to instance members through `self` – just
ports. All of the output ports **must** be set.

In order to compile a hardware system, use `System` constructing it with the
class or list of classes which are the top modules. Other modules instantiated
in the generators get generated and emitted automatically and recursively (i.e.
only the root of the hierarchy needs to be given). `name` defaults to the top
module name. `output_directory` defaults to `name`. The `compile` method outputs
a “build package” containing the generated system into `output_directory`.

Instantiating Modules [¶](#instantiating-modules)
-------------------------------------------------

Modules can be instantiated in other modules. The following example defines a module that instantiates the `AddInts` module defined above.

```
class Top(Module):
    a = Input(Bits(32))
    b = Input(Bits(32))
    c = Output(Bits(32))

    @generator
    def construct(self):
        or_ints = OrInts(a=self.a, b=self.b)
        self.c = or_ints.c


system = System([Top], name="ExampleSystem")
system.compile()
```

The constructor of a `Module` expects named keyword arguments for each input
port. These keyword arguments can be any PyCDE Signal. The above example uses
module inputs. Instances of `Module`s support named access to output port values
like `add_insts.c`. These output port values are PyCDE Signals, and can be use
to further connect modules or instantiate CIRCT dialect operations.

Types & Signals [¶](#types--signals)
------------------------------------

Since CIRCT primarily targets hardware not software, it defines its own types.
PyCDE exposes them through the `pycde.types.Type` class hierarchy. PyCDE
signals represent values on the target device. Signals have a particular `Type`
(which is distinct from the signal objects’ Python `type`) stored in their
`type` instance member. `Type`s are heretofor referred to interchangeably as
“PyCDE type”, “CIRCT type”, or “Type”.

All signals extend the `pycde.signals.Signal` class, specialized by their Type.
The various specializations often include operator overrides to perform common
operations, as demonstrated in the hello world example’s `|` bitwise or.

Note that the CIRCT type conveys information not just about the data type but
can also specify the signaling mechanism. In other words, a signal does not
necessarily imply standard wires (though it usually does).

\*\* For CIRCT/MLIR developers: “signals” map 1:1 to MLIR Values.

### Constants and Python object conversion [¶](#constants-and-python-object-conversion)

Some Python objects (e.g. int, dict, list) can be converted to constants in
hardware. PyCDE tries its best to make this automatic, but it sometimes needs to
know the Type. For instance, we don’t know the desired bitwidth of an int. In
some cases we default to the required number of bits to represent a number, but
sometimes that fails.

In those cases, you must manually specify the Type. So `Bits(16)(i)` would
create a 16-bit constant of `i`.

### Scalars [¶](#scalars)

`Bits(width)` models a bitvector. Allows indexing, slicing, bitwise operations, etc. No math operations.

`UInt(width)`, `SInt(width)` math.

### Arrays [¶](#arrays)

`Bits(32) * 10` creates an array of 32-bits of length 10.

`Bits(32) * 10 * 12` creates an array of arrays.

### Structs [¶](#structs)

```
from pycde import Input, Output, generator, System, Module
from pycde.types import Bits
from pycde.signals import Struct, BitsSignal

class ExStruct(Struct):
  a: Bits(4)
  b: Bits(32)

  def get_b_xor(self, x: int) -> BitsSignal:
    return self.b ^ Bits(32)(x)


class StructExample(Module):
  inp1 = Input(ExStruct)
  out1 = Output(Bits(32))
  out2 = Output(Bits(4))
  out3 = Output(ExStruct)

  @generator
  def build(self):
    self.out1 = self.inp1.get_b_xor(5432)
    self.out2 = self.inp1.a
    self.out3 = ExStruct(a=self.inp1.a, b=42)
```

### NumPy features [¶](#numpy-features)

PyCDE supports a subset of numpy array transformations (see `pycde/ndarray.py`)
that can be used to do complex reshaping and transformation of multidimensional
arrays.

The numpy functionality is provided by the `NDArray` class, which creates a view
on top of existing SSA values. Users may choose to perform transformations
directly on `ListSignal`s:

```
class M1(Module):
  in1 = Input(dim(Bits(32), 4, 8))
  out = Output(dim(Bits(32), 2, 16))

  @generator
  def build(self):
    self.out = self.in1.transpose((1, 0)).reshape((16, 2))
    # Under the hood, this resolves to
    # Matrix(from_value=
    #    Matrix(from_value=ports.in1).transpose((1,0)).to_circt())
    #  .reshape(16, 2).to_circt()
```

or manually manage a `NDArray` object.

```
class M1(Module):
  in1 = Input(dim(Bits(32), 4, 8))
  out = Output(dim(Bits(32), 2, 16))

  @generator
  def build(self):
    m = NDArray(from_value=self.in1).transpose((1, 0)).reshape((16, 2))
    self.out = m.to_circt()
```

Manually managing the NDArray object allows for postponing materialization
(`to_circt()`) until all transformations have been applied. In short, this
allows us to do as many transformations as possible in software, before emitting
IR. Note however, that this might reduce debugability of the generated hardware
due to the lack of `sv.wire`s in between each matrix transformation.

For further usage examples, see `PyCDE/test/test_ndarray.py`, and inspect
`ListSignal` in `pycde/signals.py` for the full list of implemented numpy
functions.

External Modules [¶](#external-modules)
---------------------------------------

External modules are how PyCDE and CIRCT support interacting with existing
System Verilog or Verilog modules. They must be declared and the ports must
match the externally defined implementation in SystemVerilog or other language.
We have no way of checking that they do indeed match so it’ll be up to the EDA
synthesizer (and they generally do a poor job reporting mismatches).

In PyCDE, an external module is any module without a generator.

```
class MulInts(Module):
    module_name = "MyMultiplier"
    a = Input(Bits(32))
    b = Input(Bits(32))
    c = Output(Bits(32))
```

The `MyMultiplier` module is declared in the default output file, `ExampleSystem/ExampleSystem.sv`.

Parameterized modules [¶](#parameterized-modules)
-------------------------------------------------

```
from pycde import modparams

@modparams
def AddInts(width: int):

  class AddInts(Module):
    a = Input(UInt(width))
    b = Input(UInt(width))
    c = Output(UInt(width + 1))

    @generator
    def build(self):
      self.c = self.a + self.b

  return AddInts


class Top(Module):
  a = Input(UInt(32))
  b = Input(UInt(32))
  c = Output(UInt(33))

  @generator
  def construct(self):
    add_ints_m = AddInts(32)
    add_ints = add_ints_m(a=self.a, b=self.b)
    self.c = add_ints.c
```

In order to “parameterize” a module, simply return one from a function. Said
function must be decorated with `modparams` to inform PyCDE that the returned
module is a parameterized one. The `modparams` decorator does several things
including: (1) memoizing the parameterization function, and (2) automatically
derive a module name which includes the parameter values (for module name
uniqueness).

PyCDE does not produce parameterized SystemVerilog modules! The specialization
happens with Python code, which is far more powerful than SystemVerilog
parameterization constructs.

External parameterized modules [¶](#external-parameterized-modules)
-------------------------------------------------------------------

Just like internally defined parameterized modules, leave off the generator and
PyCDE will output SystemVerilog instantations with the module parameters. The
parameter types are best effort based on the first instantiation encountered.

```
from pycde import modparams, Module

@modparams
def AddInts(width: int):

  class AddInts(Module):
    a = Input(UInt(width))
    b = Input(UInt(width))
    c = Output(UInt(width + 1))

  return AddInts


class Top(Module):
  a = Input(UInt(32))
  b = Input(UInt(32))
  c = Output(UInt(33))

  @generator
  def construct(self):
    add_ints_m = AddInts(32)
    add_ints = add_ints_m(a=self.a, b=self.b)
    self.c = add_ints.c
```

For the instantiation produces:

```
  AddInts #(
    .width(64'd32)
  ) AddInts (
    .a (a),
    .b (b),
    .c (c)
  );
```

Using CIRCT dialects directly (instead of with PyCDE syntactic sugar) [¶](#using-circt-dialects-directly-instead-of-with-pycde-syntactic-sugar)
-----------------------------------------------------------------------------------------------------------------------------------------------

Generally speaking, don’t.

One can directly instantiate CIRCT operations through
`pycde.dialects.<dialect_name>`. The CIRCT operations contained therein provide
thin wrappers around the CIRCT operations to adapt them to
[PyCDE
Signals](#signals) by overriding each operation’s constructor. This
auto-wrapper, however, does not always “just work” depending on the complexity
of the operation it is attemption to wrap. So don’t use it unless you know what
you’re doing. User beware. Warranty voided. Caveat emptor. etc.

 [Prev - Compiling CIRCT and PyCDE](https://circt.llvm.org/docs/PyCDE/compiling/ "Compiling CIRCT and PyCDE")
[Next - Static scheduling infrastructure](https://circt.llvm.org/docs/Scheduling/ "Static scheduling infrastructure") 

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
  + [Dialects+](https://circt.llvm.org/docs/Dialects/)

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
  + [Python CIRCT Design Entry (PyCDE)-](https://circt.llvm.org/docs/PyCDE/)
    - [Compiling CIRCT and PyCDE](https://circt.llvm.org/docs/PyCDE/compiling/)
    - [PyCDE Basics](https://circt.llvm.org/docs/PyCDE/basics/)
  + [Static scheduling infrastructure](https://circt.llvm.org/docs/Scheduling/)
  + [Symbol and Inner Symbol Rationale](https://circt.llvm.org/docs/RationaleSymbols/)
  + [Using the Python Bindings](https://circt.llvm.org/docs/PythonBindings/)
  + [Verilog and SystemVerilog Generation](https://circt.llvm.org/docs/VerilogGeneration/)