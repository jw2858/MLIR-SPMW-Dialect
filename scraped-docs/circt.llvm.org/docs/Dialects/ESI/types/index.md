ESI data types and communication types - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

ESI data types and communication types
======================================

ESI has two different classes of MLIR types: ones which represent data on the
wires (data types) and ones which specify the type of communication. From a user
perspective, communication types aren’t really types – this is just how the
communication style is modeled in MLIR and thus an implementation detail.

Data types [¶](#data-types)
---------------------------

In addition to the types in the `hw` dialect, ESI will add few:

### Void [¶](#void)

`void` translates to “no data”, meaning just a control signal.

Status: **planned**

### Lists [¶](#lists)

Lists are used to reason about variably-sized data. There are two types of
lists: those for which the size is known before transmission begins (a fixed
size list, or **fixed\_list**), and those for which it isn’t (a variably sized
list, or just **list**). `fixed_lists` are **very** roughly similar to a C
pointer/size pair (in that they contain a size upfront) whereas `lists` are
roughly similar to a simply singlely linked list wherein the size is not known
until one iterates through it.

Since it isn’t yet clear what (if any) benefit `fixed_list` will provide, it is
not plan-of-record to implement it. `list` will be implemented in the future.

Status: **planned**

Communication types [¶](#communication-types)
---------------------------------------------

As stated above, communication “types” do not represent data on the wire;
rather, they speak to the signaling.

Channels [¶](#channels)
-----------------------

ESI “channels” are streaming connections upon which *messages* can be sent and
received. They are expressed by wrapping the type (e.g. `!esi.channel<i5>`) and
using it like any other value type.

```
hw.module @add11(%clk: i1, %ints: !esi.channel<i32>) -> (%outs: !esi.channel<i32> ) {
  hw.output %ints
}
```

The value which the channel is carrying must be (un)wrapped to access it:

```
hw.module @add11(%clk: i1, %ints: !esi.channel<i32>) -> (mutatedInts: !esi.channel<i32>) {
  %i, %i_valid = esi.unwrap.vr %ints, %rdy
  %c11 = hw.constant 11
  %m = comb.add %c11, %i
  %mutInts, %rdy = esi.wrap.vr %m, %i_valid
  hw.output %mutInts
}
```

[insert diagram of wrap/unwrap]

Status: **complete and stable** (for supported data types)

Memory-mapped IO [¶](#memory-mapped-io)
---------------------------------------

*This section is not fully thought out or written about. It is certainly not
implemented. The text in this section should be considered initial thoughts.*

Status: **Planning**

### MMIO Regions [¶](#mmio-regions)

The basic idea is to present ESI memory mapped **regions** exposed by
modules. There could be any number of these regions exposed and they
would work somewhat like input (request) / output (response) port pairs,
but with implicit request-response signaling and structure. The MMIO
space itself would be defined by a statically sized struct (so lists
would be disallowed), with address offsets implicitly or explicitly
defined. The method for base address assignment is yet to be decided.
These regions can support atomic reads or writes of arbitrary size or
limited size.

### MMIO Requests (read/write) [¶](#mmio-requests-readwrite)

Other modules connected to the same MMIO bus could specify read/write
requests in several ways:

* As a data window with blinds (or similar construct) specifying the
  struct fields to read/write
* As a list of address offsets and sizes or data to read/write

Along with the request content, atomicity of the request would have to
be specified. The response on a read would correspond to the read
request – either a list of bytes read or a data window with the
requested data filled in. For a write, a simple acknowledgement or error
would suffice for the response.

### Automatic self-description [¶](#automatic-self-description)

An MMIO region’s data type can be used to automatically generate a
self-describing software data type and an access API. Additionally, a per
MMIO bus table could be generated with the base addresses for each connected
region and a descriptor for each.

### Software access [¶](#software-access)

The MMIO struct would become a software struct which software could map
to the base address as a typed pointer. This way, software access would
be simpler (just a normal struct pointer dereference – e.g. p->field1)
and safer as it knows the correct address implicitly and the type of
that field. Additionally, if the MMIO region knows the processor’s
endianness, it could respond in the correct endianness. How to initiate
an atomic read/write of multiple fields from software is undecided yet,
though there may be some merit to exposing data windows to software.

### Additional Type: any [¶](#additional-type-any)

An MMIO space can be have type **any** to allow access to memory spaces which
are fundamentally untyped (e.g. memory-mapped host RAM). In this case, the
requestor can specify the type. For instance, to access 64-bit values the
requestor would send a request with a given address and specify a uint<64> type
response or write. Alternatively, it could request a response or write type of
type struct ConfigSpace to specify it wants to read or write a configuration.

### Implementation [¶](#implementation)

The MMIO system *could* be implemented on top of the streaming portion.

Data windows
============

By default, an entire message is guaranteed to be presented to the receiving
module in one clock cycle (excepting `lists`). For particularly large messages,
this is not ideal as it requires a data path equal to the size of the message.
Data windows specify which parts of a message a module accepts on each clock
cycle. For structs, they specify which members are accepted on which cycles. For
arrays and lists, they specify how many items can be accepted each clock cycle.

Data windows do not affect port compatibility. In other words, a designer can
connect ports using two different windows into the same data type. This can be
used to connect different modules with different bandwidths. A data window
merely specifies the logical “gasket” used to connect two differently sized
ports.

The MLIR representation of data windows has yet to be determined.

Status: **planning**

 [Prev - 'esi' Dialect](https://circt.llvm.org/docs/Dialects/ESI/ "'esi' Dialect")
[Next - ESI Global Services](https://circt.llvm.org/docs/Dialects/ESI/services/ "ESI Global Services") 

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
    - ['esi' Dialect-](https://circt.llvm.org/docs/Dialects/ESI/)
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