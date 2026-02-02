'esi' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'esi' Dialect
=============

The Elastic Silicon Interconnect dialect aims to aid in accelerator system construction.

**WARNING**: The ESI dialect has evolved significantly since its inception while
these documents have not. As such, large parts are significantly out-of-date.

* [Application channels](#application-channels)
  + [ChannelBundleType](#channelbundletype)
  + [ChannelType](#channeltype)
  + [ClockType](#clocktype)
  + [ListType](#listtype)
  + [WindowType](#windowtype)
  + [FirMemType](#firmemtype)
  + [HLMemType](#hlmemtype)
  + [ImmutableType](#immutabletype)
  + [WindowFieldType](#windowfieldtype)
  + [WindowFrameType](#windowframetype)
  + [ListType](#listtype-1)
  + [`esi.buffer` (::circt::esi::ChannelBufferOp)](#esibuffer-circtesichannelbufferop)
  + [`esi.cosim.from_host` (::circt::esi::CosimFromHostEndpointOp)](#esicosimfrom_host-circtesicosimfromhostendpointop)
  + [`esi.cosim.to_host` (::circt::esi::CosimToHostEndpointOp)](#esicosimto_host-circtesicosimtohostendpointop)
  + [`esi.fifo` (::circt::esi::FIFOOp)](#esififo-circtesififoop)
  + [`esi.null` (::circt::esi::NullSourceOp)](#esinull-circtesinullsourceop)
  + [`esi.bundle.pack` (::circt::esi::PackBundleOp)](#esibundlepack-circtesipackbundleop)
  + [`esi.stage` (::circt::esi::PipelineStageOp)](#esistage-circtesipipelinestageop)
  + [`esi.snoop.xact` (::circt::esi::SnoopTransactionOp)](#esisnoopxact-circtesisnooptransactionop)
  + [`esi.snoop.vr` (::circt::esi::SnoopValidReadyOp)](#esisnoopvr-circtesisnoopvalidreadyop)
  + [`esi.bundle.unpack` (::circt::esi::UnpackBundleOp)](#esibundleunpack-circtesiunpackbundleop)
  + [`esi.unwrap.fifo` (::circt::esi::UnwrapFIFOOp)](#esiunwrapfifo-circtesiunwrapfifoop)
  + [`esi.unwrap.iface` (::circt::esi::UnwrapSVInterfaceOp)](#esiunwrapiface-circtesiunwrapsvinterfaceop)
  + [`esi.unwrap.vr` (::circt::esi::UnwrapValidReadyOp)](#esiunwrapvr-circtesiunwrapvalidreadyop)
  + [`esi.window.unwrap` (::circt::esi::UnwrapWindow)](#esiwindowunwrap-circtesiunwrapwindow)
  + [`esi.wrap.fifo` (::circt::esi::WrapFIFOOp)](#esiwrapfifo-circtesiwrapfifoop)
  + [`esi.wrap.iface` (::circt::esi::WrapSVInterfaceOp)](#esiwrapiface-circtesiwrapsvinterfaceop)
  + [`esi.wrap.vr` (::circt::esi::WrapValidReadyOp)](#esiwrapvr-circtesiwrapvalidreadyop)
  + [`esi.window.wrap` (::circt::esi::WrapWindow)](#esiwindowwrap-circtesiwrapwindow)
* [Services](#services)
  + [`esi.manifest.hier_node` (::circt::esi::AppIDHierNodeOp)](#esimanifesthier_node-circtesiappidhiernodeop)
  + [`esi.manifest.hier_root` (::circt::esi::AppIDHierRootOp)](#esimanifesthier_root-circtesiappidhierrootop)
  + [`esi.buffer` (::circt::esi::ChannelBufferOp)](#esibuffer-circtesichannelbufferop-1)
  + [`esi.manifest.compressed` (::circt::esi::CompressedManifestOp)](#esimanifestcompressed-circtesicompressedmanifestop)
  + [`esi.cosim.from_host` (::circt::esi::CosimFromHostEndpointOp)](#esicosimfrom_host-circtesicosimfromhostendpointop-1)
  + [`esi.cosim.to_host` (::circt::esi::CosimToHostEndpointOp)](#esicosimto_host-circtesicosimtohostendpointop-1)
  + [`esi.service.decl` (::circt::esi::CustomServiceDeclOp)](#esiservicedecl-circtesicustomservicedeclop)
  + [`esi.fifo` (::circt::esi::FIFOOp)](#esififo-circtesififoop-1)
  + [`esi.null` (::circt::esi::NullSourceOp)](#esinull-circtesinullsourceop-1)
  + [`esi.bundle.pack` (::circt::esi::PackBundleOp)](#esibundlepack-circtesipackbundleop-1)
  + [`esi.stage` (::circt::esi::PipelineStageOp)](#esistage-circtesipipelinestageop-1)
  + [`esi.service.req` (::circt::esi::RequestConnectionOp)](#esiservicereq-circtesirequestconnectionop)
  + [`esi.service.port` (::circt::esi::ServiceDeclPortOp)](#esiserviceport-circtesiservicedeclportop)
  + [`esi.manifest.impl_conn` (::circt::esi::ServiceImplClientRecordOp)](#esimanifestimpl_conn-circtesiserviceimplclientrecordop)
  + [`esi.manifest.service_impl` (::circt::esi::ServiceImplRecordOp)](#esimanifestservice_impl-circtesiserviceimplrecordop)
  + [`esi.service.impl_req.req` (::circt::esi::ServiceImplementConnReqOp)](#esiserviceimpl_reqreq-circtesiserviceimplementconnreqop)
  + [`esi.service.impl_req` (::circt::esi::ServiceImplementReqOp)](#esiserviceimpl_req-circtesiserviceimplementreqop)
  + [`esi.service.instance` (::circt::esi::ServiceInstanceOp)](#esiserviceinstance-circtesiserviceinstanceop)
  + [`esi.manifest.req` (::circt::esi::ServiceRequestRecordOp)](#esimanifestreq-circtesiservicerequestrecordop)
  + [`esi.snoop.xact` (::circt::esi::SnoopTransactionOp)](#esisnoopxact-circtesisnooptransactionop-1)
  + [`esi.snoop.vr` (::circt::esi::SnoopValidReadyOp)](#esisnoopvr-circtesisnoopvalidreadyop-1)
  + [`esi.manifest.constants` (::circt::esi::SymbolConstantsOp)](#esimanifestconstants-circtesisymbolconstantsop)
  + [`esi.manifest.sym` (::circt::esi::SymbolMetadataOp)](#esimanifestsym-circtesisymbolmetadataop)
  + [`esi.bundle.unpack` (::circt::esi::UnpackBundleOp)](#esibundleunpack-circtesiunpackbundleop-1)
  + [`esi.unwrap.fifo` (::circt::esi::UnwrapFIFOOp)](#esiunwrapfifo-circtesiunwrapfifoop-1)
  + [`esi.unwrap.iface` (::circt::esi::UnwrapSVInterfaceOp)](#esiunwrapiface-circtesiunwrapsvinterfaceop-1)
  + [`esi.unwrap.vr` (::circt::esi::UnwrapValidReadyOp)](#esiunwrapvr-circtesiunwrapvalidreadyop-1)
  + [`esi.window.unwrap` (::circt::esi::UnwrapWindow)](#esiwindowunwrap-circtesiunwrapwindow-1)
  + [`esi.wrap.fifo` (::circt::esi::WrapFIFOOp)](#esiwrapfifo-circtesiwrapfifoop-1)
  + [`esi.wrap.iface` (::circt::esi::WrapSVInterfaceOp)](#esiwrapiface-circtesiwrapsvinterfaceop-1)
  + [`esi.wrap.vr` (::circt::esi::WrapValidReadyOp)](#esiwrapvr-circtesiwrapvalidreadyop-1)
  + [`esi.window.wrap` (::circt::esi::WrapWindow)](#esiwindowwrap-circtesiwrapwindow-1)
  + [`esi.manifest.hier_node` (::circt::esi::AppIDHierNodeOp)](#esimanifesthier_node-circtesiappidhiernodeop-1)
  + [`esi.manifest.hier_root` (::circt::esi::AppIDHierRootOp)](#esimanifesthier_root-circtesiappidhierrootop-1)
  + [`esi.service.std.call` (::circt::esi::CallServiceDeclOp)](#esiservicestdcall-circtesicallservicedeclop)
  + [`esi.buffer` (::circt::esi::ChannelBufferOp)](#esibuffer-circtesichannelbufferop-2)
  + [`esi.manifest.compressed` (::circt::esi::CompressedManifestOp)](#esimanifestcompressed-circtesicompressedmanifestop-1)
  + [`esi.cosim.from_host` (::circt::esi::CosimFromHostEndpointOp)](#esicosimfrom_host-circtesicosimfromhostendpointop-2)
  + [`esi.cosim.to_host` (::circt::esi::CosimToHostEndpointOp)](#esicosimto_host-circtesicosimtohostendpointop-2)
  + [`esi.service.decl` (::circt::esi::CustomServiceDeclOp)](#esiservicedecl-circtesicustomservicedeclop-1)
  + [`esi.fifo` (::circt::esi::FIFOOp)](#esififo-circtesififoop-2)
  + [`esi.service.std.func` (::circt::esi::FuncServiceDeclOp)](#esiservicestdfunc-circtesifuncservicedeclop)
  + [`esi.service.std.hostmem` (::circt::esi::HostMemServiceDeclOp)](#esiservicestdhostmem-circtesihostmemservicedeclop)
  + [`esi.service.std.mmio` (::circt::esi::MMIOServiceDeclOp)](#esiservicestdmmio-circtesimmioservicedeclop)
  + [`esi.null` (::circt::esi::NullSourceOp)](#esinull-circtesinullsourceop-2)
  + [`esi.bundle.pack` (::circt::esi::PackBundleOp)](#esibundlepack-circtesipackbundleop-2)
  + [`esi.stage` (::circt::esi::PipelineStageOp)](#esistage-circtesipipelinestageop-2)
  + [`esi.mem.ram` (::circt::esi::RandomAccessMemoryDeclOp)](#esimemram-circtesirandomaccessmemorydeclop)
  + [`esi.service.req` (::circt::esi::RequestConnectionOp)](#esiservicereq-circtesirequestconnectionop-1)
  + [`esi.service.port` (::circt::esi::ServiceDeclPortOp)](#esiserviceport-circtesiservicedeclportop-1)
  + [`esi.manifest.impl_conn` (::circt::esi::ServiceImplClientRecordOp)](#esimanifestimpl_conn-circtesiserviceimplclientrecordop-1)
  + [`esi.manifest.service_impl` (::circt::esi::ServiceImplRecordOp)](#esimanifestservice_impl-circtesiserviceimplrecordop-1)
  + [`esi.service.impl_req.req` (::circt::esi::ServiceImplementConnReqOp)](#esiserviceimpl_reqreq-circtesiserviceimplementconnreqop-1)
  + [`esi.service.impl_req` (::circt::esi::ServiceImplementReqOp)](#esiserviceimpl_req-circtesiserviceimplementreqop-1)
  + [`esi.service.instance` (::circt::esi::ServiceInstanceOp)](#esiserviceinstance-circtesiserviceinstanceop-1)
  + [`esi.manifest.req` (::circt::esi::ServiceRequestRecordOp)](#esimanifestreq-circtesiservicerequestrecordop-1)
  + [`esi.snoop.xact` (::circt::esi::SnoopTransactionOp)](#esisnoopxact-circtesisnooptransactionop-2)
  + [`esi.snoop.vr` (::circt::esi::SnoopValidReadyOp)](#esisnoopvr-circtesisnoopvalidreadyop-2)
  + [`esi.manifest.constants` (::circt::esi::SymbolConstantsOp)](#esimanifestconstants-circtesisymbolconstantsop-1)
  + [`esi.manifest.sym` (::circt::esi::SymbolMetadataOp)](#esimanifestsym-circtesisymbolmetadataop-1)
  + [`esi.service.std.telemetry` (::circt::esi::TelemetryServiceDeclOp)](#esiservicestdtelemetry-circtesitelemetryservicedeclop)
  + [`esi.bundle.unpack` (::circt::esi::UnpackBundleOp)](#esibundleunpack-circtesiunpackbundleop-2)
  + [`esi.unwrap.fifo` (::circt::esi::UnwrapFIFOOp)](#esiunwrapfifo-circtesiunwrapfifoop-2)
  + [`esi.unwrap.iface` (::circt::esi::UnwrapSVInterfaceOp)](#esiunwrapiface-circtesiunwrapsvinterfaceop-2)
  + [`esi.unwrap.vr` (::circt::esi::UnwrapValidReadyOp)](#esiunwrapvr-circtesiunwrapvalidreadyop-2)
  + [`esi.window.unwrap` (::circt::esi::UnwrapWindow)](#esiwindowunwrap-circtesiunwrapwindow-2)
  + [`esi.wrap.fifo` (::circt::esi::WrapFIFOOp)](#esiwrapfifo-circtesiwrapfifoop-2)
  + [`esi.wrap.iface` (::circt::esi::WrapSVInterfaceOp)](#esiwrapiface-circtesiwrapsvinterfaceop-2)
  + [`esi.wrap.vr` (::circt::esi::WrapValidReadyOp)](#esiwrapvr-circtesiwrapvalidreadyop-2)
  + [`esi.window.wrap` (::circt::esi::WrapWindow)](#esiwindowwrap-circtesiwrapwindow-2)
* [Structural](#structural)
  + [`esi.pure_module.input` (::circt::esi::ESIPureModuleInputOp)](#esipure_moduleinput-circtesiesipuremoduleinputop)
  + [`esi.pure_module` (::circt::esi::ESIPureModuleOp)](#esipure_module-circtesiesipuremoduleop)
  + [`esi.pure_module.output` (::circt::esi::ESIPureModuleOutputOp)](#esipure_moduleoutput-circtesiesipuremoduleoutputop)
  + [`esi.pure_module.param` (::circt::esi::ESIPureModuleParamOp)](#esipure_moduleparam-circtesiesipuremoduleparamop)
* [Interfaces](#interfaces)
* [ChannelOpInterface (`ChannelOpInterface`)](#channelopinterface-channelopinterface)
  + [Methods:](#methods)
* [HasAppID (`HasAppIDOpInterface`)](#hasappid-hasappidopinterface)
  + [Methods:](#methods-1)
* [IsManifestData (`IsManifestData`)](#ismanifestdata-ismanifestdata)
  + [Methods:](#methods-2)
* [ServiceDeclOpInterface (`ServiceDeclOpInterface`)](#servicedeclopinterface-servicedeclopinterface)
  + [Methods:](#methods-3)

Application channels
--------------------

The main component of ESI are point-to-point, typed channels that allow
designers to connect modules to each other and software, then communicate by
sending messages. Channels largely abstract away the details of message
communication from the designer, though the designer can declaratively specify
how to implement the channel.

Messages have types: ints, structs, arrays, unions, and variable-length lists.
The width of a channel is not necessarily the same width as the message. ESI
“windows” can be used to break up a message into a series of “frames”. IP blocks
can emit / absorb “windowed” messages or full-sized messages, which can be
automatically broken up to save wire area at the cost of bandwidth.

Any channel which is exposed to the host will have a platform-agnostic software
API constructed for it based on the type of the channel. The software
application merely has to connect to the accelerator then invoke a method to
send or receive messages from the accelerator system.

### ChannelBundleType

*A bundle of channels*

Syntax:

```
!esi.bundle<
  ::llvm::ArrayRef<BundledChannel>,   # channels
  ::mlir::UnitAttr   # resettable
>
```

A channel bundle (sometimes referred to as just “bundle”) is a set of
channels of associated signals, along with per-channel names and directions.
The prototypical example for a bundle is a request-response channel pair.

The direction terminology is a bit confusing. Let us designate the module
which is outputting the bundle as the “sender” module and a module which has
a bundle as an input as the “receiver”. The directions “from” and “to” are
from the senders perspective. So, the “to” direction means that channel is
transmitting messages from the sender to the receiver. Then, “from” means
that the sender is getting messages from the receiver (typically responses).

When requesting a bundle from a service, the service is always considered
the sender; so, “to” means the service is sending messages to the client and
“from” means the service is receiving messages from the client.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| channels | `::llvm::ArrayRef<BundledChannel>` |  |
| resettable | `::mlir::UnitAttr` | boolean flag |

### ChannelType

*An ESI-compatible channel port*

Syntax:

```
!esi.channel<
  Type,   # inner
  ::circt::esi::ChannelSignaling,   # signaling
  uint64_t   # dataDelay
>
```

An ESI port kind which models a latency-insensitive, unidirectional,
point-to-point data stream. Channels are typed (like all of ESI). Said
type can be any MLIR type, but must be lowered to something a backend
knows how to output (i.e. something emitVerilog knows about).

Parameters:
signaling: the style of the control signals (valid/ready vs FIFO).
dataDelay: the number of cycles data takes to arrive after the control
indicates a transaction has occured. For instance, on a FIFO without read
ahead, this would be 1. Defaults to 0.

Example:

```
hw.module.extern @Sender() -> (%x: !esi.channel<i1>)
hw.module @Reciever(%a: !esi.channel<hw.array<5xi16>>) { }
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| inner | `Type` |  |
| signaling | `::circt::esi::ChannelSignaling` |  |
| dataDelay | `uint64_t` |  |

### ClockType

*A type for clock-carrying wires*

Syntax: `!seq.clock`

The `!seq.clock` type represents signals which can be used to drive the
clock input of sequential operations.

### ListType

*A runtime-variably sized list*

Syntax:

```
!esi.list<
  Type   # elementType
>
```

In software, a chunk of memory with runtime-specified length. In hardware, a
stream of runtime-specified amount of data transmitted over many cycles in
compile-time specified specified windows (chunks).

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `Type` |  |

### WindowType

*A data window*

Syntax:

```
!esi.window<
  StringAttr,   # name
  mlir::Type,   # into
  ::llvm::ArrayRef<WindowFrameType>   # frames
>
```

A ‘data window’ allows designers to break up large messages into multiple
frames (aka flits) spread across multiple cycles. Windows are specified in
terms of a mapping of struct fields to frames. The width of a window is the
maximum frame size + the union tag (log2(#frames)).

A data window does NOT imply an ESI channel.

Current restrictions:

* A field may only appear once.
* Fields may only be re-ordered (wrt the original message) within a frame.
* Array fields whose array length is not evenly divisible by ’numItems’ will
  have an implicit frame inserted directly after containing the leftover array
  items.
* A frame which contains an array with a ’numItems’ specification CANNOT
  contain another array with ’numItems’ specified or a list field.
* Likewise, a frame which contains a list field CANNOT contain another list
  field or an array with ’numItems’ specified.

When either a list or an array with ’numItems’ is specified in a frame, all
the other fields in the frame are assumed to be constant over the length of
the list/array transmission. This allows for efficient hardware
implementations where the constant fields can be held in registers while the
list/array segments stream through. In general, when a struct field appears
in multiple frames, it must be constant throughout the message transmission.

Typically lowered to a union of structs. There is a special case of one
frame with no name. In this case, it is lowered to just the struct itself.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| name | `StringAttr` |  |
| into | `mlir::Type` |  |
| frames | `::llvm::ArrayRef<WindowFrameType>` |  |

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

### WindowFieldType

*A field-in-frame specifier*

Syntax:

```
!esi.window.field<
  StringAttr,   # fieldName
  uint64_t,   # numItems
  uint64_t   # bulkCountWidth
>
```

Specify that a field should appear within the enclosing frame.

’numItems’ indicates how many items of the field should appear in the frame.
Only applicable to array or list fields. If not specified:

* For arrays, the entire array will appear in the frame.
* For lists, it is assumed to be 1.

For lists, there are two possible encodings:

* Parallel (default): a ’last’ field to the struct to indicate that the
  end of the list is this frame. If ’numItems’ is specified for a list, a
  ‘\_size’ (representing the number of valid items in the array)
  field will be added to the struct. When `<fieldName>_size` is less than
  ’numItems’, `last` MUST be true meaning that all frames must be full
  except for the last one. This format matches typical on-chip hardware
  streaming interfaces but is terribly inefficient for serial links.
* Serial (bulk transfer): in this encoding, there must be a ‘header’ frame
  immediately preceding the frame which contains the list field. That frame
  must include a special ‘count’ field which indicates how many items of the
  list will be transmitted next. Once the number of items are transmitted
  (transmitted frames == ceil(count/numItems)), the header is re-transmitted
  to set up the next bulk transfer *for the same list*. If the transfer was
  the last transfer, the header should set the count to zero. This method
  allows for efficient transfers over serial links (no ’last’ or ‘\_size’
  fields) without requiring knowledge of the total list size up front.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| fieldName | `StringAttr` |  |
| numItems | `uint64_t` | # of items in arrays or lists |
| bulkCountWidth | `uint64_t` | bitwidth of the ‘count’ field in bulk mode |

### WindowFrameType

*Declare a data window frame*

Syntax:

```
!esi.window.frame<
  StringAttr,   # name
  ::llvm::ArrayRef<WindowFieldType>   # members
>
```

A named list of fields which should appear in a given frame.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| name | `StringAttr` |  |
| members | `::llvm::ArrayRef<WindowFieldType>` |  |

### ListType

*A runtime-variably sized list*

Syntax:

```
!esi.list<
  Type   # elementType
>
```

In software, a chunk of memory with runtime-specified length. In hardware, a
stream of runtime-specified amount of data transmitted over many cycles in
compile-time specified specified windows (chunks).

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `Type` |  |

### `esi.buffer` (::circt::esi::ChannelBufferOp)

*Control options for an ESI channel.*

Syntax:

```
operation ::= `esi.buffer` $clk `,` $rst `,` $input attr-dict
              `:` qualified(type($input)) `->` qualified(type($output))
```

A channel buffer (`buffer`) is essentially a set of options on a channel.
It always adds at least one cycle of latency (pipeline stage) to the
channel, but this is configurable.

This operation is inserted on an ESI dataflow edge. It must exist
previous to SystemVerilog emission but can be added in a lowering pass.

A `stages` attribute may be provided to specify a specific number of cycles
(pipeline stages) to use on this channel. Must be greater than 0.

A `name` attribute may be provided to assigned a name to a buffered
connection.

Example:

```
%esiChan = hw.instance "sender" @Sender () : () -> (!esi.channel<i1>)
// Allow automatic selection of options.
%bufferedChan = esi.buffer %esiChan : i1
hw.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

// Alternatively, specify the number of stages.
%fourStageBufferedChan = esi.buffer %esiChan { stages = 4 } : i1
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ChannelOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `stages` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `esi.cosim.from_host` (::circt::esi::CosimFromHostEndpointOp)

*Co-simulation endpoint receiving data from the host*

Syntax:

```
operation ::= `esi.cosim.from_host` $clk `,` $rst `,` $id attr-dict `:` qualified(type($fromHost))
```

A co-simulation endpoint is a connection from the simulation to some
outside process, usually a software application responsible for driving
the simulation (driver).

It is uni-directional, in this case receiving data from the host for the
simulation.

NOTE: $id MUST be unique across all endpoints at simulation runtime.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `id` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `fromHost` | an ESI channel |

### `esi.cosim.to_host` (::circt::esi::CosimToHostEndpointOp)

*Co-simulation endpoint sending data to the host.*

Syntax:

```
operation ::= `esi.cosim.to_host` $clk `,` $rst `,` $toHost`,` $id attr-dict `:` qualified(type($toHost))
```

A co-simulation endpoint is a connection from the simulation to some
outside process, usually a software application responsible for driving
the simulation (driver).

It is uni-directional, in this case sending data from the simulation to the
host.

NOTE: $id MUST be unique across all endpoints at simulation runtime.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `id` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `toHost` | an ESI channel |

### `esi.fifo` (::circt::esi::FIFOOp)

*A FIFO with ESI channel connections*

Syntax:

```
operation ::= `esi.fifo` `in` $input `clk` $clk `rst` $rst `depth` $depth attr-dict
              `:` type($input) `->` type($output)
```

A FIFO is a first-in-first-out buffer. This operation is a simple FIFO
which can be used to connect two ESI channels. The ESI channels MUST have
FIFO signaling semantics.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `depth` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `esi.null` (::circt::esi::NullSourceOp)

*An op which never produces messages.*

Syntax:

```
operation ::= `esi.null` attr-dict `:` qualified(type($out))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `out` | an ESI channel |

### `esi.bundle.pack` (::circt::esi::PackBundleOp)

*Pack channels into a bundle*

Syntax:

```
operation ::= `esi.bundle.pack` $toChannels attr-dict `:` custom<UnPackBundleType>(
              type($toChannels), type($fromChannels), type($bundle))
```

Interfaces: `OpAsmOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `toChannels` | variadic of an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `bundle` | a bundle of channels |
| `fromChannels` | variadic of an ESI channel |

### `esi.stage` (::circt::esi::PipelineStageOp)

*An elastic buffer stage.*

Syntax:

```
operation ::= `esi.stage` $clk `,` $rst `,` $input attr-dict
              `:` qualified(type($input))
```

An individual elastic pipeline register. Generally lowered to from a
ChannelBuffer (‘buffer’), though can be inserted anywhere to add an
additional pipeline stage. Adding individually could be useful for
late-pass latency balancing.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ChannelOpInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `esi.snoop.xact` (::circt::esi::SnoopTransactionOp)

*Get the data and transaction signal from a channel*

Syntax:

```
operation ::= `esi.snoop.xact` $input attr-dict `:` qualified(type($input))
```

A snoop that observes when a transaction occurs on a channel and provides
the data being transmitted. The transaction signal indicates when data is
actually being transferred on the channel, regardless of the underlying
signaling protocol (ValidReady or FIFO). Like other snoop operations, this
does not count as another user of the channel. Useful for monitoring data
flow and debugging.

Interfaces: `InferTypeOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `transaction` | 1-bit signless integer |
| `data` | any type |

### `esi.snoop.vr` (::circt::esi::SnoopValidReadyOp)

*Get the valid, ready, and data signals from a channel*

Syntax:

```
operation ::= `esi.snoop.vr` $input attr-dict `:` qualified(type($input))
```

A snoop allows one to combinationally observe a channel’s internal signals.
It does not count as another user of the channel. Useful for constructing
control logic which can be combinationally driven. Also potentially useful
for debugging.

Interfaces: `InferTypeOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `valid` | 1-bit signless integer |
| `ready` | 1-bit signless integer |
| `data` | any type |

### `esi.bundle.unpack` (::circt::esi::UnpackBundleOp)

*Unpack channels from a bundle*

Syntax:

```
operation ::= `esi.bundle.unpack` $fromChannels `from` $bundle attr-dict `:` custom<UnPackBundleType>(
              type($toChannels), type($fromChannels), type($bundle))
```

Interfaces: `OpAsmOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `bundle` | a bundle of channels |
| `fromChannels` | variadic of an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `toChannels` | variadic of an ESI channel |

### `esi.unwrap.fifo` (::circt::esi::UnwrapFIFOOp)

*Unwrap a value from an ESI port into a FIFO interface*

Syntax:

```
operation ::= `esi.unwrap.fifo` $chanInput `,` $rden attr-dict `:` qualified(type($chanInput))
```

Interfaces: `ChannelOpInterface`, `InferTypeOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `chanInput` | an ESI channel |
| `rden` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `data` | any type |
| `empty` | 1-bit signless integer |

### `esi.unwrap.iface` (::circt::esi::UnwrapSVInterfaceOp)

*Unwrap an SV interface from an ESI port*

Syntax:

```
operation ::= `esi.unwrap.iface` $chanInput `into` $interfaceSource attr-dict `:` `(` qualified(type($chanInput)) `,` qualified(type($interfaceSource)) `)`
```

Unwrap an ESI channel into a SystemVerilog interface containing valid,
ready, and data signals.

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `chanInput` | an ESI channel |
| `interfaceSource` | sv.interface |

### `esi.unwrap.vr` (::circt::esi::UnwrapValidReadyOp)

*Unwrap a value from an ESI port*

Unwrapping a value allows operations on the contained value. Unwrap the
channel along with a ready signal that you generate. Result is the data
along with a valid signal.

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `chanInput` | an ESI channel |
| `ready` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `rawOutput` | any type |
| `valid` | 1-bit signless integer |

### `esi.window.unwrap` (::circt::esi::UnwrapWindow)

*Unwrap a data window into a union*

Syntax:

```
operation ::= `esi.window.unwrap` $window attr-dict `:` qualified(type($window))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `window` | a data window |

#### Results:

| Result | Description |
| --- | --- |
| `frame` | any type |

### `esi.wrap.fifo` (::circt::esi::WrapFIFOOp)

*Wrap a value into an ESI port with FIFO signaling*

Syntax:

```
operation ::= `esi.wrap.fifo` $data `,` $empty attr-dict `:`
              custom<WrapFIFOType>(type($data), type($chanOutput))
```

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `data` | any type |
| `empty` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `chanOutput` | an ESI channel |
| `rden` | 1-bit signless integer |

### `esi.wrap.iface` (::circt::esi::WrapSVInterfaceOp)

*Wrap an SV interface into an ESI port*

Syntax:

```
operation ::= `esi.wrap.iface` $interfaceSink attr-dict `:` qualified(type($interfaceSink)) `->` qualified(type($output))
```

Wrap a SystemVerilog interface into an ESI channel. Interface MUST look
like an interface produced by ESI meaning it MUST contain valid, ready,
and data signals. Any other signals will be discarded.

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `interfaceSink` | sv.interface |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `esi.wrap.vr` (::circt::esi::WrapValidReadyOp)

*Wrap a value into an ESI port*

Wrapping a value into an ESI port type allows modules to send values down
an ESI port. Wrap data with valid bit, result is the ESI channel and the
ready signal from the other end of the channel.

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `rawInput` | any type |
| `valid` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `chanOutput` | an ESI channel |
| `ready` | 1-bit signless integer |

### `esi.window.wrap` (::circt::esi::WrapWindow)

*Wrap a union into a data window*

Syntax:

```
operation ::= `esi.window.wrap` $frame attr-dict `:` custom<InferWindowRet>(type($frame), type($window))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `frame` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `window` | a data window |

Services
--------

ESI “services” provide device-wide connectivity and arbitration for shared
resources, which can be requested from any IP block (service “client”). Standard
services will include DRAM, clock/reset, statistical counter reporting, and
debug.

### `esi.manifest.hier_node` (::circt::esi::AppIDHierNodeOp)

*A node in the AppID hierarchy*

Syntax:

```
operation ::= `esi.manifest.hier_node` qualified($appID) `mod` $moduleRef attr-dict-with-keyword $children
```

Traits: `HasParent<circt::esi::AppIDHierRootOp, circt::esi::AppIDHierNodeOp>`, `NoTerminator`, `SingleBlock`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `appID` | ::circt::esi::AppIDAttr | An application relevant instance identifier |
| `moduleRef` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

### `esi.manifest.hier_root` (::circt::esi::AppIDHierRootOp)

*The root of an appid instance hierarchy*

Syntax:

```
operation ::= `esi.manifest.hier_root` $topModuleRef attr-dict-with-keyword $children
```

Traits: `HasParent<mlir::ModuleOp>`, `NoTerminator`, `SingleBlock`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `topModuleRef` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

### `esi.buffer` (::circt::esi::ChannelBufferOp)

*Control options for an ESI channel.*

Syntax:

```
operation ::= `esi.buffer` $clk `,` $rst `,` $input attr-dict
              `:` qualified(type($input)) `->` qualified(type($output))
```

A channel buffer (`buffer`) is essentially a set of options on a channel.
It always adds at least one cycle of latency (pipeline stage) to the
channel, but this is configurable.

This operation is inserted on an ESI dataflow edge. It must exist
previous to SystemVerilog emission but can be added in a lowering pass.

A `stages` attribute may be provided to specify a specific number of cycles
(pipeline stages) to use on this channel. Must be greater than 0.

A `name` attribute may be provided to assigned a name to a buffered
connection.

Example:

```
%esiChan = hw.instance "sender" @Sender () : () -> (!esi.channel<i1>)
// Allow automatic selection of options.
%bufferedChan = esi.buffer %esiChan : i1
hw.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

// Alternatively, specify the number of stages.
%fourStageBufferedChan = esi.buffer %esiChan { stages = 4 } : i1
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ChannelOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `stages` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `esi.manifest.compressed` (::circt::esi::CompressedManifestOp)

*A zlib-compressed JSON manifest*

Syntax:

```
operation ::= `esi.manifest.compressed` $compressedManifest attr-dict
```

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `compressedManifest` | ::circt::esi::BlobAttr | A binary blob |

### `esi.cosim.from_host` (::circt::esi::CosimFromHostEndpointOp)

*Co-simulation endpoint receiving data from the host*

Syntax:

```
operation ::= `esi.cosim.from_host` $clk `,` $rst `,` $id attr-dict `:` qualified(type($fromHost))
```

A co-simulation endpoint is a connection from the simulation to some
outside process, usually a software application responsible for driving
the simulation (driver).

It is uni-directional, in this case receiving data from the host for the
simulation.

NOTE: $id MUST be unique across all endpoints at simulation runtime.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `id` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `fromHost` | an ESI channel |

### `esi.cosim.to_host` (::circt::esi::CosimToHostEndpointOp)

*Co-simulation endpoint sending data to the host.*

Syntax:

```
operation ::= `esi.cosim.to_host` $clk `,` $rst `,` $toHost`,` $id attr-dict `:` qualified(type($toHost))
```

A co-simulation endpoint is a connection from the simulation to some
outside process, usually a software application responsible for driving
the simulation (driver).

It is uni-directional, in this case sending data from the simulation to the
host.

NOTE: $id MUST be unique across all endpoints at simulation runtime.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `id` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `toHost` | an ESI channel |

### `esi.service.decl` (::circt::esi::CustomServiceDeclOp)

*An ESI service interface declaration*

Syntax:

```
operation ::= `esi.service.decl` $sym_name $ports attr-dict
```

A declaration of an ESI service interface. Defines a contract between a
service provider and its clients.

Example:

```
esi.service.decl @HostComms {
  esi.service.port send : !esi.bundle<[!esi.any from "send"]>
  esi.service.port recieve : !esi.channel<[i8 to "recv"]>
}
```

Traits: `HasParent<::mlir::ModuleOp>`, `NoTerminator`, `SingleBlock`

Interfaces: `ServiceDeclOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `esi.fifo` (::circt::esi::FIFOOp)

*A FIFO with ESI channel connections*

Syntax:

```
operation ::= `esi.fifo` `in` $input `clk` $clk `rst` $rst `depth` $depth attr-dict
              `:` type($input) `->` type($output)
```

A FIFO is a first-in-first-out buffer. This operation is a simple FIFO
which can be used to connect two ESI channels. The ESI channels MUST have
FIFO signaling semantics.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `depth` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `esi.null` (::circt::esi::NullSourceOp)

*An op which never produces messages.*

Syntax:

```
operation ::= `esi.null` attr-dict `:` qualified(type($out))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `out` | an ESI channel |

### `esi.bundle.pack` (::circt::esi::PackBundleOp)

*Pack channels into a bundle*

Syntax:

```
operation ::= `esi.bundle.pack` $toChannels attr-dict `:` custom<UnPackBundleType>(
              type($toChannels), type($fromChannels), type($bundle))
```

Interfaces: `OpAsmOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `toChannels` | variadic of an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `bundle` | a bundle of channels |
| `fromChannels` | variadic of an ESI channel |

### `esi.stage` (::circt::esi::PipelineStageOp)

*An elastic buffer stage.*

Syntax:

```
operation ::= `esi.stage` $clk `,` $rst `,` $input attr-dict
              `:` qualified(type($input))
```

An individual elastic pipeline register. Generally lowered to from a
ChannelBuffer (‘buffer’), though can be inserted anywhere to add an
additional pipeline stage. Adding individually could be useful for
late-pass latency balancing.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ChannelOpInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `esi.service.req` (::circt::esi::RequestConnectionOp)

*Request a connection to receive data*

Syntax:

```
operation ::= `esi.service.req` $servicePort `(` qualified($appID) `)`
              attr-dict `:` qualified(type($toClient))
```

Interfaces: `HasAppID`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `servicePort` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |
| `appID` | ::circt::esi::AppIDAttr | An application relevant instance identifier |

#### Results:

| Result | Description |
| --- | --- |
| `toClient` | a bundle of channels |

### `esi.service.port` (::circt::esi::ServiceDeclPortOp)

*An ESI service bundle being received by the client*

Syntax:

```
operation ::= `esi.service.port` $inner_sym  attr-dict `:` $toClientType
```

Traits: `HasParent<::circt::esi::CustomServiceDeclOp>`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::mlir::StringAttr | string attribute |
| `toClientType` | ::mlir::TypeAttr | type attribute of a bundle of channels |

### `esi.manifest.impl_conn` (::circt::esi::ServiceImplClientRecordOp)

*Details of a service implementation client connection*

Syntax:

```
operation ::= `esi.manifest.impl_conn` $relAppIDPath `req` $servicePort `(` $typeID `)`
              (`channels` $channelAssignments^)? (`with` $implDetails^)? attr-dict
```

A record containing all the necessary details of how to connect to a client
which the parent service record is servicing. Emitted on a per-client bundle
basis. There shall be at most on of these records in the entire manifest for
a particular client.

Traits: `HasParent<ServiceImplRecordOp>`

Interfaces: `IsManifestData`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `relAppIDPath` | ::mlir::ArrayAttr | Array of AppIDs |
| `servicePort` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |
| `typeID` | ::mlir::TypeAttr | type attribute of a bundle of channels |
| `channelAssignments` | ::mlir::DictionaryAttr | dictionary of named attribute values |
| `implDetails` | ::mlir::DictionaryAttr | dictionary of named attribute values |

### `esi.manifest.service_impl` (::circt::esi::ServiceImplRecordOp)

*Record of a service implementation*

Syntax:

```
operation ::= `esi.manifest.service_impl` qualified($appID) (`svc` $service^)? (`std` $stdService^)?
              `by` $serviceImplName (`engine` $isEngine^)? `with` $implDetails
              attr-dict-with-keyword custom<ServiceImplRecordReqDetails>($reqDetails)
```

A record of a service implementation. Optionally emitted by the service
implementation. Contains information necessary to connect to the service and
service clients.

Traits: `NoTerminator`

Interfaces: `HasAppID`, `IsManifestData`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `appID` | ::circt::esi::AppIDAttr | An application relevant instance identifier |
| `isEngine` | ::mlir::UnitAttr | unit attribute |
| `service` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `stdService` | ::mlir::StringAttr | string attribute |
| `serviceImplName` | ::mlir::StringAttr | string attribute |
| `implDetails` | ::mlir::DictionaryAttr | dictionary of named attribute values |

### `esi.service.impl_req.req` (::circt::esi::ServiceImplementConnReqOp)

*The canonical form of a connection request*

Syntax:

```
operation ::= `esi.service.impl_req.req` $servicePort `(` $relativeAppIDPath `)`
              attr-dict `:` qualified(type($toClient))
```

Interfaces: `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `servicePort` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |
| `relativeAppIDPath` | ::mlir::ArrayAttr | Array of AppIDs |

#### Results:

| Result | Description |
| --- | --- |
| `toClient` | a bundle of channels |

### `esi.service.impl_req` (::circt::esi::ServiceImplementReqOp)

*Request for a service to be implemented*

Syntax:

```
operation ::= `esi.service.impl_req` qualified($appID) (`svc` $service_symbol^)? `impl` `as` $impl_type
              (`std` $stdService^)? (`opts` $impl_opts^)? `(` $inputs `)`
              attr-dict `:` functional-type($inputs, results)
              $portReqs
```

The connect services pass replaces `service.instance`s with this op. The
`portReqs` region is the set of connection requests which need to be
implemented for this service instance. Channels to/from the requests have
been added to the operands/results of this op and consumers/producers have
been redirected.

Some other pass or frontend is expected to replace this op with an actual
implementation.

Traits: `NoTerminator`

Interfaces: `HasAppID`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `appID` | ::circt::esi::AppIDAttr | An application relevant instance identifier |
| `service_symbol` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `impl_type` | ::mlir::StringAttr | string attribute |
| `stdService` | ::mlir::StringAttr | string attribute |
| `impl_opts` | ::mlir::DictionaryAttr | dictionary of named attribute values |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `esi.service.instance` (::circt::esi::ServiceInstanceOp)

*Instantiate a server module*

Syntax:

```
operation ::= `esi.service.instance` qualified($appID) (`svc` $service_symbol^)? `impl` `as` $impl_type
              (`opts` $impl_opts^)? `(` $inputs `)`
              attr-dict `:` functional-type($inputs, results)
```

Instantiate a service adhering to a service declaration interface.

A pass collects all of the connection requests to the service this op
implements from the containing modules’ descendants (in the instance
hierarchy). It bubbles them all up to the module containing this op,
creating the necessary ESI channel ports, groups them appropriately, then
replaces this op with a `service.impl_req`.

If ‘service\_symbol’ isn’t specified, this instance will be used to implement
all of the service requests which get surfaced to here. This option is
generally used at the top level to specify host connectivity.

Since implementing the server will usually need “normal” I/O, `inputs` and
`results` act like normal `hw.instance` ports.

$identifier is used by frontends to specify or remember the type of
implementation to use for this service.

Interfaces: `HasAppID`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `appID` | ::circt::esi::AppIDAttr | An application relevant instance identifier |
| `service_symbol` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `impl_type` | ::mlir::StringAttr | string attribute |
| `impl_opts` | ::mlir::DictionaryAttr | dictionary of named attribute values |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| «unnamed» | variadic of any type |

### `esi.manifest.req` (::circt::esi::ServiceRequestRecordOp)

*Record of a service request*

Syntax:

```
operation ::= `esi.manifest.req` qualified($requestor) `,` $servicePort (`std` $stdService^)?
              `,` $typeID attr-dict
```

A record of a service request, including the requestor, the service
requested, and the parameters of the request. Emitted before connecting the
service to preserve metadata about the original request.

Interfaces: `HasAppID`, `IsManifestData`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `requestor` | ::circt::esi::AppIDAttr | An application relevant instance identifier |
| `servicePort` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |
| `stdService` | ::mlir::StringAttr | string attribute |
| `typeID` | ::mlir::TypeAttr | type attribute of a bundle of channels |

### `esi.snoop.xact` (::circt::esi::SnoopTransactionOp)

*Get the data and transaction signal from a channel*

Syntax:

```
operation ::= `esi.snoop.xact` $input attr-dict `:` qualified(type($input))
```

A snoop that observes when a transaction occurs on a channel and provides
the data being transmitted. The transaction signal indicates when data is
actually being transferred on the channel, regardless of the underlying
signaling protocol (ValidReady or FIFO). Like other snoop operations, this
does not count as another user of the channel. Useful for monitoring data
flow and debugging.

Interfaces: `InferTypeOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `transaction` | 1-bit signless integer |
| `data` | any type |

### `esi.snoop.vr` (::circt::esi::SnoopValidReadyOp)

*Get the valid, ready, and data signals from a channel*

Syntax:

```
operation ::= `esi.snoop.vr` $input attr-dict `:` qualified(type($input))
```

A snoop allows one to combinationally observe a channel’s internal signals.
It does not count as another user of the channel. Useful for constructing
control logic which can be combinationally driven. Also potentially useful
for debugging.

Interfaces: `InferTypeOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `valid` | 1-bit signless integer |
| `ready` | 1-bit signless integer |
| `data` | any type |

### `esi.manifest.constants` (::circt::esi::SymbolConstantsOp)

*Constant values associated with a symbol*

Syntax:

```
operation ::= `esi.manifest.constants` $symbolRef $constants attr-dict
```

Interfaces: `IsManifestData`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `symbolRef` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `constants` | ::mlir::DictionaryAttr | dictionary of named attribute values |

### `esi.manifest.sym` (::circt::esi::SymbolMetadataOp)

*Metadata about a symbol*

Syntax:

```
operation ::= `esi.manifest.sym` $symbolRef
              (`name` $name^)?
              (`repo` $repo^)?
              (`commit` $commitHash^)?
              (`version` $version^)?
              (`summary` $summary^)?
              attr-dict
```

Metadata about a symbol, including its name, repository, commit hash,
version, and summary. All are optional, but strongly encouraged. Any
additional metadata which users wish to attach should go as discardable
attributes.

Interfaces: `IsManifestData`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `symbolRef` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `name` | ::mlir::StringAttr | string attribute |
| `repo` | ::mlir::StringAttr | string attribute |
| `commitHash` | ::mlir::StringAttr | string attribute |
| `version` | ::mlir::StringAttr | string attribute |
| `summary` | ::mlir::StringAttr | string attribute |

### `esi.bundle.unpack` (::circt::esi::UnpackBundleOp)

*Unpack channels from a bundle*

Syntax:

```
operation ::= `esi.bundle.unpack` $fromChannels `from` $bundle attr-dict `:` custom<UnPackBundleType>(
              type($toChannels), type($fromChannels), type($bundle))
```

Interfaces: `OpAsmOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `bundle` | a bundle of channels |
| `fromChannels` | variadic of an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `toChannels` | variadic of an ESI channel |

### `esi.unwrap.fifo` (::circt::esi::UnwrapFIFOOp)

*Unwrap a value from an ESI port into a FIFO interface*

Syntax:

```
operation ::= `esi.unwrap.fifo` $chanInput `,` $rden attr-dict `:` qualified(type($chanInput))
```

Interfaces: `ChannelOpInterface`, `InferTypeOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `chanInput` | an ESI channel |
| `rden` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `data` | any type |
| `empty` | 1-bit signless integer |

### `esi.unwrap.iface` (::circt::esi::UnwrapSVInterfaceOp)

*Unwrap an SV interface from an ESI port*

Syntax:

```
operation ::= `esi.unwrap.iface` $chanInput `into` $interfaceSource attr-dict `:` `(` qualified(type($chanInput)) `,` qualified(type($interfaceSource)) `)`
```

Unwrap an ESI channel into a SystemVerilog interface containing valid,
ready, and data signals.

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `chanInput` | an ESI channel |
| `interfaceSource` | sv.interface |

### `esi.unwrap.vr` (::circt::esi::UnwrapValidReadyOp)

*Unwrap a value from an ESI port*

Unwrapping a value allows operations on the contained value. Unwrap the
channel along with a ready signal that you generate. Result is the data
along with a valid signal.

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `chanInput` | an ESI channel |
| `ready` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `rawOutput` | any type |
| `valid` | 1-bit signless integer |

### `esi.window.unwrap` (::circt::esi::UnwrapWindow)

*Unwrap a data window into a union*

Syntax:

```
operation ::= `esi.window.unwrap` $window attr-dict `:` qualified(type($window))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `window` | a data window |

#### Results:

| Result | Description |
| --- | --- |
| `frame` | any type |

### `esi.wrap.fifo` (::circt::esi::WrapFIFOOp)

*Wrap a value into an ESI port with FIFO signaling*

Syntax:

```
operation ::= `esi.wrap.fifo` $data `,` $empty attr-dict `:`
              custom<WrapFIFOType>(type($data), type($chanOutput))
```

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `data` | any type |
| `empty` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `chanOutput` | an ESI channel |
| `rden` | 1-bit signless integer |

### `esi.wrap.iface` (::circt::esi::WrapSVInterfaceOp)

*Wrap an SV interface into an ESI port*

Syntax:

```
operation ::= `esi.wrap.iface` $interfaceSink attr-dict `:` qualified(type($interfaceSink)) `->` qualified(type($output))
```

Wrap a SystemVerilog interface into an ESI channel. Interface MUST look
like an interface produced by ESI meaning it MUST contain valid, ready,
and data signals. Any other signals will be discarded.

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `interfaceSink` | sv.interface |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `esi.wrap.vr` (::circt::esi::WrapValidReadyOp)

*Wrap a value into an ESI port*

Wrapping a value into an ESI port type allows modules to send values down
an ESI port. Wrap data with valid bit, result is the ESI channel and the
ready signal from the other end of the channel.

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `rawInput` | any type |
| `valid` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `chanOutput` | an ESI channel |
| `ready` | 1-bit signless integer |

### `esi.window.wrap` (::circt::esi::WrapWindow)

*Wrap a union into a data window*

Syntax:

```
operation ::= `esi.window.wrap` $frame attr-dict `:` custom<InferWindowRet>(type($frame), type($window))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `frame` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `window` | a data window |

### `esi.manifest.hier_node` (::circt::esi::AppIDHierNodeOp)

*A node in the AppID hierarchy*

Syntax:

```
operation ::= `esi.manifest.hier_node` qualified($appID) `mod` $moduleRef attr-dict-with-keyword $children
```

Traits: `HasParent<circt::esi::AppIDHierRootOp, circt::esi::AppIDHierNodeOp>`, `NoTerminator`, `SingleBlock`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `appID` | ::circt::esi::AppIDAttr | An application relevant instance identifier |
| `moduleRef` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

### `esi.manifest.hier_root` (::circt::esi::AppIDHierRootOp)

*The root of an appid instance hierarchy*

Syntax:

```
operation ::= `esi.manifest.hier_root` $topModuleRef attr-dict-with-keyword $children
```

Traits: `HasParent<mlir::ModuleOp>`, `NoTerminator`, `SingleBlock`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `topModuleRef` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

### `esi.service.std.call` (::circt::esi::CallServiceDeclOp)

*Service against which hardware can call into software*

Syntax:

```
operation ::= `esi.service.std.call` $sym_name attr-dict
```

Traits: `HasParent<::mlir::ModuleOp>`, `NoTerminator`, `SingleBlock`

Interfaces: `ServiceDeclOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `esi.buffer` (::circt::esi::ChannelBufferOp)

*Control options for an ESI channel.*

Syntax:

```
operation ::= `esi.buffer` $clk `,` $rst `,` $input attr-dict
              `:` qualified(type($input)) `->` qualified(type($output))
```

A channel buffer (`buffer`) is essentially a set of options on a channel.
It always adds at least one cycle of latency (pipeline stage) to the
channel, but this is configurable.

This operation is inserted on an ESI dataflow edge. It must exist
previous to SystemVerilog emission but can be added in a lowering pass.

A `stages` attribute may be provided to specify a specific number of cycles
(pipeline stages) to use on this channel. Must be greater than 0.

A `name` attribute may be provided to assigned a name to a buffered
connection.

Example:

```
%esiChan = hw.instance "sender" @Sender () : () -> (!esi.channel<i1>)
// Allow automatic selection of options.
%bufferedChan = esi.buffer %esiChan : i1
hw.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

// Alternatively, specify the number of stages.
%fourStageBufferedChan = esi.buffer %esiChan { stages = 4 } : i1
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ChannelOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `stages` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `esi.manifest.compressed` (::circt::esi::CompressedManifestOp)

*A zlib-compressed JSON manifest*

Syntax:

```
operation ::= `esi.manifest.compressed` $compressedManifest attr-dict
```

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `compressedManifest` | ::circt::esi::BlobAttr | A binary blob |

### `esi.cosim.from_host` (::circt::esi::CosimFromHostEndpointOp)

*Co-simulation endpoint receiving data from the host*

Syntax:

```
operation ::= `esi.cosim.from_host` $clk `,` $rst `,` $id attr-dict `:` qualified(type($fromHost))
```

A co-simulation endpoint is a connection from the simulation to some
outside process, usually a software application responsible for driving
the simulation (driver).

It is uni-directional, in this case receiving data from the host for the
simulation.

NOTE: $id MUST be unique across all endpoints at simulation runtime.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `id` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `fromHost` | an ESI channel |

### `esi.cosim.to_host` (::circt::esi::CosimToHostEndpointOp)

*Co-simulation endpoint sending data to the host.*

Syntax:

```
operation ::= `esi.cosim.to_host` $clk `,` $rst `,` $toHost`,` $id attr-dict `:` qualified(type($toHost))
```

A co-simulation endpoint is a connection from the simulation to some
outside process, usually a software application responsible for driving
the simulation (driver).

It is uni-directional, in this case sending data from the simulation to the
host.

NOTE: $id MUST be unique across all endpoints at simulation runtime.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `id` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `toHost` | an ESI channel |

### `esi.service.decl` (::circt::esi::CustomServiceDeclOp)

*An ESI service interface declaration*

Syntax:

```
operation ::= `esi.service.decl` $sym_name $ports attr-dict
```

A declaration of an ESI service interface. Defines a contract between a
service provider and its clients.

Example:

```
esi.service.decl @HostComms {
  esi.service.port send : !esi.bundle<[!esi.any from "send"]>
  esi.service.port recieve : !esi.channel<[i8 to "recv"]>
}
```

Traits: `HasParent<::mlir::ModuleOp>`, `NoTerminator`, `SingleBlock`

Interfaces: `ServiceDeclOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `esi.fifo` (::circt::esi::FIFOOp)

*A FIFO with ESI channel connections*

Syntax:

```
operation ::= `esi.fifo` `in` $input `clk` $clk `rst` $rst `depth` $depth attr-dict
              `:` type($input) `->` type($output)
```

A FIFO is a first-in-first-out buffer. This operation is a simple FIFO
which can be used to connect two ESI channels. The ESI channels MUST have
FIFO signaling semantics.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `depth` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `esi.service.std.func` (::circt::esi::FuncServiceDeclOp)

*Function service*

Syntax:

```
operation ::= `esi.service.std.func` $sym_name attr-dict
```

Declares a service which provides a function call interface to a client.

Ports:
to\_client call(args: any) -> result: any
Client exposes a function call interface to the user and does not allow
out-of-order returns.

TODO: ports for out-of-order returns

Traits: `HasParent<::mlir::ModuleOp>`, `NoTerminator`, `SingleBlock`

Interfaces: `ServiceDeclOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `esi.service.std.hostmem` (::circt::esi::HostMemServiceDeclOp)

*Host memory service*

Syntax:

```
operation ::= `esi.service.std.hostmem` $sym_name attr-dict
```

Declares a service to read/write host memory. Used for DMA services. Must be
implemented by a BSP.

Traits: `HasParent<::mlir::ModuleOp>`

Interfaces: `ServiceDeclOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `esi.service.std.mmio` (::circt::esi::MMIOServiceDeclOp)

*MMIO service*

Syntax:

```
operation ::= `esi.service.std.mmio` $sym_name attr-dict
```

Declares a service to be backed by a MMIO interface, which is platform
dependent. Must be implemented by a BSP.

Traits: `HasParent<::mlir::ModuleOp>`

Interfaces: `ServiceDeclOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `esi.null` (::circt::esi::NullSourceOp)

*An op which never produces messages.*

Syntax:

```
operation ::= `esi.null` attr-dict `:` qualified(type($out))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `out` | an ESI channel |

### `esi.bundle.pack` (::circt::esi::PackBundleOp)

*Pack channels into a bundle*

Syntax:

```
operation ::= `esi.bundle.pack` $toChannels attr-dict `:` custom<UnPackBundleType>(
              type($toChannels), type($fromChannels), type($bundle))
```

Interfaces: `OpAsmOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `toChannels` | variadic of an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `bundle` | a bundle of channels |
| `fromChannels` | variadic of an ESI channel |

### `esi.stage` (::circt::esi::PipelineStageOp)

*An elastic buffer stage.*

Syntax:

```
operation ::= `esi.stage` $clk `,` $rst `,` $input attr-dict
              `:` qualified(type($input))
```

An individual elastic pipeline register. Generally lowered to from a
ChannelBuffer (‘buffer’), though can be inserted anywhere to add an
additional pipeline stage. Adding individually could be useful for
late-pass latency balancing.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ChannelOpInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `clk` | A type for clock-carrying wires |
| `rst` | 1-bit signless integer |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `esi.mem.ram` (::circt::esi::RandomAccessMemoryDeclOp)

*Random access memory service*

Syntax:

```
operation ::= `esi.mem.ram` $sym_name $innerType `x` $depth attr-dict
```

Declares a service which is backed by a memory of some sort. Allows random
access of the inner elements.

Ports:
read(address: clog2(depth)) -> data: innerType
write({address: clog2(depth), data: innerType}) -> done: i0

Users can ensure R/W ordering by waiting for the write “done” message before
issuing a potentially dependant read. Ordering of R/W messages in flight is
undefined.

Traits: `HasParent<::mlir::ModuleOp>`, `NoTerminator`, `SingleBlock`

Interfaces: `ServiceDeclOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `innerType` | ::mlir::TypeAttr | any type attribute |
| `depth` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

### `esi.service.req` (::circt::esi::RequestConnectionOp)

*Request a connection to receive data*

Syntax:

```
operation ::= `esi.service.req` $servicePort `(` qualified($appID) `)`
              attr-dict `:` qualified(type($toClient))
```

Interfaces: `HasAppID`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `servicePort` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |
| `appID` | ::circt::esi::AppIDAttr | An application relevant instance identifier |

#### Results:

| Result | Description |
| --- | --- |
| `toClient` | a bundle of channels |

### `esi.service.port` (::circt::esi::ServiceDeclPortOp)

*An ESI service bundle being received by the client*

Syntax:

```
operation ::= `esi.service.port` $inner_sym  attr-dict `:` $toClientType
```

Traits: `HasParent<::circt::esi::CustomServiceDeclOp>`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inner_sym` | ::mlir::StringAttr | string attribute |
| `toClientType` | ::mlir::TypeAttr | type attribute of a bundle of channels |

### `esi.manifest.impl_conn` (::circt::esi::ServiceImplClientRecordOp)

*Details of a service implementation client connection*

Syntax:

```
operation ::= `esi.manifest.impl_conn` $relAppIDPath `req` $servicePort `(` $typeID `)`
              (`channels` $channelAssignments^)? (`with` $implDetails^)? attr-dict
```

A record containing all the necessary details of how to connect to a client
which the parent service record is servicing. Emitted on a per-client bundle
basis. There shall be at most on of these records in the entire manifest for
a particular client.

Traits: `HasParent<ServiceImplRecordOp>`

Interfaces: `IsManifestData`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `relAppIDPath` | ::mlir::ArrayAttr | Array of AppIDs |
| `servicePort` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |
| `typeID` | ::mlir::TypeAttr | type attribute of a bundle of channels |
| `channelAssignments` | ::mlir::DictionaryAttr | dictionary of named attribute values |
| `implDetails` | ::mlir::DictionaryAttr | dictionary of named attribute values |

### `esi.manifest.service_impl` (::circt::esi::ServiceImplRecordOp)

*Record of a service implementation*

Syntax:

```
operation ::= `esi.manifest.service_impl` qualified($appID) (`svc` $service^)? (`std` $stdService^)?
              `by` $serviceImplName (`engine` $isEngine^)? `with` $implDetails
              attr-dict-with-keyword custom<ServiceImplRecordReqDetails>($reqDetails)
```

A record of a service implementation. Optionally emitted by the service
implementation. Contains information necessary to connect to the service and
service clients.

Traits: `NoTerminator`

Interfaces: `HasAppID`, `IsManifestData`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `appID` | ::circt::esi::AppIDAttr | An application relevant instance identifier |
| `isEngine` | ::mlir::UnitAttr | unit attribute |
| `service` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `stdService` | ::mlir::StringAttr | string attribute |
| `serviceImplName` | ::mlir::StringAttr | string attribute |
| `implDetails` | ::mlir::DictionaryAttr | dictionary of named attribute values |

### `esi.service.impl_req.req` (::circt::esi::ServiceImplementConnReqOp)

*The canonical form of a connection request*

Syntax:

```
operation ::= `esi.service.impl_req.req` $servicePort `(` $relativeAppIDPath `)`
              attr-dict `:` qualified(type($toClient))
```

Interfaces: `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `servicePort` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |
| `relativeAppIDPath` | ::mlir::ArrayAttr | Array of AppIDs |

#### Results:

| Result | Description |
| --- | --- |
| `toClient` | a bundle of channels |

### `esi.service.impl_req` (::circt::esi::ServiceImplementReqOp)

*Request for a service to be implemented*

Syntax:

```
operation ::= `esi.service.impl_req` qualified($appID) (`svc` $service_symbol^)? `impl` `as` $impl_type
              (`std` $stdService^)? (`opts` $impl_opts^)? `(` $inputs `)`
              attr-dict `:` functional-type($inputs, results)
              $portReqs
```

The connect services pass replaces `service.instance`s with this op. The
`portReqs` region is the set of connection requests which need to be
implemented for this service instance. Channels to/from the requests have
been added to the operands/results of this op and consumers/producers have
been redirected.

Some other pass or frontend is expected to replace this op with an actual
implementation.

Traits: `NoTerminator`

Interfaces: `HasAppID`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `appID` | ::circt::esi::AppIDAttr | An application relevant instance identifier |
| `service_symbol` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `impl_type` | ::mlir::StringAttr | string attribute |
| `stdService` | ::mlir::StringAttr | string attribute |
| `impl_opts` | ::mlir::DictionaryAttr | dictionary of named attribute values |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `esi.service.instance` (::circt::esi::ServiceInstanceOp)

*Instantiate a server module*

Syntax:

```
operation ::= `esi.service.instance` qualified($appID) (`svc` $service_symbol^)? `impl` `as` $impl_type
              (`opts` $impl_opts^)? `(` $inputs `)`
              attr-dict `:` functional-type($inputs, results)
```

Instantiate a service adhering to a service declaration interface.

A pass collects all of the connection requests to the service this op
implements from the containing modules’ descendants (in the instance
hierarchy). It bubbles them all up to the module containing this op,
creating the necessary ESI channel ports, groups them appropriately, then
replaces this op with a `service.impl_req`.

If ‘service\_symbol’ isn’t specified, this instance will be used to implement
all of the service requests which get surfaced to here. This option is
generally used at the top level to specify host connectivity.

Since implementing the server will usually need “normal” I/O, `inputs` and
`results` act like normal `hw.instance` ports.

$identifier is used by frontends to specify or remember the type of
implementation to use for this service.

Interfaces: `HasAppID`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `appID` | ::circt::esi::AppIDAttr | An application relevant instance identifier |
| `service_symbol` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `impl_type` | ::mlir::StringAttr | string attribute |
| `impl_opts` | ::mlir::DictionaryAttr | dictionary of named attribute values |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| «unnamed» | variadic of any type |

### `esi.manifest.req` (::circt::esi::ServiceRequestRecordOp)

*Record of a service request*

Syntax:

```
operation ::= `esi.manifest.req` qualified($requestor) `,` $servicePort (`std` $stdService^)?
              `,` $typeID attr-dict
```

A record of a service request, including the requestor, the service
requested, and the parameters of the request. Emitted before connecting the
service to preserve metadata about the original request.

Interfaces: `HasAppID`, `IsManifestData`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `requestor` | ::circt::esi::AppIDAttr | An application relevant instance identifier |
| `servicePort` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |
| `stdService` | ::mlir::StringAttr | string attribute |
| `typeID` | ::mlir::TypeAttr | type attribute of a bundle of channels |

### `esi.snoop.xact` (::circt::esi::SnoopTransactionOp)

*Get the data and transaction signal from a channel*

Syntax:

```
operation ::= `esi.snoop.xact` $input attr-dict `:` qualified(type($input))
```

A snoop that observes when a transaction occurs on a channel and provides
the data being transmitted. The transaction signal indicates when data is
actually being transferred on the channel, regardless of the underlying
signaling protocol (ValidReady or FIFO). Like other snoop operations, this
does not count as another user of the channel. Useful for monitoring data
flow and debugging.

Interfaces: `InferTypeOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `transaction` | 1-bit signless integer |
| `data` | any type |

### `esi.snoop.vr` (::circt::esi::SnoopValidReadyOp)

*Get the valid, ready, and data signals from a channel*

Syntax:

```
operation ::= `esi.snoop.vr` $input attr-dict `:` qualified(type($input))
```

A snoop allows one to combinationally observe a channel’s internal signals.
It does not count as another user of the channel. Useful for constructing
control logic which can be combinationally driven. Also potentially useful
for debugging.

Interfaces: `InferTypeOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `valid` | 1-bit signless integer |
| `ready` | 1-bit signless integer |
| `data` | any type |

### `esi.manifest.constants` (::circt::esi::SymbolConstantsOp)

*Constant values associated with a symbol*

Syntax:

```
operation ::= `esi.manifest.constants` $symbolRef $constants attr-dict
```

Interfaces: `IsManifestData`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `symbolRef` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `constants` | ::mlir::DictionaryAttr | dictionary of named attribute values |

### `esi.manifest.sym` (::circt::esi::SymbolMetadataOp)

*Metadata about a symbol*

Syntax:

```
operation ::= `esi.manifest.sym` $symbolRef
              (`name` $name^)?
              (`repo` $repo^)?
              (`commit` $commitHash^)?
              (`version` $version^)?
              (`summary` $summary^)?
              attr-dict
```

Metadata about a symbol, including its name, repository, commit hash,
version, and summary. All are optional, but strongly encouraged. Any
additional metadata which users wish to attach should go as discardable
attributes.

Interfaces: `IsManifestData`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `symbolRef` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `name` | ::mlir::StringAttr | string attribute |
| `repo` | ::mlir::StringAttr | string attribute |
| `commitHash` | ::mlir::StringAttr | string attribute |
| `version` | ::mlir::StringAttr | string attribute |
| `summary` | ::mlir::StringAttr | string attribute |

### `esi.service.std.telemetry` (::circt::esi::TelemetryServiceDeclOp)

*Telemetry service*

Syntax:

```
operation ::= `esi.service.std.telemetry` $sym_name attr-dict
```

Declares a service to send telemetry data. Has one port ‘report’ for
something to request telemetry data (via a ‘get’ channel to the client and a
‘data’ channel for the return value).

Traits: `HasParent<::mlir::ModuleOp>`

Interfaces: `ServiceDeclOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `esi.bundle.unpack` (::circt::esi::UnpackBundleOp)

*Unpack channels from a bundle*

Syntax:

```
operation ::= `esi.bundle.unpack` $fromChannels `from` $bundle attr-dict `:` custom<UnPackBundleType>(
              type($toChannels), type($fromChannels), type($bundle))
```

Interfaces: `OpAsmOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `bundle` | a bundle of channels |
| `fromChannels` | variadic of an ESI channel |

#### Results:

| Result | Description |
| --- | --- |
| `toChannels` | variadic of an ESI channel |

### `esi.unwrap.fifo` (::circt::esi::UnwrapFIFOOp)

*Unwrap a value from an ESI port into a FIFO interface*

Syntax:

```
operation ::= `esi.unwrap.fifo` $chanInput `,` $rden attr-dict `:` qualified(type($chanInput))
```

Interfaces: `ChannelOpInterface`, `InferTypeOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `chanInput` | an ESI channel |
| `rden` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `data` | any type |
| `empty` | 1-bit signless integer |

### `esi.unwrap.iface` (::circt::esi::UnwrapSVInterfaceOp)

*Unwrap an SV interface from an ESI port*

Syntax:

```
operation ::= `esi.unwrap.iface` $chanInput `into` $interfaceSource attr-dict `:` `(` qualified(type($chanInput)) `,` qualified(type($interfaceSource)) `)`
```

Unwrap an ESI channel into a SystemVerilog interface containing valid,
ready, and data signals.

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `chanInput` | an ESI channel |
| `interfaceSource` | sv.interface |

### `esi.unwrap.vr` (::circt::esi::UnwrapValidReadyOp)

*Unwrap a value from an ESI port*

Unwrapping a value allows operations on the contained value. Unwrap the
channel along with a ready signal that you generate. Result is the data
along with a valid signal.

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `chanInput` | an ESI channel |
| `ready` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `rawOutput` | any type |
| `valid` | 1-bit signless integer |

### `esi.window.unwrap` (::circt::esi::UnwrapWindow)

*Unwrap a data window into a union*

Syntax:

```
operation ::= `esi.window.unwrap` $window attr-dict `:` qualified(type($window))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `window` | a data window |

#### Results:

| Result | Description |
| --- | --- |
| `frame` | any type |

### `esi.wrap.fifo` (::circt::esi::WrapFIFOOp)

*Wrap a value into an ESI port with FIFO signaling*

Syntax:

```
operation ::= `esi.wrap.fifo` $data `,` $empty attr-dict `:`
              custom<WrapFIFOType>(type($data), type($chanOutput))
```

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `data` | any type |
| `empty` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `chanOutput` | an ESI channel |
| `rden` | 1-bit signless integer |

### `esi.wrap.iface` (::circt::esi::WrapSVInterfaceOp)

*Wrap an SV interface into an ESI port*

Syntax:

```
operation ::= `esi.wrap.iface` $interfaceSink attr-dict `:` qualified(type($interfaceSink)) `->` qualified(type($output))
```

Wrap a SystemVerilog interface into an ESI channel. Interface MUST look
like an interface produced by ESI meaning it MUST contain valid, ready,
and data signals. Any other signals will be discarded.

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `interfaceSink` | sv.interface |

#### Results:

| Result | Description |
| --- | --- |
| `output` | an ESI channel |

### `esi.wrap.vr` (::circt::esi::WrapValidReadyOp)

*Wrap a value into an ESI port*

Wrapping a value into an ESI port type allows modules to send values down
an ESI port. Wrap data with valid bit, result is the ESI channel and the
ready signal from the other end of the channel.

Interfaces: `ChannelOpInterface`

#### Operands:

| Operand | Description |
| --- | --- |
| `rawInput` | any type |
| `valid` | 1-bit signless integer |

#### Results:

| Result | Description |
| --- | --- |
| `chanOutput` | an ESI channel |
| `ready` | 1-bit signless integer |

### `esi.window.wrap` (::circt::esi::WrapWindow)

*Wrap a union into a data window*

Syntax:

```
operation ::= `esi.window.wrap` $frame attr-dict `:` custom<InferWindowRet>(type($frame), type($window))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `frame` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `window` | a data window |

Structural
----------

ESI has a special module which doesn’t expose ports. All external interactions
are expected to be done through services.

### `esi.pure_module.input` (::circt::esi::ESIPureModuleInputOp)

*Inputs become input ports when the module is lowered*

Syntax:

```
operation ::= `esi.pure_module.input` $name attr-dict `:` type($value)
```

To create input ports when lowering a pure module op into an HWModuleOp, use
this op. This op is typically created by a service implementation generator.

If two ‘input’ ops exist in the same block, the names match, and the type
matches they’ll become one port during lowering. Two or more may not exist
with the same name and different types. Useful for ‘clk’ and ‘rst’.

Traits: `HasParent<ESIPureModuleOp>`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `value` | any type |

### `esi.pure_module` (::circt::esi::ESIPureModuleOp)

*ESI pure module*

Syntax:

```
operation ::= `esi.pure_module` $sym_name attr-dict-with-keyword $body
```

A module containing only ESI channels and modules with only ESI ports. All
non-local connectivity is done through ESI services. If this module is the
top level in the design, then the design’s actual top level ports are
defined by a BSP.

Useful on its own for simulation and BSPs which don’t define a top-level.

Traits: `HasParent<mlir::ModuleOp>`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

Interfaces: `HWModuleLike`, `InstanceGraphModuleOpInterface`, `PortList`, `RegionKindInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `esi.pure_module.output` (::circt::esi::ESIPureModuleOutputOp)

*Outputs become output ports when the module is lowered*

Syntax:

```
operation ::= `esi.pure_module.output` $name `,` $value attr-dict `:` type($value)
```

To create output ports when lowering a pure module op into an HWModuleOp, use
this op. This op is typically created by a service implementation generator.

Two ‘output’ ops with the same name cannot exist in the same block.

Traits: `HasParent<ESIPureModuleOp>`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `value` | any type |

### `esi.pure_module.param` (::circt::esi::ESIPureModuleParamOp)

*Params become module parameters when the module is lowered*

Syntax:

```
operation ::= `esi.pure_module.param` $name `:` $type attr-dict
```

Allows attaching parameters to modules which become HW module parameters
when lowering. Currently, they are ignored. Some low-level BSPs instantiate
modules with parameters. This allows the modules produced to accept
parameters so those BSPs can instantiate them.

Traits: `HasParent<ESIPureModuleOp>`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `type` | ::mlir::TypeAttr | any type attribute |

Interfaces
----------

Misc CIRCT interfaces.

ChannelOpInterface (`ChannelOpInterface`)
-----------------------------------------

“An interface for operations which carries channel semantics.”

### Methods:

#### `channelType`

```
circt::esi::ChannelType channelType();
```

“Returns the channel type of this operation.”

NOTE: This method *must* be implemented by the user.

#### `innerType`

```
mlir::Type innerType();
```

“Returns the inner type of this channel. This will be the type of the
data value of the channel, if the channel carries data semantics. Else,
return NoneType.”

NOTE: This method *must* be implemented by the user.

HasAppID (`HasAppIDOpInterface`)
--------------------------------

Op can be identified by an AppID.

### Methods:

#### `getAppID`

```
::circt::esi::AppIDAttr getAppID();
```

Returns the AppID of this operation.

NOTE: This method *must* be implemented by the user.

IsManifestData (`IsManifestData`)
---------------------------------

Op’s attributes should be represented in the manifest.

### Methods:

#### `getManifestClass`

```
StringRef getManifestClass();
```

Get the class name for this op.

NOTE: This method *must* be implemented by the user.

#### `getSymbolRefAttr`

```
FlatSymbolRefAttr getSymbolRefAttr();
```

Get the symbol to which this manifest data is referring, if any.

NOTE: This method *must* be implemented by the user.

#### `getDetails`

```
void getDetails(SmallVectorImpl<NamedAttribute>&results);
```

Populate results with the manifest data.

NOTE: This method *must* be implemented by the user.

#### `getDetailsAsDict`

```
DictionaryAttr getDetailsAsDict();
```

Get the manifest data from this op as an attribute.

NOTE: This method *must* be implemented by the user.

ServiceDeclOpInterface (`ServiceDeclOpInterface`)
-------------------------------------------------

Any op which represents a service declaration should implement this
interface.

### Methods:

#### `getPortList`

```
void getPortList(llvm::SmallVectorImpl<ServicePortInfo>&ports);
```

Returns the list of interface ports for this service interface.

NOTE: This method *must* be implemented by the user.

#### `getTypeName`

```
std::optional<StringRef> getTypeName();
```

Return a well-known name for this service type.

NOTE: This method *must* be implemented by the user.

#### `getPortInfo`

```
FailureOr<ServicePortInfo> getPortInfo(StringAttr portName);
```

Get info on a particular port.

NOTE: This method *must* be implemented by the user.

'esi' Dialect Docs
------------------

* [ESI data types and communication types](https://circt.llvm.org/docs/Dialects/ESI/types/)
* [ESI Global Services](https://circt.llvm.org/docs/Dialects/ESI/services/)
* [ESI Software APIs](https://circt.llvm.org/docs/Dialects/ESI/software_api/)
* [Miscellaneous Notes](https://circt.llvm.org/docs/Dialects/ESI/notes/)
* [The Elastic Silicon Interconnect dialect](https://circt.llvm.org/docs/Dialects/ESI/RationaleESI/)

 [Prev - Emission (Emit) Dialect Rationale](https://circt.llvm.org/docs/Dialects/Emit/RationaleEmit/ "Emission (Emit) Dialect Rationale")
[Next - ESI data types and communication types](https://circt.llvm.org/docs/Dialects/ESI/types/ "ESI data types and communication types") 

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