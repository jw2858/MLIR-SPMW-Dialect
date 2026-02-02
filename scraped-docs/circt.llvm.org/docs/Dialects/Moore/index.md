Moore Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

Moore Dialect
=============

This dialect provides operations and types to capture a SystemVerilog design after parsing, type checking, and elaboration.

* [Rationale](#rationale)
* [Types](#types)
  + [Simple Bit Vector Type](#simple-bit-vector-type)
  + [Event Type](#event-type)
  + [Default Values](#default-values)
  + [ArrayType](#arraytype)
  + [AssocArrayType](#assocarraytype)
  + [ChandleType](#chandletype)
  + [ClassHandleType](#classhandletype)
  + [EventType](#eventtype)
  + [FormatStringType](#formatstringtype)
  + [IntType](#inttype)
  + [NullType](#nulltype)
  + [OpenArrayType](#openarraytype)
  + [OpenUnpackedArrayType](#openunpackedarraytype)
  + [QueueType](#queuetype)
  + [RealType](#realtype)
  + [RefType](#reftype)
  + [StringType](#stringtype)
  + [StructType](#structtype)
  + [TimeType](#timetype)
  + [UnionType](#uniontype)
  + [UnpackedArrayType](#unpackedarraytype)
  + [UnpackedStructType](#unpackedstructtype)
  + [UnpackedUnionType](#unpackeduniontype)
  + [VoidType](#voidtype)
* [Operations](#operations)
  + [`moore.ashr` (::circt::moore::AShrOp)](#mooreashr-circtmooreashrop)
  + [`moore.builtin.acos` (::circt::moore::AcosBIOp)](#moorebuiltinacos-circtmooreacosbiop)
  + [`moore.builtin.acosh` (::circt::moore::AcoshBIOp)](#moorebuiltinacosh-circtmooreacoshbiop)
  + [`moore.add` (::circt::moore::AddOp)](#mooreadd-circtmooreaddop)
  + [`moore.fadd` (::circt::moore::AddRealOp)](#moorefadd-circtmooreaddrealop)
  + [`moore.and` (::circt::moore::AndOp)](#mooreand-circtmooreandop)
  + [`moore.array_create` (::circt::moore::ArrayCreateOp)](#moorearray_create-circtmoorearraycreateop)
  + [`moore.builtin.asin` (::circt::moore::AsinBIOp)](#moorebuiltinasin-circtmooreasinbiop)
  + [`moore.builtin.asinh` (::circt::moore::AsinhBIOp)](#moorebuiltinasinh-circtmooreasinhbiop)
  + [`moore.assert` (::circt::moore::AssertOp)](#mooreassert-circtmooreassertop)
  + [`moore.assigned_variable` (::circt::moore::AssignedVariableOp)](#mooreassigned_variable-circtmooreassignedvariableop)
  + [`moore.assume` (::circt::moore::AssumeOp)](#mooreassume-circtmooreassumeop)
  + [`moore.builtin.atan` (::circt::moore::AtanBIOp)](#moorebuiltinatan-circtmooreatanbiop)
  + [`moore.builtin.atanh` (::circt::moore::AtanhBIOp)](#moorebuiltinatanh-circtmooreatanhbiop)
  + [`moore.builtin.bitstoreal` (::circt::moore::BitstorealBIOp)](#moorebuiltinbitstoreal-circtmoorebitstorealbiop)
  + [`moore.builtin.bitstoshortreal` (::circt::moore::BitstoshortrealBIOp)](#moorebuiltinbitstoshortreal-circtmoorebitstoshortrealbiop)
  + [`moore.blocking_assign` (::circt::moore::BlockingAssignOp)](#mooreblocking_assign-circtmooreblockingassignop)
  + [`moore.bool_cast` (::circt::moore::BoolCastOp)](#moorebool_cast-circtmooreboolcastop)
  + [`moore.case_eq` (::circt::moore::CaseEqOp)](#moorecase_eq-circtmoorecaseeqop)
  + [`moore.case_ne` (::circt::moore::CaseNeOp)](#moorecase_ne-circtmoorecaseneop)
  + [`moore.casexz_eq` (::circt::moore::CaseXZEqOp)](#moorecasexz_eq-circtmoorecasexzeqop)
  + [`moore.casez_eq` (::circt::moore::CaseZEqOp)](#moorecasez_eq-circtmoorecasezeqop)
  + [`moore.builtin.ceil` (::circt::moore::CeilBIOp)](#moorebuiltinceil-circtmooreceilbiop)
  + [`moore.class.classdecl` (::circt::moore::ClassDeclOp)](#mooreclassclassdecl-circtmooreclassdeclop)
  + [`moore.class.methoddecl` (::circt::moore::ClassMethodDeclOp)](#mooreclassmethoddecl-circtmooreclassmethoddeclop)
  + [`moore.class.new` (::circt::moore::ClassNewOp)](#mooreclassnew-circtmooreclassnewop)
  + [`moore.class.propertydecl` (::circt::moore::ClassPropertyDeclOp)](#mooreclasspropertydecl-circtmooreclasspropertydeclop)
  + [`moore.class.property_ref` (::circt::moore::ClassPropertyRefOp)](#mooreclassproperty_ref-circtmooreclasspropertyrefop)
  + [`moore.class.upcast` (::circt::moore::ClassUpcastOp)](#mooreclassupcast-circtmooreclassupcastop)
  + [`moore.builtin.clog2` (::circt::moore::Clog2BIOp)](#moorebuiltinclog2-circtmooreclog2biop)
  + [`moore.concat` (::circt::moore::ConcatOp)](#mooreconcat-circtmooreconcatop)
  + [`moore.concat_ref` (::circt::moore::ConcatRefOp)](#mooreconcat_ref-circtmooreconcatrefop)
  + [`moore.conditional` (::circt::moore::ConditionalOp)](#mooreconditional-circtmooreconditionalop)
  + [`moore.constant` (::circt::moore::ConstantOp)](#mooreconstant-circtmooreconstantop)
  + [`moore.constant_real` (::circt::moore::ConstantRealOp)](#mooreconstant_real-circtmooreconstantrealop)
  + [`moore.constant_string` (::circt::moore::ConstantStringOp)](#mooreconstant_string-circtmooreconstantstringop)
  + [`moore.constant_time` (::circt::moore::ConstantTimeOp)](#mooreconstant_time-circtmooreconstanttimeop)
  + [`moore.assign` (::circt::moore::ContinuousAssignOp)](#mooreassign-circtmoorecontinuousassignop)
  + [`moore.conversion` (::circt::moore::ConversionOp)](#mooreconversion-circtmooreconversionop)
  + [`moore.convert_real` (::circt::moore::ConvertRealOp)](#mooreconvert_real-circtmooreconvertrealop)
  + [`moore.builtin.cos` (::circt::moore::CosBIOp)](#moorebuiltincos-circtmoorecosbiop)
  + [`moore.builtin.cosh` (::circt::moore::CoshBIOp)](#moorebuiltincosh-circtmoorecoshbiop)
  + [`moore.cover` (::circt::moore::CoverOp)](#moorecover-circtmoorecoverop)
  + [`moore.delayed_assign` (::circt::moore::DelayedContinuousAssignOp)](#mooredelayed_assign-circtmooredelayedcontinuousassignop)
  + [`moore.delayed_nonblocking_assign` (::circt::moore::DelayedNonBlockingAssignOp)](#mooredelayed_nonblocking_assign-circtmooredelayednonblockingassignop)
  + [`moore.detect_event` (::circt::moore::DetectEventOp)](#mooredetect_event-circtmooredetecteventop)
  + [`moore.builtin.display` (::circt::moore::DisplayBIOp)](#moorebuiltindisplay-circtmooredisplaybiop)
  + [`moore.fdiv` (::circt::moore::DivRealOp)](#moorefdiv-circtmooredivrealop)
  + [`moore.divs` (::circt::moore::DivSOp)](#mooredivs-circtmooredivsop)
  + [`moore.divu` (::circt::moore::DivUOp)](#mooredivu-circtmooredivuop)
  + [`moore.dyn_extract` (::circt::moore::DynExtractOp)](#mooredyn_extract-circtmooredynextractop)
  + [`moore.dyn_extract_ref` (::circt::moore::DynExtractRefOp)](#mooredyn_extract_ref-circtmooredynextractrefop)
  + [`moore.dyn_queue_extract` (::circt::moore::DynQueueExtractOp)](#mooredyn_queue_extract-circtmooredynqueueextractop)
  + [`moore.dyn_queue_extract_ref` (::circt::moore::DynQueueExtractRefOp)](#mooredyn_queue_extract_ref-circtmooredynqueueextractrefop)
  + [`moore.eq` (::circt::moore::EqOp)](#mooreeq-circtmooreeqop)
  + [`moore.feq` (::circt::moore::EqRealOp)](#moorefeq-circtmooreeqrealop)
  + [`moore.builtin.exp` (::circt::moore::ExpBIOp)](#moorebuiltinexp-circtmooreexpbiop)
  + [`moore.extract` (::circt::moore::ExtractOp)](#mooreextract-circtmooreextractop)
  + [`moore.extract_ref` (::circt::moore::ExtractRefOp)](#mooreextract_ref-circtmooreextractrefop)
  + [`moore.fge` (::circt::moore::FgeOp)](#moorefge-circtmoorefgeop)
  + [`moore.fgt` (::circt::moore::FgtOp)](#moorefgt-circtmoorefgtop)
  + [`moore.builtin.finish` (::circt::moore::FinishBIOp)](#moorebuiltinfinish-circtmoorefinishbiop)
  + [`moore.builtin.finish_message` (::circt::moore::FinishMessageBIOp)](#moorebuiltinfinish_message-circtmoorefinishmessagebiop)
  + [`moore.fle` (::circt::moore::FleOp)](#moorefle-circtmoorefleop)
  + [`moore.builtin.floor` (::circt::moore::FloorBIOp)](#moorebuiltinfloor-circtmoorefloorbiop)
  + [`moore.flt` (::circt::moore::FltOp)](#mooreflt-circtmoorefltop)
  + [`moore.fmt.concat` (::circt::moore::FormatConcatOp)](#moorefmtconcat-circtmooreformatconcatop)
  + [`moore.fmt.int` (::circt::moore::FormatIntOp)](#moorefmtint-circtmooreformatintop)
  + [`moore.fmt.literal` (::circt::moore::FormatLiteralOp)](#moorefmtliteral-circtmooreformatliteralop)
  + [`moore.fmt.real` (::circt::moore::FormatRealOp)](#moorefmtreal-circtmooreformatrealop)
  + [`moore.fmt.string` (::circt::moore::FormatStringOp)](#moorefmtstring-circtmooreformatstringop)
  + [`moore.fstring_to_string` (::circt::moore::FormatStringToStringOp)](#moorefstring_to_string-circtmooreformatstringtostringop)
  + [`moore.get_global_variable` (::circt::moore::GetGlobalVariableOp)](#mooreget_global_variable-circtmooregetglobalvariableop)
  + [`moore.global_variable` (::circt::moore::GlobalVariableOp)](#mooreglobal_variable-circtmooreglobalvariableop)
  + [`moore.handle_case_eq` (::circt::moore::HandleCaseEqOp)](#moorehandle_case_eq-circtmoorehandlecaseeqop)
  + [`moore.handle_case_ne` (::circt::moore::HandleCaseNeOp)](#moorehandle_case_ne-circtmoorehandlecaseneop)
  + [`moore.handle_eq` (::circt::moore::HandleEqOp)](#moorehandle_eq-circtmoorehandleeqop)
  + [`moore.handle_ne` (::circt::moore::HandleNeOp)](#moorehandle_ne-circtmoorehandleneop)
  + [`moore.instance` (::circt::moore::InstanceOp)](#mooreinstance-circtmooreinstanceop)
  + [`moore.int_to_logic` (::circt::moore::IntToLogicOp)](#mooreint_to_logic-circtmooreinttologicop)
  + [`moore.int_to_string` (::circt::moore::IntToStringOp)](#mooreint_to_string-circtmooreinttostringop)
  + [`moore.builtin.ln` (::circt::moore::LnBIOp)](#moorebuiltinln-circtmoorelnbiop)
  + [`moore.builtin.log10` (::circt::moore::Log10BIOp)](#moorebuiltinlog10-circtmoorelog10biop)
  + [`moore.logic_to_int` (::circt::moore::LogicToIntOp)](#moorelogic_to_int-circtmoorelogictointop)
  + [`moore.logic_to_time` (::circt::moore::LogicToTimeOp)](#moorelogic_to_time-circtmoorelogictotimeop)
  + [`moore.mods` (::circt::moore::ModSOp)](#mooremods-circtmooremodsop)
  + [`moore.modu` (::circt::moore::ModUOp)](#mooremodu-circtmooremoduop)
  + [`moore.mul` (::circt::moore::MulOp)](#mooremul-circtmooremulop)
  + [`moore.fmul` (::circt::moore::MulRealOp)](#moorefmul-circtmooremulrealop)
  + [`moore.ne` (::circt::moore::NeOp)](#moorene-circtmooreneop)
  + [`moore.fne` (::circt::moore::NeRealOp)](#moorefne-circtmoorenerealop)
  + [`moore.neg` (::circt::moore::NegOp)](#mooreneg-circtmoorenegop)
  + [`moore.fneg` (::circt::moore::NegRealOp)](#moorefneg-circtmoorenegrealop)
  + [`moore.net` (::circt::moore::NetOp)](#moorenet-circtmoorenetop)
  + [`moore.nonblocking_assign` (::circt::moore::NonBlockingAssignOp)](#moorenonblocking_assign-circtmoorenonblockingassignop)
  + [`moore.not` (::circt::moore::NotOp)](#moorenot-circtmoorenotop)
  + [`moore.null` (::circt::moore::NullOp)](#moorenull-circtmoorenullop)
  + [`moore.or` (::circt::moore::OrOp)](#mooreor-circtmooreorop)
  + [`moore.output` (::circt::moore::OutputOp)](#mooreoutput-circtmooreoutputop)
  + [`moore.packed_to_sbv` (::circt::moore::PackedToSBVOp)](#moorepacked_to_sbv-circtmoorepackedtosbvop)
  + [`moore.fpow` (::circt::moore::PowRealOp)](#moorefpow-circtmoorepowrealop)
  + [`moore.pows` (::circt::moore::PowSOp)](#moorepows-circtmoorepowsop)
  + [`moore.powu` (::circt::moore::PowUOp)](#moorepowu-circtmoorepowuop)
  + [`moore.procedure` (::circt::moore::ProcedureOp)](#mooreprocedure-circtmooreprocedureop)
  + [`moore.pop_back` (::circt::moore::QueuePopBackOp)](#moorepop_back-circtmoorequeuepopbackop)
  + [`moore.pop_front` (::circt::moore::QueuePopFrontOp)](#moorepop_front-circtmoorequeuepopfrontop)
  + [`moore.push_back` (::circt::moore::QueuePushBackOp)](#moorepush_back-circtmoorequeuepushbackop)
  + [`moore.push_front` (::circt::moore::QueuePushFrontOp)](#moorepush_front-circtmoorequeuepushfrontop)
  + [`moore.builtin.size` (::circt::moore::QueueSizeBIOp)](#moorebuiltinsize-circtmoorequeuesizebiop)
  + [`moore.builtin.random` (::circt::moore::RandomBIOp)](#moorebuiltinrandom-circtmoorerandombiop)
  + [`moore.read` (::circt::moore::ReadOp)](#mooreread-circtmoorereadop)
  + [`moore.real_to_int` (::circt::moore::RealToIntOp)](#moorereal_to_int-circtmoorerealtointop)
  + [`moore.builtin.realtobits` (::circt::moore::RealtobitsBIOp)](#moorebuiltinrealtobits-circtmoorerealtobitsbiop)
  + [`moore.reduce_and` (::circt::moore::ReduceAndOp)](#moorereduce_and-circtmoorereduceandop)
  + [`moore.reduce_or` (::circt::moore::ReduceOrOp)](#moorereduce_or-circtmoorereduceorop)
  + [`moore.reduce_xor` (::circt::moore::ReduceXorOp)](#moorereduce_xor-circtmoorereducexorop)
  + [`moore.replicate` (::circt::moore::ReplicateOp)](#moorereplicate-circtmoorereplicateop)
  + [`moore.return` (::circt::moore::ReturnOp)](#moorereturn-circtmoorereturnop)
  + [`moore.sbv_to_packed` (::circt::moore::SBVToPackedOp)](#mooresbv_to_packed-circtmooresbvtopackedop)
  + [`moore.sext` (::circt::moore::SExtOp)](#mooresext-circtmooresextop)
  + [`moore.sint_to_real` (::circt::moore::SIntToRealOp)](#mooresint_to_real-circtmooresinttorealop)
  + [`moore.module` (::circt::moore::SVModuleOp)](#mooremodule-circtmooresvmoduleop)
  + [`moore.builtin.severity` (::circt::moore::SeverityBIOp)](#moorebuiltinseverity-circtmooreseveritybiop)
  + [`moore.sge` (::circt::moore::SgeOp)](#mooresge-circtmooresgeop)
  + [`moore.sgt` (::circt::moore::SgtOp)](#mooresgt-circtmooresgtop)
  + [`moore.shl` (::circt::moore::ShlOp)](#mooreshl-circtmooreshlop)
  + [`moore.builtin.shortrealtobits` (::circt::moore::ShortrealtobitsBIOp)](#moorebuiltinshortrealtobits-circtmooreshortrealtobitsbiop)
  + [`moore.shr` (::circt::moore::ShrOp)](#mooreshr-circtmooreshrop)
  + [`moore.builtin.sin` (::circt::moore::SinBIOp)](#moorebuiltinsin-circtmooresinbiop)
  + [`moore.builtin.sinh` (::circt::moore::SinhBIOp)](#moorebuiltinsinh-circtmooresinhbiop)
  + [`moore.sle` (::circt::moore::SleOp)](#mooresle-circtmooresleop)
  + [`moore.slt` (::circt::moore::SltOp)](#mooreslt-circtmooresltop)
  + [`moore.builtin.sqrt` (::circt::moore::SqrtBIOp)](#moorebuiltinsqrt-circtmooresqrtbiop)
  + [`moore.builtin.stop` (::circt::moore::StopBIOp)](#moorebuiltinstop-circtmoorestopbiop)
  + [`moore.string_cmp` (::circt::moore::StringCmpOp)](#moorestring_cmp-circtmoorestringcmpop)
  + [`moore.string.concat` (::circt::moore::StringConcatOp)](#moorestringconcat-circtmoorestringconcatop)
  + [`moore.string.getc` (::circt::moore::StringGetCOp)](#moorestringgetc-circtmoorestringgetcop)
  + [`moore.string.len` (::circt::moore::StringLenOp)](#moorestringlen-circtmoorestringlenop)
  + [`moore.string_to_int` (::circt::moore::StringToIntOp)](#moorestring_to_int-circtmoorestringtointop)
  + [`moore.string.tolower` (::circt::moore::StringToLowerOp)](#moorestringtolower-circtmoorestringtolowerop)
  + [`moore.string.toupper` (::circt::moore::StringToUpperOp)](#moorestringtoupper-circtmoorestringtoupperop)
  + [`moore.struct_create` (::circt::moore::StructCreateOp)](#moorestruct_create-circtmoorestructcreateop)
  + [`moore.struct_extract` (::circt::moore::StructExtractOp)](#moorestruct_extract-circtmoorestructextractop)
  + [`moore.struct_extract_ref` (::circt::moore::StructExtractRefOp)](#moorestruct_extract_ref-circtmoorestructextractrefop)
  + [`moore.struct_inject` (::circt::moore::StructInjectOp)](#moorestruct_inject-circtmoorestructinjectop)
  + [`moore.sub` (::circt::moore::SubOp)](#mooresub-circtmooresubop)
  + [`moore.fsub` (::circt::moore::SubRealOp)](#moorefsub-circtmooresubrealop)
  + [`moore.builtin.tan` (::circt::moore::TanBIOp)](#moorebuiltintan-circtmooretanbiop)
  + [`moore.builtin.tanh` (::circt::moore::TanhBIOp)](#moorebuiltintanh-circtmooretanhbiop)
  + [`moore.builtin.time` (::circt::moore::TimeBIOp)](#moorebuiltintime-circtmooretimebiop)
  + [`moore.time_to_logic` (::circt::moore::TimeToLogicOp)](#mooretime_to_logic-circtmooretimetologicop)
  + [`moore.to_builtin_bool` (::circt::moore::ToBuiltinBoolOp)](#mooreto_builtin_bool-circtmooretobuiltinboolop)
  + [`moore.to_builtin_int` (::circt::moore::ToBuiltinIntOp)](#mooreto_builtin_int-circtmooretobuiltinintop)
  + [`moore.trunc` (::circt::moore::TruncOp)](#mooretrunc-circtmooretruncop)
  + [`moore.uarray_cmp` (::circt::moore::UArrayCmpOp)](#mooreuarray_cmp-circtmooreuarraycmpop)
  + [`moore.uint_to_real` (::circt::moore::UIntToRealOp)](#mooreuint_to_real-circtmooreuinttorealop)
  + [`moore.uge` (::circt::moore::UgeOp)](#mooreuge-circtmooreugeop)
  + [`moore.ugt` (::circt::moore::UgtOp)](#mooreugt-circtmooreugtop)
  + [`moore.ule` (::circt::moore::UleOp)](#mooreule-circtmooreuleop)
  + [`moore.ult` (::circt::moore::UltOp)](#mooreult-circtmooreultop)
  + [`moore.union_create` (::circt::moore::UnionCreateOp)](#mooreunion_create-circtmooreunioncreateop)
  + [`moore.union_extract` (::circt::moore::UnionExtractOp)](#mooreunion_extract-circtmooreunionextractop)
  + [`moore.union_extract_ref` (::circt::moore::UnionExtractRefOp)](#mooreunion_extract_ref-circtmooreunionextractrefop)
  + [`moore.unreachable` (::circt::moore::UnreachableOp)](#mooreunreachable-circtmooreunreachableop)
  + [`moore.builtin.urandom` (::circt::moore::UrandomBIOp)](#moorebuiltinurandom-circtmooreurandombiop)
  + [`moore.vtable_entry` (::circt::moore::VTableEntryOp)](#moorevtable_entry-circtmoorevtableentryop)
  + [`moore.vtable.load_method` (::circt::moore::VTableLoadMethodOp)](#moorevtableload_method-circtmoorevtableloadmethodop)
  + [`moore.vtable` (::circt::moore::VTableOp)](#moorevtable-circtmoorevtableop)
  + [`moore.variable` (::circt::moore::VariableOp)](#moorevariable-circtmoorevariableop)
  + [`moore.wait_delay` (::circt::moore::WaitDelayOp)](#moorewait_delay-circtmoorewaitdelayop)
  + [`moore.wait_event` (::circt::moore::WaitEventOp)](#moorewait_event-circtmoorewaiteventop)
  + [`moore.wildcard_eq` (::circt::moore::WildcardEqOp)](#moorewildcard_eq-circtmoorewildcardeqop)
  + [`moore.wildcard_ne` (::circt::moore::WildcardNeOp)](#moorewildcard_ne-circtmoorewildcardneop)
  + [`moore.xor` (::circt::moore::XorOp)](#moorexor-circtmoorexorop)
  + [`moore.yield` (::circt::moore::YieldOp)](#mooreyield-circtmooreyieldop)
  + [`moore.zext` (::circt::moore::ZExtOp)](#moorezext-circtmoorezextop)

Rationale [¶](#rationale)
-------------------------

The main goal of the `moore` dialect is to provide a set of operations and types for the `ImportVerilog` conversion to translate a fully parsed, type-checked, and elaborated Slang AST into MLIR operations.
See IEEE 1800-2017 for more details about SystemVerilog.
The dialect aims to faithfully capture the full SystemVerilog types and semantics, and provide a platform for transformation passes to resolve language quirks, analyze the design at a high level, and lower it to the core dialects.

In contrast, the `sv` dialect is geared towards emission of SystemVerilog text, and is focused on providing a good lowering target to allow for emission.
The `moore` and `sv` dialect may eventually converge into a single dialect.
As we are building out the Verilog frontend capabilities of CIRCT it is valuable to have a separate ingestion dialect, such that we do not have to make disruptive changes to the load-bearing `sv` dialect used in production.

Types [¶](#types)
-----------------

### Simple Bit Vector Type [¶](#simple-bit-vector-type)

The `moore.iN` and `moore.lN` types represent a two-valued or four-valued simple bit vector of width `N`.

| Verilog | Moore Dialect |
| --- | --- |
| `bit` | `!moore.i1` |
| `logic` | `!moore.l1` |
| `reg` | `!moore.l1` |
| `byte` | `!moore.i8` |
| `shortint` | `!moore.i16` |
| `int` | `!moore.i32` |
| `integer` | `!moore.l32` |
| `longint` | `!moore.i64` |
| `time` | `!moore.l64` |

### Event Type [¶](#event-type)

The SystemVerilog `event` type is represented as a `!moore.i1` value.
A variable `event e` is lowered to the equivalent of `bit b`.
Triggering an event through `-> e` is lowered to the equivalent of `b = ~b`.
Waiting on an event through `@(e)` is lowered to the equivalent of `@(b)`.

### Default Values [¶](#default-values)

Behavior of unconnected ports:

| Port Type | Unconnected Behavior |
| --- | --- |
| Input (Net) | High-impedance value (‘Z) |
| Input (Variable) | Default initial value |
| Output | No effect on Simulation |
| Inout (Net) | High-impedance value (‘Z) |
| Inout (Variable) | Default initial value |
| Ref | Cannot be left unconnected |
| Interconnect | Cannot be left unconnected |
| Interface | Cannot be left unconnected |

Uninitialized variables:

| Type | Default initial value |
| --- | --- |
| 4-state integral | ‘X |
| 2-state integral | ‘0 |
| `real`, `shortreal` | 0.0 |
| Enumeration | Base type default initial value |
| `string` | "" (empty string) |
| `event` | New event |
| `class` | `null` |
| `interface class` | `null` |
| `chandle` | `null` |
| `virtual interface` | `null` |

### ArrayType [¶](#arraytype)

*A packed array type*

Syntax:

```
!moore.array<
  unsigned,   # size
  PackedType   # elementType
>
```

A packed array with a fixed number of elements. This type represents packed
range dimensions (`[a:b]`) in SystemVerilog.

| Verilog | Moore Dialect |
| --- | --- |
| `T [3:0]` | `!moore.array<4 x T>` |
| `T [2:4]` | `!moore.array<3 x T>` |

#### Parameters: [¶](#parameters)

| Parameter | C++ type | Description |
| --- | --- | --- |
| size | `unsigned` |  |
| elementType | `PackedType` |  |

### AssocArrayType [¶](#assocarraytype)

*An associative array type*

Syntax:

```
!moore.assoc_array<
  UnpackedType,   # elementType
  UnpackedType   # indexType
>
```

An associative array. This type represents associative arrays (`[T]`) in
SystemVerilog.

| Verilog | Moore Dialect |
| --- | --- |
| `T foo [K]` | `!moore.assoc_array<T, K>` |

#### Parameters: [¶](#parameters-1)

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `UnpackedType` |  |
| indexType | `UnpackedType` |  |

### ChandleType [¶](#chandletype)

*The SystemVerilog `chandle` type*

Syntax: `!moore.chandle`

### ClassHandleType [¶](#classhandletype)

*Class object handle type, pointing to an object on the heap.*

Syntax:

```
!moore.class<
  ::mlir::SymbolRefAttr   # classSym
>
```

The `!moore.class<@C>` type represents a handle of a class instance of class `@C` living on the program’s heap.

#### Parameters: [¶](#parameters-2)

| Parameter | C++ type | Description |
| --- | --- | --- |
| classSym | `::mlir::SymbolRefAttr` | class symbol |

### EventType [¶](#eventtype)

*The SystemVerilog `event` type*

Syntax: `!moore.event`

### FormatStringType [¶](#formatstringtype)

*A format string type*

Syntax: `!moore.format_string`

An interpolated string produced by one of the string formatting operations.
It is used to parse format strings present in Verilog source text and
represent them as a sequence of IR operations that specify the formatting of
individual arguments.

### IntType [¶](#inttype)

*A simple bit vector type*

The `!moore.iN` and `!moore.lN` types represent a two-valued or four-valued
simple bit vector of width `N`. The predefined SystemVerilog integer types
map to this as follows:

| Verilog | Moore Dialect |
| --- | --- |
| `bit` | `!moore.i1` |
| `logic` | `!moore.l1` |
| `reg` | `!moore.l1` |
| `byte` | `!moore.i8` |
| `shortint` | `!moore.i16` |
| `int` | `!moore.i32` |
| `integer` | `!moore.l32` |
| `longint` | `!moore.i64` |
| `time` | `!moore.time` |

#### Parameters: [¶](#parameters-3)

| Parameter | C++ type | Description |
| --- | --- | --- |
| width | `unsigned` |  |
| domain | `Domain` |  |

### NullType [¶](#nulltype)

*A type that represents null literals for chandles, classhandles,
and virtual interfaces*

Syntax: `!moore.null`

### OpenArrayType [¶](#openarraytype)

*An open packed array type*

Syntax:

```
!moore.open_array<
  PackedType   # elementType
>
```

A packed array with an unspecified number of elements. This type represents
unsized/open packed arrays (`[]`) in SystemVerilog.

| Verilog | Moore Dialect |
| --- | --- |
| `T []` | `!moore.open_array<T>` |

#### Parameters: [¶](#parameters-4)

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `PackedType` |  |

### OpenUnpackedArrayType [¶](#openunpackedarraytype)

*An open unpacked array type*

Syntax:

```
!moore.open_uarray<
  UnpackedType   # elementType
>
```

An unpacked array with an unspecified number of elements. This type
represents unsized/open unpacked arrays (`[]`) in SystemVerilog.

| Verilog | Moore Dialect |
| --- | --- |
| `T foo []` | `!moore.open_uarray<T>` |

#### Parameters: [¶](#parameters-5)

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `UnpackedType` |  |

### QueueType [¶](#queuetype)

*A queue type*

Syntax:

```
!moore.queue<
  UnpackedType,   # elementType
  unsigned   # bound
>
```

A queue with an optional upper bound on the number of elements that it can
hold. This type represents queues (`[$]` and `[$:a]`) in SystemVerilog. A
`bound` of 0 indicates an unbounded queue.

| Verilog | Moore Dialect |
| --- | --- |
| `T foo [$]` | `!moore.queue<T>` |
| `T foo [$:42]` | `!moore.queue<T, 42>` |

#### Parameters: [¶](#parameters-6)

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `UnpackedType` |  |
| bound | `unsigned` |  |

### RealType [¶](#realtype)

*A SystemVerilog real type*

This type represents the SystemVerilog real type. We coalesce the
`f32`, `f64`, and `realtime` types in the SystemVerilog standard to
this common `!moore.real` type. The standard specifies these types to be of
at least 32, 64, and 64 bits, respectively.

| Verilog | Moore Dialect |
| --- | --- |
| `shortreal` | `!moore.f32` |
| `real` | `!moore.f64` |
| `realtime` | `!moore.time` |

#### Parameters: [¶](#parameters-7)

| Parameter | C++ type | Description |
| --- | --- | --- |
| width | `RealWidth` |  |

### RefType [¶](#reftype)

Syntax:

```
!moore.ref<
  UnpackedType   # nestedType
>
```

A wrapper is used to wrap any SystemVerilog type. It’s aimed to work for
‘moore.variable’, ‘moore.blocking\_assign’, and ‘moore.read’, which are
related to memory, like alloca/write/read.

#### Parameters: [¶](#parameters-8)

| Parameter | C++ type | Description |
| --- | --- | --- |
| nestedType | `UnpackedType` |  |

### StringType [¶](#stringtype)

*The SystemVerilog `string` type*

Syntax: `!moore.string`

### StructType [¶](#structtype)

*A packed struct type*

Syntax:

```
!moore.struct<
  ::llvm::ArrayRef<StructLikeMember>   # members
>
```

A packed struct. All members are guaranteed to be packed as well.

#### Parameters: [¶](#parameters-9)

| Parameter | C++ type | Description |
| --- | --- | --- |
| members | `::llvm::ArrayRef<StructLikeMember>` |  |

### TimeType [¶](#timetype)

*The SystemVerilog `time` and `realtime` type*

Syntax: `!moore.time`

The `!moore.time` type represents a time value. Internally, the value is
represented as femtoseconds in `i64`.

Time values are a problematic part of the SystemVerilog language, since they
are multiplied by a scaling factor given by the current scope’s `timeunit`
and then mapped to an 64 bit four-valued integer for `time`, or a real for
`realtime`. This means that a time value flowing from one module with one
timeunit to another module with a different timeunit will read as two
distinct delay values in the two modules.

To work around these issues, the Moore dialect maps all time values to
femtoseconds and represents them as an integer value. This allows us to have
time values exchanged between modules correctly, while locally casting from
and to `integer` and `real` with the correct local scaling factor.

### UnionType [¶](#uniontype)

*A packed union type*

Syntax:

```
!moore.union<
  ::llvm::ArrayRef<StructLikeMember>   # members
>
```

A packed union. All members are guaranteed to be packed as well.

#### Parameters: [¶](#parameters-10)

| Parameter | C++ type | Description |
| --- | --- | --- |
| members | `::llvm::ArrayRef<StructLikeMember>` |  |

### UnpackedArrayType [¶](#unpackedarraytype)

*An unpacked array type*

Syntax:

```
!moore.uarray<
  unsigned,   # size
  UnpackedType   # elementType
>
```

An unpacked array with a fixed number of elements. This type represents
unpacked range dimensions (`[a:b]`) and unpacked array dimensions (`[a]`) in
SystemVerilog.

| Verilog | Moore Dialect |
| --- | --- |
| `T foo [3:0]` | `!moore.uarray<4 x T>` |
| `T foo [2:4]` | `!moore.uarray<3 x T>` |
| `T foo [2]` | `!moore.uarray<2 x T>` |

#### Parameters: [¶](#parameters-11)

| Parameter | C++ type | Description |
| --- | --- | --- |
| size | `unsigned` |  |
| elementType | `UnpackedType` |  |

### UnpackedStructType [¶](#unpackedstructtype)

*An unpacked struct type*

Syntax:

```
!moore.ustruct<
  ::llvm::ArrayRef<StructLikeMember>   # members
>
```

An unpacked struct.

#### Parameters: [¶](#parameters-12)

| Parameter | C++ type | Description |
| --- | --- | --- |
| members | `::llvm::ArrayRef<StructLikeMember>` |  |

### UnpackedUnionType [¶](#unpackeduniontype)

*An unpacked union type*

Syntax:

```
!moore.uunion<
  ::llvm::ArrayRef<StructLikeMember>   # members
>
```

An unpacked union.

#### Parameters: [¶](#parameters-13)

| Parameter | C++ type | Description |
| --- | --- | --- |
| members | `::llvm::ArrayRef<StructLikeMember>` |  |

### VoidType [¶](#voidtype)

*The SystemVerilog `void` type*

Syntax: `!moore.void`

Operations [¶](#operations)
---------------------------

### `moore.ashr` (::circt::moore::AShrOp) [¶](#mooreashr-circtmooreashrop)

*Arithmetic right shift*

Syntax:

```
operation ::= `moore.ashr` $value `,` $amount attr-dict `:` type($value) `,` type($amount)
```

Shifts the `value` to the left or right by `amount` number of bits. The
result has the same type as the input value. The amount is always treated as
an unsigned number and has no effect on the signedness of the result. X or
Z bits in the input value are simply shifted left or right the same way 0 or
1 bits are. If the amount contains X or Z bits, all result bits are X.

`shl` shifts bits to the left, filling in 0 for the vacated least
significant bits. `shr` and `ashr` shift bits to the right; `shr` fills in
0 for the vacated most significant bits, and `ashr` copies the input’s sign
bit into the vacated most significant bits. Note that in contrast to the SV
spec, the `ashr` *always* fills in the sign bit regardless of the signedness
of the input.

`shl` corresponds to the `<<` and `<<<` operators. `shr` corresponds to the
`>>` operator, and the `>>>` operator applied to an unsigned value. `ashr`
corresponds to the `>>>` operator applied to a signed value.

See IEEE 1800-2017 § 11.4.10 “Shift operators”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands)

| Operand | Description |
| --- | --- |
| `value` | simple bit vector type |
| `amount` | simple bit vector type |

#### Results: [¶](#results)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.builtin.acos` (::circt::moore::AcosBIOp) [¶](#moorebuiltinacos-circtmooreacosbiop)

*Arc-cosine*

Syntax:

```
operation ::= `moore.builtin.acos` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-1)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-1)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.builtin.acosh` (::circt::moore::AcoshBIOp) [¶](#moorebuiltinacosh-circtmooreacoshbiop)

*Arc-hyperbolic cosine*

Syntax:

```
operation ::= `moore.builtin.acosh` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-2)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-2)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.add` (::circt::moore::AddOp) [¶](#mooreadd-circtmooreaddop)

*Addition*

Syntax:

```
operation ::= `moore.add` $lhs `,` $rhs attr-dict `:` type($result)
```

Add the operands. If any bit in the two operands is Z or X, all bits in the
result are X.

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-3)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-3)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.fadd` (::circt::moore::AddRealOp) [¶](#moorefadd-circtmooreaddrealop)

*Addition; add two real operands.*

Syntax:

```
operation ::= `moore.fadd` $lhs `,` $rhs attr-dict `:` type($result)
```

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators” and § 11.3.1 “Operators
with real operands”

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-4)

| Operand | Description |
| --- | --- |
| `lhs` | a SystemVerilog real type |
| `rhs` | a SystemVerilog real type |

#### Results: [¶](#results-4)

| Result | Description |
| --- | --- |
| `result` | a SystemVerilog real type |

### `moore.and` (::circt::moore::AndOp) [¶](#mooreand-circtmooreandop)

*Bitwise AND operation*

Syntax:

```
operation ::= `moore.and` $lhs `,` $rhs attr-dict `:` type($result)
```

Applies the boolean AND operation to each pair of corresponding bits in the
left- and right-hand side operand. Corresponds to the `&` operator.

See IEEE 1800-2017 § 11.4.8 “Bitwise operators”.

|  | 0 | 1 | X | Z |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | X | X |
| X | 0 | X | X | X |
| Z | 0 | X | X | X |

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-5)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-5)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.array_create` (::circt::moore::ArrayCreateOp) [¶](#moorearray_create-circtmoorearraycreateop)

*Create an array value from individual elements*

Syntax:

```
operation ::= `moore.array_create` $elements attr-dict `:` type($elements) `->` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-6)

| Operand | Description |
| --- | --- |
| `elements` | variadic of unpacked type |

#### Results: [¶](#results-6)

| Result | Description |
| --- | --- |
| `result` | packed or unpacked static array type |

### `moore.builtin.asin` (::circt::moore::AsinBIOp) [¶](#moorebuiltinasin-circtmooreasinbiop)

*Arc-sine*

Syntax:

```
operation ::= `moore.builtin.asin` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-7)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-7)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.builtin.asinh` (::circt::moore::AsinhBIOp) [¶](#moorebuiltinasinh-circtmooreasinhbiop)

*Arc-hyperbolic sine*

Syntax:

```
operation ::= `moore.builtin.asinh` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-8)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-8)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.assert` (::circt::moore::AssertOp) [¶](#mooreassert-circtmooreassertop)

*If cond is not true, an error should be thrown.*

Syntax:

```
operation ::= `moore.assert` $defer $cond (`label` $label^)? attr-dict `:` type($cond)
```

Traits: `HasParent<ProcedureOp>`

#### Attributes: [¶](#attributes)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `defer` | circt::moore::DeferAssertAttr | assertion deferring mode |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-9)

| Operand | Description |
| --- | --- |
| `cond` | single bit type |

### `moore.assigned_variable` (::circt::moore::AssignedVariableOp) [¶](#mooreassigned_variable-circtmooreassignedvariableop)

*A variable with a unique continuously assigned value*

Syntax:

```
operation ::= `moore.assigned_variable` `` custom<ImplicitSSAName>($name) $input attr-dict `:` type($input)
```

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`, `OpAsmOpInterface`

#### Attributes: [¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-10)

| Operand | Description |
| --- | --- |
| `input` | unpacked type |

#### Results: [¶](#results-9)

| Result | Description |
| --- | --- |
| `result` | unpacked type |

### `moore.assume` (::circt::moore::AssumeOp) [¶](#mooreassume-circtmooreassumeop)

*Verify the cond whether has the expected behavior.*

Syntax:

```
operation ::= `moore.assume` $defer $cond (`label` $label^)? attr-dict `:` type($cond)
```

Traits: `HasParent<ProcedureOp>`

#### Attributes: [¶](#attributes-2)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `defer` | circt::moore::DeferAssertAttr | assertion deferring mode |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-11)

| Operand | Description |
| --- | --- |
| `cond` | single bit type |

### `moore.builtin.atan` (::circt::moore::AtanBIOp) [¶](#moorebuiltinatan-circtmooreatanbiop)

*Arc-tangent*

Syntax:

```
operation ::= `moore.builtin.atan` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-12)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-10)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.builtin.atanh` (::circt::moore::AtanhBIOp) [¶](#moorebuiltinatanh-circtmooreatanhbiop)

*Arc-hyperbolic tangent*

Syntax:

```
operation ::= `moore.builtin.atanh` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-13)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-11)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.builtin.bitstoreal` (::circt::moore::BitstorealBIOp) [¶](#moorebuiltinbitstoreal-circtmoorebitstorealbiop)

*Convert a logic vector representation to its real-valued number*

Syntax:

```
operation ::= `moore.builtin.bitstoreal` $value attr-dict`:` type($value)
```

Corresponds to the `$bitstoreal` system function. Returns a real number
corresponding to the real number representation of the logic vector.
Note that this does not correspond to a cast to another type, but rather a no-op.

See IEEE 1800-2023 § 20.5 “Conversion functions”.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-14)

| Operand | Description |
| --- | --- |
| `value` | 64-bit two-valued integer type |

#### Results: [¶](#results-12)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.builtin.bitstoshortreal` (::circt::moore::BitstoshortrealBIOp) [¶](#moorebuiltinbitstoshortreal-circtmoorebitstoshortrealbiop)

*Convert a logic vector representation to its real-valued number*

Syntax:

```
operation ::= `moore.builtin.bitstoshortreal` $value attr-dict `:` type($value)
```

Corresponds to the `$bitstoshortreal` system function. Returns a real number
corresponding to the real number representation of the logic vector.
Note that this does not correspond to a cast to another type, but rather a no-op.

See IEEE 1800-2023 § 20.5 “Conversion functions”.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-15)

| Operand | Description |
| --- | --- |
| `value` | 32-bit two-valued integer type |

#### Results: [¶](#results-13)

| Result | Description |
| --- | --- |
| `result` | 32-bit real type, aka. shortreal |

### `moore.blocking_assign` (::circt::moore::BlockingAssignOp) [¶](#mooreblocking_assign-circtmooreblockingassignop)

*Blocking procedural assignment*

Syntax:

```
operation ::= `moore.blocking_assign` $dst `,` $src attr-dict `:` type($src)
```

A blocking procedural assignment in a sequential block, such as `x = y`. The
effects of the assignment are visible to any subsequent operations in the
block.

See IEEE 1800-2017 § 10.4.1 “Blocking procedural assignments”.

Interfaces: `PromotableMemOpInterface`

#### Operands: [¶](#operands-16)

| Operand | Description |
| --- | --- |
| `dst` |  |
| `src` | unpacked type |

### `moore.bool_cast` (::circt::moore::BoolCastOp) [¶](#moorebool_cast-circtmooreboolcastop)

*Cast a value to a single bit boolean*

Syntax:

```
operation ::= `moore.bool_cast` $input attr-dict `:` type($input) `->` type($result)
```

Convert a nonzero or true value into 1, a zero or false value into 0, and
any value containing Z or X bits into a X. This conversion is useful in
combination with the logical and, or, implication, equivalence, and negation
operators.

See IEEE 1800-2017 § 11.4.7 “Logical operators”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-17)

| Operand | Description |
| --- | --- |
| `input` | unpacked type |

#### Results: [¶](#results-14)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.case_eq` (::circt::moore::CaseEqOp) [¶](#moorecase_eq-circtmoorecaseeqop)

*Case equality*

Syntax:

```
operation ::= `moore.case_eq` $lhs `,` $rhs attr-dict `:` type($lhs)
```

Compares the bits in the left- and right-hand side operand and returns a
single bit 0 or 1 result. If all corresponding bits in the left- and
right-hand side are equal (both 0, 1, X, or Z), the two operands are
considered equal (`case_eq` returns 1, `case_ne` returns 0). If any bits are
not equal, the two operands are considered not equal (`case_eq` returns 0,
`case_ne` returns 1). `case_eq` corresponds to the `===` operator and
`case_ne` to the `!==` operator.

`casez_eq` treats Z bits in either operand as wildcards and skips them
during the comparison. `casexz_eq` treats X and Z bits as wildcards. These
are different from the `wildcard_eq` operation, which only considers X/Z in
the right-hand operand as wildcards.

Case statements use this operation to perform case comparisons:

* `case` statements use `case_eq`
* `casez` statements use `casez_eq`
* `casex` statements use `casexz_eq`

See IEEE 1800-2017 § 11.4.5 “Equality operators”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-18)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-15)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.case_ne` (::circt::moore::CaseNeOp) [¶](#moorecase_ne-circtmoorecaseneop)

*Case inequality*

Syntax:

```
operation ::= `moore.case_ne` $lhs `,` $rhs attr-dict `:` type($lhs)
```

Compares the bits in the left- and right-hand side operand and returns a
single bit 0 or 1 result. If all corresponding bits in the left- and
right-hand side are equal (both 0, 1, X, or Z), the two operands are
considered equal (`case_eq` returns 1, `case_ne` returns 0). If any bits are
not equal, the two operands are considered not equal (`case_eq` returns 0,
`case_ne` returns 1). `case_eq` corresponds to the `===` operator and
`case_ne` to the `!==` operator.

`casez_eq` treats Z bits in either operand as wildcards and skips them
during the comparison. `casexz_eq` treats X and Z bits as wildcards. These
are different from the `wildcard_eq` operation, which only considers X/Z in
the right-hand operand as wildcards.

Case statements use this operation to perform case comparisons:

* `case` statements use `case_eq`
* `casez` statements use `casez_eq`
* `casex` statements use `casexz_eq`

See IEEE 1800-2017 § 11.4.5 “Equality operators”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-19)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-16)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.casexz_eq` (::circt::moore::CaseXZEqOp) [¶](#moorecasexz_eq-circtmoorecasexzeqop)

*Case equality with X and Z as wildcard*

Syntax:

```
operation ::= `moore.casexz_eq` $lhs `,` $rhs attr-dict `:` type($lhs)
```

Compares the bits in the left- and right-hand side operand and returns a
single bit 0 or 1 result. If all corresponding bits in the left- and
right-hand side are equal (both 0, 1, X, or Z), the two operands are
considered equal (`case_eq` returns 1, `case_ne` returns 0). If any bits are
not equal, the two operands are considered not equal (`case_eq` returns 0,
`case_ne` returns 1). `case_eq` corresponds to the `===` operator and
`case_ne` to the `!==` operator.

`casez_eq` treats Z bits in either operand as wildcards and skips them
during the comparison. `casexz_eq` treats X and Z bits as wildcards. These
are different from the `wildcard_eq` operation, which only considers X/Z in
the right-hand operand as wildcards.

Case statements use this operation to perform case comparisons:

* `case` statements use `case_eq`
* `casez` statements use `casez_eq`
* `casex` statements use `casexz_eq`

See IEEE 1800-2017 § 11.4.5 “Equality operators”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-20)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-17)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.casez_eq` (::circt::moore::CaseZEqOp) [¶](#moorecasez_eq-circtmoorecasezeqop)

*Case equality with Z as wildcard*

Syntax:

```
operation ::= `moore.casez_eq` $lhs `,` $rhs attr-dict `:` type($lhs)
```

Compares the bits in the left- and right-hand side operand and returns a
single bit 0 or 1 result. If all corresponding bits in the left- and
right-hand side are equal (both 0, 1, X, or Z), the two operands are
considered equal (`case_eq` returns 1, `case_ne` returns 0). If any bits are
not equal, the two operands are considered not equal (`case_eq` returns 0,
`case_ne` returns 1). `case_eq` corresponds to the `===` operator and
`case_ne` to the `!==` operator.

`casez_eq` treats Z bits in either operand as wildcards and skips them
during the comparison. `casexz_eq` treats X and Z bits as wildcards. These
are different from the `wildcard_eq` operation, which only considers X/Z in
the right-hand operand as wildcards.

Case statements use this operation to perform case comparisons:

* `case` statements use `case_eq`
* `casez` statements use `casez_eq`
* `casex` statements use `casexz_eq`

See IEEE 1800-2017 § 11.4.5 “Equality operators”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-21)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-18)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.builtin.ceil` (::circt::moore::CeilBIOp) [¶](#moorebuiltinceil-circtmooreceilbiop)

*Ceiling*

Syntax:

```
operation ::= `moore.builtin.ceil` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-22)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-19)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.class.classdecl` (::circt::moore::ClassDeclOp) [¶](#mooreclassclassdecl-circtmooreclassdeclop)

*Class declaration*

Syntax:

```
operation ::= `moore.class.classdecl` $sym_name
              (`extends` $base^)?
              (`implements` $implementedInterfaces^)?
              attr-dict-with-keyword $body
```

Traits: `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `Symbol`

#### Attributes: [¶](#attributes-3)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `base` | ::mlir::SymbolRefAttr | symbol reference attribute |
| `implementedInterfaces` | ::mlir::ArrayAttr | symbol ref array attribute |

### `moore.class.methoddecl` (::circt::moore::ClassMethodDeclOp) [¶](#mooreclassmethoddecl-circtmooreclassmethoddeclop)

*Declare a class method*

Syntax:

```
operation ::= `moore.class.methoddecl` $sym_name (`->` $impl^)? `:` $function_type attr-dict
```

Traits: `HasParent<ClassDeclOp>`

Interfaces: `Symbol`

#### Attributes: [¶](#attributes-4)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `function_type` | ::mlir::TypeAttr | type attribute of function type |
| `impl` | ::mlir::SymbolRefAttr | symbol reference attribute |

### `moore.class.new` (::circt::moore::ClassNewOp) [¶](#mooreclassnew-circtmooreclassnewop)

*Allocate a new class instance*

Syntax:

```
operation ::= `moore.class.new` `:` type($result) attr-dict
```

Allocates a new instance of class @C. This op does not call the constructor.
The result is a non-null `!moore.class.object<@C>` handle.

Interfaces: `MemoryEffectOpInterface`

#### Results: [¶](#results-20)

| Result | Description |
| --- | --- |
| `result` | Class object handle type, pointing to an object on the heap. |

### `moore.class.propertydecl` (::circt::moore::ClassPropertyDeclOp) [¶](#mooreclasspropertydecl-circtmooreclasspropertydeclop)

*Declare a class property*

Syntax:

```
operation ::= `moore.class.propertydecl` $sym_name `:` $type  attr-dict
```

Declares a property within a class declaration.

Traits: `HasParent<ClassDeclOp>`

Interfaces: `Symbol`

#### Attributes: [¶](#attributes-5)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `type` | ::mlir::TypeAttr | any type attribute |

### `moore.class.property_ref` (::circt::moore::ClassPropertyRefOp) [¶](#mooreclassproperty_ref-circtmooreclasspropertyrefop)

*Get a !moore.ref to a class property*

Syntax:

```
operation ::= `moore.class.property_ref` $instance`[`$property`]` `:` type($instance) `->` type($propertyRef) attr-dict
```

Construct a reference to a class instance’s property field. Translates
to a fixed offset from the the class handle into its data struct.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-6)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `property` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Operands: [¶](#operands-23)

| Operand | Description |
| --- | --- |
| `instance` | Class object handle type, pointing to an object on the heap. |

#### Results: [¶](#results-21)

| Result | Description |
| --- | --- |
| `propertyRef` |  |

### `moore.class.upcast` (::circt::moore::ClassUpcastOp) [¶](#mooreclassupcast-circtmooreclassupcastop)

*Upcast a derived handle to a base handle (zero-cost)*

Syntax:

```
operation ::= `moore.class.upcast` $instance `:` type($instance) `to` type($result) attr-dict
```

Casts an instance of a class to one of its base classes.
The created instance points to the same underlying data struct as the original
instance, and modifying it will modify the spawning instance’s data as well.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-24)

| Operand | Description |
| --- | --- |
| `instance` | Class object handle type, pointing to an object on the heap. |

#### Results: [¶](#results-22)

| Result | Description |
| --- | --- |
| `result` | Class object handle type, pointing to an object on the heap. |

### `moore.builtin.clog2` (::circt::moore::Clog2BIOp) [¶](#moorebuiltinclog2-circtmooreclog2biop)

*Compute ceil(log2(x)) of x*

Syntax:

```
operation ::= `moore.builtin.clog2` $value attr-dict `:` type($value)
```

Computes the ceiling of the base-2 logarithm of the argument. The argument
is interpreted as unsigned. The result is 0 if the argument is 0. The result
corresponds to the minimum address width necessary to address a given number
of elements, or the number of bits necessary to represent a given number of
states.

If any of the bits in the argument are X or Z, the result is X.

See IEEE 1800-2017 § 20.8.1 “Integer math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-25)

| Operand | Description |
| --- | --- |
| `value` | a simple bit vector type |

#### Results: [¶](#results-23)

| Result | Description |
| --- | --- |
| `result` | a simple bit vector type |

### `moore.concat` (::circt::moore::ConcatOp) [¶](#mooreconcat-circtmooreconcatop)

*A concatenation of expressions*

Syntax:

```
operation ::= `moore.concat` $values attr-dict `:` `(` type($values) `)` `->` type($result)
```

This operation represents the SystemVerilog concatenation expression
`{x, y, z}`. See IEEE 1800-2017 §11.4.12 “Concatenation operators”.

All operands must be simple bit vector types.

The concatenation result is a simple bit vector type. The result is unsigned
regardless of the sign of the operands (see concatenation-specific rules in
IEEE 1800-2017 §11.8.1 “Rules for expression types”). The size of the result
is the sum of the sizes of all operands. If any of the operands is
four-valued, the result is four-valued; otherwise it is two-valued.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-26)

| Operand | Description |
| --- | --- |
| `values` | variadic of a simple bit vector type |

#### Results: [¶](#results-24)

| Result | Description |
| --- | --- |
| `result` | a simple bit vector type |

### `moore.concat_ref` (::circt::moore::ConcatRefOp) [¶](#mooreconcat_ref-circtmooreconcatrefop)

*The copy of concat that explicitly works on the ref type.*

Syntax:

```
operation ::= `moore.concat_ref` $values attr-dict `:` `(` type($values) `)` `->` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-27)

| Operand | Description |
| --- | --- |
| `values` | variadic of |

#### Results: [¶](#results-25)

| Result | Description |
| --- | --- |
| `result` |  |

### `moore.conditional` (::circt::moore::ConditionalOp) [¶](#mooreconditional-circtmooreconditionalop)

*Conditional operation*

Syntax:

```
operation ::= `moore.conditional` $condition attr-dict `:` type($condition) `->` type($result)
              $trueRegion $falseRegion
```

If the condition is true, this op evaluates the first region and returns its
result without evaluating the second region. If the the condition is false,
this op evaluates the second region and returns its result without
evaluating the first region.

If the condition is unknown (X or Z), *both* regions are evaluated. If both
results are equal as per `case_eq`, one of the results is returned. If the
results are not equal, this op returns a value based on the data types of
the results.

In case the results of the first and second region are of an integral type,
they are merged by applying the following bit-wise truth table:

| ?: | 0 | 1 | X | Z |
| --- | --- | --- | --- | --- |
| 0 | 0 | X | X | X |
| 1 | X | 1 | X | X |
| X | X | X | X | X |
| Z | X | X | X | X |

Non-integral data types define other rules which are not yet implemented.
See IEEE 1800-2017 § 11.4.11 “Conditional operator”.

Traits: `NoRegionArguments`, `RecursiveMemoryEffects`

#### Operands: [¶](#operands-28)

| Operand | Description |
| --- | --- |
| `condition` | single bit type |

#### Results: [¶](#results-26)

| Result | Description |
| --- | --- |
| `result` | unpacked type |

### `moore.constant` (::circt::moore::ConstantOp) [¶](#mooreconstant-circtmooreconstantop)

*A constant integer value*

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-7)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::circt::moore::FVIntegerAttr | An attribute containing a four-valued integer |

#### Results: [¶](#results-27)

| Result | Description |
| --- | --- |
| `result` | a simple bit vector type |

### `moore.constant_real` (::circt::moore::ConstantRealOp) [¶](#mooreconstant_real-circtmooreconstantrealop)

*A constant real value*

Syntax:

```
operation ::= `moore.constant_real` $value attr-dict
```

Produces a constant value of a real type.

See IEEE 1800-2017 § 5.7.2 “Real literal constants”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-8)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | mlir::FloatAttr | arbitrary float attribute |

#### Results: [¶](#results-28)

| Result | Description |
| --- | --- |
| `result` | a SystemVerilog real type |

### `moore.constant_string` (::circt::moore::ConstantStringOp) [¶](#mooreconstant_string-circtmooreconstantstringop)

*A constant integer value defined by a string of bytes*

Syntax:

```
operation ::= `moore.constant_string` $value attr-dict `:` type($result)
```

Defines a constant integer value based on the bytes of a string. The
resulting integer contains 8 bits for each byte in the string. The first,
left-most character is placed in the most significnat bits; the last,
right-most character is placed in the least significant bits.

Constant strings are represented as integers since they have a known width
and bit-pattern, which is often used in SV to assign strings to parameters,
or store strings in bit vectors. In contrast, the `string` type is a dynamic
string with runtime-known length. To obtain a constant `string` value, the
resulting integer should be converted to a `string`.

See IEEE 1800-2017 § 5.9 “String literals”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-9)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-29)

| Result | Description |
| --- | --- |
| `result` | a simple bit vector type |

### `moore.constant_time` (::circt::moore::ConstantTimeOp) [¶](#mooreconstant_time-circtmooreconstanttimeop)

*A constant time value in femtoseconds*

Syntax:

```
operation ::= `moore.constant_time` $value `fs` attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-10)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::IntegerAttr | 64-bit unsigned integer attribute |

#### Results: [¶](#results-30)

| Result | Description |
| --- | --- |
| `result` | the SystemVerilog `time` and `realtime` type |

### `moore.assign` (::circt::moore::ContinuousAssignOp) [¶](#mooreassign-circtmoorecontinuousassignop)

*Continuous assignment within a module*

Syntax:

```
operation ::= `moore.assign` $dst `,` $src attr-dict `:` type($src)
```

A continuous assignment in module scope, such as `assign x = y;`, which
continuously drives the value on the right-hand side onto the left-hand
side.

See IEEE 1800-2017 § 10.3 “Continuous assignments”.

Traits: `HasParent<SVModuleOp>`

#### Operands: [¶](#operands-29)

| Operand | Description |
| --- | --- |
| `dst` |  |
| `src` | unpacked type |

### `moore.conversion` (::circt::moore::ConversionOp) [¶](#mooreconversion-circtmooreconversionop)

*A type conversion*

Syntax:

```
operation ::= `moore.conversion` $input attr-dict `:` type($input) `->` type($result)
```

An explicit or implicit type conversion. These are either generated
automatically in order to make assignments compatible:

```
int a;
shortint b;
a = b;  // generates an implicit cast from shortint to int
```

Or explicitly by the user through a type, sign, or const cast expression:

```
byte'(a)
unsigned'(a)
signed'(a)
42'(a)
```

See IEEE 1800-2017 § 6.24 “Casting”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-30)

| Operand | Description |
| --- | --- |
| `input` | any type |

#### Results: [¶](#results-31)

| Result | Description |
| --- | --- |
| `result` | any type |

### `moore.convert_real` (::circt::moore::ConvertRealOp) [¶](#mooreconvert_real-circtmooreconvertrealop)

*Convert a real to a different bitwidth*

Syntax:

```
operation ::= `moore.convert_real` $input attr-dict `:` type($input) `->` type($result)
```

The value is rounded to the nearest value representable by the target type.
See IEEE 1800-2023 Section 6.24.1.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-31)

| Operand | Description |
| --- | --- |
| `input` | a SystemVerilog real type |

#### Results: [¶](#results-32)

| Result | Description |
| --- | --- |
| `result` | a SystemVerilog real type |

### `moore.builtin.cos` (::circt::moore::CosBIOp) [¶](#moorebuiltincos-circtmoorecosbiop)

*Cosine*

Syntax:

```
operation ::= `moore.builtin.cos` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-32)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-33)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.builtin.cosh` (::circt::moore::CoshBIOp) [¶](#moorebuiltincosh-circtmoorecoshbiop)

*Hyperbolic cosine*

Syntax:

```
operation ::= `moore.builtin.cosh` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-33)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-34)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.cover` (::circt::moore::CoverOp) [¶](#moorecover-circtmoorecoverop)

*Monitor the coverage information.*

Syntax:

```
operation ::= `moore.cover` $defer $cond (`label` $label^)? attr-dict `:` type($cond)
```

Traits: `HasParent<ProcedureOp>`

#### Attributes: [¶](#attributes-11)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `defer` | circt::moore::DeferAssertAttr | assertion deferring mode |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-34)

| Operand | Description |
| --- | --- |
| `cond` | single bit type |

### `moore.delayed_assign` (::circt::moore::DelayedContinuousAssignOp) [¶](#mooredelayed_assign-circtmooredelayedcontinuousassignop)

*Delayed continuous assignment within a module*

Syntax:

```
operation ::= `moore.delayed_assign` $dst `,` $src `,` $delay attr-dict `:` type($src)
```

A continuous assignment with a delay.

See the `moore.assign` op.
See IEEE 1800-2017 § 10.3 “Continuous assignments”.

Traits: `HasParent<SVModuleOp>`

#### Operands: [¶](#operands-35)

| Operand | Description |
| --- | --- |
| `dst` |  |
| `src` | unpacked type |
| `delay` | the SystemVerilog `time` and `realtime` type |

### `moore.delayed_nonblocking_assign` (::circt::moore::DelayedNonBlockingAssignOp) [¶](#mooredelayed_nonblocking_assign-circtmooredelayednonblockingassignop)

*Delayed nonblocking procedural assignment*

Syntax:

```
operation ::= `moore.delayed_nonblocking_assign` $dst `,` $src `,` $delay attr-dict `:` type($src)
```

A nonblocking procedural assignment with an intra-assignment delay control.

See the `moore.nonblocking_assign` op.
See IEEE 1800-2017 § 9.4.5 “Intra-assignment timing controls”.

#### Operands: [¶](#operands-36)

| Operand | Description |
| --- | --- |
| `dst` |  |
| `src` | unpacked type |
| `delay` | the SystemVerilog `time` and `realtime` type |

### `moore.detect_event` (::circt::moore::DetectEventOp) [¶](#mooredetect_event-circtmooredetecteventop)

*Check if an event occured within a `wait_event` op*

Syntax:

```
operation ::= `moore.detect_event` $edge $input (`if` $condition^)? attr-dict `:` type($input)
```

The `moore.detect_event` op is used inside the body of a `moore.wait_event`
to check if an interesting value change has occurred on its operand. The
`moore.detect_event` op implicitly stores the previous value of its operand
and compares it against the current value to detect an interesting edge:

* `posedge` checks for a low-to-high transition
* `negedge` checks for a high-to-low transition
* `edge` checks for either a `posedge` or a `negedge`
* `any` checks for any value change (including e.g. X to Z)

The edges are detected as follows:

* `0` to `1 X Z`: `posedge`
* `1` to `0 X Z`: `negedge`
* `X Z` to `1`: `posedge`
* `X Z` to `0`: `negedge`

| From | To 0 | To 1 | To X | To Z |
| --- | --- | --- | --- | --- |
| 0 | - | posedge | posedge | posedge |
| 1 | negedge | - | negedge | negedge |
| X | negedge | posedge | - | - |
| Z | negedge | posedge | - | - |

See IEEE 1800-2017 § 9.4.2 “Event control”.

Traits: `HasParent<WaitEventOp>`

#### Attributes: [¶](#attributes-12)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `edge` | circt::moore::EdgeAttr | Edge kind |

#### Operands: [¶](#operands-37)

| Operand | Description |
| --- | --- |
| `input` | unpacked type |
| `condition` | `bit` type |

### `moore.builtin.display` (::circt::moore::DisplayBIOp) [¶](#moorebuiltindisplay-circtmooredisplaybiop)

*Print a text message*

Syntax:

```
operation ::= `moore.builtin.display` $message attr-dict
```

Prints the given format string to the standard text output of the simulator.
In most cases this should be stdout. This corresponds to the `$display` and
`$write` system tasks. Message formatting is handled by `moore.fmt.*` ops.

See IEEE 1800-2017 § 21.2 “Display system tasks”.

#### Operands: [¶](#operands-38)

| Operand | Description |
| --- | --- |
| `message` | a format string type |

### `moore.fdiv` (::circt::moore::DivRealOp) [¶](#moorefdiv-circtmooredivrealop)

*Division; Divide the LHS real operand by the LHS
real operand.*

Syntax:

```
operation ::= `moore.fdiv` $lhs `,` $rhs attr-dict `:` type($result)
```

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators” and § 11.3.1 “Operators
with real operands”

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-39)

| Operand | Description |
| --- | --- |
| `lhs` | a SystemVerilog real type |
| `rhs` | a SystemVerilog real type |

#### Results: [¶](#results-35)

| Result | Description |
| --- | --- |
| `result` | a SystemVerilog real type |

### `moore.divs` (::circt::moore::DivSOp) [¶](#mooredivs-circtmooredivsop)

*Division*

Syntax:

```
operation ::= `moore.divs` $lhs `,` $rhs attr-dict `:` type($result)
```

Divide the left-hand side by the right-hand side operand. Any fractional
part is truncated toward zero. If the right-hand side is zero, all bits of
the result are X. If any bit in the two operands is Z or X, all bits in the
result are X.

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-40)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-36)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.divu` (::circt::moore::DivUOp) [¶](#mooredivu-circtmooredivuop)

*Division*

Syntax:

```
operation ::= `moore.divu` $lhs `,` $rhs attr-dict `:` type($result)
```

Divide the left-hand side by the right-hand side operand. Any fractional
part is truncated toward zero. If the right-hand side is zero, all bits of
the result are X. If any bit in the two operands is Z or X, all bits in the
result are X.

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-41)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-37)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.dyn_extract` (::circt::moore::DynExtractOp) [¶](#mooredyn_extract-circtmooredynextractop)

Syntax:

```
operation ::= `moore.dyn_extract` $input `from` $lowBit attr-dict `:`
              type($input) `,` type($lowBit) `->` type($result)
```

It’s similar with extract, but it’s used to select from a value
with a dynamic low bit.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-42)

| Operand | Description |
| --- | --- |
| `input` | unpacked type |
| `lowBit` | unpacked type |

#### Results: [¶](#results-38)

| Result | Description |
| --- | --- |
| `result` | unpacked type |

### `moore.dyn_extract_ref` (::circt::moore::DynExtractRefOp) [¶](#mooredyn_extract_ref-circtmooredynextractrefop)

Syntax:

```
operation ::= `moore.dyn_extract_ref` $input `from` $lowBit attr-dict `:`
              type($input) `,` type($lowBit) `->` type($result)
```

The copy of dyn\_extract that explicitly works on the ref type.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-43)

| Operand | Description |
| --- | --- |
| `input` |  |
| `lowBit` | unpacked type |

#### Results: [¶](#results-39)

| Result | Description |
| --- | --- |
| `result` |  |

### `moore.dyn_queue_extract` (::circt::moore::DynQueueExtractOp) [¶](#mooredyn_queue_extract-circtmooredynqueueextractop)

Syntax:

```
operation ::= `moore.dyn_queue_extract` $input `from` $lowerIdx attr-dict `:`
              type($input) `,` type($lowerIdx) `->` type($result)
```

It’s similar to extract, but it’s used to select from a value
with a dynamic lower index.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-44)

| Operand | Description |
| --- | --- |
| `input` | a queue type |
| `lowerIdx` | unpacked type |

#### Results: [¶](#results-40)

| Result | Description |
| --- | --- |
| `result` | unpacked type |

### `moore.dyn_queue_extract_ref` (::circt::moore::DynQueueExtractRefOp) [¶](#mooredyn_queue_extract_ref-circtmooredynqueueextractrefop)

Syntax:

```
operation ::= `moore.dyn_queue_extract_ref` $input `from` $lowerIdx attr-dict `:`
              type($input) `,` type($lowerIdx) `->` type($result)
```

The copy of dyn\_extract that explicitly works on the ref type.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-45)

| Operand | Description |
| --- | --- |
| `input` |  |
| `lowerIdx` | unpacked type |

#### Results: [¶](#results-41)

| Result | Description |
| --- | --- |
| `result` |  |

### `moore.eq` (::circt::moore::EqOp) [¶](#mooreeq-circtmooreeqop)

*Logical equality*

Syntax:

```
operation ::= `moore.eq` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the bits in the left- and right-hand side operand and returns a
single bit 0, 1, or X result. If all corresponding bits in the left- and
right-hand side are equal, and all are 0 or 1 (not X or Z), the two operands
are considered equal (`eq` returns 1, `ne` returns 0). If any bits are not
equal, but all are 0 or 1, the two operands are considered not equal (`eq`
returns 0, `ne` returns 1). If any bit in the two operands is Z or X,
returns X. `eq` corresponds to the `==` operator and `ne` to the `!=`
operator.

See IEEE 1800-2017 § 11.4.5 “Equality operators”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-46)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-42)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.feq` (::circt::moore::EqRealOp) [¶](#moorefeq-circtmooreeqrealop)

*Logical equality*

Syntax:

```
operation ::= `moore.feq` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the bits in the left- and right-hand side operand and returns a
single bit 0 or 1. If all corresponding bits in the left- and
right-hand side are equal, and all are 0 or 1, the two operands
are considered equal (`eq` returns 1, `ne` returns 0). If any bits are not
equal, but all are 0 or 1, the two operands are considered not equal (`eq`
returns 0, `ne` returns 1). `eq` corresponds to the `==` operator and `ne`
to the `!=` operator.

See IEEE 1800-2017 § 11.4.5 “Equality operators” and § 11.3.1 “Operators
with real operands”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-47)

| Operand | Description |
| --- | --- |
| `lhs` | a SystemVerilog real type |
| `rhs` | a SystemVerilog real type |

#### Results: [¶](#results-43)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.builtin.exp` (::circt::moore::ExpBIOp) [¶](#moorebuiltinexp-circtmooreexpbiop)

*Exponential*

Syntax:

```
operation ::= `moore.builtin.exp` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-48)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-44)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.extract` (::circt::moore::ExtractOp) [¶](#mooreextract-circtmooreextractop)

*Extract a range or single bits from a value*

Syntax:

```
operation ::= `moore.extract` $input `from` $lowBit attr-dict `:` type($input) `->` type($result)
```

It’s used to select from a value with a constant low bit.
This operation includes the vector bit/part-select, array, and memory
addressing.If the address is invalid–out of bounds or has x or z bit–
then it will produce x for 4-state or 0 for 2-state.
Bit-select results are unsigned, regardless of the operands.
Part-select results are unsigned, regardless of the operands even if
the part-select specifies the entire vector.
See IEEE 1800-2017 § 11.8.1 “Rules for expression types”

Example:

```
logic v [7:0];
v[1];                      // the bit-select addressing
v[3:0];                    // the part-select addressing
v[3-:4];  v[0+:4];         // They are equivalent to v[3:0]
```

See IEEE 1800-2017 § 11.5.1 “Vector bit-select and part-select addressing”.

Example:

```
// an array of 256-by-256 8-bit elements
logic [7:0] twod_array [0:255][0:255];
logic [7:0] mem_name [0:1023];      // a memory of 1024 8-bit words
```

See IEEE 1800-2017 § 11.5.2 “Array and memory addressing”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-13)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `lowBit` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-49)

| Operand | Description |
| --- | --- |
| `input` | unpacked type |

#### Results: [¶](#results-45)

| Result | Description |
| --- | --- |
| `result` | unpacked type |

### `moore.extract_ref` (::circt::moore::ExtractRefOp) [¶](#mooreextract_ref-circtmooreextractrefop)

Syntax:

```
operation ::= `moore.extract_ref` $input `from` $lowBit attr-dict `:` type($input) `->` type($result)
```

The copy of extract that explicitly works on the ref type.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-14)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `lowBit` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-50)

| Operand | Description |
| --- | --- |
| `input` |  |

#### Results: [¶](#results-46)

| Result | Description |
| --- | --- |
| `result` |  |

### `moore.fge` (::circt::moore::FgeOp) [¶](#moorefge-circtmoorefgeop)

*Real-valued greater than or equal comparison*

Syntax:

```
operation ::= `moore.fge` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the left- and right-hand side operand and returns a single bit 0,
or 1 result. If all bits are 0 or 1, `flt`, `fle`, `fgt`, and
`fge` return whether the left-hand side is less than, less than or equal
to, greater than, or greater than or equal to the right-hand side,
respectively. `flt` corresponds to the `<` operator, `fle` to `<=`,
`fgt` to `>`, and `fge` to `>=`.

See IEEE 1800-2017 § 11.4.4 “Relational operators” and § 11.3.1 “Operators
with real operands”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-51)

| Operand | Description |
| --- | --- |
| `lhs` | a SystemVerilog real type |
| `rhs` | a SystemVerilog real type |

#### Results: [¶](#results-47)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.fgt` (::circt::moore::FgtOp) [¶](#moorefgt-circtmoorefgtop)

*Real-valued greater than comparison*

Syntax:

```
operation ::= `moore.fgt` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the left- and right-hand side operand and returns a single bit 0,
or 1 result. If all bits are 0 or 1, `flt`, `fle`, `fgt`, and
`fge` return whether the left-hand side is less than, less than or equal
to, greater than, or greater than or equal to the right-hand side,
respectively. `flt` corresponds to the `<` operator, `fle` to `<=`,
`fgt` to `>`, and `fge` to `>=`.

See IEEE 1800-2017 § 11.4.4 “Relational operators” and § 11.3.1 “Operators
with real operands”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-52)

| Operand | Description |
| --- | --- |
| `lhs` | a SystemVerilog real type |
| `rhs` | a SystemVerilog real type |

#### Results: [¶](#results-48)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.builtin.finish` (::circt::moore::FinishBIOp) [¶](#moorebuiltinfinish-circtmoorefinishbiop)

*Exit simulation*

Syntax:

```
operation ::= `moore.builtin.finish` $exitCode attr-dict
```

Corresponds to the `$finish` system task. Causes the simulator to exit and
pass control back to the host operating system. Printing of the optional
diagnostic message is handled by the `finish_message` op.

The exit code argument of this op is not directly accessible from Verilog,
but is used to distinguish between the implicit `$finish` call in `$fatal`
and an explicit `$finish` called by the user.

See IEEE 1800-2017 § 20.2 “Simulation control system tasks”.

#### Attributes: [¶](#attributes-15)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `exitCode` | ::mlir::IntegerAttr | 8-bit signless integer attribute |

### `moore.builtin.finish_message` (::circt::moore::FinishMessageBIOp) [¶](#moorebuiltinfinish_message-circtmoorefinishmessagebiop)

*Print diagnostic message for the finish system task*

Syntax:

```
operation ::= `moore.builtin.finish_message` $withStats attr-dict
```

Prints the diagnostic message for `$stop`, `$finish`, `$exit`, and `$fatal`
mandated by the SystemVerilog standard. The exact message is controlled by
the verbosity parameter as specified in the standard:

* The absence of this op corresponds to `$finish(0)`.
* `moore.builtin.finish_message false` corresponds to `$finish(1)`.
* `moore.builtin.finish_message true` corresponds to `$finish(2)`.

The `withStats` argument controls how detailed the printed message is:

* **false**: Print simulation time and location.
* **true**: Print simulation time, location, and statistics about the memory
  and CPU usage of the simulator.

See IEEE 1800-2017 § 20.2 “Simulation control system tasks”.

#### Attributes: [¶](#attributes-16)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `withStats` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

### `moore.fle` (::circt::moore::FleOp) [¶](#moorefle-circtmoorefleop)

*Real-valued less than or equal comparison*

Syntax:

```
operation ::= `moore.fle` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the left- and right-hand side operand and returns a single bit 0,
or 1 result. If all bits are 0 or 1, `flt`, `fle`, `fgt`, and
`fge` return whether the left-hand side is less than, less than or equal
to, greater than, or greater than or equal to the right-hand side,
respectively. `flt` corresponds to the `<` operator, `fle` to `<=`,
`fgt` to `>`, and `fge` to `>=`.

See IEEE 1800-2017 § 11.4.4 “Relational operators” and § 11.3.1 “Operators
with real operands”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-53)

| Operand | Description |
| --- | --- |
| `lhs` | a SystemVerilog real type |
| `rhs` | a SystemVerilog real type |

#### Results: [¶](#results-49)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.builtin.floor` (::circt::moore::FloorBIOp) [¶](#moorebuiltinfloor-circtmoorefloorbiop)

*Floor*

Syntax:

```
operation ::= `moore.builtin.floor` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-54)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-50)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.flt` (::circt::moore::FltOp) [¶](#mooreflt-circtmoorefltop)

*Real-valued less than comparison*

Syntax:

```
operation ::= `moore.flt` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the left- and right-hand side operand and returns a single bit 0,
or 1 result. If all bits are 0 or 1, `flt`, `fle`, `fgt`, and
`fge` return whether the left-hand side is less than, less than or equal
to, greater than, or greater than or equal to the right-hand side,
respectively. `flt` corresponds to the `<` operator, `fle` to `<=`,
`fgt` to `>`, and `fge` to `>=`.

See IEEE 1800-2017 § 11.4.4 “Relational operators” and § 11.3.1 “Operators
with real operands”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-55)

| Operand | Description |
| --- | --- |
| `lhs` | a SystemVerilog real type |
| `rhs` | a SystemVerilog real type |

#### Results: [¶](#results-51)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.fmt.concat` (::circt::moore::FormatConcatOp) [¶](#moorefmtconcat-circtmooreformatconcatop)

*Concatenate string fragments*

Syntax:

```
operation ::= `moore.fmt.concat` ` ` `(` $inputs `)` attr-dict
```

Concatenates an arbitrary number of format string into one larger format
string. The strings are concatenated from left to right, with the first
operand appearing at the left start of the result string, and the last
operand appearing at the right end. Produces an empty string if no inputs
are provided.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-56)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of a format string type |

#### Results: [¶](#results-52)

| Result | Description |
| --- | --- |
| `result` | a format string type |

### `moore.fmt.int` (::circt::moore::FormatIntOp) [¶](#moorefmtint-circtmooreformatintop)

*Format an integer value*

Syntax:

```
operation ::= `moore.fmt.int` $format $value `,`
              `align` $alignment `,`
              `pad` $padding
              (`width` $specifierWidth^)?
              (`signed` $isSigned^)?
              attr-dict `:` type($value)
```

Format an integer value as a string according to the specified format.

See IEEE 1800-2017 § 21.2.1.2 “Format specifications”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-17)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `format` | circt::moore::IntFormatAttr | Integer format |
| `alignment` | circt::moore::IntAlignAttr | Integer alignment |
| `padding` | circt::moore::IntPaddingAttr | Integer alignment |
| `specifierWidth` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `isSigned` | ::mlir::UnitAttr | unit attribute |

#### Operands: [¶](#operands-57)

| Operand | Description |
| --- | --- |
| `value` | a simple bit vector type |

#### Results: [¶](#results-53)

| Result | Description |
| --- | --- |
| `result` | a format string type |

### `moore.fmt.literal` (::circt::moore::FormatLiteralOp) [¶](#moorefmtliteral-circtmooreformatliteralop)

*A constant string fragment*

Syntax:

```
operation ::= `moore.fmt.literal` $literal attr-dict
```

Creates a constant string fragment to be used as a format string. The
literal is printed as is, without any further escaping or processing of its
characters.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-18)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `literal` | ::mlir::StringAttr | string attribute |

#### Results: [¶](#results-54)

| Result | Description |
| --- | --- |
| `result` | a format string type |

### `moore.fmt.real` (::circt::moore::FormatRealOp) [¶](#moorefmtreal-circtmooreformatrealop)

*Format a real number value*

Syntax:

```
operation ::= `moore.fmt.real` $format $value `,`
              `align` $alignment
              (`fieldWidth` $fieldWidth^)?
              (`fracDigits` $fracDigits^)?
              attr-dict `:` type($value)
```

Format a real value as a string according to the specified format.

See IEEE 1800-2017 § 21.2.1.2 “Format specifications”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-19)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `format` | circt::moore::RealFormatAttr | Real format |
| `alignment` | circt::moore::IntAlignAttr | Integer alignment |
| `fieldWidth` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fracDigits` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands: [¶](#operands-58)

| Operand | Description |
| --- | --- |
| `value` | a SystemVerilog real type |

#### Results: [¶](#results-55)

| Result | Description |
| --- | --- |
| `result` | a format string type |

### `moore.fmt.string` (::circt::moore::FormatStringOp) [¶](#moorefmtstring-circtmooreformatstringop)

*A dynamic string fragment*

Syntax:

```
operation ::= `moore.fmt.string` $string
              (`,` `width` $width^)?
              (`,` `alignment` $alignment^)?
              (`,` `padding` $padding^)?
              attr-dict
```

Creates a dynamic string fragment to be used as a format string. The
string is printed as is, without any further escaping or processing of its
characters.

Use fmt.literal over this operator for any constant strings / literals.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-20)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `width` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `alignment` | circt::moore::IntAlignAttr | Integer alignment |
| `padding` | circt::moore::IntPaddingAttr | Integer alignment |

#### Operands: [¶](#operands-59)

| Operand | Description |
| --- | --- |
| `string` | the SystemVerilog `string` type |

#### Results: [¶](#results-56)

| Result | Description |
| --- | --- |
| `result` | a format string type |

### `moore.fstring_to_string` (::circt::moore::FormatStringToStringOp) [¶](#moorefstring_to_string-circtmooreformatstringtostringop)

*A dynamic string created through formatting*

Syntax:

```
operation ::= `moore.fstring_to_string` $fmtstring attr-dict
```

Creates a dynamic string from a format string.
Translates into a no-op.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-60)

| Operand | Description |
| --- | --- |
| `fmtstring` | a format string type |

#### Results: [¶](#results-57)

| Result | Description |
| --- | --- |
| `result` | the SystemVerilog `string` type |

### `moore.get_global_variable` (::circt::moore::GetGlobalVariableOp) [¶](#mooreget_global_variable-circtmooregetglobalvariableop)

*Get a reference to a global variable*

Syntax:

```
operation ::= `moore.get_global_variable` $global_name attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-21)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `global_name` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Results: [¶](#results-58)

| Result | Description |
| --- | --- |
| `result` |  |

### `moore.global_variable` (::circt::moore::GlobalVariableOp) [¶](#mooreglobal_variable-circtmooreglobalvariableop)

*A global variable declaration*

Syntax:

```
operation ::= `moore.global_variable` $sym_name attr-dict `:` $type (`init` $initRegion^)?
```

Defines a global or package variable.

See IEEE 1800-2017 § 6.8 “Variable declarations”.

Traits: `IsolatedFromAbove`, `NoRegionArguments`, `SingleBlock`

Interfaces: `Symbol`

#### Attributes: [¶](#attributes-22)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `type` | ::mlir::TypeAttr | type attribute of unpacked type |

### `moore.handle_case_eq` (::circt::moore::HandleCaseEqOp) [¶](#moorehandle_case_eq-circtmoorehandlecaseeqop)

*Case equality for handle types*

Syntax:

```
operation ::= `moore.handle_case_eq` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
```

Compares two handle-typed (ChandleType, ClassHandleType) operands and
returns a single bit 0 or 1 result. `case_eq` corresponds to the `===`
operator and `case_ne` to the `!==` operator. If the handles are not
identical, or both null, the two operands are considered not equal (`eq`
returns 0, `ne` returns 1).

See IEEE 1800-2023 § 8.4 “Objects (class instances)”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-61)

| Operand | Description |
| --- | --- |
| `lhs` | Nullable handle type |
| `rhs` | Nullable handle type |

#### Results: [¶](#results-59)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.handle_case_ne` (::circt::moore::HandleCaseNeOp) [¶](#moorehandle_case_ne-circtmoorehandlecaseneop)

*Case inequality for handle types*

Syntax:

```
operation ::= `moore.handle_case_ne` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
```

Compares two handle-typed (ChandleType, ClassHandleType) operands and
returns a single bit 0 or 1 result. `case_eq` corresponds to the `===`
operator and `case_ne` to the `!==` operator. If the handles are not
identical, or both null, the two operands are considered not equal (`eq`
returns 0, `ne` returns 1).

See IEEE 1800-2023 § 8.4 “Objects (class instances)”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-62)

| Operand | Description |
| --- | --- |
| `lhs` | Nullable handle type |
| `rhs` | Nullable handle type |

#### Results: [¶](#results-60)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.handle_eq` (::circt::moore::HandleEqOp) [¶](#moorehandle_eq-circtmoorehandleeqop)

*Handle equality*

Syntax:

```
operation ::= `moore.handle_eq` $lhs `,` $rhs attr-dict `:` type($lhs) `:` type($rhs) `->` type($result)
```

Compares two handle-typed (ChandleType, ClassHandleType) operands and
returns a single bit 0 or 1 result. If both operands correspond to the same
memory address, the two operands are considered equal (`eq` returns 1, `ne`
returns 0). If the handles are not identical, or both null, the two operands
are considered not equal (`eq` returns 0, `ne` returns 1).

See IEEE 1800-2023 § 8.4 “Objects (class instances)”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-63)

| Operand | Description |
| --- | --- |
| `lhs` | Nullable handle type |
| `rhs` | Nullable handle type |

#### Results: [¶](#results-61)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.handle_ne` (::circt::moore::HandleNeOp) [¶](#moorehandle_ne-circtmoorehandleneop)

*Handle inequality*

Syntax:

```
operation ::= `moore.handle_ne` $lhs `,` $rhs attr-dict `:` type($lhs) `:` type($rhs) `->` type($result)
```

Compares two handle-typed (ChandleType, ClassHandleType) operands and
returns a single bit 0 or 1 result. If both operands correspond to the same
memory address, the two operands are considered equal (`eq` returns 1, `ne`
returns 0). If the handles are not identical, or both null, the two operands
are considered not equal (`eq` returns 0, `ne` returns 1).

See IEEE 1800-2023 § 8.4 “Objects (class instances)”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-64)

| Operand | Description |
| --- | --- |
| `lhs` | Nullable handle type |
| `rhs` | Nullable handle type |

#### Results: [¶](#results-62)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.instance` (::circt::moore::InstanceOp) [¶](#mooreinstance-circtmooreinstanceop)

*Create an instance of a module*

The `moore.instance` operation instantiates a `moore.module` operation.

See IEEE 1800-2017 § 23.3 “Module instances”.

Interfaces: `OpAsmOpInterface`, `SymbolUserOpInterface`

#### Attributes: [¶](#attributes-23)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `instanceName` | ::mlir::StringAttr | string attribute |
| `moduleName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `inputNames` | ::mlir::ArrayAttr | string array attribute |
| `outputNames` | ::mlir::ArrayAttr | string array attribute |

#### Operands: [¶](#operands-65)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results: [¶](#results-63)

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `moore.int_to_logic` (::circt::moore::IntToLogicOp) [¶](#mooreint_to_logic-circtmooreinttologicop)

*Convert a two-valued to a four-valued integer*

Syntax:

```
operation ::= `moore.int_to_logic` $input attr-dict `:` type($input)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-66)

| Operand | Description |
| --- | --- |
| `input` | two-valued integer type |

#### Results: [¶](#results-64)

| Result | Description |
| --- | --- |
| `result` | four-valued integer type |

### `moore.int_to_string` (::circt::moore::IntToStringOp) [¶](#mooreint_to_string-circtmooreinttostringop)

*Convert an integer to a string*

Syntax:

```
operation ::= `moore.int_to_string` $input attr-dict `:` type($input)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-67)

| Operand | Description |
| --- | --- |
| `input` | two-valued integer type |

#### Results: [¶](#results-65)

| Result | Description |
| --- | --- |
| `result` | the SystemVerilog `string` type |

### `moore.builtin.ln` (::circt::moore::LnBIOp) [¶](#moorebuiltinln-circtmoorelnbiop)

*Natural logarithm*

Syntax:

```
operation ::= `moore.builtin.ln` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-68)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-66)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.builtin.log10` (::circt::moore::Log10BIOp) [¶](#moorebuiltinlog10-circtmoorelog10biop)

*Decimal logarithm*

Syntax:

```
operation ::= `moore.builtin.log10` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-69)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-67)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.logic_to_int` (::circt::moore::LogicToIntOp) [¶](#moorelogic_to_int-circtmoorelogictointop)

*Convert a four-valued to a two-valued integer*

Syntax:

```
operation ::= `moore.logic_to_int` $input attr-dict `:` type($input)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-70)

| Operand | Description |
| --- | --- |
| `input` | four-valued integer type |

#### Results: [¶](#results-68)

| Result | Description |
| --- | --- |
| `result` | two-valued integer type |

### `moore.logic_to_time` (::circt::moore::LogicToTimeOp) [¶](#moorelogic_to_time-circtmoorelogictotimeop)

*Convert an integer number of femtoseconds to a time type*

Syntax:

```
operation ::= `moore.logic_to_time` $input attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-71)

| Operand | Description |
| --- | --- |
| `input` | 64-bit four-valued integer type |

#### Results: [¶](#results-69)

| Result | Description |
| --- | --- |
| `result` | the SystemVerilog `time` and `realtime` type |

### `moore.mods` (::circt::moore::ModSOp) [¶](#mooremods-circtmooremodsop)

*Remainder*

Syntax:

```
operation ::= `moore.mods` $lhs `,` $rhs attr-dict `:` type($result)
```

Compute the remainder of the left-hand side divided by the right-hand side
operand. If the right-hand side is zero, all bits of the result are X. The
sign of the result is the sign of the left-hand side. If any bit in the two
operands is Z or X, all bits in the result are X.

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators”.

Consider the following examples:

| LHS | RHS | Result |
| --- | --- | --- |
| 11 | 3 | 2 |
| -11 | 3 | -2 |
| 11 | -3 | 2 |
| -11 | -3 | -2 |

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-72)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-70)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.modu` (::circt::moore::ModUOp) [¶](#mooremodu-circtmooremoduop)

*Remainder*

Syntax:

```
operation ::= `moore.modu` $lhs `,` $rhs attr-dict `:` type($result)
```

Compute the remainder of the left-hand side divided by the right-hand side
operand. If the right-hand side is zero, all bits of the result are X. The
sign of the result is the sign of the left-hand side. If any bit in the two
operands is Z or X, all bits in the result are X.

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators”.

Consider the following examples:

| LHS | RHS | Result |
| --- | --- | --- |
| 11 | 3 | 2 |
| -11 | 3 | -2 |
| 11 | -3 | 2 |
| -11 | -3 | -2 |

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-73)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-71)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.mul` (::circt::moore::MulOp) [¶](#mooremul-circtmooremulop)

*Multiplication*

Syntax:

```
operation ::= `moore.mul` $lhs `,` $rhs attr-dict `:` type($result)
```

Multiply the operands. If any bit in the two operands is Z or X, all bits in
the result are X.

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-74)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-72)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.fmul` (::circt::moore::MulRealOp) [¶](#moorefmul-circtmooremulrealop)

*Multiplication; Multiply the RHS real operand with the LHS
real operand.*

Syntax:

```
operation ::= `moore.fmul` $lhs `,` $rhs attr-dict `:` type($result)
```

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators” and § 11.3.1 “Operators
with real operands”

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-75)

| Operand | Description |
| --- | --- |
| `lhs` | a SystemVerilog real type |
| `rhs` | a SystemVerilog real type |

#### Results: [¶](#results-73)

| Result | Description |
| --- | --- |
| `result` | a SystemVerilog real type |

### `moore.ne` (::circt::moore::NeOp) [¶](#moorene-circtmooreneop)

*Logical inequality*

Syntax:

```
operation ::= `moore.ne` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the bits in the left- and right-hand side operand and returns a
single bit 0, 1, or X result. If all corresponding bits in the left- and
right-hand side are equal, and all are 0 or 1 (not X or Z), the two operands
are considered equal (`eq` returns 1, `ne` returns 0). If any bits are not
equal, but all are 0 or 1, the two operands are considered not equal (`eq`
returns 0, `ne` returns 1). If any bit in the two operands is Z or X,
returns X. `eq` corresponds to the `==` operator and `ne` to the `!=`
operator.

See IEEE 1800-2017 § 11.4.5 “Equality operators”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-76)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-74)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.fne` (::circt::moore::NeRealOp) [¶](#moorefne-circtmoorenerealop)

*Logical inequality*

Syntax:

```
operation ::= `moore.fne` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the bits in the left- and right-hand side operand and returns a
single bit 0 or 1. If all corresponding bits in the left- and
right-hand side are equal, and all are 0 or 1, the two operands
are considered equal (`eq` returns 1, `ne` returns 0). If any bits are not
equal, but all are 0 or 1, the two operands are considered not equal (`eq`
returns 0, `ne` returns 1). `eq` corresponds to the `==` operator and `ne`
to the `!=` operator.

See IEEE 1800-2017 § 11.4.5 “Equality operators” and § 11.3.1 “Operators
with real operands”.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-77)

| Operand | Description |
| --- | --- |
| `lhs` | a SystemVerilog real type |
| `rhs` | a SystemVerilog real type |

#### Results: [¶](#results-75)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.neg` (::circt::moore::NegOp) [¶](#mooreneg-circtmoorenegop)

*Arithmetic negation*

Syntax:

```
operation ::= `moore.neg` $input attr-dict `:` type($input)
```

Negate a value to its two’s complement form. If any bit in the input is Z or
X, all bits in the result are X.

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-78)

| Operand | Description |
| --- | --- |
| `input` | simple bit vector type |

#### Results: [¶](#results-76)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.fneg` (::circt::moore::NegRealOp) [¶](#moorefneg-circtmoorenegrealop)

*Arithmetic negation*

Syntax:

```
operation ::= `moore.fneg` $input attr-dict `:` type($input)
```

Negate a real value.

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators” and § 11.3.1 “Operators
with real operands”

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-79)

| Operand | Description |
| --- | --- |
| `input` | a SystemVerilog real type |

#### Results: [¶](#results-77)

| Result | Description |
| --- | --- |
| `result` | a SystemVerilog real type |

### `moore.net` (::circt::moore::NetOp) [¶](#moorenet-circtmoorenetop)

*A net declaration*

Syntax:

```
operation ::= `moore.net` `` custom<ImplicitSSAName>($name) $kind ($assignment^)? attr-dict
              `:` type($result)
```

The `moore.net` operation is a net declaration. Net types defines different
types of net connection in SV. There are twelve built-in net types defined
in the official standard construct of the operation:
`supply0`, `supply1`, `tri`, `triand`, `trior`, `trireg`, `tri0`, `tri1`,
`uwire`, `wire`, `wand`, `wor`.
Optional assignment argument allows net operation to be initialized with
specific values as soon as it is created. Only one net declaration
assignment can be made for a particular net. See IEEE 1800-2017 § 10.3.1
“The net declaration assignment” for the differences between net declaration
assignments and continuous assign statements. It has some features that are
not supported: declaring an interconnect net and using user-defined types in
the net operation.

See IEEE 1800-2017 § 6.7 “Net declarations”.

Interfaces: `OpAsmOpInterface`

#### Attributes: [¶](#attributes-24)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `kind` | ::circt::moore::NetKindAttr | Net type kind |

#### Operands: [¶](#operands-80)

| Operand | Description |
| --- | --- |
| `assignment` | unpacked type |

#### Results: [¶](#results-78)

| Result | Description |
| --- | --- |
| `result` |  |

### `moore.nonblocking_assign` (::circt::moore::NonBlockingAssignOp) [¶](#moorenonblocking_assign-circtmoorenonblockingassignop)

*Nonblocking procedural assignment*

Syntax:

```
operation ::= `moore.nonblocking_assign` $dst `,` $src attr-dict `:` type($src)
```

A nonblocking procedural assignment in a sequential block, such as `x <= y;`
or `x <= @(posedge y) z` or `x <= #1ns y`. The assignment does not take
effect immediately. Subsequent operations in the block do not see the
effects of this assignment. Instead, the assignment is scheduled to happen
in a subsequent time step as dictated by the delay or event control.

See IEEE 1800-2017 § 10.4.2 “Nonblocking procedural assignments”.

#### Operands: [¶](#operands-81)

| Operand | Description |
| --- | --- |
| `dst` |  |
| `src` | unpacked type |

### `moore.not` (::circt::moore::NotOp) [¶](#moorenot-circtmoorenotop)

*Bitwise unary negation*

Syntax:

```
operation ::= `moore.not` $input attr-dict `:` type($input)
```

Applies the boolean NOT operation to each bit in the input. Corresponds to
the `~` operator, as well as the negation in the `~&`, `~|`, `^~`, and `~^`
reduction operators.

See IEEE 1800-2017 § 11.4.8 “Bitwise operators”.

| Input | Result |
| --- | --- |
| 0 | 1 |
| 1 | 0 |
| X | X |
| Z | X |

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-82)

| Operand | Description |
| --- | --- |
| `input` | simple bit vector type |

#### Results: [¶](#results-79)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.null` (::circt::moore::NullOp) [¶](#moorenull-circtmoorenullop)

*SystemVerilog literal constant `null`*

Syntax:

```
operation ::= `moore.null` attr-dict
```

Represents the SystemVerilog literal constant `null`, which is a valid
primary and constant\_primary in IEEE 1800-2023.

The result type must be a null-able handle-like type (e.g., a class handle,
virtual interface, event, chandle). Verification enforces this to prevent
mis-modeling `null` as a numeric constant.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results: [¶](#results-80)

| Result | Description |
| --- | --- |
| `result` | A type that represents null literals for chandles, classhandles, |
| and virtual interfaces |  |

### `moore.or` (::circt::moore::OrOp) [¶](#mooreor-circtmooreorop)

*Bitwise OR operation*

Syntax:

```
operation ::= `moore.or` $lhs `,` $rhs attr-dict `:` type($result)
```

Applies the boolean OR operation to each pair of corresponding bits in the
left- and right-hand side operand. Corresponds to the `|` operator.

See IEEE 1800-2017 § 11.4.8 “Bitwise operators”.

|  | 0 | 1 | X | Z |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1 | X | X |
| 1 | 1 | 1 | 1 | 1 |
| X | X | 1 | X | X |
| Z | X | 1 | X | X |

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-83)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-81)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.output` (::circt::moore::OutputOp) [¶](#mooreoutput-circtmooreoutputop)

*Assign module outputs*

Syntax:

```
operation ::= `moore.output` attr-dict ($outputs^ `:` type($outputs))?
```

The `moore.output` operation marks the end of a `moore.module` body region
and specifies the values to present for the module’s output ports.

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<SVModuleOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-84)

| Operand | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `moore.packed_to_sbv` (::circt::moore::PackedToSBVOp) [¶](#moorepacked_to_sbv-circtmoorepackedtosbvop)

*Convert a packed type to its simple bit vector equivalent*

Syntax:

```
operation ::= `moore.packed_to_sbv` $input attr-dict `:` type($input)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-85)

| Operand | Description |
| --- | --- |
| `input` | packed type with known size |

#### Results: [¶](#results-82)

| Result | Description |
| --- | --- |
| `result` | a simple bit vector type |

### `moore.fpow` (::circt::moore::PowRealOp) [¶](#moorefpow-circtmoorepowrealop)

*Power; Exponentiate the LHS real base with the RHS real
exponent.*

Syntax:

```
operation ::= `moore.fpow` $lhs `,` $rhs attr-dict `:` type($result)
```

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators” and § 11.3.1 “Operators
with real operands”

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-86)

| Operand | Description |
| --- | --- |
| `lhs` | a SystemVerilog real type |
| `rhs` | a SystemVerilog real type |

#### Results: [¶](#results-83)

| Result | Description |
| --- | --- |
| `result` | a SystemVerilog real type |

### `moore.pows` (::circt::moore::PowSOp) [¶](#moorepows-circtmoorepowsop)

*Power*

Syntax:

```
operation ::= `moore.pows` $lhs `,` $rhs attr-dict `:` type($result)
```

Raise the left-hand side to the power of the right-hand side. `powu` treats
its operands as unsigned numbers, while `pows` treats them as signed
numbers.

Evaluation rules for `a ** b`:

|  | a < -1 | a = -1 | a = 0 | a = 1 | a > 1 |
| --- | --- | --- | --- | --- | --- |
| b > 0 | a \*\* b | b odd ? -1 : 1 | 0 | 1 | a \*\* b |
| b = 0 | 1 | 1 | 1 | 1 | 1 |
| b < 0 | 0 | b odd ? -1 : 1 | X | 1 | 0 |

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-87)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-84)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.powu` (::circt::moore::PowUOp) [¶](#moorepowu-circtmoorepowuop)

*Power*

Syntax:

```
operation ::= `moore.powu` $lhs `,` $rhs attr-dict `:` type($result)
```

Raise the left-hand side to the power of the right-hand side. `powu` treats
its operands as unsigned numbers, while `pows` treats them as signed
numbers.

Evaluation rules for `a ** b`:

|  | a < -1 | a = -1 | a = 0 | a = 1 | a > 1 |
| --- | --- | --- | --- | --- | --- |
| b > 0 | a \*\* b | b odd ? -1 : 1 | 0 | 1 | a \*\* b |
| b = 0 | 1 | 1 | 1 | 1 | 1 |
| b < 0 | 0 | b odd ? -1 : 1 | X | 1 | 0 |

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-88)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-85)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.procedure` (::circt::moore::ProcedureOp) [¶](#mooreprocedure-circtmooreprocedureop)

*A procedure executed at different points in time*

Syntax:

```
operation ::= `moore.procedure` $kind attr-dict-with-keyword $body
```

The `moore.procedure` operation represents the SystemVerilog `initial`,
`final`, `always`, `always_comb`, `always_latch`, and `always_ff`
procedures.

Execution times of the various procedures:

* An `initial` procedure is executed once at the start of a design’s
  lifetime, before any other procedures are executed.
* A `final` procedure is executed once at the end of a design’s lifetime,
  after all other procedures have stopped execution.
* An `always` or `always_ff` procedure is repeatedly executed during a
  design’s lifetime. Timing and event control inside the procedure can
  suspend its execution, for example to wait for a signal to change. If no
  such timing or event control is present, the procedure repeats infinitely
  at the current timestep, effectively deadlocking the design.
* An `always_comb` or `always_latch` procedure is executed once at the start
  of a design’s lifetime, after any `initial` procedures, and throughout the
  lifetime of the design whenever any of the variables read by the body of
  the procedure changes. Since the procedure is only executed when its
  change, and not repeatedly, the body generally does not contain any timing
  or event control. This behavior mitigates a shortcoming of `always`
  procedures, which commonly have an event control like `@*` that blocks
  and waits for a change of any input signals. This prevents the body from
  executing when the design is initialized and properly reacting to the
  initial values of signals. In contrast, `always_comb` and `always_latch`
  procedures have an implicit unconditional execution at design start-up.

See IEEE 1800-2017 § 9.2 “Structured procedures”.

Traits: `NoRegionArguments`, `RecursiveMemoryEffects`, `RecursivelySpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`

#### Attributes: [¶](#attributes-25)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `kind` | circt::moore::ProcedureKindAttr | Procedure kind |

### `moore.pop_back` (::circt::moore::QueuePopBackOp) [¶](#moorepop_back-circtmoorequeuepopbackop)

*Return and remove an element from the back of a queue*

Syntax:

```
operation ::= `moore.pop_back` `from` $queue attr-dict `:` type($queue)
```

See IEEE 1800-2023 § 7.10.2.4 “Pop\_back()”

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-89)

| Operand | Description |
| --- | --- |
| `queue` | a queue type |

#### Results: [¶](#results-86)

| Result | Description |
| --- | --- |
| `result` | unpacked type |

### `moore.pop_front` (::circt::moore::QueuePopFrontOp) [¶](#moorepop_front-circtmoorequeuepopfrontop)

*Return and remove an element from the front of a queue*

Syntax:

```
operation ::= `moore.pop_front` `from` $queue attr-dict `:` type($queue)
```

See IEEE 1800-2023 § 7.10.2.5 “Pop\_front()”

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-90)

| Operand | Description |
| --- | --- |
| `queue` | a queue type |

#### Results: [¶](#results-87)

| Result | Description |
| --- | --- |
| `result` | unpacked type |

### `moore.push_back` (::circt::moore::QueuePushBackOp) [¶](#moorepush_back-circtmoorequeuepushbackop)

*Push an element to the back of a queue*

Syntax:

```
operation ::= `moore.push_back` $element `into` $queue attr-dict `:` type($queue)
```

See IEEE 1800-2023 § 7.10.2.7 “Push\_back()”

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-91)

| Operand | Description |
| --- | --- |
| `queue` | a queue type |
| `element` | unpacked type |

#### Results: [¶](#results-88)

| Result | Description |
| --- | --- |
| `out` | the SystemVerilog `void` type |

### `moore.push_front` (::circt::moore::QueuePushFrontOp) [¶](#moorepush_front-circtmoorequeuepushfrontop)

*Push an element to the front of a queue*

Syntax:

```
operation ::= `moore.push_front` $element `into` $queue attr-dict `:` type($queue)
```

See IEEE 1800-2023 § 7.10.2.6 “Push\_front()”

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-92)

| Operand | Description |
| --- | --- |
| `queue` | a queue type |
| `element` | unpacked type |

#### Results: [¶](#results-89)

| Result | Description |
| --- | --- |
| `out` | the SystemVerilog `void` type |

### `moore.builtin.size` (::circt::moore::QueueSizeBIOp) [¶](#moorebuiltinsize-circtmoorequeuesizebiop)

*Return the number of elements in a queue*

Syntax:

```
operation ::= `moore.builtin.size` $queue `:` type($queue) attr-dict
```

Corresponds to the `size` system function. Returns the number of items in a queue.
According to spec the returned size fits an `int`.

See IEEE 1800-2023 § 7.10.2.1 “Size()”.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-93)

| Operand | Description |
| --- | --- |
| `queue` | a queue type |

#### Results: [¶](#results-90)

| Result | Description |
| --- | --- |
| `result` | 32-bit two-valued integer type |

### `moore.builtin.random` (::circt::moore::RandomBIOp) [¶](#moorebuiltinrandom-circtmoorerandombiop)

*Generate a true random signed integer (optionally seeded)*

Syntax:

```
operation ::= `moore.builtin.random` ($seed^)? attr-dict
```

Corresponds to the `$random` system function. Returns a 32-bit
true random integer. The seed is optional; when provided, it initializes
the generator. If not provided, treat it as 0 in semantics/lowering.

`$random` is largely considered to be deprecated since it leads to
non-reproducible simulation results. Consider using `$urandom` instead.

See IEEE 1800-2023 § 20.14 “Probablistic distribution functions”.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-94)

| Operand | Description |
| --- | --- |
| `seed` | 32-bit two-valued integer type |

#### Results: [¶](#results-91)

| Result | Description |
| --- | --- |
| `result` | 32-bit two-valued integer type |

### `moore.read` (::circt::moore::ReadOp) [¶](#mooreread-circtmoorereadop)

*Read the current value of a declaration*

Syntax:

```
operation ::= `moore.read` $input attr-dict `:` type($input)
```

Samples the current value of a declaration. This is a helper to capture the
exact point at which declarations that can be targeted by all possible
expressions are read. It’s similar to llvm.load.

Interfaces: `InferTypeOpInterface`, `PromotableMemOpInterface`

#### Operands: [¶](#operands-95)

| Operand | Description |
| --- | --- |
| `input` |  |

#### Results: [¶](#results-92)

| Result | Description |
| --- | --- |
| `result` | unpacked type |

### `moore.real_to_int` (::circt::moore::RealToIntOp) [¶](#moorereal_to_int-circtmoorerealtointop)

*Convert a real value to a two-valued integer*

Syntax:

```
operation ::= `moore.real_to_int` $input attr-dict `:` type($input) `->` type($result)
```

See IEEE 1800-2023 Section 6.24.1: “Cast operator” and 6.12:
“Real, shortreal and realtime data types”. Accordingly,
output values are rounded.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-96)

| Operand | Description |
| --- | --- |
| `input` | a SystemVerilog real type |

#### Results: [¶](#results-93)

| Result | Description |
| --- | --- |
| `result` | two-valued integer type |

### `moore.builtin.realtobits` (::circt::moore::RealtobitsBIOp) [¶](#moorebuiltinrealtobits-circtmoorerealtobitsbiop)

*Convert a real-valued number to its logic vector representation*

Syntax:

```
operation ::= `moore.builtin.realtobits` $value attr-dict
```

Corresponds to the `$realtobits` system function. Returns a 64-bit
logic vector corresponding to the bit representation of the real number.
Note that this does not correspond to a cast to another type, but rather a no-op.

See IEEE 1800-2023 § 20.5 “Conversion functions”.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-97)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-94)

| Result | Description |
| --- | --- |
| `result` | 64-bit two-valued integer type |

### `moore.reduce_and` (::circt::moore::ReduceAndOp) [¶](#moorereduce_and-circtmoorereduceandop)

*Reduction AND operator*

Syntax:

```
operation ::= `moore.reduce_and` $input attr-dict `:` type($input) `->` type($result)
```

Reduces all bits in the input to a single result bit by iteratively applying
the boolean AND operator. If the input has only a single bit, that bit is
returned.

See IEEE 1800-2017 § 11.4.9 “Reduction operators”. See the corresponding
`and`, `or`, and `xor` operations for the truth table.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-98)

| Operand | Description |
| --- | --- |
| `input` | simple bit vector type |

#### Results: [¶](#results-95)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.reduce_or` (::circt::moore::ReduceOrOp) [¶](#moorereduce_or-circtmoorereduceorop)

*Reduction OR operator*

Syntax:

```
operation ::= `moore.reduce_or` $input attr-dict `:` type($input) `->` type($result)
```

Reduces all bits in the input to a single result bit by iteratively applying
the boolean OR operator. If the input has only a single bit, that bit is
returned.

See IEEE 1800-2017 § 11.4.9 “Reduction operators”. See the corresponding
`and`, `or`, and `xor` operations for the truth table.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-99)

| Operand | Description |
| --- | --- |
| `input` | simple bit vector type |

#### Results: [¶](#results-96)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.reduce_xor` (::circt::moore::ReduceXorOp) [¶](#moorereduce_xor-circtmoorereducexorop)

*Reduction XOR operator*

Syntax:

```
operation ::= `moore.reduce_xor` $input attr-dict `:` type($input) `->` type($result)
```

Reduces all bits in the input to a single result bit by iteratively applying
the boolean XOR operator. If the input has only a single bit, that bit is
returned.

See IEEE 1800-2017 § 11.4.9 “Reduction operators”. See the corresponding
`and`, `or`, and `xor` operations for the truth table.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-100)

| Operand | Description |
| --- | --- |
| `input` | simple bit vector type |

#### Results: [¶](#results-97)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.replicate` (::circt::moore::ReplicateOp) [¶](#moorereplicate-circtmoorereplicateop)

*Multiple concatenation of expressions*

Syntax:

```
operation ::= `moore.replicate` $value attr-dict `:` type($value) `->` type($result)
```

This operation indicates a joining together of that many copies of the
concatenation `{constant{w}}`. Which enclosed together within brace.
The ‘constant’ must a non-negative, non-x, and non-z constant expression.
The ‘constant’ may be a value of zero, but it only exists in parameterized
code, and it will be ignored(type is changed to the void).

Example:

```
  {0{w}}   // empty! ignore it.
  {4{w}}   // the same as {w, w, w, w}
```

See IEEE 1800-2017 §11.4.12 “Concatenation operators”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-101)

| Operand | Description |
| --- | --- |
| `value` | a simple bit vector type |

#### Results: [¶](#results-98)

| Result | Description |
| --- | --- |
| `result` | a simple bit vector type |

### `moore.return` (::circt::moore::ReturnOp) [¶](#moorereturn-circtmoorereturnop)

*Return from a procedure*

Syntax:

```
operation ::= `moore.return` attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<ProcedureOp>`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

### `moore.sbv_to_packed` (::circt::moore::SBVToPackedOp) [¶](#mooresbv_to_packed-circtmooresbvtopackedop)

*Convert a simple bit vector to an equivalent packed type*

Syntax:

```
operation ::= `moore.sbv_to_packed` $input attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-102)

| Operand | Description |
| --- | --- |
| `input` | a simple bit vector type |

#### Results: [¶](#results-99)

| Result | Description |
| --- | --- |
| `result` | packed type with known size |

### `moore.sext` (::circt::moore::SExtOp) [¶](#mooresext-circtmooresextop)

*Sign-extend a value*

Syntax:

```
operation ::= `moore.sext` $input attr-dict `:` type($input) `->` type($result)
```

Increase the bit width of a value by replicating its most significant bit.
This keeps the signed value constant.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-103)

| Operand | Description |
| --- | --- |
| `input` | a simple bit vector type |

#### Results: [¶](#results-100)

| Result | Description |
| --- | --- |
| `result` | a simple bit vector type |

### `moore.sint_to_real` (::circt::moore::SIntToRealOp) [¶](#mooresint_to_real-circtmooresinttorealop)

*Convert an integer value to a real*

Syntax:

```
operation ::= `moore.sint_to_real` $input attr-dict `:` type($input) `->` type($result)
```

See IEEE 1800-2023 Section 6.24.1: “Cast operator” and 6.12:
“Real, shortreal and realtime data types”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-104)

| Operand | Description |
| --- | --- |
| `input` | two-valued integer type |

#### Results: [¶](#results-101)

| Result | Description |
| --- | --- |
| `result` | a SystemVerilog real type |

### `moore.module` (::circt::moore::SVModuleOp) [¶](#mooremodule-circtmooresvmoduleop)

*A module definition*

The `moore.module` operation represents a SystemVerilog module, including
its name, port list, and the constituent parts that make up its body. The
module’s body is a graph region.

See IEEE 1800-2017 § 3.3 “Modules” and § 23.2 “Module definitions”.

Traits: `IsolatedFromAbove`, `SingleBlockImplicitTerminator<OutputOp>`, `SingleBlock`

Interfaces: `OpAsmOpInterface`, `RegionKindInterface`, `Symbol`

#### Attributes: [¶](#attributes-26)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `module_type` | ::mlir::TypeAttr | type attribute of module type |
| `sym_visibility` | ::mlir::StringAttr | string attribute |

### `moore.builtin.severity` (::circt::moore::SeverityBIOp) [¶](#moorebuiltinseverity-circtmooreseveritybiop)

*Print a diagnostic message*

Syntax:

```
operation ::= `moore.builtin.severity` $severity $message attr-dict
```

Prints the given format string to the standard diagnostic output of the
simulator. In most cases this should be stderr. This corresponds to the
`$info`, `$warning`, `$error`, and `$fatal` system tasks. Message formatting
is handled by `moore.fmt.*` ops. This only handles the message printing of
`$fatal`; printing of the additional statistics and the call to `$finish`
must be done through the `finish_message` and `finish` ops.

See IEEE 1800-2017 § 20.10 “Severity tasks”.

#### Attributes: [¶](#attributes-27)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `severity` | circt::moore::SeverityAttr | Diagnostic severity |

#### Operands: [¶](#operands-105)

| Operand | Description |
| --- | --- |
| `message` | a format string type |

### `moore.sge` (::circt::moore::SgeOp) [¶](#mooresge-circtmooresgeop)

*Signed greater than or equal comparison*

Syntax:

```
operation ::= `moore.sge` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the left- and right-hand side operand and returns a single bit 0,
1, or X result. If any bit in the two operands is Z or X, returns X.
Otherwise, if all bits are 0 or 1, `ult/slt`, `ule/sle`, `ugt/sgt`, and
`uge/sge` return whether the left-hand side is less than, less than or equal
to, greater than, or greater than or equal to the right-hand side,
respectively. `ult/slt` corresponds to the `<` operator, `ule/sle` to `<=`,
`ugt/sgt` to `>`, and `uge/sge` to `>=`.

See IEEE 1800-2017 § 11.4.4 “Relational operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-106)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-102)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.sgt` (::circt::moore::SgtOp) [¶](#mooresgt-circtmooresgtop)

*Signed greater than comparison*

Syntax:

```
operation ::= `moore.sgt` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the left- and right-hand side operand and returns a single bit 0,
1, or X result. If any bit in the two operands is Z or X, returns X.
Otherwise, if all bits are 0 or 1, `ult/slt`, `ule/sle`, `ugt/sgt`, and
`uge/sge` return whether the left-hand side is less than, less than or equal
to, greater than, or greater than or equal to the right-hand side,
respectively. `ult/slt` corresponds to the `<` operator, `ule/sle` to `<=`,
`ugt/sgt` to `>`, and `uge/sge` to `>=`.

See IEEE 1800-2017 § 11.4.4 “Relational operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-107)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-103)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.shl` (::circt::moore::ShlOp) [¶](#mooreshl-circtmooreshlop)

*Logical left shift*

Syntax:

```
operation ::= `moore.shl` $value `,` $amount attr-dict `:` type($value) `,` type($amount)
```

Shifts the `value` to the left or right by `amount` number of bits. The
result has the same type as the input value. The amount is always treated as
an unsigned number and has no effect on the signedness of the result. X or
Z bits in the input value are simply shifted left or right the same way 0 or
1 bits are. If the amount contains X or Z bits, all result bits are X.

`shl` shifts bits to the left, filling in 0 for the vacated least
significant bits. `shr` and `ashr` shift bits to the right; `shr` fills in
0 for the vacated most significant bits, and `ashr` copies the input’s sign
bit into the vacated most significant bits. Note that in contrast to the SV
spec, the `ashr` *always* fills in the sign bit regardless of the signedness
of the input.

`shl` corresponds to the `<<` and `<<<` operators. `shr` corresponds to the
`>>` operator, and the `>>>` operator applied to an unsigned value. `ashr`
corresponds to the `>>>` operator applied to a signed value.

See IEEE 1800-2017 § 11.4.10 “Shift operators”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-108)

| Operand | Description |
| --- | --- |
| `value` | simple bit vector type |
| `amount` | simple bit vector type |

#### Results: [¶](#results-104)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.builtin.shortrealtobits` (::circt::moore::ShortrealtobitsBIOp) [¶](#moorebuiltinshortrealtobits-circtmooreshortrealtobitsbiop)

*Convert a real-valued number to its logic vector representation*

Syntax:

```
operation ::= `moore.builtin.shortrealtobits` $value attr-dict
```

Corresponds to the `$shortrealtobits` system function. Returns a 64-bit
logic vector corresponding to the bit representation of the real number.
Note that this does not correspond to a cast to another type, but rather a no-op.

See IEEE 1800-2023 § 20.5 “Conversion functions”.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-109)

| Operand | Description |
| --- | --- |
| `value` | 32-bit real type, aka. shortreal |

#### Results: [¶](#results-105)

| Result | Description |
| --- | --- |
| `result` | 32-bit two-valued integer type |

### `moore.shr` (::circt::moore::ShrOp) [¶](#mooreshr-circtmooreshrop)

*Logical right shift*

Syntax:

```
operation ::= `moore.shr` $value `,` $amount attr-dict `:` type($value) `,` type($amount)
```

Shifts the `value` to the left or right by `amount` number of bits. The
result has the same type as the input value. The amount is always treated as
an unsigned number and has no effect on the signedness of the result. X or
Z bits in the input value are simply shifted left or right the same way 0 or
1 bits are. If the amount contains X or Z bits, all result bits are X.

`shl` shifts bits to the left, filling in 0 for the vacated least
significant bits. `shr` and `ashr` shift bits to the right; `shr` fills in
0 for the vacated most significant bits, and `ashr` copies the input’s sign
bit into the vacated most significant bits. Note that in contrast to the SV
spec, the `ashr` *always* fills in the sign bit regardless of the signedness
of the input.

`shl` corresponds to the `<<` and `<<<` operators. `shr` corresponds to the
`>>` operator, and the `>>>` operator applied to an unsigned value. `ashr`
corresponds to the `>>>` operator applied to a signed value.

See IEEE 1800-2017 § 11.4.10 “Shift operators”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-110)

| Operand | Description |
| --- | --- |
| `value` | simple bit vector type |
| `amount` | simple bit vector type |

#### Results: [¶](#results-106)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.builtin.sin` (::circt::moore::SinBIOp) [¶](#moorebuiltinsin-circtmooresinbiop)

*Sine*

Syntax:

```
operation ::= `moore.builtin.sin` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-111)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-107)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.builtin.sinh` (::circt::moore::SinhBIOp) [¶](#moorebuiltinsinh-circtmooresinhbiop)

*Hyperbolic sine*

Syntax:

```
operation ::= `moore.builtin.sinh` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-112)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-108)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.sle` (::circt::moore::SleOp) [¶](#mooresle-circtmooresleop)

*Signed less than or equal comparison*

Syntax:

```
operation ::= `moore.sle` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the left- and right-hand side operand and returns a single bit 0,
1, or X result. If any bit in the two operands is Z or X, returns X.
Otherwise, if all bits are 0 or 1, `ult/slt`, `ule/sle`, `ugt/sgt`, and
`uge/sge` return whether the left-hand side is less than, less than or equal
to, greater than, or greater than or equal to the right-hand side,
respectively. `ult/slt` corresponds to the `<` operator, `ule/sle` to `<=`,
`ugt/sgt` to `>`, and `uge/sge` to `>=`.

See IEEE 1800-2017 § 11.4.4 “Relational operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-113)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-109)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.slt` (::circt::moore::SltOp) [¶](#mooreslt-circtmooresltop)

*Signed less than comparison*

Syntax:

```
operation ::= `moore.slt` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the left- and right-hand side operand and returns a single bit 0,
1, or X result. If any bit in the two operands is Z or X, returns X.
Otherwise, if all bits are 0 or 1, `ult/slt`, `ule/sle`, `ugt/sgt`, and
`uge/sge` return whether the left-hand side is less than, less than or equal
to, greater than, or greater than or equal to the right-hand side,
respectively. `ult/slt` corresponds to the `<` operator, `ule/sle` to `<=`,
`ugt/sgt` to `>`, and `uge/sge` to `>=`.

See IEEE 1800-2017 § 11.4.4 “Relational operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-114)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-110)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.builtin.sqrt` (::circt::moore::SqrtBIOp) [¶](#moorebuiltinsqrt-circtmooresqrtbiop)

*Square root*

Syntax:

```
operation ::= `moore.builtin.sqrt` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-115)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-111)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.builtin.stop` (::circt::moore::StopBIOp) [¶](#moorebuiltinstop-circtmoorestopbiop)

*Suspend simulation*

Syntax:

```
operation ::= `moore.builtin.stop` attr-dict
```

Corresponds to the `$stop` system task. Causes the simulation to be
suspended but the simulator does not exit. Printing of the optional
diagnostic message is handled by the `finish_message` op.

See IEEE 1800-2017 § 20.2 “Simulation control system tasks”.

### `moore.string_cmp` (::circt::moore::StringCmpOp) [¶](#moorestring_cmp-circtmoorestringcmpop)

Syntax:

```
operation ::= `moore.string_cmp` $predicate $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares two string operands using the specified
predicate and returns a single bit result. Supported predicates:

* eq : equal
* ne : not equal
* lt : less than
* le : less or equal
* gt : greater than
* ge : greater or equal

The equality operator yields 1 if its operands are equal and 0 otherwise.
Relational operators compare two strings lexicographically,
returning 1 when the specified condition holds and 0 when it does not.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-28)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `predicate` | circt::moore::StringCmpPredicateAttr | allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5 |

#### Operands: [¶](#operands-116)

| Operand | Description |
| --- | --- |
| `lhs` | the SystemVerilog `string` type |
| `rhs` | the SystemVerilog `string` type |

#### Results: [¶](#results-112)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.string.concat` (::circt::moore::StringConcatOp) [¶](#moorestringconcat-circtmoorestringconcatop)

*Concatenate strings*

Syntax:

```
operation ::= `moore.string.concat` ` ` `(` $inputs `)` attr-dict
```

Concatenates an arbitrary number of strings into one larger string. The
strings are concatenated from left to right, with the first operand
appearing at the left start of the result string, and the last operand
appearing at the right end. Produces an empty string if no inputs are
provided.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-117)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of the SystemVerilog `string` type |

#### Results: [¶](#results-113)

| Result | Description |
| --- | --- |
| `result` | the SystemVerilog `string` type |

### `moore.string.getc` (::circt::moore::StringGetCOp) [¶](#moorestringgetc-circtmoorestringgetcop)

*Get a character from a string*

Syntax:

```
operation ::= `moore.string.getc` $str `[` $index `]` attr-dict
```

Returns the character at the specified index in the given string.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-118)

| Operand | Description |
| --- | --- |
| `str` | the SystemVerilog `string` type |
| `index` | 32-bit two-valued integer type |

#### Results: [¶](#results-114)

| Result | Description |
| --- | --- |
| `result` | 8-bit two-valued integer type |

### `moore.string.len` (::circt::moore::StringLenOp) [¶](#moorestringlen-circtmoorestringlenop)

*Get the length of a string*

Syntax:

```
operation ::= `moore.string.len` $str attr-dict
```

Returns the number of characters in the given string.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-119)

| Operand | Description |
| --- | --- |
| `str` | the SystemVerilog `string` type |

#### Results: [¶](#results-115)

| Result | Description |
| --- | --- |
| `result` | 32-bit two-valued integer type |

### `moore.string_to_int` (::circt::moore::StringToIntOp) [¶](#moorestring_to_int-circtmoorestringtointop)

*Convert a string to an integer*

Syntax:

```
operation ::= `moore.string_to_int` $input attr-dict `:` type($result)
```

If the width of the result type is smaller than the input string, it is
truncated. If it is larger, it is appended with 0s.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-120)

| Operand | Description |
| --- | --- |
| `input` | the SystemVerilog `string` type |

#### Results: [¶](#results-116)

| Result | Description |
| --- | --- |
| `result` | two-valued integer type |

### `moore.string.tolower` (::circt::moore::StringToLowerOp) [¶](#moorestringtolower-circtmoorestringtolowerop)

*Convert a string to lowercase*

Syntax:

```
operation ::= `moore.string.tolower` $str attr-dict
```

Converts all characters in the given string to lowercase.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-121)

| Operand | Description |
| --- | --- |
| `str` | the SystemVerilog `string` type |

#### Results: [¶](#results-117)

| Result | Description |
| --- | --- |
| `result` | the SystemVerilog `string` type |

### `moore.string.toupper` (::circt::moore::StringToUpperOp) [¶](#moorestringtoupper-circtmoorestringtoupperop)

*Convert a string to uppercase*

Syntax:

```
operation ::= `moore.string.toupper` $str attr-dict
```

Converts all characters in the given string to uppercase.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-122)

| Operand | Description |
| --- | --- |
| `str` | the SystemVerilog `string` type |

#### Results: [¶](#results-118)

| Result | Description |
| --- | --- |
| `result` | the SystemVerilog `string` type |

### `moore.struct_create` (::circt::moore::StructCreateOp) [¶](#moorestruct_create-circtmoorestructcreateop)

*Create a struct value from individual fields*

Syntax:

```
operation ::= `moore.struct_create` $fields attr-dict `:` type($fields) `->` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-123)

| Operand | Description |
| --- | --- |
| `fields` | variadic of unpacked type |

#### Results: [¶](#results-119)

| Result | Description |
| --- | --- |
| `result` | packed or unpacked struct type |

### `moore.struct_extract` (::circt::moore::StructExtractOp) [¶](#moorestruct_extract-circtmoorestructextractop)

*Obtain the value of a struct field*

Syntax:

```
operation ::= `moore.struct_extract` $input `,` $fieldName attr-dict `:` type($input) `->` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-29)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldName` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-124)

| Operand | Description |
| --- | --- |
| `input` | packed or unpacked struct type |

#### Results: [¶](#results-120)

| Result | Description |
| --- | --- |
| `result` | unpacked type |

### `moore.struct_extract_ref` (::circt::moore::StructExtractRefOp) [¶](#moorestruct_extract_ref-circtmoorestructextractrefop)

*Create a reference to a struct field*

Syntax:

```
operation ::= `moore.struct_extract_ref` $input `,` $fieldName attr-dict `:` type($input) `->` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `DestructurableAccessorOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-30)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldName` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-125)

| Operand | Description |
| --- | --- |
| `input` | ref of packed or unpacked struct type |

#### Results: [¶](#results-121)

| Result | Description |
| --- | --- |
| `result` |  |

### `moore.struct_inject` (::circt::moore::StructInjectOp) [¶](#moorestruct_inject-circtmoorestructinjectop)

*Update the value of a struct field*

Syntax:

```
operation ::= `moore.struct_inject` $input `,` $fieldName `,` $newValue attr-dict
              `:` type($input) `,` type($newValue)
```

Takes an existing struct value, sets one of its fields to a new value, and
returns the resulting struct value.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-31)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldName` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-126)

| Operand | Description |
| --- | --- |
| `input` | packed or unpacked struct type |
| `newValue` | unpacked type |

#### Results: [¶](#results-122)

| Result | Description |
| --- | --- |
| `result` | packed or unpacked struct type |

### `moore.sub` (::circt::moore::SubOp) [¶](#mooresub-circtmooresubop)

*Subtraction*

Syntax:

```
operation ::= `moore.sub` $lhs `,` $rhs attr-dict `:` type($result)
```

Subtract the right-hand side from the left-hand side operand. If any bit in
the two operands is Z or X, all bits in the result are X.

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-127)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-123)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.fsub` (::circt::moore::SubRealOp) [¶](#moorefsub-circtmooresubrealop)

*Subtraction; subtract the RHS real operand from the LHS
real operand.*

Syntax:

```
operation ::= `moore.fsub` $lhs `,` $rhs attr-dict `:` type($result)
```

See IEEE 1800-2017 § 11.4.3 “Arithmetic operators” and § 11.3.1 “Operators
with real operands”

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-128)

| Operand | Description |
| --- | --- |
| `lhs` | a SystemVerilog real type |
| `rhs` | a SystemVerilog real type |

#### Results: [¶](#results-124)

| Result | Description |
| --- | --- |
| `result` | a SystemVerilog real type |

### `moore.builtin.tan` (::circt::moore::TanBIOp) [¶](#moorebuiltintan-circtmooretanbiop)

*Tangent*

Syntax:

```
operation ::= `moore.builtin.tan` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-129)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-125)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.builtin.tanh` (::circt::moore::TanhBIOp) [¶](#moorebuiltintanh-circtmooretanhbiop)

*Hyperbolic tangent*

Syntax:

```
operation ::= `moore.builtin.tanh` $value attr-dict `:` type($value)
```

The system real math functions shall accept real value arguments and return
a real result type. Their behavior shall match the equivalent C language
standard math library function indicated.

See IEEE 1800-2017 § 20.8.2 “Real math functions”.

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-130)

| Operand | Description |
| --- | --- |
| `value` | 64-bit real type, aka. real |

#### Results: [¶](#results-126)

| Result | Description |
| --- | --- |
| `result` | 64-bit real type, aka. real |

### `moore.builtin.time` (::circt::moore::TimeBIOp) [¶](#moorebuiltintime-circtmooretimebiop)

*Return the current simulation time*

Syntax:

```
operation ::= `moore.builtin.time` attr-dict
```

Corresponds to the `$time` system function. Returns a int-rounded 64-bit
integer corresponding to the elapsed simulation time scaled by the
simulation’s timescale.

See IEEE 1800-2023 § 20.3 “Simulation time system functions”.

Interfaces: `InferTypeOpInterface`

#### Results: [¶](#results-127)

| Result | Description |
| --- | --- |
| `result` | the SystemVerilog `time` and `realtime` type |

### `moore.time_to_logic` (::circt::moore::TimeToLogicOp) [¶](#mooretime_to_logic-circtmooretimetologicop)

*Convert a time type to an integer number of femtoseconds*

Syntax:

```
operation ::= `moore.time_to_logic` $input attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-131)

| Operand | Description |
| --- | --- |
| `input` | the SystemVerilog `time` and `realtime` type |

#### Results: [¶](#results-128)

| Result | Description |
| --- | --- |
| `result` | 64-bit four-valued integer type |

### `moore.to_builtin_bool` (::circt::moore::ToBuiltinBoolOp) [¶](#mooreto_builtin_bool-circtmooretobuiltinboolop)

*Convert a `!moore.l1` or `!moore.i1` to a builtin `i1`*

Syntax:

```
operation ::= `moore.to_builtin_bool` $input attr-dict `:` type($input)
```

Maps `X` and `Z` to 0, and passes through 1 and 0 unchanged.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-132)

| Operand | Description |
| --- | --- |
| `input` | single bit type |

#### Results: [¶](#results-129)

| Result | Description |
| --- | --- |
| `result` | 1-bit signless integer |

### `moore.to_builtin_int` (::circt::moore::ToBuiltinIntOp) [¶](#mooreto_builtin_int-circtmooretobuiltinintop)

*Convert a `!moore.i<n>` to a builtin `i<n>`*

Syntax:

```
operation ::= `moore.to_builtin_int` $input attr-dict `:` type($input)
```

Casts from a two-valued Moore integer to a builtin integer.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-133)

| Operand | Description |
| --- | --- |
| `input` | two-valued integer type |

#### Results: [¶](#results-130)

| Result | Description |
| --- | --- |
| `result` | integer |

### `moore.trunc` (::circt::moore::TruncOp) [¶](#mooretrunc-circtmooretruncop)

*Truncate a value*

Syntax:

```
operation ::= `moore.trunc` $input attr-dict `:` type($input) `->` type($result)
```

Reduce the bit width of a value by removing some of its most significant
bits. This can only change the bit width of an integer type.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-134)

| Operand | Description |
| --- | --- |
| `input` | a simple bit vector type |

#### Results: [¶](#results-131)

| Result | Description |
| --- | --- |
| `result` | a simple bit vector type |

### `moore.uarray_cmp` (::circt::moore::UArrayCmpOp) [¶](#mooreuarray_cmp-circtmooreuarraycmpop)

Syntax:

```
operation ::= `moore.uarray_cmp` $predicate $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Performs an elementwise comparison of two unpacked arrays
using the specified predicate (for example, “eq” for equality or “ne” for inequality)
and returns a single bit result.
Its first argument is an attribute that defines which type of comparison is
performed. The following comparisons are supported:

* equal (mnemonic: `"eq"`; integer value: `0`)
* not equal (mnemonic: `"ne"`; integer value: `1`)

The result is `1` if the comparison is true and `0` otherwise.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-32)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `predicate` | circt::moore::UArrayCmpPredicateAttr | allowed 64-bit signless integer cases: 0, 1 |

#### Operands: [¶](#operands-135)

| Operand | Description |
| --- | --- |
| `lhs` | an unpacked array type |
| `rhs` | an unpacked array type |

#### Results: [¶](#results-132)

| Result | Description |
| --- | --- |
| `result` | `bit` type |

### `moore.uint_to_real` (::circt::moore::UIntToRealOp) [¶](#mooreuint_to_real-circtmooreuinttorealop)

*Convert an integer value to a real*

Syntax:

```
operation ::= `moore.uint_to_real` $input attr-dict `:` type($input) `->` type($result)
```

See IEEE 1800-2023 Section 6.24.1: “Cast operator” and 6.12:
“Real, shortreal and realtime data types”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-136)

| Operand | Description |
| --- | --- |
| `input` | two-valued integer type |

#### Results: [¶](#results-133)

| Result | Description |
| --- | --- |
| `result` | a SystemVerilog real type |

### `moore.uge` (::circt::moore::UgeOp) [¶](#mooreuge-circtmooreugeop)

*Unsigned greater than or equal comparison*

Syntax:

```
operation ::= `moore.uge` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the left- and right-hand side operand and returns a single bit 0,
1, or X result. If any bit in the two operands is Z or X, returns X.
Otherwise, if all bits are 0 or 1, `ult/slt`, `ule/sle`, `ugt/sgt`, and
`uge/sge` return whether the left-hand side is less than, less than or equal
to, greater than, or greater than or equal to the right-hand side,
respectively. `ult/slt` corresponds to the `<` operator, `ule/sle` to `<=`,
`ugt/sgt` to `>`, and `uge/sge` to `>=`.

See IEEE 1800-2017 § 11.4.4 “Relational operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-137)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-134)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.ugt` (::circt::moore::UgtOp) [¶](#mooreugt-circtmooreugtop)

*Unsigned greater than comparison*

Syntax:

```
operation ::= `moore.ugt` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the left- and right-hand side operand and returns a single bit 0,
1, or X result. If any bit in the two operands is Z or X, returns X.
Otherwise, if all bits are 0 or 1, `ult/slt`, `ule/sle`, `ugt/sgt`, and
`uge/sge` return whether the left-hand side is less than, less than or equal
to, greater than, or greater than or equal to the right-hand side,
respectively. `ult/slt` corresponds to the `<` operator, `ule/sle` to `<=`,
`ugt/sgt` to `>`, and `uge/sge` to `>=`.

See IEEE 1800-2017 § 11.4.4 “Relational operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-138)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-135)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.ule` (::circt::moore::UleOp) [¶](#mooreule-circtmooreuleop)

*Unsigned less than or equal comparison*

Syntax:

```
operation ::= `moore.ule` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the left- and right-hand side operand and returns a single bit 0,
1, or X result. If any bit in the two operands is Z or X, returns X.
Otherwise, if all bits are 0 or 1, `ult/slt`, `ule/sle`, `ugt/sgt`, and
`uge/sge` return whether the left-hand side is less than, less than or equal
to, greater than, or greater than or equal to the right-hand side,
respectively. `ult/slt` corresponds to the `<` operator, `ule/sle` to `<=`,
`ugt/sgt` to `>`, and `uge/sge` to `>=`.

See IEEE 1800-2017 § 11.4.4 “Relational operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-139)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-136)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.ult` (::circt::moore::UltOp) [¶](#mooreult-circtmooreultop)

*Unsigned less than comparison*

Syntax:

```
operation ::= `moore.ult` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the left- and right-hand side operand and returns a single bit 0,
1, or X result. If any bit in the two operands is Z or X, returns X.
Otherwise, if all bits are 0 or 1, `ult/slt`, `ule/sle`, `ugt/sgt`, and
`uge/sge` return whether the left-hand side is less than, less than or equal
to, greater than, or greater than or equal to the right-hand side,
respectively. `ult/slt` corresponds to the `<` operator, `ule/sle` to `<=`,
`ugt/sgt` to `>`, and `uge/sge` to `>=`.

See IEEE 1800-2017 § 11.4.4 “Relational operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-140)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-137)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.union_create` (::circt::moore::UnionCreateOp) [¶](#mooreunion_create-circtmooreunioncreateop)

*Union Create operation*

Syntax:

```
operation ::= `moore.union_create` $input attr-dict `:` type($input) `->` type($result)
```

A union is a data type that represents a single piece
of storage that can be accessed using one of
the named member data types. Only one of the
data types in the union can be used at a time.
By default, a union is unpacked, meaning there
is no required representation for how members
of the union are stored. Dynamic types and chandle
types can only be used in tagged unions.
See IEEE 1800-2017 § 7.3 “Unions”

Example:

```
typedef union { int i; shortreal f; } num; // named union type
num n;
n.f = 0.0; // set n in floating point format
typedef struct {
bit isfloat;
union { int i; shortreal f; } n;           // anonymous union type
} tagged_st;                               // named structure
```

See IEEE 1800-2017 § 7.3 “Unions”

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-33)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldName` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-141)

| Operand | Description |
| --- | --- |
| `input` | unpacked type |

#### Results: [¶](#results-138)

| Result | Description |
| --- | --- |
| `result` | unpacked type |

### `moore.union_extract` (::circt::moore::UnionExtractOp) [¶](#mooreunion_extract-circtmooreunionextractop)

*Union Extract operation*

Syntax:

```
operation ::= `moore.union_extract` $input `,`   $fieldName  attr-dict `:`
              type($input) `->` type($result)
```

With packed unions, writing one member and reading another is
independent of the byte ordering of the machine,
unlike an unpacked union of unpacked structures,
which are C-compatible and have members in ascending address order.
See IEEE 1800-2017 § 7.3.1 “Packed unions”

Example:

```
typedef union packed { // default unsigned
s_atmcell acell;
bit [423:0] bit_slice;
bit [52:0][7:0] byte_slice;
} u_atmcell;
u_atmcell u1;
byte b; bit [3:0] nib;
b = u1.bit_slice[415:408]; // same as b = u1.byte_slice[51];
nib = u1.bit_slice [423:420];
```

See IEEE 1800-2017 § 7.3.1 “Packed unions”

#### Attributes: [¶](#attributes-34)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldName` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-142)

| Operand | Description |
| --- | --- |
| `input` | unpacked type |

#### Results: [¶](#results-139)

| Result | Description |
| --- | --- |
| `result` | unpacked type |

### `moore.union_extract_ref` (::circt::moore::UnionExtractRefOp) [¶](#mooreunion_extract_ref-circtmooreunionextractrefop)

*Union Extract operation*

Syntax:

```
operation ::= `moore.union_extract_ref` $input `,`   $fieldName  attr-dict `:`
              type($input) `->` type($result)
```

#### Attributes: [¶](#attributes-35)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldName` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-143)

| Operand | Description |
| --- | --- |
| `input` |  |

#### Results: [¶](#results-140)

| Result | Description |
| --- | --- |
| `result` |  |

### `moore.unreachable` (::circt::moore::UnreachableOp) [¶](#mooreunreachable-circtmooreunreachableop)

*Terminates a block as unreachable*

Syntax:

```
operation ::= `moore.unreachable` attr-dict
```

The `moore.unreachable` op is used to indicate that control flow never
reaches the end of a block. This is useful for operations such as `$fatal`
which never return as they cause the simulator to shut down. Behavior is
undefined if control actually *does* reach this terminator, but should
probably crash the process with a useful error message.

Traits: `Terminator`

### `moore.builtin.urandom` (::circt::moore::UrandomBIOp) [¶](#moorebuiltinurandom-circtmooreurandombiop)

*Generate a pseudo-random unsigned integer (optionally seeded)*

Syntax:

```
operation ::= `moore.builtin.urandom` ($seed^)? attr-dict
```

Corresponds to the `$urandom` system function. Returns a 32-bit
pseudo-random integer. The seed is optional; when provided, it initializes
the generator. If not provided, treat it as 0 in semantics/lowering.

See IEEE 1800-2023 § 18.13 “Random number system functions and methods”.

Interfaces: `InferTypeOpInterface`

#### Operands: [¶](#operands-144)

| Operand | Description |
| --- | --- |
| `seed` | 32-bit two-valued integer type |

#### Results: [¶](#results-141)

| Result | Description |
| --- | --- |
| `result` | 32-bit two-valued integer type |

### `moore.vtable_entry` (::circt::moore::VTableEntryOp) [¶](#moorevtable_entry-circtmoorevtableentryop)

*One vtable slot pointing to the selected implementation*

Syntax:

```
operation ::= `moore.vtable_entry` $name `->` $target attr-dict
```

Each entry references the resolved implementation with a SymbolRefAttr.

Interfaces: `SymbolUserOpInterface`

#### Attributes: [¶](#attributes-36)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `target` | ::mlir::SymbolRefAttr | symbol reference attribute |

### `moore.vtable.load_method` (::circt::moore::VTableLoadMethodOp) [¶](#moorevtableload_method-circtmoorevtableloadmethodop)

*Load a virtual method entry from a vtable.*

Syntax:

```
operation ::= `moore.vtable.load_method` $object `:` $methodSym `of` type($object) `->` type($result) attr-dict
```

Loads a virtual method function pointer from a vtable.
The verifier resolves the symbols in the symbol table and ensures that the
result function type matches the erased ABI.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-37)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `methodSym` | ::mlir::SymbolRefAttr | symbol reference attribute |

#### Operands: [¶](#operands-145)

| Operand | Description |
| --- | --- |
| `object` | Class object handle type, pointing to an object on the heap. |

#### Results: [¶](#results-142)

| Result | Description |
| --- | --- |
| `result` |  |

### `moore.vtable` (::circt::moore::VTableOp) [¶](#moorevtable-circtmoorevtableop)

*Virtual dispatch table*

Syntax:

```
operation ::= `moore.vtable` $sym_name attr-dict-with-keyword $body
```

Lives at module top-level. Contains `moore.vtable_entry` and nested
`moore.vtable` ops for base segments. Holds the dispatch targets for every
possible virtual method call of a class and its ancestors.

Traits: `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`

Interfaces: `SymbolUserOpInterface`

#### Attributes: [¶](#attributes-38)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::SymbolRefAttr | symbol reference attribute |

### `moore.variable` (::circt::moore::VariableOp) [¶](#moorevariable-circtmoorevariableop)

*A variable declaration*

Syntax:

```
operation ::= `moore.variable` `` custom<ImplicitSSAName>($name) ($initial^)? attr-dict
              `:` type($result)
```

See IEEE 1800-2017 § 6.8 “Variable declarations”.

Interfaces: `DestructurableAllocationOpInterface`, `OpAsmOpInterface`, `PromotableAllocationOpInterface`

#### Attributes: [¶](#attributes-39)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-146)

| Operand | Description |
| --- | --- |
| `initial` | unpacked type |

#### Results: [¶](#results-143)

| Result | Description |
| --- | --- |
| `result` |  |

### `moore.wait_delay` (::circt::moore::WaitDelayOp) [¶](#moorewait_delay-circtmoorewaitdelayop)

*Suspend execution for a given amount of time*

Syntax:

```
operation ::= `moore.wait_delay` $delay attr-dict
```

The `moore.wait_delay` op suspends execution of the current process for the
amount of time specified by its operand. Corresponds to the `#` delay
control in SystemVerilog.

See IEEE 1800-2017 § 9.4.1 “Delay control”.

#### Operands: [¶](#operands-147)

| Operand | Description |
| --- | --- |
| `delay` | the SystemVerilog `time` and `realtime` type |

### `moore.wait_event` (::circt::moore::WaitEventOp) [¶](#moorewait_event-circtmoorewaiteventop)

*Suspend execution until an event occurs*

Syntax:

```
operation ::= `moore.wait_event` attr-dict-with-keyword $body
```

The `moore.wait_event` op suspends execution of the current process until
its body signals that an event has been the detected. Conceptually, the body
of this op is executed whenever any potentially relevant signal has changed.
If one of the contained `moore.detect_event` ops detect an event, execution
resumes after the `moore.wait_event` operation. If no event is detected, the
current process remains suspended.

Example corresponding to the SystemVerilog `@(posedge x, negedge y iff z)`:

```
moore.wait_event {
  %0 = moore.read %x : <i1>
  %1 = moore.read %y : <i1>
  %2 = moore.read %z : <i1>
  moore.detect_event posedge %0 : i1
  moore.detect_event negedge %1 if %2 : i1
}
```

The body may also contain any operations necessary to evaluate the event
conditions. For example, the SV `@(posedge ~x iff i == 42)`:

```
moore.wait_event {
  %0 = moore.read %x : <i1>
  %1 = moore.not %0 : i1
  %2 = moore.read %i : <i19>
  %3 = moore.constant 42 : i19
  %4 = moore.eq %2, %3 : i19
  moore.detect_event posedge %0 if %4 : i1
}
```

See IEEE 1800-2017 § 9.4.2 “Event control”.

Traits: `NoRegionArguments`, `NoTerminator`, `SingleBlock`

### `moore.wildcard_eq` (::circt::moore::WildcardEqOp) [¶](#moorewildcard_eq-circtmoorewildcardeqop)

*Wildcard equality*

Syntax:

```
operation ::= `moore.wildcard_eq` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the bits in the left- and right-hand side operand and returns a
single bit 0, 1, or X result. If any bit in the left-hand side is Z or X,
returns X. Performs the same comparison as the `eq` and `ne` operations, but
all right-hand side bits that are X or Z are skipped. Therefore, X and Z in
the right-hand side act as wildcards or “don’t care” values. `wildcard_eq`
corresponds to the `==?` operator and `wildcard_ne` to the `!=?` operator.

See IEEE 1800-2017 § 11.4.6 “Wildcard equality operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-148)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-144)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.wildcard_ne` (::circt::moore::WildcardNeOp) [¶](#moorewildcard_ne-circtmoorewildcardneop)

*Wildcard inequality*

Syntax:

```
operation ::= `moore.wildcard_ne` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```

Compares the bits in the left- and right-hand side operand and returns a
single bit 0, 1, or X result. If any bit in the left-hand side is Z or X,
returns X. Performs the same comparison as the `eq` and `ne` operations, but
all right-hand side bits that are X or Z are skipped. Therefore, X and Z in
the right-hand side act as wildcards or “don’t care” values. `wildcard_eq`
corresponds to the `==?` operator and `wildcard_ne` to the `!=?` operator.

See IEEE 1800-2017 § 11.4.6 “Wildcard equality operators”.

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-149)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-145)

| Result | Description |
| --- | --- |
| `result` | single bit type |

### `moore.xor` (::circt::moore::XorOp) [¶](#moorexor-circtmoorexorop)

*Bitwise XOR operation*

Syntax:

```
operation ::= `moore.xor` $lhs `,` $rhs attr-dict `:` type($result)
```

Applies the boolean XOR operation to each pair of corresponding bits in the
left- and right-hand side operand. Corresponds to the `^` operator.

See IEEE 1800-2017 § 11.4.8 “Bitwise operators”.

|  | 0 | 1 | X | Z |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1 | X | X |
| 1 | 1 | 0 | X | X |
| X | X | X | X | X |
| Z | X | X | X | X |

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-150)

| Operand | Description |
| --- | --- |
| `lhs` | simple bit vector type |
| `rhs` | simple bit vector type |

#### Results: [¶](#results-146)

| Result | Description |
| --- | --- |
| `result` | simple bit vector type |

### `moore.yield` (::circt::moore::YieldOp) [¶](#mooreyield-circtmooreyieldop)

*Conditional yield and termination operation*

Syntax:

```
operation ::= `moore.yield` attr-dict $result `:` type($result)
```

“moore.yield” yields an SSA value from the Moore dialect op region and
terminates the regions. The semantics of how the values are yielded is
defined by the parent operation.
If “moore.yield” has any operands, the operands must match the parent
operation’s results.
If the parent operation defines no values, then the “moore.yield” may be
left out in the custom syntax and the builders will insert one implicitly.
Otherwise, it has to be present in the syntax to indicate which values are
yielded.

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<ConditionalOp, GlobalVariableOp>`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-151)

| Operand | Description |
| --- | --- |
| `result` | unpacked type |

### `moore.zext` (::circt::moore::ZExtOp) [¶](#moorezext-circtmoorezextop)

*Zero-extend a value*

Syntax:

```
operation ::= `moore.zext` $input attr-dict `:` type($input) `->` type($result)
```

Increase the bit width of a value by inserting additional zero most
significant bits. This keeps the unsigned value constant.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-152)

| Operand | Description |
| --- | --- |
| `input` | a simple bit vector type |

#### Results: [¶](#results-147)

| Result | Description |
| --- | --- |
| `result` | a simple bit vector type |

 [Prev - LTL Dialect](https://circt.llvm.org/docs/Dialects/LTL/ "LTL Dialect")
[Next - Random Test Generation (RTG) Rationale](https://circt.llvm.org/docs/Dialects/RTG/ "Random Test Generation (RTG) Rationale") 

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