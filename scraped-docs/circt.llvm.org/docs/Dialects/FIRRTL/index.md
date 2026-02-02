'firrtl' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'firrtl' Dialect
================

This dialect defines the `firrtl` dialect, which is used to lower from Chisel
code to Verilog. For more information, see the
[FIRRTL GitHub
page](https://github.com/freechipsproject/firrtl).

* [Operation Definitions – Structure](#operation-definitions----structure)
  + [`firrtl.circuit` (::circt::firrtl::CircuitOp)](#firrtlcircuit-circtfirrtlcircuitop)
  + [`firrtl.class` (::circt::firrtl::ClassOp)](#firrtlclass-circtfirrtlclassop)
  + [`firrtl.domain` (::circt::firrtl::DomainOp)](#firrtldomain-circtfirrtldomainop)
  + [`firrtl.extclass` (::circt::firrtl::ExtClassOp)](#firrtlextclass-circtfirrtlextclassop)
  + [`firrtl.extmodule` (::circt::firrtl::FExtModuleOp)](#firrtlextmodule-circtfirrtlfextmoduleop)
  + [`firrtl.intmodule` (::circt::firrtl::FIntModuleOp)](#firrtlintmodule-circtfirrtlfintmoduleop)
  + [`firrtl.memmodule` (::circt::firrtl::FMemModuleOp)](#firrtlmemmodule-circtfirrtlfmemmoduleop)
  + [`firrtl.module` (::circt::firrtl::FModuleOp)](#firrtlmodule-circtfirrtlfmoduleop)
  + [`firrtl.formal` (::circt::firrtl::FormalOp)](#firrtlformal-circtfirrtlformalop)
  + [`firrtl.layer` (::circt::firrtl::LayerOp)](#firrtllayer-circtfirrtllayerop)
  + [`firrtl.option_case` (::circt::firrtl::OptionCaseOp)](#firrtloption_case-circtfirrtloptioncaseop)
  + [`firrtl.option` (::circt::firrtl::OptionOp)](#firrtloption-circtfirrtloptionop)
  + [`firrtl.simulation` (::circt::firrtl::SimulationOp)](#firrtlsimulation-circtfirrtlsimulationop)
* [Operation Definitions – Declarations](#operation-definitions----declarations)
  + [`firrtl.contract` (::circt::firrtl::ContractOp)](#firrtlcontract-circtfirrtlcontractop)
  + [`firrtl.domain.anon` (::circt::firrtl::DomainCreateAnonOp)](#firrtldomainanon-circtfirrtldomaincreateanonop)
  + [`firrtl.instance_choice` (::circt::firrtl::InstanceChoiceOp)](#firrtlinstance_choice-circtfirrtlinstancechoiceop)
  + [`firrtl.instance` (::circt::firrtl::InstanceOp)](#firrtlinstance-circtfirrtlinstanceop)
  + [`firrtl.mem` (::circt::firrtl::MemOp)](#firrtlmem-circtfirrtlmemop)
  + [`firrtl.node` (::circt::firrtl::NodeOp)](#firrtlnode-circtfirrtlnodeop)
  + [`firrtl.object` (::circt::firrtl::ObjectOp)](#firrtlobject-circtfirrtlobjectop)
  + [`firrtl.reg` (::circt::firrtl::RegOp)](#firrtlreg-circtfirrtlregop)
  + [`firrtl.regreset` (::circt::firrtl::RegResetOp)](#firrtlregreset-circtfirrtlregresetop)
  + [`firrtl.wire` (::circt::firrtl::WireOp)](#firrtlwire-circtfirrtlwireop)
* [Statement Operation – Statements](#statement-operation----statements)
  + [`firrtl.assert` (::circt::firrtl::AssertOp)](#firrtlassert-circtfirrtlassertop)
  + [`firrtl.assume` (::circt::firrtl::AssumeOp)](#firrtlassume-circtfirrtlassumeop)
  + [`firrtl.attach` (::circt::firrtl::AttachOp)](#firrtlattach-circtfirrtlattachop)
  + [`firrtl.bind` (::circt::firrtl::BindOp)](#firrtlbind-circtfirrtlbindop)
  + [`firrtl.connect` (::circt::firrtl::ConnectOp)](#firrtlconnect-circtfirrtlconnectop)
  + [`firrtl.cover` (::circt::firrtl::CoverOp)](#firrtlcover-circtfirrtlcoverop)
  + [`firrtl.domain.define` (::circt::firrtl::DomainDefineOp)](#firrtldomaindefine-circtfirrtldomaindefineop)
  + [`firrtl.fflush` (::circt::firrtl::FFlushOp)](#firrtlfflush-circtfirrtlfflushop)
  + [`firrtl.fprintf` (::circt::firrtl::FPrintFOp)](#firrtlfprintf-circtfirrtlfprintfop)
  + [`firrtl.force` (::circt::firrtl::ForceOp)](#firrtlforce-circtfirrtlforceop)
  + [`firrtl.layerblock` (::circt::firrtl::LayerBlockOp)](#firrtllayerblock-circtfirrtllayerblockop)
  + [`firrtl.match` (::circt::firrtl::MatchOp)](#firrtlmatch-circtfirrtlmatchop)
  + [`firrtl.matchingconnect` (::circt::firrtl::MatchingConnectOp)](#firrtlmatchingconnect-circtfirrtlmatchingconnectop)
  + [`firrtl.printf` (::circt::firrtl::PrintFOp)](#firrtlprintf-circtfirrtlprintfop)
  + [`firrtl.propassign` (::circt::firrtl::PropAssignOp)](#firrtlpropassign-circtfirrtlpropassignop)
  + [`firrtl.ref.define` (::circt::firrtl::RefDefineOp)](#firrtlrefdefine-circtfirrtlrefdefineop)
  + [`firrtl.ref.force_initial` (::circt::firrtl::RefForceInitialOp)](#firrtlrefforce_initial-circtfirrtlrefforceinitialop)
  + [`firrtl.ref.force` (::circt::firrtl::RefForceOp)](#firrtlrefforce-circtfirrtlrefforceop)
  + [`firrtl.ref.release_initial` (::circt::firrtl::RefReleaseInitialOp)](#firrtlrefrelease_initial-circtfirrtlrefreleaseinitialop)
  + [`firrtl.ref.release` (::circt::firrtl::RefReleaseOp)](#firrtlrefrelease-circtfirrtlrefreleaseop)
  + [`firrtl.skip` (::circt::firrtl::SkipOp)](#firrtlskip-circtfirrtlskipop)
  + [`firrtl.stop` (::circt::firrtl::StopOp)](#firrtlstop-circtfirrtlstopop)
  + [`firrtl.int.verif.assert` (::circt::firrtl::VerifAssertIntrinsicOp)](#firrtlintverifassert-circtfirrtlverifassertintrinsicop)
  + [`firrtl.int.verif.assume` (::circt::firrtl::VerifAssumeIntrinsicOp)](#firrtlintverifassume-circtfirrtlverifassumeintrinsicop)
  + [`firrtl.int.verif.cover` (::circt::firrtl::VerifCoverIntrinsicOp)](#firrtlintverifcover-circtfirrtlverifcoverintrinsicop)
  + [`firrtl.int.verif.ensure` (::circt::firrtl::VerifEnsureIntrinsicOp)](#firrtlintverifensure-circtfirrtlverifensureintrinsicop)
  + [`firrtl.int.verif.require` (::circt::firrtl::VerifRequireIntrinsicOp)](#firrtlintverifrequire-circtfirrtlverifrequireintrinsicop)
  + [`firrtl.when` (::circt::firrtl::WhenOp)](#firrtlwhen-circtfirrtlwhenop)
* [Operation Definitions – Expressions](#operation-definitions----expressions)
  + [`firrtl.add` (::circt::firrtl::AddPrimOp)](#firrtladd-circtfirrtladdprimop)
  + [`firrtl.aggregateconstant` (::circt::firrtl::AggregateConstantOp)](#firrtlaggregateconstant-circtfirrtlaggregateconstantop)
  + [`firrtl.and` (::circt::firrtl::AndPrimOp)](#firrtland-circtfirrtlandprimop)
  + [`firrtl.andr` (::circt::firrtl::AndRPrimOp)](#firrtlandr-circtfirrtlandrprimop)
  + [`firrtl.asAsyncReset` (::circt::firrtl::AsAsyncResetPrimOp)](#firrtlasasyncreset-circtfirrtlasasyncresetprimop)
  + [`firrtl.asClock` (::circt::firrtl::AsClockPrimOp)](#firrtlasclock-circtfirrtlasclockprimop)
  + [`firrtl.asSInt` (::circt::firrtl::AsSIntPrimOp)](#firrtlassint-circtfirrtlassintprimop)
  + [`firrtl.asUInt` (::circt::firrtl::AsUIntPrimOp)](#firrtlasuint-circtfirrtlasuintprimop)
  + [`firrtl.bitcast` (::circt::firrtl::BitCastOp)](#firrtlbitcast-circtfirrtlbitcastop)
  + [`firrtl.bits` (::circt::firrtl::BitsPrimOp)](#firrtlbits-circtfirrtlbitsprimop)
  + [`firrtl.bool` (::circt::firrtl::BoolConstantOp)](#firrtlbool-circtfirrtlboolconstantop)
  + [`firrtl.bundlecreate` (::circt::firrtl::BundleCreateOp)](#firrtlbundlecreate-circtfirrtlbundlecreateop)
  + [`firrtl.cat` (::circt::firrtl::CatPrimOp)](#firrtlcat-circtfirrtlcatprimop)
  + [`firrtl.constCast` (::circt::firrtl::ConstCastOp)](#firrtlconstcast-circtfirrtlconstcastop)
  + [`firrtl.constant` (::circt::firrtl::ConstantOp)](#firrtlconstant-circtfirrtlconstantop)
  + [`firrtl.cvt` (::circt::firrtl::CvtPrimOp)](#firrtlcvt-circtfirrtlcvtprimop)
  + [`firrtl.dshl` (::circt::firrtl::DShlPrimOp)](#firrtldshl-circtfirrtldshlprimop)
  + [`firrtl.dshlw` (::circt::firrtl::DShlwPrimOp)](#firrtldshlw-circtfirrtldshlwprimop)
  + [`firrtl.dshr` (::circt::firrtl::DShrPrimOp)](#firrtldshr-circtfirrtldshrprimop)
  + [`firrtl.div` (::circt::firrtl::DivPrimOp)](#firrtldiv-circtfirrtldivprimop)
  + [`firrtl.double` (::circt::firrtl::DoubleConstantOp)](#firrtldouble-circtfirrtldoubleconstantop)
  + [`firrtl.eq` (::circt::firrtl::EQPrimOp)](#firrtleq-circtfirrtleqprimop)
  + [`firrtl.elementwise_and` (::circt::firrtl::ElementwiseAndPrimOp)](#firrtlelementwise_and-circtfirrtlelementwiseandprimop)
  + [`firrtl.elementwise_or` (::circt::firrtl::ElementwiseOrPrimOp)](#firrtlelementwise_or-circtfirrtlelementwiseorprimop)
  + [`firrtl.elementwise_xor` (::circt::firrtl::ElementwiseXorPrimOp)](#firrtlelementwise_xor-circtfirrtlelementwisexorprimop)
  + [`firrtl.enumcreate` (::circt::firrtl::FEnumCreateOp)](#firrtlenumcreate-circtfirrtlfenumcreateop)
  + [`firrtl.integer` (::circt::firrtl::FIntegerConstantOp)](#firrtlinteger-circtfirrtlfintegerconstantop)
  + [`firrtl.geq` (::circt::firrtl::GEQPrimOp)](#firrtlgeq-circtfirrtlgeqprimop)
  + [`firrtl.gt` (::circt::firrtl::GTPrimOp)](#firrtlgt-circtfirrtlgtprimop)
  + [`firrtl.hwStructCast` (::circt::firrtl::HWStructCastOp)](#firrtlhwstructcast-circtfirrtlhwstructcastop)
  + [`firrtl.head` (::circt::firrtl::HeadPrimOp)](#firrtlhead-circtfirrtlheadprimop)
  + [`firrtl.fstring.hierarchicalmodulename` (::circt::firrtl::HierarchicalModuleNameOp)](#firrtlfstringhierarchicalmodulename-circtfirrtlhierarchicalmodulenameop)
  + [`firrtl.integer.add` (::circt::firrtl::IntegerAddOp)](#firrtlintegeradd-circtfirrtlintegeraddop)
  + [`firrtl.integer.mul` (::circt::firrtl::IntegerMulOp)](#firrtlintegermul-circtfirrtlintegermulop)
  + [`firrtl.integer.shl` (::circt::firrtl::IntegerShlOp)](#firrtlintegershl-circtfirrtlintegershlop)
  + [`firrtl.integer.shr` (::circt::firrtl::IntegerShrOp)](#firrtlintegershr-circtfirrtlintegershrop)
  + [`firrtl.invalidvalue` (::circt::firrtl::InvalidValueOp)](#firrtlinvalidvalue-circtfirrtlinvalidvalueop)
  + [`firrtl.istag` (::circt::firrtl::IsTagOp)](#firrtlistag-circtfirrtlistagop)
  + [`firrtl.leq` (::circt::firrtl::LEQPrimOp)](#firrtlleq-circtfirrtlleqprimop)
  + [`firrtl.int.ltl.and` (::circt::firrtl::LTLAndIntrinsicOp)](#firrtlintltland-circtfirrtlltlandintrinsicop)
  + [`firrtl.int.ltl.clock` (::circt::firrtl::LTLClockIntrinsicOp)](#firrtlintltlclock-circtfirrtlltlclockintrinsicop)
  + [`firrtl.int.ltl.concat` (::circt::firrtl::LTLConcatIntrinsicOp)](#firrtlintltlconcat-circtfirrtlltlconcatintrinsicop)
  + [`firrtl.int.ltl.delay` (::circt::firrtl::LTLDelayIntrinsicOp)](#firrtlintltldelay-circtfirrtlltldelayintrinsicop)
  + [`firrtl.int.ltl.eventually` (::circt::firrtl::LTLEventuallyIntrinsicOp)](#firrtlintltleventually-circtfirrtlltleventuallyintrinsicop)
  + [`firrtl.int.ltl.goto_repeat` (::circt::firrtl::LTLGoToRepeatIntrinsicOp)](#firrtlintltlgoto_repeat-circtfirrtlltlgotorepeatintrinsicop)
  + [`firrtl.int.ltl.implication` (::circt::firrtl::LTLImplicationIntrinsicOp)](#firrtlintltlimplication-circtfirrtlltlimplicationintrinsicop)
  + [`firrtl.int.ltl.intersect` (::circt::firrtl::LTLIntersectIntrinsicOp)](#firrtlintltlintersect-circtfirrtlltlintersectintrinsicop)
  + [`firrtl.int.ltl.non_consecutive_repeat` (::circt::firrtl::LTLNonConsecutiveRepeatIntrinsicOp)](#firrtlintltlnon_consecutive_repeat-circtfirrtlltlnonconsecutiverepeatintrinsicop)
  + [`firrtl.int.ltl.not` (::circt::firrtl::LTLNotIntrinsicOp)](#firrtlintltlnot-circtfirrtlltlnotintrinsicop)
  + [`firrtl.int.ltl.or` (::circt::firrtl::LTLOrIntrinsicOp)](#firrtlintltlor-circtfirrtlltlorintrinsicop)
  + [`firrtl.int.ltl.repeat` (::circt::firrtl::LTLRepeatIntrinsicOp)](#firrtlintltlrepeat-circtfirrtlltlrepeatintrinsicop)
  + [`firrtl.int.ltl.until` (::circt::firrtl::LTLUntilIntrinsicOp)](#firrtlintltluntil-circtfirrtlltluntilintrinsicop)
  + [`firrtl.lt` (::circt::firrtl::LTPrimOp)](#firrtllt-circtfirrtlltprimop)
  + [`firrtl.list.concat` (::circt::firrtl::ListConcatOp)](#firrtllistconcat-circtfirrtllistconcatop)
  + [`firrtl.list.create` (::circt::firrtl::ListCreateOp)](#firrtllistcreate-circtfirrtllistcreateop)
  + [`firrtl.mul` (::circt::firrtl::MulPrimOp)](#firrtlmul-circtfirrtlmulprimop)
  + [`firrtl.multibit_mux` (::circt::firrtl::MultibitMuxOp)](#firrtlmultibit_mux-circtfirrtlmultibitmuxop)
  + [`firrtl.int.mux2cell` (::circt::firrtl::Mux2CellIntrinsicOp)](#firrtlintmux2cell-circtfirrtlmux2cellintrinsicop)
  + [`firrtl.int.mux4cell` (::circt::firrtl::Mux4CellIntrinsicOp)](#firrtlintmux4cell-circtfirrtlmux4cellintrinsicop)
  + [`firrtl.mux` (::circt::firrtl::MuxPrimOp)](#firrtlmux-circtfirrtlmuxprimop)
  + [`firrtl.neq` (::circt::firrtl::NEQPrimOp)](#firrtlneq-circtfirrtlneqprimop)
  + [`firrtl.neg` (::circt::firrtl::NegPrimOp)](#firrtlneg-circtfirrtlnegprimop)
  + [`firrtl.not` (::circt::firrtl::NotPrimOp)](#firrtlnot-circtfirrtlnotprimop)
  + [`firrtl.object.anyref_cast` (::circt::firrtl::ObjectAnyRefCastOp)](#firrtlobjectanyref_cast-circtfirrtlobjectanyrefcastop)
  + [`firrtl.object.subfield` (::circt::firrtl::ObjectSubfieldOp)](#firrtlobjectsubfield-circtfirrtlobjectsubfieldop)
  + [`firrtl.opensubfield` (::circt::firrtl::OpenSubfieldOp)](#firrtlopensubfield-circtfirrtlopensubfieldop)
  + [`firrtl.opensubindex` (::circt::firrtl::OpenSubindexOp)](#firrtlopensubindex-circtfirrtlopensubindexop)
  + [`firrtl.or` (::circt::firrtl::OrPrimOp)](#firrtlor-circtfirrtlorprimop)
  + [`firrtl.orr` (::circt::firrtl::OrRPrimOp)](#firrtlorr-circtfirrtlorrprimop)
  + [`firrtl.pad` (::circt::firrtl::PadPrimOp)](#firrtlpad-circtfirrtlpadprimop)
  + [`firrtl.path` (::circt::firrtl::PathOp)](#firrtlpath-circtfirrtlpathop)
  + [`firrtl.ref.rwprobe` (::circt::firrtl::RWProbeOp)](#firrtlrefrwprobe-circtfirrtlrwprobeop)
  + [`firrtl.ref.cast` (::circt::firrtl::RefCastOp)](#firrtlrefcast-circtfirrtlrefcastop)
  + [`firrtl.ref.resolve` (::circt::firrtl::RefResolveOp)](#firrtlrefresolve-circtfirrtlrefresolveop)
  + [`firrtl.ref.send` (::circt::firrtl::RefSendOp)](#firrtlrefsend-circtfirrtlrefsendop)
  + [`firrtl.ref.sub` (::circt::firrtl::RefSubOp)](#firrtlrefsub-circtfirrtlrefsubop)
  + [`firrtl.rem` (::circt::firrtl::RemPrimOp)](#firrtlrem-circtfirrtlremprimop)
  + [`firrtl.shl` (::circt::firrtl::ShlPrimOp)](#firrtlshl-circtfirrtlshlprimop)
  + [`firrtl.shr` (::circt::firrtl::ShrPrimOp)](#firrtlshr-circtfirrtlshrprimop)
  + [`firrtl.int.sizeof` (::circt::firrtl::SizeOfIntrinsicOp)](#firrtlintsizeof-circtfirrtlsizeofintrinsicop)
  + [`firrtl.specialconstant` (::circt::firrtl::SpecialConstantOp)](#firrtlspecialconstant-circtfirrtlspecialconstantop)
  + [`firrtl.string` (::circt::firrtl::StringConstantOp)](#firrtlstring-circtfirrtlstringconstantop)
  + [`firrtl.sub` (::circt::firrtl::SubPrimOp)](#firrtlsub-circtfirrtlsubprimop)
  + [`firrtl.subaccess` (::circt::firrtl::SubaccessOp)](#firrtlsubaccess-circtfirrtlsubaccessop)
  + [`firrtl.subfield` (::circt::firrtl::SubfieldOp)](#firrtlsubfield-circtfirrtlsubfieldop)
  + [`firrtl.subindex` (::circt::firrtl::SubindexOp)](#firrtlsubindex-circtfirrtlsubindexop)
  + [`firrtl.subtag` (::circt::firrtl::SubtagOp)](#firrtlsubtag-circtfirrtlsubtagop)
  + [`firrtl.tagextract` (::circt::firrtl::TagExtractOp)](#firrtltagextract-circtfirrtltagextractop)
  + [`firrtl.tail` (::circt::firrtl::TailPrimOp)](#firrtltail-circtfirrtltailprimop)
  + [`firrtl.fstring.time` (::circt::firrtl::TimeOp)](#firrtlfstringtime-circtfirrtltimeop)
  + [`firrtl.resetCast` (::circt::firrtl::UninferredResetCastOp)](#firrtlresetcast-circtfirrtluninferredresetcastop)
  + [`firrtl.unknown` (::circt::firrtl::UnknownValueOp)](#firrtlunknown-circtfirrtlunknownvalueop)
  + [`firrtl.unresolved_path` (::circt::firrtl::UnresolvedPathOp)](#firrtlunresolved_path-circtfirrtlunresolvedpathop)
  + [`firrtl.unsafe_domain_cast` (::circt::firrtl::UnsafeDomainCastOp)](#firrtlunsafe_domain_cast-circtfirrtlunsafedomaincastop)
  + [`firrtl.vectorcreate` (::circt::firrtl::VectorCreateOp)](#firrtlvectorcreate-circtfirrtlvectorcreateop)
  + [`firrtl.verbatim.expr` (::circt::firrtl::VerbatimExprOp)](#firrtlverbatimexpr-circtfirrtlverbatimexprop)
  + [`firrtl.verbatim.wire` (::circt::firrtl::VerbatimWireOp)](#firrtlverbatimwire-circtfirrtlverbatimwireop)
  + [`firrtl.xmr.deref` (::circt::firrtl::XMRDerefOp)](#firrtlxmrderef-circtfirrtlxmrderefop)
  + [`firrtl.xmr.ref` (::circt::firrtl::XMRRefOp)](#firrtlxmrref-circtfirrtlxmrrefop)
  + [`firrtl.xor` (::circt::firrtl::XorPrimOp)](#firrtlxor-circtfirrtlxorprimop)
  + [`firrtl.xorr` (::circt::firrtl::XorRPrimOp)](#firrtlxorr-circtfirrtlxorrprimop)
* [Operation Definitions – Intrinsics](#operation-definitions----intrinsics)
  + [`firrtl.int.clock_div` (::circt::firrtl::ClockDividerIntrinsicOp)](#firrtlintclock_div-circtfirrtlclockdividerintrinsicop)
  + [`firrtl.int.clock_gate` (::circt::firrtl::ClockGateIntrinsicOp)](#firrtlintclock_gate-circtfirrtlclockgateintrinsicop)
  + [`firrtl.int.clock_inv` (::circt::firrtl::ClockInverterIntrinsicOp)](#firrtlintclock_inv-circtfirrtlclockinverterintrinsicop)
  + [`firrtl.int.dpi.call` (::circt::firrtl::DPICallIntrinsicOp)](#firrtlintdpicall-circtfirrtldpicallintrinsicop)
  + [`firrtl.int.fpga_probe` (::circt::firrtl::FPGAProbeIntrinsicOp)](#firrtlintfpga_probe-circtfirrtlfpgaprobeintrinsicop)
  + [`firrtl.int.generic` (::circt::firrtl::GenericIntrinsicOp)](#firrtlintgeneric-circtfirrtlgenericintrinsicop)
  + [`firrtl.int.has_been_reset` (::circt::firrtl::HasBeenResetIntrinsicOp)](#firrtlinthas_been_reset-circtfirrtlhasbeenresetintrinsicop)
  + [`firrtl.int.isX` (::circt::firrtl::IsXIntrinsicOp)](#firrtlintisx-circtfirrtlisxintrinsicop)
  + [`firrtl.int.plusargs.test` (::circt::firrtl::PlusArgsTestIntrinsicOp)](#firrtlintplusargstest-circtfirrtlplusargstestintrinsicop)
  + [`firrtl.int.plusargs.value` (::circt::firrtl::PlusArgsValueIntrinsicOp)](#firrtlintplusargsvalue-circtfirrtlplusargsvalueintrinsicop)
  + [`firrtl.int.unclocked_assume` (::circt::firrtl::UnclockedAssumeIntrinsicOp)](#firrtlintunclocked_assume-circtfirrtlunclockedassumeintrinsicop)
  + [`firrtl.view` (::circt::firrtl::ViewIntrinsicOp)](#firrtlview-circtfirrtlviewintrinsicop)
* [Type Definitions](#type-definitions)
  + [AnalogType](#analogtype)
  + [AnyRefType](#anyreftype)
  + [AsyncResetType](#asyncresettype)
  + [BaseTypeAliasType](#basetypealiastype)
  + [BoolType](#booltype)
  + [BundleType](#bundletype)
  + [ClassType](#classtype)
  + [ClockType](#clocktype)
  + [DomainType](#domaintype)
  + [DoubleType](#doubletype)
  + [FEnumType](#fenumtype)
  + [FStringType](#fstringtype)
  + [FVectorType](#fvectortype)
  + [FIntegerType](#fintegertype)
  + [LHSType](#lhstype)
  + [ListType](#listtype)
  + [OpenBundleType](#openbundletype)
  + [OpenVectorType](#openvectortype)
  + [PathType](#pathtype)
  + [RefType](#reftype)
  + [ResetType](#resettype)
  + [SIntType](#sinttype)
  + [StringType](#stringtype)
  + [UIntType](#uinttype)
* [Attribute Definitions](#attribute-definitions)
  + [AugmentedBundleTypeAttr](#augmentedbundletypeattr)
  + [AugmentedGroundTypeAttr](#augmentedgroundtypeattr)
  + [AugmentedVectorTypeAttr](#augmentedvectortypeattr)
  + [DomainFieldAttr](#domainfieldattr)
  + [MemoryInitAttr](#memoryinitattr)
  + [ParamDeclAttr](#paramdeclattr)
* [ClassLike (`ClassLike`)](#classlike-classlike)
  + [Methods:](#methods)
* [FConnectLike (`FConnectLike`)](#fconnectlike-fconnectlike)
  + [Methods:](#methods-1)
* [FInstanceLike (`FInstanceLike`)](#finstancelike-finstancelike)
  + [Methods:](#methods-2)
* [FModuleLike (`FModuleLike`)](#fmodulelike-fmodulelike)
  + [Methods:](#methods-3)
* [FNamableOp (`FNamableOp`)](#fnamableop-fnamableop)
  + [Methods:](#methods-4)
* [Forceable (`Forceable`)](#forceable-forceable)
  + [Methods:](#methods-5)

Operation Definitions – Structure
---------------------------------

### `firrtl.circuit` (::circt::firrtl::CircuitOp)

*FIRRTL Circuit*

Syntax:

```
operation ::= `firrtl.circuit` $name `` custom<CircuitOpAttrs>(attr-dict) $body
```

The “firrtl.circuit” operation represents an overall Verilog circuit,
containing a list of modules.

Traits: `InnerRefNamespace`, `IsolatedFromAbove`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `enable_layers` | ::mlir::ArrayAttr | symbol ref array attribute |
| `disable_layers` | ::mlir::ArrayAttr | symbol ref array attribute |
| `default_layer_specialization` | ::circt::firrtl::LayerSpecializationAttr | layer specialization |
| `select_inst_choice` | ::mlir::ArrayAttr | array attribute |

### `firrtl.class` (::circt::firrtl::ClassOp)

*FIRRTL Class*

The “firrtl.class” operation defines a class of property-only objects,
including a given name, a list of ports, and a body that represents the
connections within the class.

A class may only have property ports, and its body may only be ops that act
on properties, such as propassign ops.

Traits: `HasParent<CircuitOp>`, `InnerSymbolTable`, `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`

Interfaces: `ClassLike`, `FModuleLike`, `InstanceGraphModuleOpInterface`, `ModuleOpInterface`, `OpAsmOpInterface`, `PortList`, `SymbolOpInterface`, `SymbolUserOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `portDirections` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |
| `portNames` | ::mlir::ArrayAttr | array attribute |
| `portTypes` | ::mlir::ArrayAttr | array attribute |
| `portSymbols` | ::mlir::ArrayAttr | array attribute |
| `portLocations` | ::mlir::ArrayAttr | array attribute |
| `domainInfo` | ::mlir::ArrayAttr | array attribute |

### `firrtl.domain` (::circt::firrtl::DomainOp)

*Define a domain type*

Syntax:

```
operation ::= `firrtl.domain` $sym_name ( $fields^ )? attr-dict-with-keyword
```

A `firrtl.domain` operation defines a type of domain that exists in this
circuit. E.g., this can be used to declare a `ClockDomain` type when
modeling clocks in FIRRTL Dialect.

Traits: `HasParent<firrtl::CircuitOp>`

Interfaces: `SymbolOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `fields` | ::mlir::ArrayAttr | an array of domain fields |

### `firrtl.extclass` (::circt::firrtl::ExtClassOp)

*FIRRTL external class*

The “firrtl.extclass” operation represents a reference to an external
firrtl class, and includes a given name, as well as a list of ports.
Just as usual firrtl.class definitions, An ext.class may only have property
ports.

example:

```
firrtl.extclass @MyImportedClass(in in_str: !firrtl.string, out out_str: !firrtl.string)
```

Traits: `HasParent<CircuitOp>`, `InnerSymbolTable`, `IsolatedFromAbove`

Interfaces: `ClassLike`, `FModuleLike`, `InstanceGraphModuleOpInterface`, `ModuleOpInterface`, `OpAsmOpInterface`, `PortList`, `SymbolOpInterface`, `SymbolUserOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `portDirections` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |
| `portNames` | ::mlir::ArrayAttr | array attribute |
| `portTypes` | ::mlir::ArrayAttr | array attribute |
| `portSymbols` | ::mlir::ArrayAttr | array attribute |
| `portLocations` | ::mlir::ArrayAttr | array attribute |
| `domainInfo` | ::mlir::ArrayAttr | array attribute |

### `firrtl.extmodule` (::circt::firrtl::FExtModuleOp)

*FIRRTL external module*

The “firrtl.extmodule” operation represents an external reference to a
Verilog module, including a given name and a list of ports.

Traits: `HasParent<CircuitOp>`, `InnerSymbolTable`, `IsolatedFromAbove`

Interfaces: `FModuleLike`, `InstanceGraphModuleOpInterface`, `ModuleOpInterface`, `OpAsmOpInterface`, `PortList`, `SymbolUserOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `defname` | ::mlir::StringAttr | string attribute |
| `parameters` | ::mlir::ArrayAttr | parameter array attribute |
| `convention` | ::circt::firrtl::ConventionAttr | lowering convention |
| `portDirections` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |
| `portLocations` | ::mlir::ArrayAttr | array attribute |
| `portAnnotations` | ::mlir::ArrayAttr | array attribute |
| `portSymbols` | ::mlir::ArrayAttr | array attribute |
| `portNames` | ::mlir::ArrayAttr | array attribute |
| `portTypes` | ::mlir::ArrayAttr | array attribute |
| `knownLayers` | ::mlir::ArrayAttr | an array of layers |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `layers` | ::mlir::ArrayAttr | an array of layers |
| `domainInfo` | ::mlir::ArrayAttr | array attribute |
| `externalRequirements` | ::mlir::ArrayAttr | array attribute |

### `firrtl.intmodule` (::circt::firrtl::FIntModuleOp)

*FIRRTL intrinsic module*

The “firrtl.intmodule” operation represents a compiler intrinsic.

Traits: `HasParent<CircuitOp>`, `InnerSymbolTable`, `IsolatedFromAbove`

Interfaces: `FModuleLike`, `InstanceGraphModuleOpInterface`, `ModuleOpInterface`, `OpAsmOpInterface`, `PortList`, `SymbolUserOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `intrinsic` | ::mlir::StringAttr | string attribute |
| `parameters` | ::mlir::ArrayAttr | parameter array attribute |
| `portDirections` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |
| `portLocations` | ::mlir::ArrayAttr | array attribute |
| `portAnnotations` | ::mlir::ArrayAttr | array attribute |
| `portSymbols` | ::mlir::ArrayAttr | array attribute |
| `portNames` | ::mlir::ArrayAttr | array attribute |
| `portTypes` | ::mlir::ArrayAttr | array attribute |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `layers` | ::mlir::ArrayAttr | an array of layers |
| `domainInfo` | ::mlir::ArrayAttr | array attribute |

### `firrtl.memmodule` (::circt::firrtl::FMemModuleOp)

*FIRRTL Generated Memory Module*

The “firrtl.memmodule” operation represents an external reference to a
memory module. See the “firrtl.mem” op for a deeper explantation of the
parameters.

A “firrtl.mem” operation is typically lowered to this operation when they
are not directly lowered to registers by the compiler.

Traits: `HasParent<CircuitOp>`, `InnerSymbolTable`, `IsolatedFromAbove`

Interfaces: `FModuleLike`, `InstanceGraphModuleOpInterface`, `ModuleOpInterface`, `OpAsmOpInterface`, `PortList`, `SymbolUserOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `numReadPorts` | ::mlir::IntegerAttr | 32-bit unsigned integer attribute |
| `numWritePorts` | ::mlir::IntegerAttr | 32-bit unsigned integer attribute |
| `numReadWritePorts` | ::mlir::IntegerAttr | 32-bit unsigned integer attribute |
| `dataWidth` | ::mlir::IntegerAttr | 32-bit unsigned integer attribute |
| `maskBits` | ::mlir::IntegerAttr | 32-bit unsigned integer attribute |
| `readLatency` | ::mlir::IntegerAttr | 32-bit unsigned integer attribute |
| `writeLatency` | ::mlir::IntegerAttr | 32-bit unsigned integer attribute |
| `depth` | ::mlir::IntegerAttr | 64-bit unsigned integer attribute |
| `extraPorts` | ::mlir::ArrayAttr | array attribute |
| `portDirections` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |
| `portLocations` | ::mlir::ArrayAttr | array attribute |
| `portAnnotations` | ::mlir::ArrayAttr | array attribute |
| `portSymbols` | ::mlir::ArrayAttr | array attribute |
| `portNames` | ::mlir::ArrayAttr | array attribute |
| `portTypes` | ::mlir::ArrayAttr | array attribute |
| `ruw` | ::circt::firrtl::RUWBehaviorAttr | read under write behavior |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `layers` | ::mlir::ArrayAttr | an array of layers |
| `domainInfo` | ::mlir::ArrayAttr | array attribute |

### `firrtl.module` (::circt::firrtl::FModuleOp)

*FIRRTL Module*

The “firrtl.module” operation represents a Verilog module, including a given
name, a list of ports, and a body that represents the connections within
the module.

Traits: `HasParent<CircuitOp>`, `InnerSymbolTable`, `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`

Interfaces: `FModuleLike`, `InstanceGraphModuleOpInterface`, `ModuleOpInterface`, `OpAsmOpInterface`, `PortList`, `SymbolUserOpInterface`, `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `convention` | ::circt::firrtl::ConventionAttr | lowering convention |
| `portDirections` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |
| `portLocations` | ::mlir::ArrayAttr | array attribute |
| `portAnnotations` | ::mlir::ArrayAttr | array attribute |
| `portSymbols` | ::mlir::ArrayAttr | array attribute |
| `portNames` | ::mlir::ArrayAttr | array attribute |
| `portTypes` | ::mlir::ArrayAttr | array attribute |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `layers` | ::mlir::ArrayAttr | an array of layers |
| `domainInfo` | ::mlir::ArrayAttr | array attribute |

### `firrtl.formal` (::circt::firrtl::FormalOp)

*Define a formal unit test*

Syntax:

```
operation ::= `firrtl.formal` $sym_name `,` $moduleName $parameters attr-dict-with-keyword
```

The `firrtl.formal` operation defines a formal verification unit test. The
op defines a unique symbol name that can be used to refer to it. The design
to be tested and any necessary test harness is defined by the separate
`firrtl.module` op referenced by `moduleName`. Additional parameters may be
specified for the unit test. Input ports of the target module are considered
to be symbolic values during the test; output ports are ignored.

This operation may be used to mark unit tests in a FIRRTL design, which
other tools may later pick up and run automatically. It is intended to lower
to the `verif.formal` operation. See `verif.formal` for more details.

Example:

```
firrtl.module @MyTest(in %a: !firrtl.uint<42>) {}
firrtl.formal @myTestA, @MyTest {bound = 42}
firrtl.formal @myTestB, @MyTest {mode = "induction", userAttr = 9001}
```

Traits: `HasParent<firrtl::CircuitOp>`

Interfaces: `SymbolOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `moduleName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `parameters` | ::mlir::DictionaryAttr | dictionary of named attribute values |

### `firrtl.layer` (::circt::firrtl::LayerOp)

*A layer definition*

Syntax:

```
operation ::= `firrtl.layer` $sym_name `` $convention attr-dict-with-keyword $body
```

The `firrtl.layer` operation defines a layer and a lowering convention for
that layer. Layers are a feature of FIRRTL that add verification or
debugging code to an existing module at runtime.

A `firrtl.layer` operation only defines the layer and any layers nested
under it. Functionality is added to modules using the `firrtl.group`
operation.

Traits: `HasParent<firrtl::CircuitOp, firrtl::LayerOp>`, `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `convention` | ::circt::firrtl::LayerConventionAttr | layer convention |

### `firrtl.option_case` (::circt::firrtl::OptionCaseOp)

*A configuration option value definition*

Syntax:

```
operation ::= `firrtl.option_case` $sym_name attr-dict
```

`firrtl.option_case` defines an acceptable value to be provided for an
option. Ops reference it to define their behavior when this case is active.

Traits: `HasParent<firrtl::OptionOp>`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `firrtl.option` (::circt::firrtl::OptionOp)

*An option group definition*

Syntax:

```
operation ::= `firrtl.option` $sym_name attr-dict-with-keyword $body
```

The `firrtl.option` operation defines a specializable parameter with a
known set of values, represented by the `firrtl.option_case` operations
nested underneath.

Operations which support specialization reference the option and its
cases to define the specializations they support.

Example:

```
firrtl.circuit {
  firrtl.option @Target {
    firrtl.option_case @FPGA
    firrtl.option_case @ASIC
  }
}
```

Traits: `HasParent<firrtl::CircuitOp>`, `IsolatedFromAbove`, `NoTerminator`, `SymbolTable`

Interfaces: `Symbol`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `firrtl.simulation` (::circt::firrtl::SimulationOp)

*Define a simulation unit test*

Syntax:

```
operation ::= `firrtl.simulation` $sym_name `,` $moduleName $parameters attr-dict-with-keyword
```

The `firrtl.simulation` operation defines a simulation unit test. The op
defines a unique symbol name that can be used to refer to it. The design to
be tested and any necessary test harness is defined by the separate
`firrtl.module` op referenced by `moduleName`. Additional parameters may be
specified for the unit test. The target module’s first four ports must be:

```
(
  in %clock: !firrtl.clock,
  in %init: !firrtl.uint<1>,
  out %done: !firrtl.uint<1>,
  out %success: !firrtl.uint<1>
)
```

Any additional ports (index 4 and beyond) may only be property types and
are ignored.

This operation may be used to mark unit tests in a FIRRTL design, which
other tools may later pick up and run automatically. It is intended to lower
to the `verif.simulation` operation. See `verif.simulation` for more
details.

Example:

```
firrtl.extmodule @MyTest(
  in clock: !firrtl.clock,
  in init: !firrtl.uint<1>,
  out done: !firrtl.uint<1>,
  out success: !firrtl.uint<1>
)
firrtl.simulation @myTestA, @MyTest {bound = 1000}
firrtl.simulation @myTestB, @MyTest {asserts = "error", userAttr = 9001}
```

Traits: `HasParent<firrtl::CircuitOp>`

Interfaces: `SymbolOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `moduleName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `parameters` | ::mlir::DictionaryAttr | dictionary of named attribute values |

Operation Definitions – Declarations
------------------------------------

### `firrtl.contract` (::circt::firrtl::ContractOp)

*Contract declaration*

Syntax:

```
operation ::= `firrtl.contract` ($inputs^ `:` type($inputs))? attr-dict-with-keyword $body
```

The `firrtl.contract` operation defines values that adhere to a set of
formal assertions and assumptions outlined in the contract’s body.

See the `verif.contract` operation for more details.

Traits: `NoTerminator`, `SingleBlock`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of a passive base type (contain no flips) |

#### Results:

| Result | Description |
| --- | --- |
| `outputs` | variadic of a passive base type (contain no flips) |

### `firrtl.domain.anon` (::circt::firrtl::DomainCreateAnonOp)

*Create an anonymous domain, without parameters set*

Syntax:

```
operation ::= `firrtl.domain.anon` attr-dict `:` type($result) `of` $domain
```

Traits: `HasParent<firrtl::ContractOp, firrtl::FModuleOp, firrtl::LayerBlockOp, firrtl::MatchOp, firrtl::WhenOp, sv::IfDefOp>`

Interfaces: `InferTypeOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `domain` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a domain type |

### `firrtl.instance_choice` (::circt::firrtl::InstanceChoiceOp)

*Creates an instance of a module based on a option*

The instance choice operation creates an instance choosing the target based
on the value of an option if one is specified, instantiating a default
target otherwise.

The port lists of all instance targets must match exactly.

Examples:

```
%0 = firrtl.instance_choice foo @Foo alternatives @Opt { @FPGA -> @FPGAFoo }
  (in io: !firrtl.uint)
```

Traits: `HasParent<firrtl::ContractOp, firrtl::FModuleOp, firrtl::LayerBlockOp, firrtl::MatchOp, firrtl::WhenOp, sv::IfDefOp>`

Interfaces: `FNamableOp`, `HasCustomSSAName`, `InnerSymbolOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `moduleNames` | ::mlir::ArrayAttr | flat symbol ref array attribute |
| `caseNames` | ::mlir::ArrayAttr | symbol ref array attribute |
| `name` | ::mlir::StringAttr | string attribute |
| `nameKind` | ::circt::firrtl::NameKindEnumAttr | name kind |
| `portDirections` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |
| `portNames` | ::mlir::ArrayAttr | string array attribute |
| `domainInfo` | ::mlir::ArrayAttr | domain attributes must be an empty array or array-of-arrays |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `portAnnotations` | ::mlir::ArrayAttr | Port annotations attribute |
| `layers` | ::mlir::ArrayAttr | an array of layers |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `firrtl.instance` (::circt::firrtl::InstanceOp)

*Instantiate a module*

This represents an instance of a module. The results are the modules inputs
and outputs. The inputs have flip type, the outputs do not.

Examples:

```
%0 = firrtl.instance foo @Foo(in io: !firrtl.uint)
```

Traits: `HasParent<firrtl::ContractOp, firrtl::FModuleOp, firrtl::LayerBlockOp, firrtl::MatchOp, firrtl::WhenOp, sv::IfDefOp>`

Interfaces: `FInstanceLike`, `FNamableOp`, `HasCustomSSAName`, `InnerSymbolOpInterface`, `InstanceGraphInstanceOpInterface`, `InstanceOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `moduleName` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `name` | ::mlir::StringAttr | string attribute |
| `nameKind` | ::circt::firrtl::NameKindEnumAttr | name kind |
| `portDirections` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |
| `portNames` | ::mlir::ArrayAttr | string array attribute |
| `domainInfo` | ::mlir::ArrayAttr | domain attributes must be an empty array or array-of-arrays |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `portAnnotations` | ::mlir::ArrayAttr | Port annotations attribute |
| `layers` | ::mlir::ArrayAttr | an array of layers |
| `lowerToBind` | ::mlir::UnitAttr | unit attribute |
| `doNotPrint` | ::mlir::UnitAttr | unit attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of any type |

### `firrtl.mem` (::circt::firrtl::MemOp)

*Define a new mem*

Syntax:

```
operation ::= `firrtl.mem` (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
              $ruw `` custom<MemOp>(attr-dict) `:` qualified(type($results))
```

Traits: `HasParent<firrtl::ContractOp, firrtl::FModuleOp, firrtl::LayerBlockOp, firrtl::MatchOp, firrtl::WhenOp, sv::IfDefOp>`

Interfaces: `CombDataFlow`, `FNamableOp`, `HasCustomSSAName`, `InnerSymbolOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `readLatency` | ::mlir::IntegerAttr | 32-bit signless integer attribute whose minimum value is 0 |
| `writeLatency` | ::mlir::IntegerAttr | 32-bit signless integer attribute whose minimum value is 1 |
| `depth` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 1 |
| `ruw` | ::circt::firrtl::RUWBehaviorAttr | read under write behavior |
| `portNames` | ::mlir::ArrayAttr | string array attribute |
| `name` | ::mlir::StringAttr | string attribute |
| `nameKind` | ::circt::firrtl::NameKindEnumAttr | name kind |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `portAnnotations` | ::mlir::ArrayAttr | Port annotations attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `init` | ::circt::firrtl::MemoryInitAttr | Information about the initial state of a memory |
| `prefix` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `results` | variadic of FIRRTLType |

### `firrtl.node` (::circt::firrtl::NodeOp)

*No-op to name a value*

Syntax:

```
operation ::= `firrtl.node` (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
              $input (`forceable` $forceable^)? `` custom<FIRRTLImplicitSSAName>(attr-dict) `:` qualified(type($input))
```

A node is simply a named intermediate value in a circuit. The node must
be initialized to a value with a passive type and cannot be connected to.
Nodes are often used to split a complicated compound expression into named
subexpressions.

```
  %result = firrtl.node %input : t1
```

Traits: `HasParent<firrtl::ContractOp, firrtl::FModuleOp, firrtl::LayerBlockOp, firrtl::MatchOp, firrtl::WhenOp, sv::IfDefOp>`

Interfaces: `FNamableOp`, `Forceable`, `HasCustomSSAName`, `InferTypeOpInterface`, `InnerSymbolOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `nameKind` | ::circt::firrtl::NameKindEnumAttr | name kind |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `forceable` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a passive base type (contain no flips) |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a passive base type (contain no flips) |
| `ref` | reference type |

### `firrtl.object` (::circt::firrtl::ObjectOp)

*Instantiate a class to produce an object*

Syntax:

```
operation ::= `firrtl.object` `` custom<ImplicitSSAName>(attr-dict) custom<ClassInterface>(type($result))
```

This represents an instance of a class. The results is the instantiated
object.

Examples:

```
%0 = firrtl.object @Foo(in io: !firrtl.uint)
```

Traits: `HasParent<firrtl::FModuleOp, firrtl::ClassOp>`

Interfaces: `FInstanceLike`, `InstanceGraphInstanceOpInterface`, `InstanceOpInterface`, `OpAsmOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | class type |

### `firrtl.reg` (::circt::firrtl::RegOp)

*Define a new register*

Syntax:

```
operation ::= `firrtl.reg` (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
              operands (`forceable` $forceable^)? `` custom<FIRRTLImplicitSSAName>(attr-dict) `:` type($clockVal) `,` qualified(type($result)) (`,` qualified(type($ref))^)?
```

Declare a new register:

```
%name = firrtl.reg %clockVal : !firrtl.clock, t1
```

Traits: `HasParent<firrtl::ContractOp, firrtl::FModuleOp, firrtl::LayerBlockOp, firrtl::MatchOp, firrtl::WhenOp, sv::IfDefOp>`

Interfaces: `CombDataFlow`, `FNamableOp`, `Forceable`, `HasCustomSSAName`, `InnerSymbolOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `nameKind` | ::circt::firrtl::NameKindEnumAttr | name kind |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `forceable` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clockVal` | clock |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a passive non-‘const’ base type that does not contain analog |
| `ref` | reference type |

### `firrtl.regreset` (::circt::firrtl::RegResetOp)

*Define a new register with a reset*

Syntax:

```
operation ::= `firrtl.regreset` (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
              operands (`forceable` $forceable^)? `` custom<FIRRTLImplicitSSAName>(attr-dict)
              `:` type($clockVal) `,` qualified(type($resetSignal)) `,` qualified(type($resetValue)) `,` qualified(type($result)) (`,` qualified(type($ref))^)?
```

Declare a new register:

```
  %name = firrtl.regreset %clockVal, %resetSignal, %resetValue : !firrtl.clock, t1
```

Traits: `HasParent<firrtl::ContractOp, firrtl::FModuleOp, firrtl::LayerBlockOp, firrtl::MatchOp, firrtl::WhenOp, sv::IfDefOp>`

Interfaces: `CombDataFlow`, `FNamableOp`, `Forceable`, `HasCustomSSAName`, `InnerSymbolOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `nameKind` | ::circt::firrtl::NameKindEnumAttr | name kind |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `forceable` | ::mlir::UnitAttr | unit attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clockVal` | clock |
| `resetSignal` | Reset |
| `resetValue` | a base type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a passive non-‘const’ base type that does not contain analog |
| `ref` | reference type |

### `firrtl.wire` (::circt::firrtl::WireOp)

*Define a new wire*

Syntax:

```
operation ::= `firrtl.wire` (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
              (`forceable` $forceable^)? `` custom<FIRRTLImplicitSSAName>(attr-dict) `:`
              qualified(type($result)) (`,` qualified(type($ref))^)?
```

Declare a new wire:

```
  %name = firrtl.wire : t1
```

Traits: `HasParent<firrtl::ContractOp, firrtl::FModuleOp, firrtl::LayerBlockOp, firrtl::MatchOp, firrtl::WhenOp, sv::IfDefOp>`

Interfaces: `FNamableOp`, `Forceable`, `HasCustomSSAName`, `InnerSymbolOpInterface`, `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `nameKind` | ::circt::firrtl::NameKindEnumAttr | name kind |
| `annotations` | ::mlir::ArrayAttr | Annotation array attribute |
| `inner_sym` | ::circt::hw::InnerSymAttr | Inner symbol definition |
| `forceable` | ::mlir::UnitAttr | unit attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |
| `ref` | reference type |

Statement Operation – Statements
--------------------------------

### `firrtl.assert` (::circt::firrtl::AssertOp)

*Assert Verification Statement*

Syntax:

```
operation ::= `firrtl.assert` $clock `,` $predicate `,` $enable `,`
              $message (`(` $substitutions^ `)`)? `:` type($clock) `,` type($predicate) `,` type($enable) (`,` qualified(type($substitutions))^)?
              custom<VerifAttrs>(attr-dict)
```

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `message` | ::mlir::StringAttr | string attribute |
| `name` | ::mlir::StringAttr | string attribute |
| `isConcurrent` | ::mlir::BoolAttr | bool attribute |
| `eventControl` | ::circt::firrtl::EventControlAttr | edge control trigger |

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | clock |
| `predicate` | 1-bit uint |
| `enable` | 1-bit uint |
| `substitutions` | variadic of any type |

### `firrtl.assume` (::circt::firrtl::AssumeOp)

*Assume Verification Statement*

Syntax:

```
operation ::= `firrtl.assume` $clock `,` $predicate `,` $enable `,`
              $message (`(` $substitutions^ `)`)? `:` type($clock) `,` type($predicate) `,` type($enable) (`,` qualified(type($substitutions))^)?
              custom<VerifAttrs>(attr-dict)
```

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `message` | ::mlir::StringAttr | string attribute |
| `name` | ::mlir::StringAttr | string attribute |
| `isConcurrent` | ::mlir::BoolAttr | bool attribute |
| `eventControl` | ::circt::firrtl::EventControlAttr | edge control trigger |

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | clock |
| `predicate` | 1-bit uint |
| `enable` | 1-bit uint |
| `substitutions` | variadic of any type |

### `firrtl.attach` (::circt::firrtl::AttachOp)

*Analog Attach Statement*

Syntax:

```
operation ::= `firrtl.attach` $attached attr-dict `:` qualified(type($attached))
```

#### Operands:

| Operand | Description |
| --- | --- |
| `attached` | variadic of analog type |

### `firrtl.bind` (::circt::firrtl::BindOp)

*Indirect instantiation statement*

Syntax:

```
operation ::= `firrtl.bind` $instance attr-dict
```

Indirectly instantiate a module from the context of another module. BindOp
pairs with a `firrtl.instance` which tracks all information except the
emission point for the bind.

This op exists to aid in the progressive lowering of FIRRTL surface level
constructs, such as layers, to system verilog (sv dialect).

Interfaces: `InnerRefUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `instance` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |

### `firrtl.connect` (::circt::firrtl::ConnectOp)

*Connect two signals*

Syntax:

```
operation ::= `firrtl.connect` $dest `,` $src  attr-dict `:` custom<OptionalBinaryOpTypes>(type($dest), type($src))
```

Connect Operation:

```
  firrtl.connect %dest, %src : t1, t2
```

Interfaces: `FConnectLike`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | a base type or foreign type |
| `src` | a base type or foreign type |

### `firrtl.cover` (::circt::firrtl::CoverOp)

*Cover Verification Statement*

Syntax:

```
operation ::= `firrtl.cover` $clock `,` $predicate `,` $enable `,`
              $message (`(` $substitutions^ `)`)? `:` type($clock) `,` type($predicate) `,` type($enable) (`,` qualified(type($substitutions))^)?
              custom<VerifAttrs>(attr-dict)
```

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `message` | ::mlir::StringAttr | string attribute |
| `name` | ::mlir::StringAttr | string attribute |
| `isConcurrent` | ::mlir::BoolAttr | bool attribute |
| `eventControl` | ::circt::firrtl::EventControlAttr | edge control trigger |

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | clock |
| `predicate` | 1-bit uint |
| `enable` | 1-bit uint |
| `substitutions` | variadic of any type |

### `firrtl.domain.define` (::circt::firrtl::DomainDefineOp)

*FIRRTL Define Domains.*

Syntax:

```
operation ::= `firrtl.domain.define` $dest `,` $src  attr-dict
```

Define a target domain to the source domain:

```
  firrtl.domain.define %dest, %src
```

Used to statically route domains from source to destination
through the design, one module at a time.

Similar to “connect” but cannot have multiple define’s to same
destination, and the define is never conditional even if under
a “when”.

Source and destination must resolve statically.

Traits: `SameTypeOperands`

Interfaces: `FConnectLike`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | a domain type |
| `src` | a domain type |

### `firrtl.fflush` (::circt::firrtl::FFlushOp)

*FFlush statement*

Syntax:

```
operation ::= `firrtl.fflush` $clock `,` $cond (`,` $outputFile `(` $outputFileSubstitutions^ `)`)?
              attr-dict `:` type($clock) `,` type($cond)
              (`,` qualified(type($outputFileSubstitutions))^)?
```

This operation flushes the output buffer of the specified file descriptor. If
no file descriptor is specified, the output buffer of the default output
file descriptor is flushed.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `outputFile` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | clock |
| `cond` | 1-bit uint |
| `outputFileSubstitutions` | variadic of a printf operand type (a FIRRTL base type or a format string type) |

### `firrtl.fprintf` (::circt::firrtl::FPrintFOp)

*Formatted File Print Statement*

This operation is similar to the “firrtl.printf” operation, but it prints
to a file instead of stdout.

Traits: `AttrSizedOperandSegments`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `outputFile` | ::mlir::StringAttr | string attribute |
| `formatString` | ::mlir::StringAttr | string attribute |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | clock |
| `cond` | 1-bit uint |
| `outputFileSubstitutions` | variadic of a printf operand type (a FIRRTL base type or a format string type) |
| `substitutions` | variadic of a printf operand type (a FIRRTL base type or a format string type) |

### `firrtl.force` (::circt::firrtl::ForceOp)

*Force procedural statement*

Syntax:

```
operation ::= `firrtl.force` $dest `,` $src attr-dict `:` qualified(type($dest)) `,` qualified(type($src))
```

Maps to the corresponding `sv.force` operation.

Traits: `SameTypeOperands`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | a base type |
| `src` | a base type |

### `firrtl.layerblock` (::circt::firrtl::LayerBlockOp)

*A definition of a layer block*

Syntax:

```
operation ::= `firrtl.layerblock` $layerName $region attr-dict
```

The `firrtl.layerblock` operation defines optional code that is
conditionally part of a `firrtl.module` if its referenced `firrtl.layer` is
enabled. This is typically used to store verification or debugging code
that is lowered to a module that is “enabled” using the `firrtl.layer`’s
convention (e.g., remote instantiation via SystemVerilog `bind`). A layer
block can read from (capture) values defined in parent layer blocks or in
the parent module, but may not write to hardware declared outside the layer
block.

A `firrtl.layerblock` must refer to an existing layer definition
(`firrtl.layer`) via a symbol reference. A nested `firrtl.layerblock`
refers to a nested layer definition via a nested symbol reference.

Traits: `HasParent<firrtl::FModuleOp, firrtl::LayerBlockOp, firrtl::WhenOp, firrtl::MatchOp>`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

Interfaces: `SymbolUserOpInterface`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `layerName` | ::mlir::SymbolRefAttr | symbol reference attribute |

### `firrtl.match` (::circt::firrtl::MatchOp)

*Match Statement*

The “firrtl.match” operation represents a pattern matching statement on a
enumeration. This operation does not return a value and cannot be used as an
expression. Last connect semantics work similarly to a when statement.

Example:

```
  firrtl.match %in : !firrtl.enum<Some: uint<1>, None: uint<0>> {
    case Some(%arg0) {
      !firrtl.matchingconnect %w, %arg0 : !firrtl.uint<1>
    }
    case None(%arg0) {
      !firrt.matchingconnect %w, %c1 : !firrtl.uint<1>
    }
  }
```

Traits: `NoTerminator`, `RecursiveMemoryEffects`, `RecursivelySpeculatableImplTrait`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `tags` | ::mlir::ArrayAttr | 32-bit integer array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | enum type |

### `firrtl.matchingconnect` (::circt::firrtl::MatchingConnectOp)

*Connect two signals*

Syntax:

```
operation ::= `firrtl.matchingconnect` $dest `,` $src  attr-dict `:`
              custom<OptionalBinaryOpTypes>(type($dest), type($src))
```

Connect two values with matching type constraints. The types of the lhs and
rhs must match:

```
  firrtl.matchingconnect %dest, %src : t1
  firrtl.matchingconnect %dest, %src : t1, !firrtl.alias<foo, t1>
```

Interfaces: `FConnectLike`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | a sized passive base type (contains no uninferred widths, or flips) or foreign type |
| `src` | a sized passive base type (contains no uninferred widths, or flips) or foreign type |

### `firrtl.printf` (::circt::firrtl::PrintFOp)

*Formatted Print Statement*

Syntax:

```
operation ::= `firrtl.printf` $clock `,` $cond `,` $formatString `` custom<PrintfAttrs>(attr-dict) ` `
              (`(` $substitutions^ `)`)? `:` type($clock) `,` type($cond) (`,` qualified(type($substitutions))^)?
```

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `formatString` | ::mlir::StringAttr | string attribute |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | clock |
| `cond` | 1-bit uint |
| `substitutions` | variadic of a printf operand type (a FIRRTL base type or a format string type) |

### `firrtl.propassign` (::circt::firrtl::PropAssignOp)

*Assign to a sink property value.*

Syntax:

```
operation ::= `firrtl.propassign` $dest `,` $src attr-dict `:` qualified(type($dest))
```

Assign an output property value. The types must match exactly.

Example:

```
  firrtl.propassign %dest, %src : !firrtl.string
```

Traits: `HasParent<FModuleOp, ClassOp, LayerBlockOp>`, `SameTypeOperands`

Interfaces: `FConnectLike`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | property type |
| `src` | property type |

### `firrtl.ref.define` (::circt::firrtl::RefDefineOp)

*FIRRTL Define References*

Syntax:

```
operation ::= `firrtl.ref.define` $dest `,` $src  attr-dict `:` qualified(type($dest))
```

Define a target reference to the source reference:

```
  firrtl.ref.define %dest, %src : ref<t1>
```

Used to statically route reference from source to destination
through the design, one module at a time.

Similar to “connect” but cannot have multiple define’s to same
destination and the define is never conditional even if under
a “when”.

Source and destination must resolve statically.

Traits: `SameTypeOperands`

Interfaces: `FConnectLike`

#### Operands:

| Operand | Description |
| --- | --- |
| `dest` | reference type |
| `src` | reference type |

### `firrtl.ref.force_initial` (::circt::firrtl::RefForceInitialOp)

*FIRRTL force\_initial statement*

Syntax:

```
operation ::= `firrtl.ref.force_initial` $predicate `,` $dest `,` $src attr-dict `:` type($predicate) `,` qualified(type($dest)) `,` qualified(type($src))
```

Force a RWProbe to the specified value continuously.

#### Operands:

| Operand | Description |
| --- | --- |
| `predicate` | 1-bit uint |
| `dest` | rwprobe type |
| `src` | a base type |

### `firrtl.ref.force` (::circt::firrtl::RefForceOp)

*FIRRTL force statement*

Syntax:

```
operation ::= `firrtl.ref.force` $clock `,` $predicate `,` $dest `,` $src attr-dict `:` type($clock) `,` type($predicate) `,` qualified(type($dest)) `,` qualified(type($src))
```

Force a RWProbe to the specified value using the specified clock and predicate.

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | clock |
| `predicate` | 1-bit uint |
| `dest` | rwprobe type |
| `src` | a base type |

### `firrtl.ref.release_initial` (::circt::firrtl::RefReleaseInitialOp)

*FIRRTL release\_initial statement*

Syntax:

```
operation ::= `firrtl.ref.release_initial` $predicate `,` $dest attr-dict `:` type($predicate) `,` qualified(type($dest))
```

Release the target RWProbe.

#### Operands:

| Operand | Description |
| --- | --- |
| `predicate` | 1-bit uint |
| `dest` | rwprobe type |

### `firrtl.ref.release` (::circt::firrtl::RefReleaseOp)

*FIRRTL release statement*

Syntax:

```
operation ::= `firrtl.ref.release` $clock `,` $predicate `,` $dest attr-dict `:` type($clock) `,` type($predicate) `,` qualified(type($dest))
```

Release the target RWProbe using the specified clock and predicate.

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | clock |
| `predicate` | 1-bit uint |
| `dest` | rwprobe type |

### `firrtl.skip` (::circt::firrtl::SkipOp)

*Skip statement*

Syntax:

```
operation ::= `firrtl.skip` attr-dict
```

Skip Statement:

```
   %firrtl.skip
```

This is a no-op statement.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

### `firrtl.stop` (::circt::firrtl::StopOp)

*Stop Statement*

Syntax:

```
operation ::= `firrtl.stop` $clock `,` $cond `,` $exitCode `` custom<StopAttrs>(attr-dict) `:` type($clock) `,` type($cond)
```

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `exitCode` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | clock |
| `cond` | 1-bit uint |

### `firrtl.int.verif.assert` (::circt::firrtl::VerifAssertIntrinsicOp)

*FIRRTL variant of `verif.assert`*

Syntax:

```
operation ::= `firrtl.int.verif.assert` operands attr-dict `:` type(operands)
```

See `verif.assert` op in the Verif dialect.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `property` | 1-bit uint |
| `enable` | 1-bit uint |

### `firrtl.int.verif.assume` (::circt::firrtl::VerifAssumeIntrinsicOp)

*FIRRTL variant of `verif.assume`*

Syntax:

```
operation ::= `firrtl.int.verif.assume` operands attr-dict `:` type(operands)
```

See `verif.assume` op in the Verif dialect.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `property` | 1-bit uint |
| `enable` | 1-bit uint |

### `firrtl.int.verif.cover` (::circt::firrtl::VerifCoverIntrinsicOp)

*FIRRTL variant of `verif.cover`*

Syntax:

```
operation ::= `firrtl.int.verif.cover` operands attr-dict `:` type(operands)
```

See `verif.cover` op in the Verif dialect.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `property` | 1-bit uint |
| `enable` | 1-bit uint |

### `firrtl.int.verif.ensure` (::circt::firrtl::VerifEnsureIntrinsicOp)

*FIRRTL variant of `verif.ensure`*

Syntax:

```
operation ::= `firrtl.int.verif.ensure` operands attr-dict `:` type(operands)
```

See `verif.ensure` op in the Verif dialect.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `property` | 1-bit uint |
| `enable` | 1-bit uint |

### `firrtl.int.verif.require` (::circt::firrtl::VerifRequireIntrinsicOp)

*FIRRTL variant of `verif.require`*

Syntax:

```
operation ::= `firrtl.int.verif.require` operands attr-dict `:` type(operands)
```

See `verif.require` op in the Verif dialect.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `property` | 1-bit uint |
| `enable` | 1-bit uint |

### `firrtl.when` (::circt::firrtl::WhenOp)

*When Statement*

Syntax:

```
operation ::= `firrtl.when` $condition `:` type($condition) $thenRegion (`else` $elseRegion^)? attr-dict-with-keyword
```

The “firrtl.when” operation represents a conditional. Connections within
a conditional statement that connect to previously declared components hold
only when the given condition is high. The condition must have a 1-bit
unsigned integer type.

Traits: `NoRegionArguments`, `NoTerminator`, `SingleBlock`

#### Operands:

| Operand | Description |
| --- | --- |
| `condition` | 1-bit uint |

Operation Definitions – Expressions
-----------------------------------

### `firrtl.add` (::circt::firrtl::AddPrimOp)

Syntax:

```
operation ::= `firrtl.add` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint or uint type |

### `firrtl.aggregateconstant` (::circt::firrtl::AggregateConstantOp)

*Produce a constant of a passive aggregate value*

Syntax:

```
operation ::= `firrtl.aggregateconstant` $fields attr-dict `:` type($result)
```

The constant operation produces a constant value of an aggregate type. The
type must be passive. Clock and reset values are supported.
For nested aggregates, embedded arrays are used.

```
  %result = firrtl.aggregateconstant [1, 2, 3] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
  %result = firrtl.aggregateconstant [1, 2, 3] : !firrtl.vector<uint<8>, 3>
  %result = firrtl.aggregateconstant [[1, 2], [3, 5]] : !firrtl.vector<!firrtl.bundle<a: uint<8>, b: uint<5>>, 2>
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fields` | ::mlir::ArrayAttr | array attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a aggregate type |

### `firrtl.and` (::circt::firrtl::AndPrimOp)

Syntax:

```
operation ::= `firrtl.and` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | uint type |

### `firrtl.andr` (::circt::firrtl::AndRPrimOp)

Syntax:

```
operation ::= `firrtl.andr` $input attr-dict `:` functional-type($input, $result)
```

Horizontally reduce a value to one bit, using the ‘and’ operation to merge
bits. `andr(x)` is equivalent to `concat(x, 1b1) == ~0`. As such, it
returns 1 for zero-bit-wide operands.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.asAsyncReset` (::circt::firrtl::AsAsyncResetPrimOp)

Syntax:

```
operation ::= `firrtl.asAsyncReset` $input attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | 1-bit uint/sint/analog, reset, asyncreset, or clock |

#### Results:

| Result | Description |
| --- | --- |
| `result` | async reset type |

### `firrtl.asClock` (::circt::firrtl::AsClockPrimOp)

Syntax:

```
operation ::= `firrtl.asClock` $input attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | 1-bit uint/sint/analog, reset, asyncreset, or clock |

#### Results:

| Result | Description |
| --- | --- |
| `result` | clock |

### `firrtl.asSInt` (::circt::firrtl::AsSIntPrimOp)

Syntax:

```
operation ::= `firrtl.asSInt` $input attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a base type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint type |

### `firrtl.asUInt` (::circt::firrtl::AsUIntPrimOp)

Syntax:

```
operation ::= `firrtl.asUInt` $input attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a base type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | uint type |

### `firrtl.bitcast` (::circt::firrtl::BitCastOp)

*Reinterpret one value to another value of the same size and
potentially different type. This op is lowered to hw::BitCastOp.*

Syntax:

```
operation ::= `firrtl.bitcast` $input attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a base type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a passive base type (contain no flips) |

### `firrtl.bits` (::circt::firrtl::BitsPrimOp)

Syntax:

```
operation ::= `firrtl.bits` $input $hi `to` $lo attr-dict `:` functional-type($input, $result)
```

The `bits` operation extracts the bits between `hi` (inclusive) and `lo`
(inclusive) from `input`. `hi` must be greater than or equal to `lo`. Both
`hi` and `lo` must be non-negative and less than the bit width of `input`.
The result is `hi - lo + 1` bits wide.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `hi` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `lo` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | uint type |

### `firrtl.bool` (::circt::firrtl::BoolConstantOp)

*Produce a constant boolean value*

Syntax:

```
operation ::= `firrtl.bool` $value attr-dict
```

Produces a constant value of boolean type.

Example:

```
%0 = firrtl.bool true
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::BoolAttr | bool attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | boolean type |

### `firrtl.bundlecreate` (::circt::firrtl::BundleCreateOp)

*Produce a bundle value*

Syntax:

```
operation ::= `firrtl.bundlecreate` $fields attr-dict `:` functional-type($fields, $result)
```

Create an bundle from component values. This is equivalent in terms of
flow to creating a node.

```
  %result = firrtl.bundlecreate %1, %2, %3 : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
```

#### Operands:

| Operand | Description |
| --- | --- |
| `fields` | variadic of a base type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | bundle type |

### `firrtl.cat` (::circt::firrtl::CatPrimOp)

Syntax:

```
operation ::= `firrtl.cat` $inputs attr-dict `:` functional-type($inputs, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | uint type |

### `firrtl.constCast` (::circt::firrtl::ConstCastOp)

Syntax:

```
operation ::= `firrtl.constCast` $input attr-dict `:` functional-type($input, $result)
```

Cast from a ‘const’ to a non-‘const’ type.

```
%result = firrtl.constCast %in : (!firrtl.const.t1) -> !firrtl.t1
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a passive base type (contain no flips) |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a passive base type (contain no flips) |

### `firrtl.constant` (::circt::firrtl::ConstantOp)

*Produce a constant value*

The constant operation produces a constant value of SInt or UInt type, it
never produces a zero bit wide result.

```
  %result = firrtl.constant 42 : t1
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`, `FirstAttrDerivedResultType`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::IntegerAttr | arbitrary integer attribute with sign |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint or uint type |

### `firrtl.cvt` (::circt::firrtl::CvtPrimOp)

Syntax:

```
operation ::= `firrtl.cvt` $input attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint type |

### `firrtl.dshl` (::circt::firrtl::DShlPrimOp)

Syntax:

```
operation ::= `firrtl.dshl` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

A dynamic shift left operation. The width of `$result` is expanded to
`width($lhs) + 1 << width($rhs) - 1`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint or uint type |

### `firrtl.dshlw` (::circt::firrtl::DShlwPrimOp)

Syntax:

```
operation ::= `firrtl.dshlw` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

A dynamic shift left operation same as ‘dshl’ but with different width rule.
The width of `$result` is equal to `$lhs`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint or uint type |

### `firrtl.dshr` (::circt::firrtl::DShrPrimOp)

Syntax:

```
operation ::= `firrtl.dshr` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint or uint type |

### `firrtl.div` (::circt::firrtl::DivPrimOp)

Syntax:

```
operation ::= `firrtl.div` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Divides the first argument (the numerator) by the second argument
(the denominator) truncating the result (rounding towards zero).

**If the denominator is zero, the result is undefined.**

The compiler may optimize this undefined behavior in any way it
wants. Notably `div(a, a)` will be optimized to `1`. This may cause
erroneous formal equivalence mismatches between unoptimized and
optimized FIRRTL dialects that are separately converted to Verilog.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint or uint type |

### `firrtl.double` (::circt::firrtl::DoubleConstantOp)

*Produce a constant double value*

Syntax:

```
operation ::= `firrtl.double` $value attr-dict
```

Produces a constant value of double type.

Example:

```
%0 = firrtl.double 3.2
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::FloatAttr | double-precision float |

#### Results:

| Result | Description |
| --- | --- |
| `result` | double type |

### `firrtl.eq` (::circt::firrtl::EQPrimOp)

Syntax:

```
operation ::= `firrtl.eq` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.elementwise_and` (::circt::firrtl::ElementwiseAndPrimOp)

Syntax:

```
operation ::= `firrtl.elementwise_and` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | 1d vector with Int element type |
| `rhs` | 1d vector with Int element type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1d vector with UInt element type |

### `firrtl.elementwise_or` (::circt::firrtl::ElementwiseOrPrimOp)

Syntax:

```
operation ::= `firrtl.elementwise_or` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | 1d vector with Int element type |
| `rhs` | 1d vector with Int element type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1d vector with UInt element type |

### `firrtl.elementwise_xor` (::circt::firrtl::ElementwiseXorPrimOp)

Syntax:

```
operation ::= `firrtl.elementwise_xor` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | 1d vector with UInt element type |
| `rhs` | 1d vector with Int element type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1d vector with UInt element type |

### `firrtl.enumcreate` (::circt::firrtl::FEnumCreateOp)

*Produce a enum value*

Create an enum from tag and value.

```
  %result = firrtl.enumcreate field-name(%input) : !firrtl.enum<None: uint<0>, Some: uint<8>>
```

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a base type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | enum type |

### `firrtl.integer` (::circt::firrtl::FIntegerConstantOp)

*Produce a constant integer value*

Produces a constant value of integer type.

Example:

```
%0 = firrtl.integer 42
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::IntegerAttr | arbitrary integer attribute with sign |

#### Results:

| Result | Description |
| --- | --- |
| `result` | integer type |

### `firrtl.geq` (::circt::firrtl::GEQPrimOp)

Syntax:

```
operation ::= `firrtl.geq` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.gt` (::circt::firrtl::GTPrimOp)

Syntax:

```
operation ::= `firrtl.gt` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.hwStructCast` (::circt::firrtl::HWStructCastOp)

Syntax:

```
operation ::= `firrtl.hwStructCast` $input attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any type |

### `firrtl.head` (::circt::firrtl::HeadPrimOp)

Syntax:

```
operation ::= `firrtl.head` $input `,` $amount attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `amount` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | uint type |

### `firrtl.fstring.hierarchicalmodulename` (::circt::firrtl::HierarchicalModuleNameOp)

*An operation that represents the module name with its full path*

Syntax:

```
operation ::= `firrtl.fstring.hierarchicalmodulename` attr-dict `:` type($result)
```

This operation represents the name of the current module with a
fully-qualified path in front of it. It returns a format string type which
can only be used as a substitution inside a printf. This operation is not
represented in the FIRRTL spec.

This operation is the FIRRTL Dialect representation of the special
substitution `{{HierarchicalModuleName}}`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `result` | a format string type |

### `firrtl.integer.add` (::circt::firrtl::IntegerAddOp)

*Add two FIntegerType values*

Syntax:

```
operation ::= `firrtl.integer.add` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

The add operation result is the arbitrary precision signed integer
arithmetic sum of the two operands.

Example:

```
%2 = firrtl.integer.add %0, %1 : (!firrtl.integer, !firrtl.integer) ->
                                     !firrtl.integer
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | integer type |
| `rhs` | integer type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | integer type |

### `firrtl.integer.mul` (::circt::firrtl::IntegerMulOp)

*Multiply two FIntegerType values*

Syntax:

```
operation ::= `firrtl.integer.mul` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

The multiply operation result is the arbitrary precision signed integer
arithmetic product of the two operands.

Example:

```
%2 = firrtl.integer.mul %0, %1 : (!firrtl.integer, !firrtl.integer) ->
                                     !firrtl.integer
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | integer type |
| `rhs` | integer type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | integer type |

### `firrtl.integer.shl` (::circt::firrtl::IntegerShlOp)

*Shift an FIntegerType value left by an FIntegerType value*

Syntax:

```
operation ::= `firrtl.integer.shl` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

The shift left operation result is the arbitrary precision signed integer
shift left of the lhs operand by the rhs operand. The rhs operand must be
non-negative.

Example:

```
%2 = firrtl.integer.shl %0, %1 : (!firrtl.integer, !firrtl.integer) ->
                                     !firrtl.integer
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | integer type |
| `rhs` | integer type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | integer type |

### `firrtl.integer.shr` (::circt::firrtl::IntegerShrOp)

*Shift an FIntegerType value right by an FIntegerType value*

Syntax:

```
operation ::= `firrtl.integer.shr` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

The shift right operation result is the arbitrary precision signed integer
arithmetic shift right of the lhs operand by the rhs operand. The rhs
operand must be non-negative.

Example:

```
%2 = firrtl.integer.shr %0, %1 : (!firrtl.integer, !firrtl.integer) ->
                                     !firrtl.integer
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | integer type |
| `rhs` | integer type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | integer type |

### `firrtl.invalidvalue` (::circt::firrtl::InvalidValueOp)

*InvalidValue primitive*

Syntax:

```
operation ::= `firrtl.invalidvalue` attr-dict `:` qualified(type($result))
```

The InvalidValue operation returns an invalid value of a specified type:

```
  %result = firrtl.invalid : !firrtl.uint<1>
```

This corresponds to the FIRRTL invalidate operation without the implicit
connect semantics. Each invalid op produces a unique invalid value.
InvalidOp is not constant-like

Interfaces: `HasCustomSSAName`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Allocate on ::mlir::SideEffects::DefaultResource}`

#### Results:

| Result | Description |
| --- | --- |
| `result` | a base type |

### `firrtl.istag` (::circt::firrtl::IsTagOp)

*Test the active variant of an enumeration*

This operation is used to test the active variant of an enumeration. The
tag tested for must be one of the possible variants of the input type. If
the tag is the currently active variant the result will be 1, otherwise the
result will be 0.

Example:

```
  %0 = firrtl.istag A %v : !firrtl.enum<A: UInt<0>, B: UInt<0>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | enum type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.leq` (::circt::firrtl::LEQPrimOp)

Syntax:

```
operation ::= `firrtl.leq` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.and` (::circt::firrtl::LTLAndIntrinsicOp)

*FIRRTL variant of `ltl.and`*

Syntax:

```
operation ::= `firrtl.int.ltl.and` operands attr-dict `:` functional-type(operands, results)
```

See `ltl.and` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | 1-bit uint |
| `rhs` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.clock` (::circt::firrtl::LTLClockIntrinsicOp)

*FIRRTL variant of `ltl.clock`*

Syntax:

```
operation ::= `firrtl.int.ltl.clock` $input `,` $clock attr-dict `:` functional-type(operands, results)
```

See `ltl.clock` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | 1-bit uint |
| `clock` | clock |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.concat` (::circt::firrtl::LTLConcatIntrinsicOp)

*FIRRTL variant of `ltl.concat`*

Syntax:

```
operation ::= `firrtl.int.ltl.concat` operands attr-dict `:` functional-type(operands, results)
```

See `ltl.concat` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | 1-bit uint |
| `rhs` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.delay` (::circt::firrtl::LTLDelayIntrinsicOp)

*FIRRTL variant of `ltl.delay`*

Syntax:

```
operation ::= `firrtl.int.ltl.delay` $input `,` $delay (`,` $length^)? attr-dict `:`
              functional-type(operands, results)
```

See `ltl.delay` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `delay` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `length` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.eventually` (::circt::firrtl::LTLEventuallyIntrinsicOp)

*FIRRTL variant of `ltl.eventually`*

Syntax:

```
operation ::= `firrtl.int.ltl.eventually` operands attr-dict `:` functional-type(operands, results)
```

See `ltl.eventually` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.goto_repeat` (::circt::firrtl::LTLGoToRepeatIntrinsicOp)

*FIRRTL variant of `ltl.goto_repeat`*

Syntax:

```
operation ::= `firrtl.int.ltl.goto_repeat` $input `,` $base `,` $more attr-dict `:`
              functional-type(operands, results)
```

See `ltl.goto_repeat` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `base` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `more` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.implication` (::circt::firrtl::LTLImplicationIntrinsicOp)

*FIRRTL variant of `ltl.implication`*

Syntax:

```
operation ::= `firrtl.int.ltl.implication` operands attr-dict `:` functional-type(operands, results)
```

See `ltl.implication` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | 1-bit uint |
| `rhs` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.intersect` (::circt::firrtl::LTLIntersectIntrinsicOp)

*FIRRTL variant of `ltl.intersect`*

Syntax:

```
operation ::= `firrtl.int.ltl.intersect` operands attr-dict `:` functional-type(operands, results)
```

See `ltl.intersect` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | 1-bit uint |
| `rhs` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.non_consecutive_repeat` (::circt::firrtl::LTLNonConsecutiveRepeatIntrinsicOp)

*FIRRTL variant of `ltl.non_consecutive_repeat`*

Syntax:

```
operation ::= `firrtl.int.ltl.non_consecutive_repeat` $input `,` $base `,` $more attr-dict `:`
              functional-type(operands, results)
```

See `ltl.non_consecutive_repeat` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `base` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `more` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.not` (::circt::firrtl::LTLNotIntrinsicOp)

*FIRRTL variant of `ltl.not`*

Syntax:

```
operation ::= `firrtl.int.ltl.not` operands attr-dict `:` functional-type(operands, results)
```

See `ltl.not` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.or` (::circt::firrtl::LTLOrIntrinsicOp)

*FIRRTL variant of `ltl.or`*

Syntax:

```
operation ::= `firrtl.int.ltl.or` operands attr-dict `:` functional-type(operands, results)
```

See `ltl.or` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | 1-bit uint |
| `rhs` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.repeat` (::circt::firrtl::LTLRepeatIntrinsicOp)

*FIRRTL variant of `ltl.repeat`*

Syntax:

```
operation ::= `firrtl.int.ltl.repeat` $input `,` $base (`,` $more^)? attr-dict `:`
              functional-type(operands, results)
```

See `ltl.repeat` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `base` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `more` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.ltl.until` (::circt::firrtl::LTLUntilIntrinsicOp)

*FIRRTL variant of `ltl.until`*

Syntax:

```
operation ::= `firrtl.int.ltl.until` operands attr-dict `:` functional-type(operands, results)
```

See `ltl.until` op in the LTL dialect.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | 1-bit uint |
| `rhs` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.lt` (::circt::firrtl::LTPrimOp)

Syntax:

```
operation ::= `firrtl.lt` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.list.concat` (::circt::firrtl::ListConcatOp)

*Concatenate multiple lists to produce a new list*

Syntax:

```
operation ::= `firrtl.list.concat` $subLists attr-dict `:` type($result)
```

Produces a value of list type by concatenating the provided lists.

Example:

```
%3 = firrtl.list.concat %0, %1, %2 : !firrtl.list<string>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `subLists` | variadic of list type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | list type |

### `firrtl.list.create` (::circt::firrtl::ListCreateOp)

*Produce a list value*

Produces a value of list type containing the provided elements.

Example:

```
%3 = firrtl.list.create %0, %1, %2 : !firrtl.list<string>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `elements` | variadic of property type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | list type |

### `firrtl.mul` (::circt::firrtl::MulPrimOp)

Syntax:

```
operation ::= `firrtl.mul` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint or uint type |

### `firrtl.multibit_mux` (::circt::firrtl::MultibitMuxOp)

*Multibit multiplexer*

The multibit mux expression dynamically selects operands. The
index must be an expression with an unsigned integer type.

```
  %result = firrtl.multibit_mux %index,
            %v_{n-1}, ..., %v_2, %v_1, %v_0  : t1, t2
```

The order of operands is defined in the same way as hw dialect.
For the example above, if `%index` is 0, then the value is `%v_0`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `index` | uint type |
| `inputs` | variadic of a base type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a base type |

### `firrtl.int.mux2cell` (::circt::firrtl::Mux2CellIntrinsicOp)

*An intrinsic lowered into 2-to-1 MUX cell in synthesis tools.*

Syntax:

```
operation ::= `firrtl.int.mux2cell` `(` operands `)` attr-dict `:` functional-type(operands, $result)
```

This intrinsic exposes a low-level API to use 2-to-1 MUX cell in backend
synthesis tool. At FIRRTL level, this operation participates
the inference process in the same way as a normal mux operation.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `sel` | uint with width less than or equal to 1 bits or uint with uninferred width |
| `high` | a passive base type (contain no flips) |
| `low` | a passive base type (contain no flips) |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a passive base type (contain no flips) |

### `firrtl.int.mux4cell` (::circt::firrtl::Mux4CellIntrinsicOp)

*An intrinsic lowered into 4-to-1 MUX cell in synthesis tools.*

Syntax:

```
operation ::= `firrtl.int.mux4cell` `(` operands `)` attr-dict `:` functional-type(operands, $result)
```

This intrinsic exposes a low-level API to use 4-to-1 MUX cell in backend
synthesis tool. At FIRRTL level, this operation participates
the inference process as a sugar of mux operation chains.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `sel` | uint with width less than or equal to 2 bits or uint with uninferred width |
| `v3` | a passive base type (contain no flips) |
| `v2` | a passive base type (contain no flips) |
| `v1` | a passive base type (contain no flips) |
| `v0` | a passive base type (contain no flips) |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a passive base type (contain no flips) |

### `firrtl.mux` (::circt::firrtl::MuxPrimOp)

Syntax:

```
operation ::= `firrtl.mux` `(` operands `)` attr-dict `:` functional-type(operands, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `sel` | uint with width less than or equal to 1 bits or uint with uninferred width |
| `high` | a passive base type (contain no flips) |
| `low` | a passive base type (contain no flips) |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a passive base type (contain no flips) |

### `firrtl.neq` (::circt::firrtl::NEQPrimOp)

Syntax:

```
operation ::= `firrtl.neq` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.neg` (::circt::firrtl::NegPrimOp)

Syntax:

```
operation ::= `firrtl.neg` $input attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint type |

### `firrtl.not` (::circt::firrtl::NotPrimOp)

Syntax:

```
operation ::= `firrtl.not` $input attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | uint type |

### `firrtl.object.anyref_cast` (::circt::firrtl::ObjectAnyRefCastOp)

*Cast object reference to anyref.*

Syntax:

```
operation ::= `firrtl.object.anyref_cast` $input attr-dict `:` type($input)
```

Cast any object reference to AnyRef type. This is needed for passing objects
of a known class to sinks that accept any reference.

Example

```
  %0= firrtl.object.anyref_cast %object : !firrtl.class<@Foo()>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | class type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | any reference type |

### `firrtl.object.subfield` (::circt::firrtl::ObjectSubfieldOp)

*Extract an element of an object*

The object.subfield expression refers to a subelement of an object.

```
%field = firrtl.object.subfield %object[field] : !firrt.class<@Class(field: !firrtl.string)>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `index` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | class type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | property type |

### `firrtl.opensubfield` (::circt::firrtl::OpenSubfieldOp)

*Extract a subfield of another value*

The subfield expression refers to a subelement of an expression with a
bundle type.

```
  %result = firrtl.opensubfield %input[field-name] : !input-type
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | open bundle type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | FIRRTLType |

### `firrtl.opensubindex` (::circt::firrtl::OpenSubindexOp)

*Extract an element of a vector value*

Syntax:

```
operation ::= `firrtl.opensubindex` $input `[` $index `]` attr-dict `:` qualified(type($input))
```

The subindex expression statically refers, by index, to a subelement
of an expression with a vector type. The index must be a non-negative
integer and cannot be equal to or exceed the length of the vector it
indexes.

```
  %result = firrtl.opensubindex %input[index] : t1
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `index` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | open vector type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | FIRRTLType |

### `firrtl.or` (::circt::firrtl::OrPrimOp)

Syntax:

```
operation ::= `firrtl.or` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | uint type |

### `firrtl.orr` (::circt::firrtl::OrRPrimOp)

Syntax:

```
operation ::= `firrtl.orr` $input attr-dict `:` functional-type($input, $result)
```

Horizontally reduce a value to one bit, using the ‘or’ operation to merge
bits. `orr(x)` is equivalent to `concat(x, 1b0) != 0`. As such, it
returns 0 for zero-bit-wide operands.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.pad` (::circt::firrtl::PadPrimOp)

Syntax:

```
operation ::= `firrtl.pad` $input `,` $amount attr-dict `:` functional-type($input, $result)
```

Pad the input out to an `amount` wide integer, sign extending or zero
extending according to `input`s type. If `amount` is less than the existing
width of `input`, then input is unmodified.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `amount` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint or uint type |

### `firrtl.path` (::circt::firrtl::PathOp)

*Produce a path value*

Syntax:

```
operation ::= `firrtl.path` $targetKind $target attr-dict
```

Produces a value which represents a path to the target in a design.

Example:

```
hw.hierpath @Path [@Foo::@bar, @Bar]
%wire = firrtl.wire {annotations = [ {class = "circt.tracker", id = distinct[0]<>, circt.nonlocal = @Path} ]} : !firrtl.uint<1>
%0 = firrtl.path reference distinct[0]<>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `targetKind` | ::circt::firrtl::TargetKindAttr | object model target kind |
| `target` | ::mlir::DistinctAttr | distinct attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | path type |

### `firrtl.ref.rwprobe` (::circt::firrtl::RWProbeOp)

*FIRRTL RWProbe*

Syntax:

```
operation ::= `firrtl.ref.rwprobe` $target attr-dict `:` type($result)
```

Create a RWProbe for the target.
Target must be local.

```
  %result = firrtl.ref.rwprobe @mod::@sym : firrtl.rwprobe<t>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InnerRefUserOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `target` | ::circt::hw::InnerRefAttr | Refer to a name inside a module |

#### Results:

| Result | Description |
| --- | --- |
| `result` | rwprobe type |

### `firrtl.ref.cast` (::circt::firrtl::RefCastOp)

*Cast between compatible reference types*

Syntax:

```
operation ::= `firrtl.ref.cast` $input attr-dict `:` functional-type($input, $result)
```

Losslessly cast between compatible reference types.
Source and destination must be recursively identical or destination
has uninferred variants of the corresponding element in source.

```
  %result = firrtl.ref.cast %ref : (t1) -> t2
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | reference type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | reference type |

### `firrtl.ref.resolve` (::circt::firrtl::RefResolveOp)

*FIRRTL Resolve a Reference*

Syntax:

```
operation ::= `firrtl.ref.resolve` $ref attr-dict `:` qualified(type($ref))
```

Resolve a remote reference for reading a remote value.
It takes a RefType input and returns the corresponding BaseType value.
If an XMR is emitted for this reference, it will be at the location
of this operation.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `ref` | reference type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a base type |

### `firrtl.ref.send` (::circt::firrtl::RefSendOp)

*FIRRTL Send through Reference*

Syntax:

```
operation ::= `firrtl.ref.send` $base attr-dict `:` qualified(type($base))
```

Endpoint of a remote reference. Send a value through a reference
to be read from the firrtl.ref.resolve op.
It takes a BaseType input and returns the corresponding RefType value.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `base` | a base type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | reference type |

### `firrtl.ref.sub` (::circt::firrtl::RefSubOp)

*Extract an element of an aggregate RefType value*

Syntax:

```
operation ::= `firrtl.ref.sub` $input `[` $index `]` attr-dict `:` qualified(type($input))
```

The refsub expression statically refers, by index, to a sub-element
of an expression with a RefType. The index must be a non-negative
integer and cannot be equal to or exceed the underlying vector size
or number of elements in bundle.

```
  %result = firrtl.ref.sub %input[index] : t1
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `index` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | reference type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | reference type |

### `firrtl.rem` (::circt::firrtl::RemPrimOp)

Syntax:

```
operation ::= `firrtl.rem` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint or uint type |

### `firrtl.shl` (::circt::firrtl::ShlPrimOp)

Syntax:

```
operation ::= `firrtl.shl` $input `,` $amount attr-dict `:` functional-type($input, $result)
```

The `shl` operation concatenates `amount` zero bits to the least significant
end of `input`. `amount` must be non-negative.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `amount` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint or uint type |

### `firrtl.shr` (::circt::firrtl::ShrPrimOp)

Syntax:

```
operation ::= `firrtl.shr` $input `,` $amount attr-dict `:` functional-type($input, $result)
```

The `shr` operation truncates least significant `amount` bits from `input`.
If `amount` is greater than of equal to `width(input)`, the value will be
zero for unsigned types and the sign bit for signed types. `amount` must be
non-negative.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `amount` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint or uint type |

### `firrtl.int.sizeof` (::circt::firrtl::SizeOfIntrinsicOp)

Syntax:

```
operation ::= `firrtl.int.sizeof` $input attr-dict `:` functional-type($input, $result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a base type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 32-bit uint |

### `firrtl.specialconstant` (::circt::firrtl::SpecialConstantOp)

*Produce a constant Reset or Clock value*

The constant operation produces a constant value of Reset, AsyncReset, or
Clock type. The value can only be 0 or 1.

```
  %result = firrtl.specialconstant 1 : !firrtl.clock
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::BoolAttr | bool attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | clock or reset type or async reset type |

### `firrtl.string` (::circt::firrtl::StringConstantOp)

*Produce a constant string value*

Syntax:

```
operation ::= `firrtl.string` $value attr-dict
```

Produces a constant value of string type.

Example:

```
%0 = firrtl.string "hello world"
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | string type |

### `firrtl.sub` (::circt::firrtl::SubPrimOp)

Syntax:

```
operation ::= `firrtl.sub` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | sint or uint type |

### `firrtl.subaccess` (::circt::firrtl::SubaccessOp)

*Extract a dynamic element of a vector value*

Syntax:

```
operation ::= `firrtl.subaccess` $input `[` $index `]` attr-dict `:` qualified(type($input)) `,` qualified(type($index))
```

The subaccess expression dynamically refers to a subelement of a
vector-typed expression using a calculated index. The index must be an
expression with an unsigned integer type.

```
  %result = firrtl.subaccess %input[%idx] : t1, t2
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | vector type |
| `index` | uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a base type |

### `firrtl.subfield` (::circt::firrtl::SubfieldOp)

*Extract a subfield of another value*

The subfield expression refers to a subelement of an expression with a
bundle type.

```
  %result = firrtl.subfield %input[field-name] : !input-type
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | bundle type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a base type |

### `firrtl.subindex` (::circt::firrtl::SubindexOp)

*Extract an element of a vector value*

Syntax:

```
operation ::= `firrtl.subindex` $input `[` $index `]` attr-dict `:` qualified(type($input))
```

The subindex expression statically refers, by index, to a subelement
of an expression with a vector type. The index must be a non-negative
integer and cannot be equal to or exceed the length of the vector it
indexes.

```
  %result = firrtl.subindex %input[index] : t1
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `index` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | vector type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a base type |

### `firrtl.subtag` (::circt::firrtl::SubtagOp)

*Extract an element of a enum value*

The subtag expression refers to a subelement of a
enum-typed expression.

```
  %result = firrtl.subtag %input[field-name] : !input-type
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fieldIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | enum type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a base type |

### `firrtl.tagextract` (::circt::firrtl::TagExtractOp)

*Extract the tag from a value*

Syntax:

```
operation ::= `firrtl.tagextract` $input attr-dict `:` qualified(type($input))
```

The tagextract expression returns the binary value of the current tag of an
enum value.

```
  %result = firrtl.tagextract %input : !input-type
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | enum type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | uint type |

### `firrtl.tail` (::circt::firrtl::TailPrimOp)

Syntax:

```
operation ::= `firrtl.tail` $input `,` $amount attr-dict `:` functional-type($input, $result)
```

The `tail` operation truncates the `amount` most significant bits from
`input`. `amount` must be non-negative and less than or equal to the bit
width of e. The result is `width(input)-amount` bits wide.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `amount` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | uint type |

### `firrtl.fstring.time` (::circt::firrtl::TimeOp)

*An operation that represents the current simulation time*

Syntax:

```
operation ::= `firrtl.fstring.time` attr-dict `:` type($result)
```

This operation represents the current simulation time. It returns a format
string type which can only be used as a substitution inside a printf. This
operation is not represented in the FIRRTL spec.

This operation is the FIRRTL Dialect representation of the special
substitution `{{SimulationTime}}`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| --- | --- |
| `result` | a format string type |

### `firrtl.resetCast` (::circt::firrtl::UninferredResetCastOp)

Syntax:

```
operation ::= `firrtl.resetCast` $input attr-dict `:` functional-type($input, $result)
```

Cast between reset types. This is used to enable matchingconnects early in
the pipeline by isolating all uninferred reset connections to a single op.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | Reset |

#### Results:

| Result | Description |
| --- | --- |
| `result` | Reset |

### `firrtl.unknown` (::circt::firrtl::UnknownValueOp)

*A property with an unknown value used to tie-off connections*

Syntax:

```
operation ::= `firrtl.unknown` attr-dict `:` type($result)
```

An operation that is used to tie-off property connections where we do not
care about the value.

This lowers to an `om::UnknownOp`.

Interfaces: `SymbolUserOpInterface`

#### Results:

| Result | Description |
| --- | --- |
| `result` | property type |

### `firrtl.unresolved_path` (::circt::firrtl::UnresolvedPathOp)

*Produce a path value*

Syntax:

```
operation ::= `firrtl.unresolved_path` $target attr-dict
```

Produces a value which represents a path to the target in a design.

Example:

```
0 = firrtl.unresolved_path "~Circuit|Module>w"
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `target` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `result` | path type |

### `firrtl.unsafe_domain_cast` (::circt::firrtl::UnsafeDomainCastOp)

*Cast an operand from one domain to another*

Syntax:

```
operation ::= `firrtl.unsafe_domain_cast` $input (`domains` $domains^)? attr-dict-with-keyword `:` qualified(type($input))
```

This operation is used to *unsafely* convert from the domain of an operand
to another domain. This operation is inherently unsafe and unchecked.

This operation is necessary in order to describe certain constructs like
synchronizers which, from a domain type system perspective, violate normal
constraints.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | a passive base type (contain no flips) |
| `domains` | variadic of a domain type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a passive base type (contain no flips) |

### `firrtl.vectorcreate` (::circt::firrtl::VectorCreateOp)

*Produce a vector value*

Syntax:

```
operation ::= `firrtl.vectorcreate` $fields attr-dict `:` functional-type($fields, $result)
```

Create a vector from component values. This is equivalent in terms of
flow to creating a node. The first operand indicates 0-th element of
the result.

```
  %result = firrtl.vectorcreate %1, %2, %3 : !firrtl.vector<uint<8>, 3>
```

#### Operands:

| Operand | Description |
| --- | --- |
| `fields` | variadic of a base type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | vector type |

### `firrtl.verbatim.expr` (::circt::firrtl::VerbatimExprOp)

*Expression that expands to a value given SystemVerilog text*

Syntax:

```
operation ::= `firrtl.verbatim.expr` $text (`(` $substitutions^ `)`)?
              `:` functional-type($substitutions, $result) attr-dict
```

This operation produces a typed value expressed by a string of
SystemVerilog. This can be used to access macros and other values that are
only sensible as Verilog text.

The text string is expected to have the highest precedence, so you should
include parentheses in the string if it isn’t a single token. This is also
assumed to not have side effects (use `sv.verbatim` if you need them).

`firrtl.verbatim.expr` allows operand substitutions with `{{0}}` syntax.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `text` | ::mlir::StringAttr | string attribute |
| `symbols` | ::mlir::ArrayAttr | name reference array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `substitutions` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a base type |

### `firrtl.verbatim.wire` (::circt::firrtl::VerbatimWireOp)

*Expression with wire semantics that expands to a value given SystemVerilog text*

Syntax:

```
operation ::= `firrtl.verbatim.wire` $text (`(` $substitutions^ `)`)?
              `:` functional-type($substitutions, $result) attr-dict
```

This operation produces a typed value with wire semantics, expressed by a
string of SystemVerilog. This can be used to access macros and other values
that are only sensible as Verilog text.

The text string is expected to have the highest precedence, so you should
include parentheses in the string if it isn’t a single token. This is also
assumed to not have side effects (use `sv.verbatim` if you need them).

`firrtl.verbatim.wire` allows operand substitutions with `{{0}}` syntax.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `text` | ::mlir::StringAttr | string attribute |
| `symbols` | ::mlir::ArrayAttr | name reference array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `substitutions` | variadic of any type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a base type |

### `firrtl.xmr.deref` (::circt::firrtl::XMRDerefOp)

*FIRRTL XMR operation, reading an XMR target.*

Syntax:

```
operation ::= `firrtl.xmr.deref` $ref (`,` $verbatimSuffix^)? attr-dict `:` qualified(type($dest))
```

A “read at a distance”, taking a (symbol-ref to a) hierarchical path,
returning the value of the target.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `ref` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `verbatimSuffix` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `dest` | a passive base type (contain no flips) |

### `firrtl.xmr.ref` (::circt::firrtl::XMRRefOp)

*FIRRTL XMR operation, targetable by ref ops.*

Syntax:

```
operation ::= `firrtl.xmr.ref` $ref (`,` $verbatimSuffix^)? attr-dict `:` qualified(type($dest))
```

Takes a (symbol-ref to a) hierarchical path and returns a reference to
the target. Through this reference, the target can be read or written
“at a distance”.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `ref` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `verbatimSuffix` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `dest` | reference type |

### `firrtl.xor` (::circt::firrtl::XorPrimOp)

Syntax:

```
operation ::= `firrtl.xor` $lhs `,` $rhs  attr-dict `:`
              `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsIntTypeKind`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `lhs` | sint or uint type |
| `rhs` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | uint type |

### `firrtl.xorr` (::circt::firrtl::XorRPrimOp)

Syntax:

```
operation ::= `firrtl.xorr` $input attr-dict `:` functional-type($input, $result)
```

Horizontally reduce a value to one bit, using the ‘xor’ operation to merge
bits. `xorr(x)` is equivalent to `popcount(concat(x, 1b0)) & 1`. As
such, it returns 0 for zero-bit-wide operands.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | sint or uint type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

Operation Definitions – Intrinsics
----------------------------------

### `firrtl.int.clock_div` (::circt::firrtl::ClockDividerIntrinsicOp)

*Produces a clock divided by a power of two*

Syntax:

```
operation ::= `firrtl.int.clock_div` $input `by` $pow2 attr-dict
```

The `firrtl.int.clock_div` takes a clock signal and divides it by a
power-of-two ratio. The output clock is phase-aligned to the input clock.

```
%div_clock = seq.clock_div %clock by 1
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `pow2` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | clock |

#### Results:

| Result | Description |
| --- | --- |
| `output` | clock |

### `firrtl.int.clock_gate` (::circt::firrtl::ClockGateIntrinsicOp)

*Safely gates a clock with an enable signal*

Syntax:

```
operation ::= `firrtl.int.clock_gate` $input `,` $enable (`,` $test_enable^)? attr-dict
```

The `int.clock_gate` enables and disables a clock safely, without glitches,
based on a boolean enable value. If the enable input is 1, the output clock
produced by the clock gate is identical to the input clock. If the enable
input is 0, the output clock is a constant zero.

The enable input is sampled at the rising edge of the input clock; any
changes on the enable before or after that edge are ignored and do not
affect the output clock.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | clock |
| `enable` | 1-bit uint |
| `test_enable` | 1-bit uint |

#### Results:

| Result | Description |
| --- | --- |
| `output` | clock |

### `firrtl.int.clock_inv` (::circt::firrtl::ClockInverterIntrinsicOp)

*Inverts the clock signal*

Syntax:

```
operation ::= `firrtl.int.clock_inv` $input attr-dict
```

The `firrtl.int.clock.inv` intrinsic takes a clock signal and inverts it.
It can be used to build registers and other operations which are triggered
by a negative clock edge relative to a reference signal. The compiler is
free to optimize inverters (particularly double inverters).

See the corresponding `seq.clock_inv` operation.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | clock |

#### Results:

| Result | Description |
| --- | --- |
| `output` | clock |

### `firrtl.int.dpi.call` (::circt::firrtl::DPICallIntrinsicOp)

*Import and call DPI function*

Syntax:

```
operation ::= `firrtl.int.dpi.call` $functionName `(` $inputs `)` (`clock` $clock^)? (`enable` $enable^)?
              attr-dict `:` functional-type($inputs, results)
```

The `int.dpi.call` intrinsic calls an external function.
See Sim dialect DPI call op.

Traits: `AttrSizedOperandSegments`

Interfaces: `CombDataFlow`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `functionName` | ::mlir::StringAttr | string attribute |
| `inputNames` | ::mlir::ArrayAttr | string array attribute |
| `outputName` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | clock |
| `enable` | 1-bit uint |
| `inputs` | variadic of a passive base type (contain no flips) |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a passive base type (contain no flips) |

### `firrtl.int.fpga_probe` (::circt::firrtl::FPGAProbeIntrinsicOp)

*Mark a value to be observed through FPGA debugging facilities*

Syntax:

```
operation ::= `firrtl.int.fpga_probe` $clock `,` $input attr-dict `:` type($input)
```

The `firrtl.int.fpga_probe` intrinsic marks a value in
the IR to be made observable through FPGA debugging facilities. Most FPGAs
offer a form of signal observation or logic analyzer to debug a design. This
operation allows the IR to indicate which signals should be made observable
for debugging. Later FPGA-specific passes may then pick this information up
and materialize the necessary logic analyzers or tool scripts.

#### Operands:

| Operand | Description |
| --- | --- |
| `input` | any type |
| `clock` | clock |

### `firrtl.int.generic` (::circt::firrtl::GenericIntrinsicOp)

*Generic intrinsic operation for FIRRTL intrinsics.*

Syntax:

```
operation ::= `firrtl.int.generic` $intrinsic custom<ParameterList>($parameters) ($operands^)? attr-dict-with-keyword `:` functional-type($operands, $result)
```

Interfaces: `HasCustomSSAName`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `intrinsic` | ::mlir::StringAttr | string attribute |
| `parameters` | ::mlir::ArrayAttr | parameter array attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `operands` | variadic of a passive base type (contain no flips) |

#### Results:

| Result | Description |
| --- | --- |
| `result` | a passive base type (contain no flips) |

### `firrtl.int.has_been_reset` (::circt::firrtl::HasBeenResetIntrinsicOp)

*Check that a proper reset has been seen.*

Syntax:

```
operation ::= `firrtl.int.has_been_reset` $clock `,` $reset attr-dict `:` type($reset)
```

The result of `firrtl.int.has_been_reset` reads as 0 immediately after simulation
startup and after each power-cycle in a power-aware simulation. The result
remains 0 before and during reset and only switches to 1 after the reset is
deasserted again.

See the corresponding `verif.has_been_reset` operation.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `clock` | clock |
| `reset` | Reset |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.isX` (::circt::firrtl::IsXIntrinsicOp)

*Test for ‘x*

Syntax:

```
operation ::= `firrtl.int.isX` $arg attr-dict `:` type($arg)
```

The `int.isX` expression checks that the operand is not a verilog literal
‘x. FIRRTL doesn’t have a notion of ‘x per-se, but x can come in to the
system from external modules and from SV constructs. Verification
constructs need to explicitly test for ‘x.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| --- | --- |
| `arg` | a base type |

#### Results:

| Result | Description |
| --- | --- |
| `result` | 1-bit uint |

### `firrtl.int.plusargs.test` (::circt::firrtl::PlusArgsTestIntrinsicOp)

*SystemVerilog `$test$plusargs` call*

Syntax:

```
operation ::= `firrtl.int.plusargs.test` $formatString attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `formatString` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `found` | 1-bit uint |

### `firrtl.int.plusargs.value` (::circt::firrtl::PlusArgsValueIntrinsicOp)

*SystemVerilog `$value$plusargs` call*

Syntax:

```
operation ::= `firrtl.int.plusargs.value` $formatString attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasCustomSSAName`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `formatString` | ::mlir::StringAttr | string attribute |

#### Results:

| Result | Description |
| --- | --- |
| `found` | 1-bit uint |
| `result` | any type |

### `firrtl.int.unclocked_assume` (::circt::firrtl::UnclockedAssumeIntrinsicOp)

*Special Assume Verification Statement to assume predicate*

Syntax:

```
operation ::= `firrtl.int.unclocked_assume` $predicate `,` $enable `,` $message (`(` $substitutions^ `)`)? `:`
              type($predicate) `,` type($enable) (`,` qualified(type($substitutions))^)?
              custom<VerifAttrs>(attr-dict)
```

The `firrtl.int.unclocked_assume` intrinsic is a special assume statement
lowered into a SV concurrent assertion within always block that has the assumed
predicate in a sensitivity list.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `message` | ::mlir::StringAttr | string attribute |
| `name` | ::mlir::StringAttr | string attribute |

#### Operands:

| Operand | Description |
| --- | --- |
| `predicate` | 1-bit uint |
| `enable` | 1-bit uint |
| `substitutions` | variadic of any type |

### `firrtl.view` (::circt::firrtl::ViewIntrinsicOp)

*A SystemVerilog Interface only usable for waveform debugging*

Syntax:

```
operation ::= `firrtl.view` $name `,` (`yaml` $yamlFile^ `,`)? $augmentedType (`,` $inputs^)? attr-dict (`:` type($inputs)^)?
```

This will become a SystemVerilog Interface that is driven by its arguments.
This is *not* intended to be used for anything other than assistance when
debugging in a waveform. This is *not* a true SystemVerilog Interface, it
is only lowered to one.

#### Attributes:

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `yamlFile` | ::mlir::StringAttr | string attribute |
| `augmentedType` | ::circt::firrtl::AugmentedBundleTypeAttr | GrandCentral AugmentedBundleType |

#### Operands:

| Operand | Description |
| --- | --- |
| `inputs` | variadic of a ground type |

Type Definitions
----------------

### AnalogType

*Analog signal*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| widthOrSentinel | `int32_t` |  |
| isConst | `bool` |  |

### AnyRefType

*A reference to an instance of any class.*

A reference of type AnyRef can be used in ports, property assignments, and
any other Property “plumbing” ops. But it is opaque, and references to
objects of AnyRef type cannot be “dereferenced”. There is no information
about the referred to object’s fields, so subfield access, etc. is illegal.

### AsyncResetType

*AsyncReset signal*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| isConst | `bool` |  |

### BaseTypeAliasType

*Type alias for firrtl base types*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| name | `StringAttr` |  |
| innerType | `::circt::firrtl::FIRRTLBaseType` | An inner type |

### BoolType

*A boolean property. Not representable in hardware.*

### BundleType

*An aggregate of named elements. This is effectively a struct.*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elements | `ArrayRef<BundleElement>` |  |
| isConst | `bool` |  |

### ClassType

\_An instance of a class.

```
Example:
```mlir
!firrtl.class<@Module(in p0: !firrtl.uint<8>, out p1: !firrtl.uint<8>)>
```_
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| name | `FlatSymbolRefAttr` |  |
| elements | `::llvm::ArrayRef<ClassElement>` |  |

### ClockType

*Clock signal*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| isConst | `bool` |  |

### DomainType

*A domain type*

### DoubleType

*A double property. Not representable in hardware.*

### FEnumType

*A sum type of named elements.*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elements | `ArrayRef<EnumElement>` |  |
| isConst | `bool` |  |

### FStringType

*A format string type*

### FVectorType

*A fixed size collection of elements, like an array.*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `::circt::firrtl::FIRRTLBaseType` | Type of vector elements |
| numElements | `size_t` |  |
| isConst | `bool` |  |

### FIntegerType

*An unlimited length signed integer type. Not representable in hardware.*

### LHSType

*A wrapper for LHS types.*

A LHS type is a type usable for the destination of a strict connect
and for field indexing. No other operations are valid. Any passive,
strict-connectable types are valid inside LHS types.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| type | `FIRRTLBaseType` |  |

### ListType

*A typed property list of any length. Not representable in hardware.*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `circt::firrtl::PropertyType` | element type |

### OpenBundleType

*An aggregate of named elements. This is effectively a struct.*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elements | `ArrayRef<BundleElement>` |  |
| isConst | `bool` |  |

### OpenVectorType

*A fixed size collection of elements, like an array.*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| elementType | `::circt::firrtl::FIRRTLType` | Type of vector elements |
| numElements | `size_t` |  |
| isConst | `bool` |  |

### PathType

*A path to a hardware entity. Not representable in hardware.*

### RefType

*A reference to a signal elsewhere.*

A reference type, such as `firrtl.probe<uint<1>>` or `firrtl.rwprobe<uint<2>>`.

Used for remote reads and writes of the wrapped base type.

Parameterized over the referenced base type, with flips removed.

Not a base type.

Values of this type are used to capture dataflow paths,
and do not represent a circuit element or entity.

Generally read-only (probe), optionally forceable (rwprobe).

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| type | `::circt::firrtl::FIRRTLBaseType` | Type of reference target |
| forceable | `bool` |  |
| layer | `::mlir::SymbolRefAttr` |  |

### ResetType

*Reset Signal*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| isConst | `bool` |  |

### SIntType

*A signed integer type, whose width may not be known.*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| widthOrSentinel | `int32_t` |  |
| isConst | `bool` |  |

### StringType

*An unlimited length string type. Not representable in hardware.*

### UIntType

*An unsigned integer type, whose width may not be known.*

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| widthOrSentinel | `int32_t` |  |
| isConst | `bool` |  |

Attribute Definitions
---------------------

### AugmentedBundleTypeAttr

*GrandCentral AugmentedBundleType*

Syntax:

```
#firrtl.augmentedBundle<
  DictionaryAttr   # underlying
>
```

Used in the GrandCentralPass.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| underlying | `DictionaryAttr` |  |

### AugmentedGroundTypeAttr

*GrandCentral AugmentedGroundType*

Syntax:

```
#firrtl.augmentedGround<
  DictionaryAttr   # underlying
>
```

Used in the GrandCentralPass.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| underlying | `DictionaryAttr` |  |

### AugmentedVectorTypeAttr

*GrandCentral AugmentedVectorType*

Syntax:

```
#firrtl.augmentedVector<
  DictionaryAttr   # underlying
>
```

Used in the GrandCentralPass.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| underlying | `DictionaryAttr` |  |

### DomainFieldAttr

*A single field of a domain*

Syntax:

```
#firrtl.domain.field<
  ::mlir::StringAttr,   # name
  ::circt::firrtl::PropertyType   # type
>
```

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| name | `::mlir::StringAttr` |  |
| type | `::circt::firrtl::PropertyType` |  |

### MemoryInitAttr

*Information about the initial state of a memory*

Syntax:

```
#firrtl.meminit<
  ::mlir::StringAttr,   # filename
  bool,   # isBinary
  bool   # isInline
>
```

This attribute captures information about the external initialization of a
memory. This is the FIRRTL Dialect representation of both
“firrtl.annotations.LoadMemoryFromFile” and
“firrtl.annotations.MemoryFileInlineAnnotation”.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| filename | `::mlir::StringAttr` |  |
| isBinary | `bool` |  |
| isInline | `bool` |  |

### ParamDeclAttr

*Module or instance parameter definition*

An attribute describing a module parameter, or instance parameter
specification.

#### Parameters:

| Parameter | C++ type | Description |
| --- | --- | --- |
| name | `::mlir::StringAttr` |  |
| type | `::mlir::Type` |  |
| value | `::mlir::Attribute` |  |

ClassLike (`ClassLike`)
-----------------------

Provide common class information.

### Methods:

#### `getInstanceType`

```
ClassType getInstanceType();
```

Get the type for instances of this class

NOTE: This method *must* be implemented by the user.

#### `verifyType`

```
LogicalResult verifyType(::circt::firrtl::ClassType type, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
```

Verify that the given type agrees with this class

NOTE: This method *must* be implemented by the user.

FConnectLike (`FConnectLike`)
-----------------------------

Provide common connection information.

### Methods:

#### `getDest`

```
Value getDest();
```

Return a destination of connection.

NOTE: This method *must* be implemented by the user.

#### `getSrc`

```
Value getSrc();
```

Return a source of connection.

NOTE: This method *must* be implemented by the user.

#### `getConnectBehaviorKind`

```
static ConnectBehaviorKind getConnectBehaviorKind();
```

Return connection behavior kind.

NOTE: This method *must* be implemented by the user.

#### `hasStaticSingleConnectBehavior`

```
static bool hasStaticSingleConnectBehavior();
```

Returns true if ConnectBehavior is StaticSingleConnect.

#### `hasLastConnectBehavior`

```
static bool hasLastConnectBehavior();
```

Returns true if ConnectBehavior is LastConnect.

FInstanceLike (`FInstanceLike`)
-------------------------------

Provide common instance information.

### Methods:

#### `getReferencedModuleName`

```
::llvm::StringRef getReferencedModuleName();
```

Get the name of the instantiated module

NOTE: This method *must* be implemented by the user.

#### `getReferencedModuleNameAttr`

```
::mlir::StringAttr getReferencedModuleNameAttr();
```

Get the name of the instantiated module

NOTE: This method *must* be implemented by the user.

#### `getReferencedOperation`

```
::mlir::Operation *getReferencedOperation(const SymbolTable&symtbl);
```

Get the referenced module via a symbol table.

NOTE: This method *must* be implemented by the user.

FModuleLike (`FModuleLike`)
---------------------------

Provide common module information.

### Methods:

#### `getParameters`

```
ArrayAttr getParameters();
```

Get the parameters

NOTE: This method *must* be implemented by the user.

#### `getConventionAttr`

```
ConventionAttr getConventionAttr();
```

Get the module’s instantiation convention

NOTE: This method *must* be implemented by the user.

#### `getConvention`

```
Convention getConvention();
```

Get the module’s instantiation convention

NOTE: This method *must* be implemented by the user.

#### `getLayersAttr`

```
ArrayAttr getLayersAttr();
```

Get the module’s enabled layers

NOTE: This method *must* be implemented by the user.

#### `getLayers`

```
ArrayRef<Attribute> getLayers();
```

Get the module’s enabled layers.

NOTE: This method *must* be implemented by the user.

#### `getPortDirectionsAttr`

```
mlir::DenseBoolArrayAttr getPortDirectionsAttr();
```

Get the port directions attribute

NOTE: This method *must* be implemented by the user.

#### `getPortDirections`

```
ArrayRef<bool> getPortDirections();
```

Get the port directions

NOTE: This method *must* be implemented by the user.

#### `getPortDirection`

```
Direction getPortDirection(size_t portIndex);
```

Get a port direction

NOTE: This method *must* be implemented by the user.

#### `getPortNamesAttr`

```
ArrayAttr getPortNamesAttr();
```

Get the port names attribute

NOTE: This method *must* be implemented by the user.

#### `getPortNames`

```
ArrayRef<Attribute> getPortNames();
```

Get the port names

NOTE: This method *must* be implemented by the user.

#### `getPortNameAttr`

```
StringAttr getPortNameAttr(size_t portIndex);
```

Get a port name

NOTE: This method *must* be implemented by the user.

#### `getPortName`

```
StringRef getPortName(size_t portIndex);
```

Get a port name

NOTE: This method *must* be implemented by the user.

#### `getPortTypesAttr`

```
ArrayAttr getPortTypesAttr();
```

Get the port types attribute

NOTE: This method *must* be implemented by the user.

#### `getPortTypes`

```
ArrayRef<Attribute> getPortTypes();
```

Get the port types

NOTE: This method *must* be implemented by the user.

#### `setPortTypesAttr`

```
void setPortTypesAttr(::mlir::ArrayAttr portTypes);
```

Set the types of all ports

NOTE: This method *must* be implemented by the user.

#### `getPortTypeAttr`

```
TypeAttr getPortTypeAttr(size_t portIndex);
```

Get a port type

NOTE: This method *must* be implemented by the user.

#### `getPortType`

```
Type getPortType(size_t portIndex);
```

Get a port type

NOTE: This method *must* be implemented by the user.

#### `getPortAnnotationsAttr`

```
ArrayAttr getPortAnnotationsAttr();
```

Get the port annotations attribute

NOTE: This method *must* be implemented by the user.

#### `getPortAnnotations`

```
ArrayRef<Attribute> getPortAnnotations();
```

Get the port annotations attribute

NOTE: This method *must* be implemented by the user.

#### `setPortAnnotationsAttr`

```
void setPortAnnotationsAttr(::mlir::ArrayAttr annotations);
```

Set the port annotations attribute

NOTE: This method *must* be implemented by the user.

#### `getAnnotationsAttrForPort`

```
ArrayAttr getAnnotationsAttrForPort(size_t portIndex);
```

Get a port’s annotations attribute

NOTE: This method *must* be implemented by the user.

#### `getAnnotationsForPort`

```
ArrayRef<Attribute> getAnnotationsForPort(size_t portIndex);
```

Get a port’s annotations

NOTE: This method *must* be implemented by the user.

#### `getPortSymbolsAttr`

```
ArrayAttr getPortSymbolsAttr();
```

Get the port symbols attribute

NOTE: This method *must* be implemented by the user.

#### `getPortSymbols`

```
ArrayRef<Attribute> getPortSymbols();
```

Get the port symbols

NOTE: This method *must* be implemented by the user.

#### `getPortSymbolAttr`

```
circt::hw::InnerSymAttr getPortSymbolAttr(size_t portIndex);
```

Get the port symbol attribute

NOTE: This method *must* be implemented by the user.

#### `setPortSymbolsAttr`

```
void setPortSymbolsAttr(::mlir::ArrayAttr symbols);
```

Set the symbols of all ports and their fields

NOTE: This method *must* be implemented by the user.

#### `setPortSymbols`

```
void setPortSymbols(::llvm::ArrayRef<Attribute> symbols);
```

Set the symbols of all ports and their fields

NOTE: This method *must* be implemented by the user.

#### `setPortSymbolAttr`

```
void setPortSymbolAttr(size_t portIndex, circt::hw::InnerSymAttr symbol);
```

Set the symbols for a port including its fields

NOTE: This method *must* be implemented by the user.

#### `getPortLocationsAttr`

```
ArrayAttr getPortLocationsAttr();
```

Get the port locations attribute

NOTE: This method *must* be implemented by the user.

#### `getPortLocations`

```
ArrayRef<Attribute> getPortLocations();
```

Get the port locations attribute

NOTE: This method *must* be implemented by the user.

#### `getPortLocationAttr`

```
LocationAttr getPortLocationAttr(size_t portIndex);
```

Get a port’s location attribute

NOTE: This method *must* be implemented by the user.

#### `getPortLocation`

```
Location getPortLocation(size_t portIndex);
```

Get a port’s location

NOTE: This method *must* be implemented by the user.

#### `getDomainInfoAttr`

```
ArrayAttr getDomainInfoAttr();
```

Get domain information attribute

NOTE: This method *must* be implemented by the user.

#### `getDomainInfo`

```
ArrayRef<Attribute> getDomainInfo();
```

Get domain information

#### `getDomainInfoAttrForPort`

```
Attribute getDomainInfoAttrForPort(size_t portIndex);
```

Get a port’s domain info attribute

NOTE: This method *must* be implemented by the user.

#### `setDomainInfoAttr`

```
void setDomainInfoAttr(::mlir::ArrayAttr domainInfo);
```

Set the domain info of all ports

NOTE: This method *must* be implemented by the user.

#### `getPorts`

```
SmallVector<PortInfo> getPorts();
```

Get information about all ports

NOTE: This method *must* be implemented by the user.

#### `insertPorts`

```
void insertPorts(ArrayRef<std::pair<unsigned, PortInfo>> ports);
```

Inserts the given ports at the corresponding indices

NOTE: This method *must* be implemented by the user.

#### `erasePorts`

```
void erasePorts(const llvm::BitVector&portIndices);
```

Erases the ports that have their corresponding bit set in `portIndices`

NOTE: This method *must* be implemented by the user.

FNamableOp (`FNamableOp`)
-------------------------

This interface provides common methods for namable operations.

### Methods:

#### `getNameAttr`

```
::mlir::StringAttr getNameAttr();
```

Return the name.

NOTE: This method *must* be implemented by the user.

#### `getName`

```
::llvm::StringRef getName();
```

Return the name.

NOTE: This method *must* be implemented by the user.

#### `setNameAttr`

```
void setNameAttr(::mlir::StringAttr name);
```

Set the name.

NOTE: This method *must* be implemented by the user.

#### `setName`

```
void setName(::llvm::StringRef name);
```

Set the name.

NOTE: This method *must* be implemented by the user.

#### `getNameKindAttr`

```
::circt::firrtl::NameKindEnumAttr getNameKindAttr();
```

Get the namekind

NOTE: This method *must* be implemented by the user.

#### `getNameKind`

```
::circt::firrtl::NameKindEnum getNameKind();
```

Get the namekind

NOTE: This method *must* be implemented by the user.

#### `setNameKindAttr`

```
void setNameKindAttr(::circt::firrtl::NameKindEnumAttr nameKind);
```

Set a namekind.

NOTE: This method *must* be implemented by the user.

#### `setNameKind`

```
void setNameKind(::circt::firrtl::NameKindEnum nameKind);
```

Set a namekind.

NOTE: This method *must* be implemented by the user.

#### `hasDroppableName`

```
bool hasDroppableName();
```

Return true if the name is droppable.

NOTE: This method *must* be implemented by the user.

Forceable (`Forceable`)
-----------------------

Interaction with declarations of forceable hardware components,
and managing references to them.

### Methods:

#### `isForceable`

```
bool isForceable();
```

Return true if the operation is forceable.

NOTE: This method *must* be implemented by the user.

#### `getData`

```
mlir::TypedValue<FIRRTLBaseType> getData();
```

Return data value that will be targeted.

NOTE: This method *must* be implemented by the user.

#### `getDataRaw`

```
Value getDataRaw();
```

Return raw data value that will be targeted.

NOTE: This method *must* be implemented by the user.

#### `getDataType`

```
FIRRTLBaseType getDataType();
```

Return data type that will be referenced.

NOTE: This method *must* be implemented by the user.

#### `getDataRef`

```
mlir::TypedValue<RefType> getDataRef();
```

Return reference result, or null if not active.

NOTE: This method *must* be implemented by the user.

'firrtl' Dialect Docs
---------------------

* [FIRRTL Annotations](https://circt.llvm.org/docs/Dialects/FIRRTL/FIRRTLAnnotations/)
* [FIRRTL Dialect Rationale](https://circt.llvm.org/docs/Dialects/FIRRTL/RationaleFIRRTL/)
* [Intrinsics](https://circt.llvm.org/docs/Dialects/FIRRTL/FIRRTLIntrinsics/)

 [Prev - The Elastic Silicon Interconnect dialect](https://circt.llvm.org/docs/Dialects/ESI/RationaleESI/ "The Elastic Silicon Interconnect dialect")
[Next - FIRRTL Annotations](https://circt.llvm.org/docs/Dialects/FIRRTL/FIRRTLAnnotations/ "FIRRTL Annotations") 

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