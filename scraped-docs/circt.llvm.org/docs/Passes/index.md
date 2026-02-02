Passes - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

Passes
======

This document describes the available CIRCT passes and their contracts.

* [General Passes](#general-passes)
  + [`-convert-index-to-uint`](#-convert-index-to-uint)
  + [`-flatten-memref`](#-flatten-memref)
  + [`-flatten-memref-calls`](#-flatten-memref-calls)
  + [`-hierarchical-runner`](#-hierarchical-runner)
  + [`-insert-merge-blocks`](#-insert-merge-blocks)
  + [`-map-arith-to-comb`](#-map-arith-to-comb)
  + [`-maximize-ssa`](#-maximize-ssa)
  + [`-memory-banking`](#-memory-banking)
  + [`-print-op-count`](#-print-op-count)
  + [`-strip-debuginfo-with-pred`](#-strip-debuginfo-with-pred)
  + [`-switch-to-if`](#-switch-to-if)
* [Conversion Passes](#conversion-passes)
  + [`-calyx-native`](#-calyx-native)
  + [`-calyx-remove-groups-fsm`](#-calyx-remove-groups-fsm)
  + [`-convert-affine-to-loopschedule`](#-convert-affine-to-loopschedule)
  + [`-convert-comb-to-arith`](#-convert-comb-to-arith)
  + [`-convert-comb-to-datapath`](#-convert-comb-to-datapath)
  + [`-convert-comb-to-smt`](#-convert-comb-to-smt)
  + [`-convert-comb-to-synth`](#-convert-comb-to-synth)
  + [`-convert-core-to-fsm`](#-convert-core-to-fsm)
  + [`-convert-datapath-to-comb`](#-convert-datapath-to-comb)
  + [`-convert-datapath-to-smt`](#-convert-datapath-to-smt)
  + [`-convert-fsm-to-core`](#-convert-fsm-to-core)
  + [`-convert-fsm-to-sv`](#-convert-fsm-to-sv)
  + [`-convert-hw-to-btor2`](#-convert-hw-to-btor2)
  + [`-convert-hw-to-llvm`](#-convert-hw-to-llvm)
  + [`-convert-hw-to-smt`](#-convert-hw-to-smt)
  + [`-convert-hw-to-systemc`](#-convert-hw-to-systemc)
  + [`-convert-moore-to-core`](#-convert-moore-to-core)
  + [`-convert-synth-to-comb`](#-convert-synth-to-comb)
  + [`-convert-to-arcs`](#-convert-to-arcs)
  + [`-convert-to-llvm`](#-convert-to-llvm)
  + [`-convert-verif-to-smt`](#-convert-verif-to-smt)
  + [`-export-split-verilog`](#-export-split-verilog)
  + [`-export-verilog`](#-export-verilog)
  + [`-handshake-remove-block-structure`](#-handshake-remove-block-structure)
  + [`-hw-lower-instance-choices`](#-hw-lower-instance-choices)
  + [`-legalize-anon-enums`](#-legalize-anon-enums)
  + [`-lower-arc-to-llvm`](#-lower-arc-to-llvm)
  + [`-lower-calyx-to-fsm`](#-lower-calyx-to-fsm)
  + [`-lower-calyx-to-hw`](#-lower-calyx-to-hw)
  + [`-lower-cf-to-handshake`](#-lower-cf-to-handshake)
  + [`-lower-dc-to-hw`](#-lower-dc-to-hw)
  + [`-lower-firrtl-to-hw`](#-lower-firrtl-to-hw)
  + [`-lower-handshake-to-dc`](#-lower-handshake-to-dc)
  + [`-lower-handshake-to-hw`](#-lower-handshake-to-hw)
  + [`-lower-hw-to-sv`](#-lower-hw-to-sv)
  + [`-lower-hwarith-to-hw`](#-lower-hwarith-to-hw)
  + [`-lower-loopschedule-to-calyx`](#-lower-loopschedule-to-calyx)
  + [`-lower-ltl-to-core`](#-lower-ltl-to-core)
  + [`-lower-pipeline-to-hw`](#-lower-pipeline-to-hw)
  + [`-lower-scf-to-calyx`](#-lower-scf-to-calyx)
  + [`-lower-seq-firmem`](#-lower-seq-firmem)
  + [`-lower-seq-to-sv`](#-lower-seq-to-sv)
  + [`-lower-sim-to-sv`](#-lower-sim-to-sv)
  + [`-lower-smt-to-z3-llvm`](#-lower-smt-to-z3-llvm)
  + [`-lower-verif-to-sv`](#-lower-verif-to-sv)
  + [`-materialize-calyx-to-fsm`](#-materialize-calyx-to-fsm)
  + [`-prepare-for-emission`](#-prepare-for-emission)
  + [`-test-apply-lowering-options`](#-test-apply-lowering-options)
* [Arc Dialect Passes](#arc-dialect-passes)
  + [`-arc-add-taps`](#-arc-add-taps)
  + [`-arc-allocate-state`](#-arc-allocate-state)
  + [`-arc-canonicalizer`](#-arc-canonicalizer)
  + [`-arc-dedup`](#-arc-dedup)
  + [`-arc-find-initial-vectors`](#-arc-find-initial-vectors)
  + [`-arc-infer-memories`](#-arc-infer-memories)
  + [`-arc-infer-state-properties`](#-arc-infer-state-properties)
  + [`-arc-inline`](#-arc-inline)
  + [`-arc-insert-runtime`](#-arc-insert-runtime)
  + [`-arc-isolate-clocks`](#-arc-isolate-clocks)
  + [`-arc-latency-retiming`](#-arc-latency-retiming)
  + [`-arc-lower-arcs-to-funcs`](#-arc-lower-arcs-to-funcs)
  + [`-arc-lower-clocks-to-funcs`](#-arc-lower-clocks-to-funcs)
  + [`-arc-lower-lut`](#-arc-lower-lut)
  + [`-arc-lower-state`](#-arc-lower-state)
  + [`-arc-lower-vectorizations`](#-arc-lower-vectorizations)
  + [`-arc-lower-verif-simulations`](#-arc-lower-verif-simulations)
  + [`-arc-make-tables`](#-arc-make-tables)
  + [`-arc-merge-ifs`](#-arc-merge-ifs)
  + [`-arc-merge-taps`](#-arc-merge-taps)
  + [`-arc-mux-to-control-flow`](#-arc-mux-to-control-flow)
  + [`-arc-print-cost-model`](#-arc-print-cost-model)
  + [`-arc-simplify-variadic-ops`](#-arc-simplify-variadic-ops)
  + [`-arc-split-funcs`](#-arc-split-funcs)
  + [`-arc-split-loops`](#-arc-split-loops)
  + [`-arc-strip-sv`](#-arc-strip-sv)
* [Calyx Dialect Passes](#calyx-dialect-passes)
  + [`-affine-parallel-unroll`](#-affine-parallel-unroll)
  + [`-affine-ploop-unparallelize`](#-affine-ploop-unparallelize)
  + [`-calyx-affine-to-scf`](#-calyx-affine-to-scf)
  + [`-calyx-clk-insertion`](#-calyx-clk-insertion)
  + [`-calyx-compile-control`](#-calyx-compile-control)
  + [`-calyx-gicm`](#-calyx-gicm)
  + [`-calyx-go-insertion`](#-calyx-go-insertion)
  + [`-calyx-remove-comb-groups`](#-calyx-remove-comb-groups)
* [Example](#example)
  + [`-calyx-remove-groups`](#-calyx-remove-groups)
  + [`-calyx-reset-insertion`](#-calyx-reset-insertion)
  + [`-exclude-exec-region-canonicalize`](#-exclude-exec-region-canonicalize)
* [Comb Dialect Passes](#comb-dialect-passes)
  + [`-comb-assume-two-valued`](#-comb-assume-two-valued)
  + [`-comb-balance-mux`](#-comb-balance-mux)
  + [`-comb-int-range-narrowing`](#-comb-int-range-narrowing)
  + [`-comb-overflow-annotating`](#-comb-overflow-annotating)
  + [`-comb-simplify-tt`](#-comb-simplify-tt)
  + [`-lower-comb`](#-lower-comb)
* [Datapath Dialect Passes](#datapath-dialect-passes)
  + [`-datapath-reduce-delay`](#-datapath-reduce-delay)
* [DC Dialect Passes](#dc-dialect-passes)
  + [`-dc-dematerialize-forks-sinks`](#-dc-dematerialize-forks-sinks)
  + [`-dc-materialize-forks-sinks`](#-dc-materialize-forks-sinks)
  + [`-dc-print-dot`](#-dc-print-dot)
* [ESI Dialect Passes](#esi-dialect-passes)
  + [`-esi-appid-hier`](#-esi-appid-hier)
  + [`-esi-build-manifest`](#-esi-build-manifest)
  + [`-esi-clean-metadata`](#-esi-clean-metadata)
  + [`-esi-connect-services`](#-esi-connect-services)
  + [`-lower-esi-bundles`](#-lower-esi-bundles)
  + [`-lower-esi-ports`](#-lower-esi-ports)
  + [`-lower-esi-to-hw`](#-lower-esi-to-hw)
  + [`-lower-esi-to-physical`](#-lower-esi-to-physical)
  + [`-lower-esi-types`](#-lower-esi-types)
  + [`-verify-esi-connections`](#-verify-esi-connections)
* [FIRRTL Dialect Passes](#firrtl-dialect-passes)
  + [`-firrtl-add-seqmem-ports`](#-firrtl-add-seqmem-ports)
  + [`-firrtl-annotate-input-only-modules`](#-firrtl-annotate-input-only-modules)
  + [`-firrtl-assign-output-dirs`](#-firrtl-assign-output-dirs)
  + [`-firrtl-blackbox-reader`](#-firrtl-blackbox-reader)
  + [`-firrtl-check-comb-loops`](#-firrtl-check-comb-loops)
  + [`-firrtl-check-layers`](#-firrtl-check-layers)
  + [`-firrtl-check-recursive-instantiation`](#-firrtl-check-recursive-instantiation)
  + [`-firrtl-dedup`](#-firrtl-dedup)
  + [`-firrtl-drop-const`](#-firrtl-drop-const)
  + [`-firrtl-drop-names`](#-firrtl-drop-names)
  + [`-firrtl-eliminate-wires`](#-firrtl-eliminate-wires)
  + [`-firrtl-emit-metadata`](#-firrtl-emit-metadata)
  + [`-firrtl-expand-whens`](#-firrtl-expand-whens)
  + [`-firrtl-extract-instances`](#-firrtl-extract-instances)
  + [`-firrtl-finalize-ir`](#-firrtl-finalize-ir)
  + [`-firrtl-flatten-memory`](#-firrtl-flatten-memory)
  + [`-firrtl-grand-central`](#-firrtl-grand-central)
  + [`-firrtl-imconstprop`](#-firrtl-imconstprop)
  + [`-firrtl-imdeadcodeelim`](#-firrtl-imdeadcodeelim)
  + [`-firrtl-infer-domains`](#-firrtl-infer-domains)
  + [`-firrtl-infer-resets`](#-firrtl-infer-resets)
  + [`-firrtl-infer-rw`](#-firrtl-infer-rw)
  + [`-firrtl-infer-widths`](#-firrtl-infer-widths)
  + [`-firrtl-inject-dut-hier`](#-firrtl-inject-dut-hier)
  + [`-firrtl-inliner`](#-firrtl-inliner)
  + [`-firrtl-inner-symbol-dce`](#-firrtl-inner-symbol-dce)
  + [`-firrtl-layer-merge`](#-firrtl-layer-merge)
  + [`-firrtl-layer-sink`](#-firrtl-layer-sink)
  + [`-firrtl-link-circuits`](#-firrtl-link-circuits)
  + [`-firrtl-lint`](#-firrtl-lint)
  + [`-firrtl-lower-annotations`](#-firrtl-lower-annotations)
  + [`-firrtl-lower-chirrtl`](#-firrtl-lower-chirrtl)
  + [`-firrtl-lower-classes`](#-firrtl-lower-classes)
  + [`-firrtl-lower-domains`](#-firrtl-lower-domains)
  + [`-firrtl-lower-dpi`](#-firrtl-lower-dpi)
  + [`-firrtl-lower-intmodules`](#-firrtl-lower-intmodules)
  + [`-firrtl-lower-intrinsics`](#-firrtl-lower-intrinsics)
  + [`-firrtl-lower-layers`](#-firrtl-lower-layers)
  + [`-firrtl-lower-matches`](#-firrtl-lower-matches)
  + [`-firrtl-lower-memory`](#-firrtl-lower-memory)
  + [`-firrtl-lower-open-aggs`](#-firrtl-lower-open-aggs)
  + [`-firrtl-lower-signatures`](#-firrtl-lower-signatures)
  + [`-firrtl-lower-types`](#-firrtl-lower-types)
  + [`-firrtl-lower-xmr`](#-firrtl-lower-xmr)
  + [`-firrtl-materialize-debug-info`](#-firrtl-materialize-debug-info)
  + [`-firrtl-mem-to-reg-of-vec`](#-firrtl-mem-to-reg-of-vec)
  + [`-firrtl-merge-connections`](#-firrtl-merge-connections)
  + [`-firrtl-module-summary`](#-firrtl-module-summary)
  + [`-firrtl-passive-wires`](#-firrtl-passive-wires)
  + [`-firrtl-print-field-source`](#-firrtl-print-field-source)
  + [`-firrtl-print-instance-graph`](#-firrtl-print-instance-graph)
  + [`-firrtl-print-nla-table`](#-firrtl-print-nla-table)
  + [`-firrtl-probes-to-signals`](#-firrtl-probes-to-signals)
  + [`-firrtl-randomize-register-init`](#-firrtl-randomize-register-init)
  + [`-firrtl-register-optimizer`](#-firrtl-register-optimizer)
  + [`-firrtl-remove-unused-ports`](#-firrtl-remove-unused-ports)
  + [`-firrtl-resolve-paths`](#-firrtl-resolve-paths)
  + [`-firrtl-resolve-traces`](#-firrtl-resolve-traces)
  + [`-firrtl-sfc-compat`](#-firrtl-sfc-compat)
  + [`-firrtl-specialize-layers`](#-firrtl-specialize-layers)
  + [`-firrtl-specialize-option`](#-firrtl-specialize-option)
  + [`-firrtl-vb-to-bv`](#-firrtl-vb-to-bv)
  + [`-firrtl-vectorization`](#-firrtl-vectorization)
* [FSM Dialect Passes](#fsm-dialect-passes)
  + [`-fsm-print-graph`](#-fsm-print-graph)
* [Handshake Dialect Passes](#handshake-dialect-passes)
  + [`-handshake-add-ids`](#-handshake-add-ids)
  + [`-handshake-dematerialize-forks-sinks`](#-handshake-dematerialize-forks-sinks)
  + [`-handshake-insert-buffers`](#-handshake-insert-buffers)
  + [`-handshake-legalize-memrefs`](#-handshake-legalize-memrefs)
  + [`-handshake-lock-functions`](#-handshake-lock-functions)
  + [`-handshake-lower-extmem-to-hw`](#-handshake-lower-extmem-to-hw)
  + [`-handshake-materialize-forks-sinks`](#-handshake-materialize-forks-sinks)
  + [`-handshake-op-count`](#-handshake-op-count)
  + [`-handshake-print-dot`](#-handshake-print-dot)
  + [`-handshake-remove-buffers`](#-handshake-remove-buffers)
  + [`-handshake-split-merges`](#-handshake-split-merges)
* [HW Dialect Passes](#hw-dialect-passes)
  + [`-hw-aggregate-to-comb`](#-hw-aggregate-to-comb)
  + [`-hw-bypass-inner-symbols`](#-hw-bypass-inner-symbols)
  + [`-hw-convert-bitcasts`](#-hw-convert-bitcasts)
  + [`-hw-flatten-io`](#-hw-flatten-io)
  + [`-hw-flatten-modules`](#-hw-flatten-modules)
  + [`-hw-foo-wires`](#-hw-foo-wires)
  + [`-hw-parameterize-constant-ports`](#-hw-parameterize-constant-ports)
  + [`-hw-print-instance-graph`](#-hw-print-instance-graph)
  + [`-hw-print-module-graph`](#-hw-print-module-graph)
  + [`-hw-specialize`](#-hw-specialize)
  + [`-hw-verify-irn`](#-hw-verify-irn)
* [Kanagawa Dialect Passes](#kanagawa-dialect-passes)
  + [`-kanagawa-add-operator-library`](#-kanagawa-add-operator-library)
  + [`-kanagawa-argify-blocks`](#-kanagawa-argify-blocks)
  + [`-kanagawa-call-prep`](#-kanagawa-call-prep)
  + [`-kanagawa-clean-selfdrivers`](#-kanagawa-clean-selfdrivers)
  + [`-kanagawa-containerize`](#-kanagawa-containerize)
  + [`-kanagawa-convert-cf-to-handshake`](#-kanagawa-convert-cf-to-handshake)
  + [`-kanagawa-convert-containers-to-hw`](#-kanagawa-convert-containers-to-hw)
  + [`-kanagawa-convert-handshake-to-dc`](#-kanagawa-convert-handshake-to-dc)
  + [`-kanagawa-convert-methods-to-containers`](#-kanagawa-convert-methods-to-containers)
  + [`-kanagawa-eliminate-redundant-ops`](#-kanagawa-eliminate-redundant-ops)
  + [`-kanagawa-inline-sblocks`](#-kanagawa-inline-sblocks)
  + [`-kanagawa-lower-portrefs`](#-kanagawa-lower-portrefs)
  + [`-kanagawa-prepare-scheduling`](#-kanagawa-prepare-scheduling)
  + [`-kanagawa-reblock`](#-kanagawa-reblock)
  + [`-kanagawa-tunneling`](#-kanagawa-tunneling)
* [LLHD Dialect Passes](#llhd-dialect-passes)
  + [`-llhd-combine-drives`](#-llhd-combine-drives)
  + [`-llhd-deseq`](#-llhd-deseq)
  + [`-llhd-hoist-signals`](#-llhd-hoist-signals)
  + [`-llhd-inline-calls`](#-llhd-inline-calls)
  + [`-llhd-lower-processes`](#-llhd-lower-processes)
  + [`-llhd-mem2reg`](#-llhd-mem2reg)
  + [`-llhd-remove-control-flow`](#-llhd-remove-control-flow)
  + [`-llhd-sig2reg`](#-llhd-sig2reg)
  + [`-llhd-unroll-loops`](#-llhd-unroll-loops)
  + [`-llhd-wrap-procedural-ops`](#-llhd-wrap-procedural-ops)
* [MSFT Dialect Passes](#msft-dialect-passes)
  + [`-msft-export-tcl`](#-msft-export-tcl)
  + [`-msft-lower-constructs`](#-msft-lower-constructs)
  + [`-msft-lower-instances`](#-msft-lower-instances)
* [OM Dialect Passes](#om-dialect-passes)
  + [`-om-freeze-paths`](#-om-freeze-paths)
  + [`-om-link-modules`](#-om-link-modules)
  + [`-om-verify-object-fields`](#-om-verify-object-fields)
  + [`-strip-om`](#-strip-om)
* [Pipeline Dialect Passes](#pipeline-dialect-passes)
  + [`-pipeline-explicit-regs`](#-pipeline-explicit-regs)
  + [`-pipeline-schedule-linear`](#-pipeline-schedule-linear)
* [Seq Dialect Passes](#seq-dialect-passes)
  + [`-externalize-clock-gate`](#-externalize-clock-gate)
  + [`-hw-memory-sim`](#-hw-memory-sim)
  + [`-lower-seq-fifo`](#-lower-seq-fifo)
  + [`-lower-seq-hlmem`](#-lower-seq-hlmem)
  + [`-lower-seq-shiftreg`](#-lower-seq-shiftreg)
  + [`-seq-reg-of-vec-to-mem`](#-seq-reg-of-vec-to-mem)
* [SSP Dialect Passes](#ssp-dialect-passes)
  + [`-ssp-print`](#-ssp-print)
  + [`-ssp-roundtrip`](#-ssp-roundtrip)
  + [`-ssp-schedule`](#-ssp-schedule)
* [SV Dialect Passes](#sv-dialect-passes)
  + [`-hw-cleanup`](#-hw-cleanup)
  + [`-hw-eliminate-inout-ports`](#-hw-eliminate-inout-ports)
  + [`-hw-export-module-hierarchy`](#-hw-export-module-hierarchy)
  + [`-hw-generator-callout`](#-hw-generator-callout)
  + [`-hw-legalize-modules`](#-hw-legalize-modules)
  + [`-hw-stub-external-modules`](#-hw-stub-external-modules)
  + [`-prettify-verilog`](#-prettify-verilog)
  + [`-sv-extract-test-code`](#-sv-extract-test-code)
  + [`-sv-trace-iverilog`](#-sv-trace-iverilog)
* [Synth Dialect Passes](#synth-dialect-passes)
  + [`-synth-abc-runner`](#-synth-abc-runner)
  + [`-synth-aiger-runner`](#-synth-aiger-runner)
  + [`-synth-generic-lut-mapper`](#-synth-generic-lut-mapper)
  + [`-synth-lower-variadic`](#-synth-lower-variadic)
  + [`-synth-lower-word-to-bits`](#-synth-lower-word-to-bits)
  + [`-synth-maximum-and-cover`](#-synth-maximum-and-cover)
  + [`-synth-print-longest-path-analysis`](#-synth-print-longest-path-analysis)
  + [`-synth-sop-balancing`](#-synth-sop-balancing)
  + [`-synth-structural-hash`](#-synth-structural-hash)
  + [`-synth-tech-mapper`](#-synth-tech-mapper)
  + [`-synth-test-priority-cuts`](#-synth-test-priority-cuts)
* [SystemC Dialect Passes](#systemc-dialect-passes)
  + [`-systemc-lower-instance-interop`](#-systemc-lower-instance-interop)
* [LEC (logical equivalence checking) Passes](#lec-logical-equivalence-checking-passes)
  + [`-construct-lec`](#-construct-lec)
* [BMC (bounded model checking) Passes](#bmc-bounded-model-checking-passes)
  + [`-externalize-registers`](#-externalize-registers)
  + [`-lower-to-bmc`](#-lower-to-bmc)

General Passes [¶](#general-passes)
-----------------------------------

### `-convert-index-to-uint` [¶](#-convert-index-to-uint)

*Rewrite index-based switch comparisons into unsigned integer ops.*

Replace `arith.cmpi` operations whose operands are `index` values (often
produced when lowering `scf.index_switch`) with comparisons over the
original integer type so that downstream hardware mapping passes (e.g.
`--map-arith-to-comb`) do not encounter unsupported index-typed arithmetic.
The pass converts any associated index constants and erases the redundant
casts that become dead afterwards. For example, it rewrites:

`arith.cmpi eq, (arith.index_cast %v : i4 to index), (arith.constant 5 : index)`

into:

`arith.cmpi eq, %v, (arith.constant 5 : i4)`

### `-flatten-memref` [¶](#-flatten-memref)

*Flatten memrefs*

Flattens multidimensional memories and accesses to them into
single-dimensional memories.

### `-flatten-memref-calls` [¶](#-flatten-memref-calls)

*Flatten memref calls*

Flattens calls to functions which have multidimensional memrefs as arguments.
This is done by casting memref arguments through memref.subview operations.
Any called functions which had their type signatures changes will be replaced
by a private function definition with the new signature.
It is up to users of this pass to define how these rewritten functions are
to be implemented.

### `-hierarchical-runner` [¶](#-hierarchical-runner)

*Run passes under hierarchy*

This pass runs a specified pipeline of passes on the hierarchy of modules
starting from a given top-level module. It allows for hierarchical
application of transformations, which can be useful for targeting specific
parts of a design or for applying different optimizations at different
levels of the module hierarchy.

#### Options [¶](#options)

```
-pipeline                : The pipeline to run under hierarchy
-top-name                : The name of the top-level module to run the pass on
-include-bound-instances : Whether to include bound instances in the hierarchy
```

### `-insert-merge-blocks` [¶](#-insert-merge-blocks)

*Insert explicit merge blocks*

This pass inserts additional merge blocks for each block with more than
two successors. A merge block is a block that only contains one operation,
a terminator, and has two predecessors.
The order successors are merged together mirrors the order different control
paths are created. Thus, each block with two successors will have a corresponding
merge block.

This pass brings the CFG into a canonical form for further transformations.

Treats loops and sub-CFGs with irregular control flow like single blocks.

### `-map-arith-to-comb` [¶](#-map-arith-to-comb)

*Map arith ops to combinational logic*

A pass which does a simple `arith` to `comb` mapping wherever possible.
This pass will not convert:

* floating point operations
* operations using `vector`-typed values

This **does not** intend to be the definitive lowering/HLS pass of `arith`
operations in CIRCT (hence the name “map” instead of e.g. “lower”).
Rather, it provides a simple way (mostly for testing purposes) to map
`arith` operations.

#### Options [¶](#options-1)

```
-enable-best-effort-lowering : Enable best effort lowering of operations(eg, existing arith operations that have no lowering stay as is)
```

### `-maximize-ssa` [¶](#-maximize-ssa)

*Convert every function in the module into maximal SSA form*

Convert the region within every function into maximal SSA form. This
ensures that every value used within a block is also defined within the
block, making dataflow explicit and removing block dominance-based dataflow
semantics. The pass achieves this by adding block arguments wherever
necessary to forward values to the block(s) where they are used.

### `-memory-banking` [¶](#-memory-banking)

*Partition the memories used in affine parallel loops into banks*

#### Options [¶](#options-2)

```
-factors    : Use banking factors to partition all memories that don't have banking attributes.The elements specified in banking factors should be greater than 1;The elements specified in banking factors will be paired with the ones specified in banking dimensions.In principle, the number of elements in banking factors should be equal to banking dimensions',with a single exception case: there is one banking factor with zero banking dimensions.
-dimensions : The dimensions along which to bank the memory. If unspecified andthere is only one factor, the innermost dimension with size > 1 is used.
```

### `-print-op-count` [¶](#-print-op-count)

*Print operation count analysis results*

This pass prints data on operation counts in a builtin.module.

#### Options [¶](#options-3)

```
-emission-format : Specify the format to emit op count info in
```

### `-strip-debuginfo-with-pred` [¶](#-strip-debuginfo-with-pred)

*Selectively strip debug info from all operations*

This pass extends mlir::StripDebugInfoPass to selectively strip locations with a
given predicate.

#### Options [¶](#options-4)

```
-drop-suffix : Drop file location info with the specified suffix. This option isintended to be used for testing.
```

### `-switch-to-if` [¶](#-switch-to-if)

*Index switch to if*

Transform `scf.index_switch` to a series of `scf.if` operations.
This is necessary for dialects that don’t support switch statements, e.g., Calyx.
An example:

```
  %0 = scf.index_switch %cond -> i32
    case 0 { ... }
    case 1 { ... }
    ...

  =>

  %c0 = arith.cmpi eq %0, 0
  %c1 = arith.cmpi eq %0, 1
  %0 = scf.if %c0 {
   ...
  } else {
    %1 = scf.if %c1 {
      ...
    } else {
      ...
    }
  }
```

Conversion Passes [¶](#conversion-passes)
-----------------------------------------

### `-calyx-native` [¶](#-calyx-native)

*Callout to the Calyx native compiler and run a pass pipeline*

This pass calls out to the native, Rust-based Calyx compiler to run passes
with it and generate a new, valid, calyx dialect program.

#### Options [¶](#options-5)

```
-pass-pipeline : Passes to run with the native compiler
```

### `-calyx-remove-groups-fsm` [¶](#-calyx-remove-groups-fsm)

*Perform FSM outlining and group removal*

This pass will outline the FSM into module scope and replace any SSA value references
from within the FSM body with additional inputs. Given this, the FSM
is instantiated as a `fsm.hw_module` operation within the Calyx component.
Using the FSM I/O (which is the group go/done signals), the `calyx.group`
operations are removed from the component, with the group go and done signals
being wired up to the FSM instance.
Example:

```
calyx.component {
    %reg, ... = calyx.register ... : i1
    calyx.wires {
        // Groups have explicit done signals, and assignments are not guarded
        // by a group go signal.
        calyx.group @A {
            ...
            calyx.assign %reg = ...
            ...
            calyx.group_done %foo ? %bar
        }
    }
    calyx.control {
        // Machine is defined inside the `calyx.control` operation and references
        // SSA values defined outside the machine.
        fsm.machine @control(%A_done : i1) -> (%A_go : i1) {
            ...
            %0 = comb.not %reg // reference some SSA value defined outside the machine
            ...
        }
    }
}
```

into

```
// The machine has been outlined into module scope, and no longer references
// any SSA values defined outside the machine. It is now fully independent
// from any notion of Calyx.
fsm.machine @control(%A_done : i1, %reg : i1) -> (%A_go : i1) {
    ...
    %0 = comb.not %reg // reference some SSA value defined outside the machine
    ...
}

calyx.component {
    %reg, ... = calyx.register ...
    // Done signals are now wires
    %A_done_in, %A_done_out = calyx.wire : i1
    // The FSM is now instantiated as an `fsm.hwinstance` module
    %A_go = fsm.hwinstance @control(%A_done_out, %reg) : ...
    calyx.wires {
        // Groups have been inlined, the group go signal is now a guard for
        // all assignments, and `calyx.group_done` operations have been
        // replaced by wire assignments.
        ...
        calyx.assign %reg = %A_go ? ...
        ...
        calyx.assign %A_done_in = %foo ? %bar
    }
    calyx.control {
    }
}
```

### `-convert-affine-to-loopschedule` [¶](#-convert-affine-to-loopschedule)

*Convert Affine dialect to LoopSchedule scheduled loops*

This pass analyzes Affine loops and control flow, creates a Scheduling
problem using the Calyx operator library, solves the problem, and lowers
the loops to a LoopSchedule.

### `-convert-comb-to-arith` [¶](#-convert-comb-to-arith)

*Convert combinational ops and constants into arith ops*

### `-convert-comb-to-datapath` [¶](#-convert-comb-to-datapath)

*Lower Comb ops to Datapath ops*

This pass converts arithmetic Comb operations into Datapath operations that
leverage redundant number representations (carry save). Primarily for use
in the circt-synth flow.

### `-convert-comb-to-smt` [¶](#-convert-comb-to-smt)

*Convert combinational ops and constants to SMT ops*

### `-convert-comb-to-synth` [¶](#-convert-comb-to-synth)

*Lower Comb ops to Synth ops.*

#### Options [¶](#options-6)

```
-additional-legal-ops       : Specify additional legal ops to partially legalize Comb
-target-ir                  : Target IR kind
-max-emulation-unknown-bits : Maximum number of unknown bits to emulate in a table lookup
```

### `-convert-core-to-fsm` [¶](#-convert-core-to-fsm)

*Convert Core to FSM*

This pass extracts FSM structure from an RTL netlist. State registers
can be specified explicitly via the `state-regs` option, or they will
be inferred as registers whose names contain “state”.

#### Options [¶](#options-7)

```
-state-regs : Comma-separated list of names of registers to use as stateregisters. If not specified, registers with names containing'state' are used.
```

### `-convert-datapath-to-comb` [¶](#-convert-datapath-to-comb)

*Convert Datapath ops to Comb ops*

#### Options [¶](#options-8)

```
-lower-compress-to-add          : Lower compress operators to variadic add.
-lower-partial-product-to-booth : Force all partial products to be lowered to Booth arrays.
-timing-aware                   : Use timing-aware compressor synthesis algorithm.
```

### `-convert-datapath-to-smt` [¶](#-convert-datapath-to-smt)

*Convert datapath ops to SMT ops*

### `-convert-fsm-to-core` [¶](#-convert-fsm-to-core)

*Convert FSM to Core*

### `-convert-fsm-to-sv` [¶](#-convert-fsm-to-sv)

*Convert FSM to SV and HW*

### `-convert-hw-to-btor2` [¶](#-convert-hw-to-btor2)

*Convert HW to BTOR2*

This pass converts a HW module into a state transition system that is then
directly used to emit btor2. The output of this pass is thus a btor2 string.

### `-convert-hw-to-llvm` [¶](#-convert-hw-to-llvm)

*Convert HW to LLVM*

This pass translates HW to LLVM.

#### Options [¶](#options-9)

```
-spill-arrays-early : If true, array values will be materialized on the stack at thepoint of definition. This can reduce redundant stack allocations.
```

### `-convert-hw-to-smt` [¶](#-convert-hw-to-smt)

*Convert HW ops and constants to SMT ops*

#### Options [¶](#options-10)

```
-for-smtlib-export : Produce output for SMTLIB export - this will replace hw.moduleoperations with smt.solver operations and assert the equivalenceof each module output with a symbolic constant.
```

### `-convert-hw-to-systemc` [¶](#-convert-hw-to-systemc)

*Convert HW to SystemC*

This pass translates a HW design into an equivalent SystemC design.

### `-convert-moore-to-core` [¶](#-convert-moore-to-core)

*Convert Moore to Core*

This pass translates Moore to the core dialects.

### `-convert-synth-to-comb` [¶](#-convert-synth-to-comb)

*Lower Synth ops to Comb ops*

This pass converts Synth operations to Comb operations. This is mostly
used for verifying post-synthesis results.

### `-convert-to-arcs` [¶](#-convert-to-arcs)

*Outline logic between registers into state transfer arcs*

This pass outlines combinational logic between registers into state transfer
arc definitions. The the original combinational logic and register is
replaced with an arc invocation, where the register is now represented as a
latency.

#### Options [¶](#options-11)

```
-tap-registers : Make registers observable
```

### `-convert-to-llvm` [¶](#-convert-to-llvm)

*Convert Comb and HW to LLVM*

This pass translates Comb and HW operations inside func.func to LLVM IR.
It combines both HW and Comb conversion patterns to provide a complete
lowering from hardware description to LLVM IR.

### `-convert-verif-to-smt` [¶](#-convert-verif-to-smt)

*Convert Verif ops to SMT ops*

#### Options [¶](#options-12)

```
-rising-clocks-only : When lowering verif.bmc ops, only consider the circuit and propertyon rising clock edges.
```

### `-export-split-verilog` [¶](#-export-split-verilog)

*Emit the IR to a (System)Verilog directory of files*

This pass generates (System)Verilog for the current design, mutating it
where necessary to be valid Verilog.

#### Options [¶](#options-13)

```
-dir-name : Directory to emit into
```

### `-export-verilog` [¶](#-export-verilog)

*Emit the IR to a (System)Verilog file*

This pass creates empty module bodies for external modules. This is
useful for linting to eliminate missing file errors.

### `-handshake-remove-block-structure` [¶](#-handshake-remove-block-structure)

*Remove block structure in Handshake IR*

### `-hw-lower-instance-choices` [¶](#-hw-lower-instance-choices)

*Prepare the collateral for instance choice emission*

This pass runs as part of verilog emission.
It introduces the macros & file lists to which instance choices lower to.

### `-legalize-anon-enums` [¶](#-legalize-anon-enums)

*Prepare anonymous enumeration types for ExportVerilog*

This pass transforms all anonymous enumeration types into typedecls to work
around difference in how anonymous enumerations work in SystemVerilog.

### `-lower-arc-to-llvm` [¶](#-lower-arc-to-llvm)

*Lower state transfer arc representation to LLVM*

### `-lower-calyx-to-fsm` [¶](#-lower-calyx-to-fsm)

*Lower Calyx to FSM*

This pass lowers a Calyx control schedule to an FSM representation.
An `fsm.machine` operation is nested within the `control` region of the Calyx
component. This machine is itself in an intermediate format wherein it has
no I/O ports and solely contains output statements with `calyx.enable`s
referencing `calyx.group` and transition logic guarded by the SSA values
specified in the source control schedule.
This intermediate state facilitates transformation of the FSM, given that
top-level I/O has yet to be materialized (one input and output per activated
group) as well as guard transition logic (every transition must be guarded
on all groups active within the state having finished). As such, `calyx.enable`
operations can easily be moved between states without worrying about updating
transition guards while doing so.

Eventually, the FSM must be materialized (materialize I/O ports, remove
`calyx.enable` operations in favor of asserting output ports, guarding
transitions by input `done` ports) and outlined to a separate module.

### `-lower-calyx-to-hw` [¶](#-lower-calyx-to-hw)

*Lower Calyx to HW*

This pass lowers Calyx to HW.

### `-lower-cf-to-handshake` [¶](#-lower-cf-to-handshake)

*Lower func and CF into Handshake IR*

#### Options [¶](#options-14)

```
-source-constants        : If true, will connect constants to source operations instead of to the control network. May reduce the size of the final circuit.
-disable-task-pipelining : If true, will disable support for task pipelining. This relaxes the restrictions put on the structure of the input CDFG. Disabling task pipelining may severely reduce kernel II.
```

### `-lower-dc-to-hw` [¶](#-lower-dc-to-hw)

*Lower DC to HW*

Lower DC to ESI/hw/comb/seq operations.
In case the IR contains DC operations that need to be clocked (fork, buffer),
there must exist a clock and reset signal in the parent `FunctionLike`
operation. These arguments are to be marked with a `dc.clock` and `dc.reset`
attribute, respectively.

### `-lower-firrtl-to-hw` [¶](#-lower-firrtl-to-hw)

*Lower FIRRTL to HW*

Lower a module of FIRRTL dialect to the HW dialect family.

#### Options [¶](#options-15)

```
-warn-on-unprocessed-annotations : Emit warnings on unprocessed annotations during lower-to-hw pass
-verification-flavor             : Specify a verification flavor used in the lowering
```

### `-lower-handshake-to-dc` [¶](#-lower-handshake-to-dc)

*Lower Handshake to DC*

Lower Handshake to DC operations.
Currently, a `handshake.func` will be converted into a `hw.module`. This
is principally an incorrect jump of abstraction - DC does not imply any
RTL/hardware semantics. However, DC does not define a container operation,
and there does not exist an e.g. `func.graph_func` which would be a generic
function with graph region behaviour. Thus, for now, we just use `hw.module`
as a container operation.

#### Options [¶](#options-16)

```
-clk-name : Name of the clock signal to use in the generated DC module
-rst-name : Name of the reset signal to use in the generated DC module
```

### `-lower-handshake-to-hw` [¶](#-lower-handshake-to-hw)

*Lower Handshake to ESI/HW/Comb/Seq*

Lower Handshake to ESI/HW/Comb/Seq.

### `-lower-hw-to-sv` [¶](#-lower-hw-to-sv)

*Convert HW to SV*

This pass converts various HW contructs to SV.

### `-lower-hwarith-to-hw` [¶](#-lower-hwarith-to-hw)

*Lower HWArith to HW/Comb*

This pass lowers HWArith to HW/Comb.

### `-lower-loopschedule-to-calyx` [¶](#-lower-loopschedule-to-calyx)

*Lower LoopSchedule to Calyx*

This pass lowers LoopSchedule to Calyx.

#### Options [¶](#options-17)

```
-top-level-function             : Identifier of top-level function to be the entry-point component of the Calyx program.
-cider-source-location-metadata : Whether to track source location for the Cider debugger.
```

### `-lower-ltl-to-core` [¶](#-lower-ltl-to-core)

*Convert LTL and Verif to Core*

This pass converts ltl and verif operations to core ones. This can be done directly
without going through FSM if we’re only working with overlapping properties (no delays).

### `-lower-pipeline-to-hw` [¶](#-lower-pipeline-to-hw)

*Lower Pipeline to HW*

This pass lowers `pipeline.rtp` operations to HW.

#### Options [¶](#options-18)

```
-clock-gate-regs       : Clock gate each register instead of (default) input muxing  (ASIC optimization).
-enable-poweron-values : Add power-on values to the pipeline control registers
```

### `-lower-scf-to-calyx` [¶](#-lower-scf-to-calyx)

*Lower SCF/Standard to Calyx*

This pass lowers SCF / standard to Calyx.

#### Options [¶](#options-19)

```
-top-level-function             : Identifier of top-level function to be the entry-point component of the Calyx program.
-cider-source-location-metadata : Whether to track source location for the Cider debugger.
-write-json                     : Whether to write memory contents to the json file.
```

### `-lower-seq-firmem` [¶](#-lower-seq-firmem)

*Lower seq.firmem ops to instances of hw.module.generated ops*

### `-lower-seq-to-sv` [¶](#-lower-seq-to-sv)

*Lower sequential firrtl ops to SV.*

#### Options [¶](#options-20)

```
-disable-reg-randomization   : Disable emission of register randomization code
-disable-mem-randomization   : Disable emission of memory randomization code
-emit-separate-always-blocks : Emit assigments to registers in separate always blocks
-lower-to-always-ff          : Place assignments to registers into `always_ff` blocks if possible
```

#### Statistics [¶](#statistics)

```
num-subaccess-restored : Number of lhs subaccess operations restored
```

### `-lower-sim-to-sv` [¶](#-lower-sim-to-sv)

*Lower simulator-specific `sim` ops to SV.*

### `-lower-smt-to-z3-llvm` [¶](#-lower-smt-to-z3-llvm)

*Lower the SMT dialect to LLVM IR calling the Z3 API*

#### Options [¶](#options-21)

```
-debug : Insert additional printf calls printing the solver's state to stdout (e.g. at check-sat operations) for debugging purposes
```

### `-lower-verif-to-sv` [¶](#-lower-verif-to-sv)

*Convert Verif to SV*

This pass converts various Verif contructs to SV.

### `-materialize-calyx-to-fsm` [¶](#-materialize-calyx-to-fsm)

*Materializes an FSM embedded inside the control of this Calyx component.*

Materializes the FSM in the control of the component. This materializes the
top-level I/O of the FSM to receive `group_done` signals as input and
`group_go` signals as output, based on the `calyx.enable` operations
used within the states of the FSM.
Each transition of the FSM is predicated on the enabled groups within a
state being done, or, for static groups, a separate sub-FSM is instantiated
to await the group finishing.

Given an FSM that enables N unique groups, the top-level FSM will have N+1
in- and output ports, wherein:

* Input # 0 to N-1 are `group_done` signals
* Input N is the top-level `go` port
* Output 0 to N-1 are `group_go` signals
* Output N is the top-level `done` port

### `-prepare-for-emission` [¶](#-prepare-for-emission)

*Prepare IR for ExportVerilog*

This pass runs only PrepareForEmission.
It is not necessary for users to run this pass explicitly since
ExportVerilog internally runs PrepareForEmission.

### `-test-apply-lowering-options` [¶](#-test-apply-lowering-options)

*Apply lowering options*

This pass allows overriding lowering options. It is intended for test
construction.

#### Options [¶](#options-22)

```
-options : Lowering Options
```

Arc Dialect Passes [¶](#arc-dialect-passes)
-------------------------------------------

### `-arc-add-taps` [¶](#-arc-add-taps)

*Add taps to ports and wires such that they remain observable*

#### Options [¶](#options-23)

```
-ports        : Make module ports observable
-wires        : Make wires observable
-named-values : Make values with `sv.namehint` observable
```

### `-arc-allocate-state` [¶](#-arc-allocate-state)

*Allocate and layout the global simulation state*

#### Options [¶](#options-24)

```
-trace-taps : Insert TraceTap attributes for signal tracing
```

### `-arc-canonicalizer` [¶](#-arc-canonicalizer)

*Simulation centric canonicalizations*

#### Statistics [¶](#statistics-1)

```
num-arc-args-removed : Number of arguments removed from DefineOps
```

### `-arc-dedup` [¶](#-arc-dedup)

*Deduplicate identical arc definitions*

This pass deduplicates identical arc definitions. If two arcs differ only by
constants, the constants are outlined such that the arc can be deduplicated.

#### Statistics [¶](#statistics-2)

```
dedupPassNumArcsDeduped : Number of arcs deduped
dedupPassTotalOps       : Total number of ops deduped
```

### `-arc-find-initial-vectors` [¶](#-arc-find-initial-vectors)

*Find initial groups of vectorizable ops*

#### Statistics [¶](#statistics-3)

```
vectorizedOps       : Total number of ops that were vectorized
numOfSavedOps       : Total number of ops saved after FindInitialVectors pass
biggestSeedVector   : Size of the biggest seed vector
numOfVectorsCreated : Total number of VectorizeOps the pass inserted
```

### `-arc-infer-memories` [¶](#-arc-infer-memories)

*Convert `FIRRTL_Memory` instances to dedicated memory ops*

#### Options [¶](#options-25)

```
-tap-ports    : Make memory ports observable
-tap-memories : Make memory contents observable
```

### `-arc-infer-state-properties` [¶](#-arc-infer-state-properties)

*Add resets and enables explicitly to the state operations*

#### Options [¶](#options-26)

```
-enables : Infer enable signals
-resets  : Infer reset signals
```

#### Statistics [¶](#statistics-4)

```
added-enables  : Enables added explicitly to a StateOp
added-resets   : Resets added explicitly to a StateOp
missed-enables : Detected enables that could not be added explicitly to a StateOp
missed-resets  : Detected resets that could not be added explicitly to a StateOp
```

### `-arc-inline` [¶](#-arc-inline)

*Inline very small arcs*

#### Options [¶](#options-27)

```
-into-arcs-only : Call operations to inline
-max-body-ops   : Max number of non-trivial ops in the region to be inlined
```

#### Statistics [¶](#statistics-5)

```
inlined-arcs    : Arcs inlined at a use site
removed-arcs    : Arcs removed after full inlining
trivial-arcs    : Arcs with very few ops
single-use-arcs : Arcs with a single use
```

### `-arc-insert-runtime` [¶](#-arc-insert-runtime)

*Insert structs and function calls for the ArcRuntime library*

#### Options [¶](#options-28)

```
-extra-args : Extra arguments passed to the runtime when creating simulation instances
-trace-file : Output file name for signal trace
```

### `-arc-isolate-clocks` [¶](#-arc-isolate-clocks)

*Group clocked operations into clock domains*

### `-arc-latency-retiming` [¶](#-arc-latency-retiming)

*Push latencies through the design*

#### Statistics [¶](#statistics-6)

```
num-ops-removed     : Number of zero-latency passthrough states removed
latency-units-saved : Number of latency units saved by merging them in a successor state
```

### `-arc-lower-arcs-to-funcs` [¶](#-arc-lower-arcs-to-funcs)

*Lower arc definitions into functions*

### `-arc-lower-clocks-to-funcs` [¶](#-arc-lower-clocks-to-funcs)

*Lower clock trees into functions*

### `-arc-lower-lut` [¶](#-arc-lower-lut)

*Lowers arc.lut into a comb and hw only representation.*

### `-arc-lower-state` [¶](#-arc-lower-state)

*Split state into read and write ops grouped by clock tree*

### `-arc-lower-vectorizations` [¶](#-arc-lower-vectorizations)

*Lower `arc.vectorize` operations*

This pass lowers `arc.vectorize` operations. By default, the operation will
be fully lowered (i.e., the op disappears in the IR). Alternatively, it can
be partially lowered.

The “mode” pass option allows to only lower the boundary, only the body, or
only inline the body given that both the boundary and the body are already
lowered.

The pass supports vectorization within scalar registers and SIMD
vectorization and prioritizes vectorization by packing the vector elements
into a scalar value if it can fit into 64 bits.

Example:

```
hw.module @example(%in0: i8, %in1: i8, %in2: i8) -> (out0: i8, out1: i8) {
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in2) :
    (i8, i8, i8, i8) -> (i8, i8) {
  ^bb0(%arg0: i8, %arg1: i8):
    %1 = comb.and %arg0, %arg1 : i8
    arc.vectorize.return %1 : i8
  }
  hw.output %0#0, %0#1 : i8, i8
}
```

This piece of IR is lowered to the following fully vectorized IR:

```
hw.module @example(%in0: i8, %in1: i8, %in2: i8) -> (out0: i8, out1: i8) {
  %0 = comb.concat %in0, %in1 : i8, i8
  %1 = comb.concat %in2, %in2 : i8, i8
  %2 = comb.and %0, %1 : i16
  %3 = comb.extract %2 from 0 : (i16) -> i8
  %4 = comb.extract %2 from 8 : (i16) -> i8
  hw.output %3, %4 : i8, i8
}
```

#### Options [¶](#options-29)

```
-mode : Select what should be lowered.
```

### `-arc-lower-verif-simulations` [¶](#-arc-lower-verif-simulations)

*Lower verif.simulation ops to main functions*

### `-arc-make-tables` [¶](#-arc-make-tables)

*Transform appropriate arc logic into lookup tables*

### `-arc-merge-ifs` [¶](#-arc-merge-ifs)

*Merge control flow structures*

This pass optimizes control flow in a few ways. It moves operations closer
to their earliest user, if possible sinking them into blocks if all uses are
nested in the same block. It merges adjacent `scf.if` operations with the
same condition. And it moves operations in between two `scf.if` operations
ahead of the first if op to allow them to be merged. The pass runs on any
SSACFG regions nested under the operation it is applied to.

Note that this pass assumes that `!arc.state` and `!arc.memory` values can
never alias. That is, different values are assumed to never point to the
same storage location in simulation memory.

#### Statistics [¶](#statistics-7)

```
sunk                   : Ops sunk into blocks
moved-to-user          : Ops moved to first user
ifs-merged             : Adjacent scf.if ops merged
moved-from-between-ifs : Ops moved from between ifs to enable merging
iterations             : Number of iterations until no more ops were sunk/merged
```

### `-arc-merge-taps` [¶](#-arc-merge-taps)

*Merge TapOps observing the same value in the same Block.*

### `-arc-mux-to-control-flow` [¶](#-arc-mux-to-control-flow)

*Convert muxes with large independent fan-ins to if-statements*

### `-arc-print-cost-model` [¶](#-arc-print-cost-model)

*A dummy pass to test analysis passes*

#### Statistics [¶](#statistics-8)

```
Operation(s)           : Number of operations in the module
Pack operations(s)     : Number of scalar to vector packking in the module
Shuffle operation(s)   : Number of shuffles done to set up the VectorizeOps
VectorizeOps Body Cost : Number of operations inside the body of the VectorizeOps
All VectorizeOps Cost  : Total Cost of all VectorizeOps in the module
```

### `-arc-simplify-variadic-ops` [¶](#-arc-simplify-variadic-ops)

*Convert variadic ops into distributed binary ops*

#### Statistics [¶](#statistics-9)

```
skipped-multiple-blocks : Ops skipped due to operands in different blocks
simplified              : Ops simplified into binary ops
created                 : Ops created as part of simplification
reordered               : Ops where simplification reordered operands
```

### `-arc-split-funcs` [¶](#-arc-split-funcs)

*Split large funcs into multiple smaller funcs*

#### Options [¶](#options-30)

```
-split-bound : Size threshold (in ops) above which to split funcs
```

#### Statistics [¶](#statistics-10)

```
funcs-created : Number of new functions created
```

### `-arc-split-loops` [¶](#-arc-split-loops)

*Split arcs to break zero latency loops*

#### Statistics [¶](#statistics-11)

```
created : Arcs created during the splitting
removed : Arcs removed during the splitting
```

### `-arc-strip-sv` [¶](#-arc-strip-sv)

*Remove SV wire, reg, and assigns*

#### Options [¶](#options-31)

```
-async-resets-as-sync : Treat asynchronous resets as synchronous.
```

Calyx Dialect Passes [¶](#calyx-dialect-passes)
-----------------------------------------------

### `-affine-parallel-unroll` [¶](#-affine-parallel-unroll)

*Unrolls affine.parallel operations in a way suitable for Calyx.*

This pass unrolls `affine.parallel` operations completely and wrap each unrolled body
with `scf.execute_region` operations to better align with `calyx.par`’s representation.
Moreover, the newly created `affine.parallel` will be attached with attributes to
indicate that this is different from normal `affine.parallel`. Behavior is undefined if
there is a data race or memory banking contention.
An example:

```
affine.parallel (%x, %y) from (0, 0) to (2, 2) {
  %0 = memref.load %alloc_0[%x, %y] : memref<2x2xf32>
  %1 = memref.load %alloc_1[%x, %y] : memref<2x2xf32>
  %2 = arith.addf %0, %1 : f32
  memref.store %2, %alloc_2[%x, %y] : memref<2x2xf32>
}
=>
affine.parallel _ from 0 to 1 {
  scf.execute_region {
    %0 = memref.load %alloc_0[%c0, %c0] : memref<2x2xf32>
    %1 = memref.load %alloc_1[%c0, %c0] : memref<2x2xf32>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %alloc_2[%c0, %c0] : memref<2x2xf32>
  }
  scf.execute_region {
    %0 = memref.load %alloc_0[%c0, %c1] : memref<2x2xf32>
    %1 = memref.load %alloc_1[%c0, %c1] : memref<2x2xf32>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %alloc_2[%c0, %c1] : memref<2x2xf32>
  }
  scf.execute_region {
    %0 = memref.load %alloc_0[%c1, %c0] : memref<2x2xf32>
    %1 = memref.load %alloc_1[%c1, %c0] : memref<2x2xf32>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %alloc_2[%c1, %c0] : memref<2x2xf32>
  }
  scf.execute_region {
    %0 = memref.load %alloc_0[%c1, %c1] : memref<2x2xf32>
    %1 = memref.load %alloc_1[%c1, %c1] : memref<2x2xf32>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %alloc_2[%c1, %c1] : memref<2x2xf32>
  }
} {calyx.unroll = true}
```

### `-affine-ploop-unparallelize` [¶](#-affine-ploop-unparallelize)

*Unparallelize `affine.parallel` op to `affine.for`*

Unparallelize `affine.parallel` op to `affine.for` by the factor
specified in the attribute. For example:

```
#map_mod = affine_map<d0 -> d0 mod 2>
affine.parallel (%ip) = (0) to (6) {
  %mod = affine.apply #map_mod(%ip)
  scf.index_switch %mod:
  case 1 {
    affine.store %cst, %mem0[%ip floordiv 2] : memref<3xf32>
  }
  case 0 {
    affine.store %cst, %mem1[%ip floordiv 2] : memref<3xf32>
  }
  default {}
} {unparallelize.factor = 2}

=>

#map_sum = affine_map<(d0, d1) -> (d0 + d1)>
#map_mod = affine_map<d0 -> d0 mod 2>
affine.for %if = 0 to 6 step 2 {
  affine.parallel (%ip) = (0) to (2) {
    %i = affine.apply #map_sum(%if, %ip)
    %mod = affine.apply #map_mod(%i)
    scf.index_switch %mod:
    case 0 {
      affine.store %cst, %mem0[%i floordiv 2] : memref<3xf32>
    }
    case 1 {
      affine.store %cst, %mem1[%i floordiv 2] : memref<3xf32>
    }
    default {}
  }
}
```

### `-calyx-affine-to-scf` [¶](#-calyx-affine-to-scf)

*Lowers the Affine dialect to the SCF dialect while attaching relevant information used for the SCFToCalyx or the LoopScheduleToCalyx pass.*

### `-calyx-clk-insertion` [¶](#-calyx-clk-insertion)

*Inserts assignments from component clock to sub-component clock.*

### `-calyx-compile-control` [¶](#-calyx-compile-control)

*Generates latency-insensitive finite state machines to realize control.*

This pass performs a bottom-up traversal of the control program and does the following:

1. For each control statement such as “calyx.seq”, create a new GroupOp to contain all
   the structure to realize the schedule.
2. Implement the schedule by setting the constituent groups’ GoOp and DoneOp.
3. Replace the control statement in the control program with the corresponding
   compilation group.

### `-calyx-gicm` [¶](#-calyx-gicm)

*Lift group-invariant operations to wire-scope.*

This pass performs GICM (group-invariant code motion) of operations which are
deemed to be invariant of the group in which they are placed. In practice,
this amounts to anything which is not a `calyx.group_done/assign/group_go`
operation. GICM’d operations are lifted to wire-scope.

After GICM, a Calyx component has the following properties:

* No values are being defined within groups (excluding `calyx.group_go`).
  As such, groups will only contain group-level assignments
  (calyx.assign/group\_done).
* Any value referenced by operations within the group may safely be
  referenced by other groups, or operations in wire scope.
* A group does not define anything structural; it exclusively describes
  wiring between existing structures.

### `-calyx-go-insertion` [¶](#-calyx-go-insertion)

*Insert go signals into the guards of a group’s non-hole assignments*

This pass inserts the operation “calyx.group\_go” into the guards of all
assignments housed in the group, with the exception of the “calyx.group\_done”
terminator. For example,

Before:

```
calyx.group @Group1 {
  calyx.assign %in = %out1, %guard ? : i8
  %done = calyx.group_done %out2 : i1
}
```

After:

```
// The `go` assignment takes on an undefined
// value until the Compile Control pass.
%undef = calyx.undef : i1
...
calyx.group @Group1 {
  %go = calyx.group_go %undef : i1

  %and = comb.and %guard, %go : i1
  calyx.assign %in = %out1, %and ? : i8

  %done = calyx.group_done %out2 : i1
}
```

### `-calyx-remove-comb-groups` [¶](#-calyx-remove-comb-groups)

*Removes combinational groups from a Calyx component.*

Transforms combinational groups, which have a constant done condition,
into proper groups by registering the values read from the ports of cells
used within the combinational group.

It also transforms (invoke,if,while)-with into semantically equivalent
control programs that first enable a group that calculates and registers the
ports defined by the combinational group execute the respective cond group
and then execute the control operator.

Example [¶](#example)
---------------------

```
group comb_cond<"static"=0> {
    lt.right = 32'd10;
    lt.left = 32'd1;
    eq.right = r.out;
    eq.left = x.out;
    comb_cond[done] = 1'd1;
}
control {
    invoke comp(left = lt.out, ..)(..) with comb_cond;
    if lt.out with comb_cond {
        ...
    }
    while eq.out with comb_cond {
        ...
    }
}
```

into:

```
group comb_cond<"static"=1> {
    lt.right = 32'd10;
    lt.left = 32'd1;
    eq.right = r.out;
    eq.left = x.out;
    lt_reg.in = lt.out
    lt_reg.write_en = 1'd1;
    eq_reg.in = eq.out;
    eq_reg.write_en = 1'd1;
    comb_cond[done] = lt_reg.done & eq_reg.done ? 1'd1;
}
control {
    seq {
      comb_cond;
      invoke comp(left = lt_reg.out, ..)(..);
    }
    seq {
      comb_cond;
      if lt_reg.out {
          ...
      }
    }
    seq {
      comb_cond;
      while eq_reg.out {
          ...
          comb_cond;
      }
    }
}
```

### `-calyx-remove-groups` [¶](#-calyx-remove-groups)

*Inlines the groups in a Calyx component.*

This pass removes the Group interface from the Calyx program, and inlines all
assignments. This is done in the following manner:

1. Assign values to the ‘done’ signal of the component, corresponding with the
   top-level control group’s DoneOp. Add the ‘go’ signal of the component to
   all assignments.
2. TODO(Calyx): If there are multiple writes to a signal, replace the reads
   with the disjunction.
3. Remove all groups.

### `-calyx-reset-insertion` [¶](#-calyx-reset-insertion)

*Connect component reset to sub-component reset for applicable components.*

### `-exclude-exec-region-canonicalize` [¶](#-exclude-exec-region-canonicalize)

*Canonicalize all legal operations except `scf.execute_region`*

The AffineParallelUnroll pass unrolls the body of `affine.parallel`
to multiple copies of `scf.execute_region`s. Since the semantics
of those `scf.execute_region` has deviated from the original MLIR
definition, a new canonicalization pass that does not operate on
`scf.execute_region` is needed.

Comb Dialect Passes [¶](#comb-dialect-passes)
---------------------------------------------

### `-comb-assume-two-valued` [¶](#-comb-assume-two-valued)

*Simplify under assumption of two-valued logic*

Performs simplifying transformations that are valid if all integer values
are always two-valued (i.e., all bits are either 1 or 0).

### `-comb-balance-mux` [¶](#-comb-balance-mux)

*Balance and optimize mux chains*

Optimizes mux chains through enhanced folding and priority mux
rebalancing. Converts index comparisons to arrays and rebalances
linear chains into balanced trees, reducing depth from O(n) to O(log n).

#### Options [¶](#options-32)

```
-mux-chain-threshold : Minimum number of linear mux chains to trigger rebalancing
```

### `-comb-int-range-narrowing` [¶](#-comb-int-range-narrowing)

*Reduce comb op bitwidth based on integer range analysis.*

Compute a basic value range analysis, by propagating integer intervals
through the domain. The analysis is limited by a lack of sign-extension
operator in the comb dialect, leading to an over-approximation.
Particularly for signed arithmetic, a single interval is often an
over-approximation, a more precise analysis would require a union of
intervals.

### `-comb-overflow-annotating` [¶](#-comb-overflow-annotating)

*Annotate comb ops with no overflow signal (nuw).*

Using integer range analysis, annotate comb ops which cannot overflow with
LLVMs no unsigned wrap (nuw) signal. This is useful for downstream passes
which need to handle overflow conditions.

### `-comb-simplify-tt` [¶](#-comb-simplify-tt)

*Simplify truth tables that depend on 1 or less inputs*

Simplifies truth tables that are constant or depend only on single input.
For truth tables that depend on no input, reduce them to hw.constant. For
truth tables that depend on single input, reduce to identity of input or
negation of input.

### `-lower-comb` [¶](#-lower-comb)

*Lowers the some of the comb ops*

Some operations in the comb dialect (e.g. `comb.truth_table`) are not
directly supported by ExportVerilog. They need to be lowered into ops which
are supported. There are many ways to lower these ops so we do this in a
separate pass. This also allows the lowered form to participate in
optimizations like the comb canonicalizers.

Datapath Dialect Passes [¶](#datapath-dialect-passes)
-----------------------------------------------------

### `-datapath-reduce-delay` [¶](#-datapath-reduce-delay)

*Reduce datapath delay at a cost of potentially greater area.*

This pass performs datapath transformations to reduce the delay of any
matching paths. The transformations include logic replication/duplication to
operate on carry-save values, potentially increasing area usage.

DC Dialect Passes [¶](#dc-dialect-passes)
-----------------------------------------

### `-dc-dematerialize-forks-sinks` [¶](#-dc-dematerialize-forks-sinks)

*Dematerialize fork and sink operations.*

This pass analyses a function-like operation and removes all fork and sink
operations.

### `-dc-materialize-forks-sinks` [¶](#-dc-materialize-forks-sinks)

*Materialize fork and sink operations.*

This pass analyses a function-like operation and inserts fork and sink
operations ensuring that all values have exactly one use.

### `-dc-print-dot` [¶](#-dc-print-dot)

*Print .dot graph of a DC function.*

This pass analyses a dc modulen and prints a .dot graph of the
structure. If multiple functions are present in the IR, the top level
function will be printed, and called functions will be subgraphs within
the main graph.

ESI Dialect Passes [¶](#esi-dialect-passes)
-------------------------------------------

### `-esi-appid-hier` [¶](#-esi-appid-hier)

*Build an AppID based hierarchy rooted at top module ’top’*

#### Options [¶](#options-33)

```
-top : Root module of the instance hierarchy
```

### `-esi-build-manifest` [¶](#-esi-build-manifest)

*Build a manifest of an ESI system*

#### Options [¶](#options-34)

```
-top : Root module of the instance hierarchy
```

### `-esi-clean-metadata` [¶](#-esi-clean-metadata)

*Clean up ESI service metadata*

### `-esi-connect-services` [¶](#-esi-connect-services)

*Connect up ESI service requests to service providers*

### `-lower-esi-bundles` [¶](#-lower-esi-bundles)

*Lower ESI bundles to channels.*

### `-lower-esi-ports` [¶](#-lower-esi-ports)

*Lower ESI input and/or output ports.*

### `-lower-esi-to-hw` [¶](#-lower-esi-to-hw)

*Lower ESI to HW where possible and SV elsewhere.*

#### Options [¶](#options-35)

```
-platform : Target this platform
```

### `-lower-esi-to-physical` [¶](#-lower-esi-to-physical)

*Lower ESI abstract Ops to ESI physical ops.*

### `-lower-esi-types` [¶](#-lower-esi-types)

*Lower ESI high level types.*

### `-verify-esi-connections` [¶](#-verify-esi-connections)

*Verify that channels and bundles are only used once*

FIRRTL Dialect Passes [¶](#firrtl-dialect-passes)
-------------------------------------------------

### `-firrtl-add-seqmem-ports` [¶](#-firrtl-add-seqmem-ports)

*Add extra ports to memory modules*

This pass looks for `AddSeqMemPortAnnotation` annotations and adds extra
ports to memories. It will emit metadata based if the
`AddSeqMemPortsFileAnnotation` annotation is specified.

This pass requires that FIRRTL MemOps have been lowered to modules to add
the extra ports.

#### Statistics [¶](#statistics-12)

```
num-added-ports : Number of extra ports added
```

### `-firrtl-annotate-input-only-modules` [¶](#-firrtl-annotate-input-only-modules)

*Annotate input-only modules with InlineAnnotation*

This pass identifies modules that have only input ports (no output or inout
ports) and annotates them with `firrtl.passes.InlineAnnotation`. These
modules don’t produce any hardware outputs.

Inlining these modules is necessary because some synthesis tools treat
modules without outputs as blackboxes, which can cause synthesis errors.

#### Statistics [¶](#statistics-13)

```
num-annotated : Number of modules annotated
```

### `-firrtl-assign-output-dirs` [¶](#-firrtl-assign-output-dirs)

*Assign output directories to modules.*

While some modules are assigned output directories by the user, many modules
“don’t care” what output directory they are placed into. This pass uses the
instance graph to assign these modules to the “deepest” output directory
possible.

#### Options [¶](#options-36)

```
-output-dir : The default output directory.
```

### `-firrtl-blackbox-reader` [¶](#-firrtl-blackbox-reader)

*Load source files for black boxes into the IR*

This pass reads the Verilog source files for black boxes and adds them as
`sv.verbatim.file` operations into the IR. Later passes can then write
these files back to disk to ensure that they can be accessed by other tools
down the line in a well-known location. Supports inline and path
annotations for black box source files.

The supported `firrtl.circuit` annotations are:

* `{class = "firrtl.transforms.BlackBoxTargetDirAnno", targetDir = "..."}`
  Overrides the target directory into which black box source files are
  emitted.

The supported `firrtl.extmodule` annotations are:

* ```
  {
    class = "firrtl.transforms.BlackBoxInlineAnno",
    name = "myfile.v",
    text = "..."
  }
  ```

  Specifies the black box source code (`text`) inline. Generates a file with
  the given `name` in the target directory.
* ```
  {
    class = "firrtl.transforms.BlackBoxPathAnno",
    path = "myfile.v"
  }
  ```

  Specifies the file `path` as source code for the module. Copies the file
  to the target directory.

#### Options [¶](#options-37)

```
-input-prefix : Prefix for input paths in black box annotations. This should be the directory where the input file was located, to allow for annotations relative to the input file.
```

### `-firrtl-check-comb-loops` [¶](#-firrtl-check-comb-loops)

*Check combinational cycles and emit errors*

This pass checks combinational cycles in the IR and emit errors.

### `-firrtl-check-layers` [¶](#-firrtl-check-layers)

*Check for illegal layers instantiated under layers*

This pass checks for illegal instantiation of a module with a layer
underneath another layer.

### `-firrtl-check-recursive-instantiation` [¶](#-firrtl-check-recursive-instantiation)

*Check for illegal recursive instantiation*

This pass checks for illegal recursive module instantion. Recursive
instantiation is when a module instantiates itself, either directly or
indirectly through other modules it instantiates. Recursive module
instantiation is illegal because it would require infinite hardware to
synthesize. Recursive class instantiation is illegal as it would create an
infinite loop.

### `-firrtl-dedup` [¶](#-firrtl-dedup)

*Deduplicate modules which are structurally equivalent*

This pass detects modules which are structurally equivalent and removes the
duplicate module by replacing all instances of one with the other.
Structural equivalence ignores the naming of operations and fields in
bundles, and any annotations. Deduplicating a module may cause the result
type of instances to change if the field names of a bundle type change. To
handle this, the pass will update any bulk-connections so that the correct
fields are legally connected. Deduplicated modules will have their
annotations merged, which tends to create many non-local annotations.

#### Options [¶](#options-38)

```
-dedup-classes : Deduplicate classes, violating their nominal typing.
```

#### Statistics [¶](#statistics-14)

```
num-erased-modules : Number of modules which were erased by deduplication
```

### `-firrtl-drop-const` [¶](#-firrtl-drop-const)

*Drop ‘const’ modifier from types*

This pass drops the ‘const’ modifier from all types and removes all
const-cast ops.

This simplifies downstream passes and folds so that they do not need to
take ‘const’ into account.

### `-firrtl-drop-names` [¶](#-firrtl-drop-names)

*Drop interesting names*

This pass changes names of namable ops to droppable so that we can disable
full name preservation. For example,
before:

```
%a = firrtl.node interesting_name %input
```

after:

```
%a = firrtl.node droppable_name %input
```

#### Options [¶](#options-39)

```
-preserve-values : specify the values which can be optimized away
```

#### Statistics [¶](#statistics-15)

```
num-names-dropped   : Number of names dropped
num-names-converted : Number of interesting names made droppable
```

### `-firrtl-eliminate-wires` [¶](#-firrtl-eliminate-wires)

*Eliminate Unneeded Wires*

This pass eliminates wires for which the write dominates the reads and
for which there are no other preservation reasons.

#### Statistics [¶](#statistics-16)

```
num-erased-wires         : Number of wires erased
num-complex-wires        : Number of wires not erased due to complex writes
num-not-dominating-wires : Number of wires not erased due to dominance issues
num-not-passive-wires    : Number of wires not erased due to type issues
```

### `-firrtl-emit-metadata` [¶](#-firrtl-emit-metadata)

*Emit metadata of the FIRRTL modules*

This pass handles the emission of several different kinds of metadata.

#### Options [¶](#options-40)

```
-repl-seq-mem      : Lower the seq mem for macro replacement and emit relevant metadata
-repl-seq-mem-file : File to which emit seq meme metadata
```

### `-firrtl-expand-whens` [¶](#-firrtl-expand-whens)

*Remove all when conditional blocks.*

This pass will:

1. Resolve last connect semantics.
2. Remove all when operations.

When a wire has multiple connections, only the final connection is used,
all previous connections are overwritten. When there is a conditional
connect, the previous connect is only overwritten when the condition
holds:

```
w <= a
when c :
  w <= b

; Equivalent to:
w <= mux(c, b, a)
```

This pass requires that all connects are expanded.

### `-firrtl-extract-instances` [¶](#-firrtl-extract-instances)

*Move annotated instances upwards in the module hierarchy*

This pass takes instances in the design annotated with one out of a
particular set of annotations and pulls them upwards to a location further
up in the module hierarchy.

The annotations that control the behaviour of this pass are:

* `MarkDUTAnnotation`
* `ExtractBlackBoxAnnotation`
* `ExtractClockGatesFileAnnotation`

### `-firrtl-finalize-ir` [¶](#-firrtl-finalize-ir)

*Perform final IR mutations after ExportVerilog*

This pass finalizes the IR after it has been exported with ExportVerilog,
and before firtool emits the final IR. This includes mutations like dropping
verbatim ops that represent sideband files and are not required in the IR.

### `-firrtl-flatten-memory` [¶](#-firrtl-flatten-memory)

*Flatten aggregate memory data to a UInt*

This pass converts the data type of memories into a flat UInt, and inserts
appropriate bitcasts to access the data.

#### Statistics [¶](#statistics-17)

```
num-flatten-mems : Number of memories flattened
```

### `-firrtl-grand-central` [¶](#-firrtl-grand-central)

*Remove Grand Central Annotations*

Processes annotations associated with SiFive’s Grand Central utility.

#### Options [¶](#options-41)

```
-companion-mode : specify the handling of companion modules
-no-views       : Delete FIRRTL view intrinsics
```

#### Statistics [¶](#statistics-18)

```
num-views-created       : Number of top-level SystemVerilog interfaces that were created
num-interfaces-created  : Number of SystemVerilog interfaces that were created
num-xmrs-created        : Number of SystemVerilog XMRs added
num-annotations-removed : Number of annotations removed
```

### `-firrtl-imconstprop` [¶](#-firrtl-imconstprop)

*Intermodule constant propagation and dead code elimination*

Use optimistic constant propagation to delete ports and unreachable IR.

#### Statistics [¶](#statistics-19)

```
num-folded-op : Number of operations folded
num-erased-op : Number of operations erased
```

### `-firrtl-imdeadcodeelim` [¶](#-firrtl-imdeadcodeelim)

*Intermodule dead code elimination*

This pass performs inter-module liveness analysis and deletes dead code
aggressively. A value is considered as alive if it is connected to a port
of public modules or a value with a symbol. We first populate alive values
into a set, and then propagate the liveness by looking at their dataflow.

#### Statistics [¶](#statistics-20)

```
num-erased-ops     : Number of operations erased
num-erased-modules : Number of modules erased
num-removed-ports  : Number of ports erased
```

### `-firrtl-infer-domains` [¶](#-firrtl-infer-domains)

*Infer and type check all firrtl domains*

This pass does domain inference on a FIRRTL circuit. The end result of this
is either a corrrctly domain-checked FIRRTL circuit or failure with verbose
error messages indicating why the FIRRTL circuit has illegal domain
constructs.

E.g., this pass can be used to check for illegal clock-domain-crossings if
clock domains are specified for signals in the design.

#### Options [¶](#options-42)

```
-mode : infer, check, infer-all.
```

### `-firrtl-infer-resets` [¶](#-firrtl-infer-resets)

*Infer reset synchronicity and add implicit resets*

This pass infers whether resets are synchronous or asynchronous, and extends
reset-less registers with a reset based on the following
annotations:

* `circt.FullResetAnnotation`
* `circt.ExcludeFromFullResetAnnotation`
* `sifive.enterprise.firrtl.FullAsyncResetAnnotation` (deprecated)
* `sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation` (deprecated)

### `-firrtl-infer-rw` [¶](#-firrtl-infer-rw)

*Infer the read-write memory port*

This pass merges the read and write ports of a memory, using a simple
module-scoped heuristic. The heuristic checks if the read and write enable
conditions are mutually exclusive.
The heuristic tries to break up the read enable and write enable logic into an
`AND` expression tree. It then compares the read and write `AND` terms,
looking for a situation where the read/write is the complement of the write/read.

#### Statistics [¶](#statistics-21)

```
num-rw-port-mems-inferred : Number of memories inferred to use RW port
```

### `-firrtl-infer-widths` [¶](#-firrtl-infer-widths)

*Infer the width of types*

This pass infers the widths of all types throughout a FIRRTL module, and
emits diagnostics for types that could not be inferred.

### `-firrtl-inject-dut-hier` [¶](#-firrtl-inject-dut-hier)

*Add a level of hierarchy outside the DUT*

This pass takes the DUT (as indicated by the presence of a
MarkDUTAnnotation) and moves all the contents of it into a new module
insided the DUT named by an InjectDUTHierarchyAnnotation. This pass is
intended to be used in conjunction with passes that pull things out of the
DUT, e.g., SRAM extraction, to give the extracted modules a new home that is
still inside the original DUT.

### `-firrtl-inliner` [¶](#-firrtl-inliner)

*Performs inlining, flattening, and dead module elimination*

This inliner pass will inline any instance of module marked as inline, and
recursively inline all instances inside of a module marked with flatten.
This pass performs renaming of every entity with a name that is inlined by
prefixing it with the instance name. This pass also will remove any module
which is not reachable from the top level module.

The inline and flatten annotation attributes are attached to module
definitions, and they are:

```
  {class = "firrtl.passes.InlineAnnotation"}
  {class = "firrtl.transforms.FlattenAnnotation"}
```

### `-firrtl-inner-symbol-dce` [¶](#-firrtl-inner-symbol-dce)

*Eliminate dead inner symbols*

This pass deletes all inner symbols which have no uses. This is necessary to
unblock optimizations and removal of the operations which have these unused
inner symbols.

#### Statistics [¶](#statistics-22)

```
num-inner-refs-found  : Number of inner-refs found
num-inner-sym-found   : Number of inner symbols found
num-inner-sym-removed : Number of inner symbols removed
```

### `-firrtl-layer-merge` [¶](#-firrtl-layer-merge)

*Merge layer blocks*

Combine all layer blocks in a module which reference the same layer
definition.

#### Statistics [¶](#statistics-23)

```
num-merged : Number of layers merged
```

### `-firrtl-layer-sink` [¶](#-firrtl-layer-sink)

*Sink operations into layer blocks*

This pass sinks ops into layers, whenever possible, to minimize unused
hardware in the design.

### `-firrtl-link-circuits` [¶](#-firrtl-link-circuits)

*Links multiple circuits into a single one*

This pass concatenates all circuits into one, iterates through all extmodules
to find matching public module implementations, replacing them when possible
and any remaining extmodules are treated as blackboxes.

#### Options [¶](#options-43)

```
-base-circuit : The base circuit name.
-no-mangle    : Do not perform private symbol mangling.
```

### `-firrtl-lint` [¶](#-firrtl-lint)

*An analysis pass to detect static simulation failures.*

This pass detects operations that will trivially fail any simulation.
Currently it detects assertions whose predicate condition can be statically
inferred to be false. The pass emits error on such failing ops.

#### Options [¶](#options-44)

```
-lint-static-asserts : enable linting of static assertions
-lint-xmrs-in-design : enable linting of XMRs that exist in the design
```

### `-firrtl-lower-annotations` [¶](#-firrtl-lower-annotations)

*Lower FIRRTL annotations to usable entities*

Lower FIRRTL annotations to usable forms. FIRRTL annotations are a big bag
of semi-structured, irregular JSON. This pass normalizes all supported
annotations and annotation paths.

#### Options [¶](#options-45)

```
-disable-annotation-classless : Ignore classless annotations.
-disable-annotation-unknown   : Ignore unknown annotations.
-no-ref-type-ports            : Create normal ports, not ref type ports.
```

#### Statistics [¶](#statistics-24)

```
num-raw-annos       : Number of raw annotations on circuit
num-added-annos     : Number of additional annotations
num-annos           : Total number of annotations processed
num-unhandled-annos : Number of unhandled annotations
num-reused-hierpath : Number of reused HierPathOp's
```

### `-firrtl-lower-chirrtl` [¶](#-firrtl-lower-chirrtl)

*Infer the memory ports of SeqMem and CombMem*

This pass finds the CHIRRTL behavioral memories and their ports, and
transforms them into standard FIRRTL memory operations. For each
`seqmem` or `combmem`, a new memory is created. For every `memoryport`
operation using a CHIRRTL memory, a memory port is defined on the
new standard memory.

The direction or kind of the port is inferred from how each of the memory
ports is used in the IR. If a memory port is only written to, it becomes
a `Write` port. If a memory port is only read from, it become a `Read`
port. If it is used both ways, it becomes a `ReadWrite` port.

`Write`, `ReadWrite` and combinational `Read` ports are disabled by
default, but then enabled when the CHIRRTL memory port is declared.
Sequential `Read` ports have more complicated enable inference:

1. If a wire or register is used as the index of the memory port, then
   the memory is enabled whenever a non-invalid value is driven to the
   address.
2. If a node is used as the index of the memory port, then the memory is
   enabled at the declaration of the node.
3. In all other cases, the memory is never enabled.

In the first two cases, they can easily produce a situation where we try
to enable the memory before it is even declared. This produces a
compilation error.

#### Statistics [¶](#statistics-25)

```
num-created-mems  : Number of memories created
num-lowered-mems  : Number of memories lowered
num-portless-mems : Number of memories dropped as having no valid ports
```

### `-firrtl-lower-classes` [¶](#-firrtl-lower-classes)

*Lower FIRRTL classes and objects to OM classes and objects*

This pass walks all FIRRTL classes and creates OM classes. It also lowers
FIRRTL objects to OM objects. OM classes are created with parameters
corresponding to FIRRTL input properties, and OM class fields corresponding
to FIRRTL output properties. FIRRTL operations are converted to OM
operations.

### `-firrtl-lower-domains` [¶](#-firrtl-lower-domains)

*Lower domain information to properties*

Lower all domain information into FIRRTL properties. This has the effect of
erasing all domain information.

#### Statistics [¶](#statistics-26)

```
num-domains : Number of domains lowered
```

### `-firrtl-lower-dpi` [¶](#-firrtl-lower-dpi)

*Lower DPI intrinsic into Sim DPI operations*

### `-firrtl-lower-intmodules` [¶](#-firrtl-lower-intmodules)

*Lower instances instances of intrinsic modules to ops.*

This pass replaces instances of intrinsic modules (intmodule) with
`firrtl.int.generic` operations, and removes the intmodule’s from
the circuit.

Use this before running LowerIntrinsics.

#### Options [¶](#options-46)

```
-fixup-eicg-wrapper : Lower `EICG_wrapper` modules into clock gate intrinsics
```

#### Statistics [¶](#statistics-27)

```
num-instances  : Number of intmodules instances lowered
num-intmodules : Number of intmodules lowered
```

### `-firrtl-lower-intrinsics` [¶](#-firrtl-lower-intrinsics)

*Lower intrinsics*

This pass lowers generic intrinsic ops to their implementation or op.

#### Statistics [¶](#statistics-28)

```
num-converted : Number of intrinsic operations lowered
```

### `-firrtl-lower-layers` [¶](#-firrtl-lower-layers)

*Lower layers and layer blocks to instances*

This pass lowers FIRRTL layers as defined by their convention. After this
pass, all layer blocks and layers are removed.

#### Options [¶](#options-47)

```
-emit-all-bind-files : Emit bind files for private modules.
```

### `-firrtl-lower-matches` [¶](#-firrtl-lower-matches)

*Remove all matchs conditional blocks*

Lowers FIRRTL match statements in to when statements, which can later be
lowered with ExpandWhens.

### `-firrtl-lower-memory` [¶](#-firrtl-lower-memory)

*Lower memories to generated modules*

This pass lowers FIRRTL memory operations to generated modules.

#### Statistics [¶](#statistics-29)

```
num-created-mem-modules : Number of modules created
num-lowered-mems        : Number of memories lowered
```

### `-firrtl-lower-open-aggs` [¶](#-firrtl-lower-open-aggs)

*Lower ‘Open’ aggregates by splitting out non-hardware elements*

This pass lowers aggregates of the more open varieties into their equivalents
using only hardware types, by pulling out non-hardware to other locations.

### `-firrtl-lower-signatures` [¶](#-firrtl-lower-signatures)

*Lower FIRRTL module signatures*

Lower aggregate FIRRTL types in Modules as indicated by the calling
convention.

### `-firrtl-lower-types` [¶](#-firrtl-lower-types)

*Lower FIRRTL types to ground types*

Lower aggregate FIRRTL types to ground types. Memories, ports, wires, etc
are split apart by elements of aggregate types. The only aggregate types
which exist after this pass are memory ports, though memory data types are
split.

Connect and expansion and canonicalization happen in this pass.

#### Options [¶](#options-48)

```
-preserve-aggregate : Specify aggregate preservation mode
-preserve-memories  : Specify memory preservation mode
```

### `-firrtl-lower-xmr` [¶](#-firrtl-lower-xmr)

*Lower ref ports to XMR*

This pass lowers RefType ops and ports to verbatim encoded XMRs.

### `-firrtl-materialize-debug-info` [¶](#-firrtl-materialize-debug-info)

*Generate debug ops to track FIRRTL values*

This pass creates debug ops to track FIRRTL-level ports, nodes, wires,
registers, and instances throughout the pipeline. The `DebugInfo` analysis
can then be used at a later point in the pipeline to obtain a source
language view into the lowered IR.

### `-firrtl-mem-to-reg-of-vec` [¶](#-firrtl-mem-to-reg-of-vec)

*Convert combinational memories to a vector of registers*

This pass generates the logic to implement a memory using Registers.

#### Options [¶](#options-49)

```
-repl-seq-mem           : Prepare seq mems for macro replacement
-ignore-read-enable-mem : ignore the read enable signal, instead of assigning X on read disable
```

#### Statistics [¶](#statistics-30)

```
num-converted-mems : Number of memories converted to registers
```

### `-firrtl-merge-connections` [¶](#-firrtl-merge-connections)

*Merge field-level connections into full bundle connections*

#### Options [¶](#options-50)

```
-aggressive-merging : Merge connections even when source values won't be simplified.
```

### `-firrtl-module-summary` [¶](#-firrtl-module-summary)

*Print a summary of modules.*

The pass produces a summary of modules.

### `-firrtl-passive-wires` [¶](#-firrtl-passive-wires)

*Make FIRRTL wires have passive type*

Eliminate flips from aggregate types on wires.

### `-firrtl-print-field-source` [¶](#-firrtl-print-field-source)

*Print field source information.*

### `-firrtl-print-instance-graph` [¶](#-firrtl-print-instance-graph)

*Print a DOT graph of the module hierarchy.*

### `-firrtl-print-nla-table` [¶](#-firrtl-print-nla-table)

*Print the NLA Table.*

### `-firrtl-probes-to-signals` [¶](#-firrtl-probes-to-signals)

*Convert probes to signals.*

This pass transforms the design to replace probes with non-probe signals
that are explicitly routed through the design instead of remote access.
The result will not be ABI-compatible but provide similar behavior without the
use of potentially unsynthesizable cross-module references.

The pass will do a complete conversion or produce an error.

Not all FIRRTL designs with probes can be equivalently represented
with non-probes.

Run after LowerOpenAggs, ExpandWhens (so ref.define can always be replaced),
and any colored probes are removed through specialization.

Due to differences in width and reset inference, it is suggested to run
those first as well but not required.

RWProbes and related operations are not supported.

### `-firrtl-randomize-register-init` [¶](#-firrtl-randomize-register-init)

*Randomize register initialization.*

This pass eagerly creates large vectors of randomized bits for initializing
registers, and marks each register with attributes indicating which bits to
read. If the registers survive until LowerToHW, their initialization logic
will pick up the correct bits.

This ensures a stable initialization, so registers should always see the
same initial value for the same seed, regardless of optimization levels.

### `-firrtl-register-optimizer` [¶](#-firrtl-register-optimizer)

*Optimizer Registers*

This pass applies classic FIRRTL register optimizations. These
optimizations are isolated to this pass as they can change the visible
behavior of the register, especially before reset.

### `-firrtl-remove-unused-ports` [¶](#-firrtl-remove-unused-ports)

*Remove unused ports*

This pass removes unused ports without annotations or symbols. Implementation
wise, this pass iterates over the instance graph in a topological order from
leaves to the top so that we can remove unused ports optimally.

#### Options [¶](#options-51)

```
-ignore-dont-touch : remove unused ports even if they have a symbol or annotation
```

#### Statistics [¶](#statistics-31)

```
num-removed-ports : Number of ports erased
```

### `-firrtl-resolve-paths` [¶](#-firrtl-resolve-paths)

*Lowers UnresolvedPathOps to PathOps*

FIRRTL path operations are initially create as unresolved path operations,
which encode their target as a string. This pass parses and resolves those
target strings to actual path operations. Path operations refer to their
targets using annotations with a unique identifier.

### `-firrtl-resolve-traces` [¶](#-firrtl-resolve-traces)

*Write out TraceAnnotations to an output annotation file*

This pass implements Chisel’s Trace API. It collects all TraceAnnotations
that exist in the circuit, updates them with information about the final
target in a design, and writes these to an output annotation file. This
exists for Chisel users to build tooling around them that needs to query the
final output name/path of some component in a Chisel circuit.

Note: this pass and API are expected to be eventually replaced via APIs and
language bindings that enable users to directly query the MLIR.

#### Options [¶](#options-52)

```
-file : Output file for the JSON-serialized Trace Annotations
```

### `-firrtl-sfc-compat` [¶](#-firrtl-sfc-compat)

*Perform SFC Compatibility fixes*

### `-firrtl-specialize-layers` [¶](#-firrtl-specialize-layers)

*Specialize a configurable design for a given layer*

This pass performs specialization of layers, layerblocks, and probe
operations by permanantly enabling or disabling layers. The layers are
specified by attributes on the `firrtl.circuit` operation in the
`enable_layers` and `disable_layers` attributes which are both lists of
SymbolRefAttrs. When a layer is specialized, it will be removed from the
circuit and all references to it should be cleaned up. If a layer is
disabled, all layerblock are referencing that layer will be deleted, and all
probe colored by that layer will be removed. When a layer is enabled, all
contents of the layer will be inlined in to the outer context, and probes
colored by that layer will no longer be colored.

### `-firrtl-specialize-option` [¶](#-firrtl-specialize-option)

*Specialize a configurable design for a given option.*

If the design has option groups, instead of preserving the information
up to the emitted SystemVerilog, this pass specializes it early on for a
given choice.

#### Options [¶](#options-53)

```
-select-default-for-unspecified-instance-choice : Specialize instance choice to default, if no option selected.
```

#### Statistics [¶](#statistics-32)

```
num-instances : Number of instances specialized
```

### `-firrtl-vb-to-bv` [¶](#-firrtl-vb-to-bv)

*Transform vector-of-bundles to bundle-of-vectors*

This pass converts vectors containing bundles, into bundles containing
vectors.

### `-firrtl-vectorization` [¶](#-firrtl-vectorization)

*Transform firrtl primitive operations into vector operations*

FSM Dialect Passes [¶](#fsm-dialect-passes)
-------------------------------------------

### `-fsm-print-graph` [¶](#-fsm-print-graph)

*Print a DOT graph of an FSM’s structure.*

Handshake Dialect Passes [¶](#handshake-dialect-passes)
-------------------------------------------------------

### `-handshake-add-ids` [¶](#-handshake-add-ids)

*Add an ID to each operation in a handshake function.*

This pass adds an ID to each operation in a handshake function. This id can
be used in lowerings facilitate mapping lowered IR back to the handshake code
which it originated from. An ID is unique with respect to other operations
of the same type in the function. The tuple of the operation name and the
operation ID denotes a unique identifier for the operation within the
`handshake.func` operation.

### `-handshake-dematerialize-forks-sinks` [¶](#-handshake-dematerialize-forks-sinks)

*Dematerialize fork and sink operations.*

This pass analyses a handshake.func operation and removes all fork and sink
operations.

### `-handshake-insert-buffers` [¶](#-handshake-insert-buffers)

*Insert buffers to break graph cycles*

#### Options [¶](#options-54)

```
-strategy    : Strategy to apply. Possible values are: cycles, allFIFO, all (default)
-buffer-size : Number of slots in each buffer
```

### `-handshake-legalize-memrefs` [¶](#-handshake-legalize-memrefs)

*Memref legalization and lowering pass.*

Lowers various memref operations to a state suitable for passing to the
CFToHandshake lowering.

### `-handshake-lock-functions` [¶](#-handshake-lock-functions)

*Lock each function to only allow single invocations.*

This pass adds a locking mechanism to each handshake function. This mechanism
ensures that only one control token can be active in a function at each point
in time.

### `-handshake-lower-extmem-to-hw` [¶](#-handshake-lower-extmem-to-hw)

*Lowers handshake.extmem and memref inputs to ports.*

Lowers handshake.extmem and memref inputs to a hardware-targeting
memory accessing scheme (explicit load- and store ports on the top
level interface).

#### Options [¶](#options-55)

```
-wrap-esi : Create an ESI wrapper for the module. Any extmem will be served by an esi.mem.ram service
```

### `-handshake-materialize-forks-sinks` [¶](#-handshake-materialize-forks-sinks)

*Materialize fork and sink operations.*

This pass analyses a handshake.func operation and inserts fork and sink
operations ensuring that all values have exactly one use.

### `-handshake-op-count` [¶](#-handshake-op-count)

*Count the number of operations (resources) in a handshake function.*

This pass analyses a handshake.func operation and prints the number of
operations (resources) used the function.

### `-handshake-print-dot` [¶](#-handshake-print-dot)

*Print .dot graph of a handshake function.*

This pass analyses a handshake.func operation and prints a .dot graph of the
structure. If multiple functions are present in the IR, the top level
function will be printed, and called functions will be subgraphs within
the main graph.

### `-handshake-remove-buffers` [¶](#-handshake-remove-buffers)

*Remove buffers from handshake functions.*

This pass analyses a handshake.func operation and removes any buffers from
the function.

### `-handshake-split-merges` [¶](#-handshake-split-merges)

*Deconstruct >2 input merge operations into 2-input merges*

This pass deconstructs the (rather complex) semantics of a >2 input merge
and control merge operation into a series of 2-input merge operations +
supporting logic.

HW Dialect Passes [¶](#hw-dialect-passes)
-----------------------------------------

### `-hw-aggregate-to-comb` [¶](#-hw-aggregate-to-comb)

*Lower aggregate operations to comb operations*

This pass lowers aggregate *operations* to comb operations within modules.
Note that this pass does not lower ports. Ports lowering is handled
by FlattenIO.

This pass will change the behavior of out-of-bounds access of arrays,
specifically the last element of the array is used as a value for
out-of-bounds access.

### `-hw-bypass-inner-symbols` [¶](#-hw-bypass-inner-symbols)

*Pass through values through inner symbols*

This pass moves inner symbols from ports to wires, then bypasses wire
operations with inner symbols by replacing uses with their inputs while
keeping the wire to preserve the symbol. This enables optimizations to
cross symbol boundaries while maintaining symbol references.

Warning: This transformation assumes that values associated with inner
symbols are not mutated through inner symbols (e.g. force). This assumption
may not hold in simulation, but is safe in synthesis. This pass treats
inner symbols differently from the optimization-blocking semantics that
other parts of CIRCT use, so it is opt-in and should only be used when
the above assumptions hold.

#### Statistics [¶](#statistics-33)

```
num-ports-moved : Number of inner symbols moved from ports to wires
```

### `-hw-convert-bitcasts` [¶](#-hw-convert-bitcasts)

*Convert hw.bitcast operations to bit extracts and concats*

This passes attempts to convert `hw.bitcast` operations on HW aggregates to
functionally identical sequences of explicit aggregate and bit-access
operations. It currently supports `!hw.array` and `!hw.struct` types.

#### Options [¶](#options-56)

```
-allow-partial-conversion : Ignore bitcasts involving unsupported types
```

### `-hw-flatten-io` [¶](#-hw-flatten-io)

*Flattens hw::Structure typed in- and output ports.*

#### Options [¶](#options-57)

```
-recursive      : Recursively flatten nested structs.
-flatten-extern : Flatten the extern modules also.
-join-char      : Use a custom character to construct the flattened names.
```

### `-hw-flatten-modules` [¶](#-hw-flatten-modules)

*Eagerly inline private modules*

This pass eagerly inlines private HW modules into their instantiation sites.
This is necessary for verification purposes, as model checking backends do not
require or support the use of module hierarchy. For simulation, module hierarchies
degenerate into a purely cosmetic construct, at which point it is beneficial
to fully flatten the module hierarchy to simplify further analysis and
optimization of state transfer arcs.

By default, all private modules are inlined. The pass supports heuristics to
control which modules are inlined based on their characteristics.

#### Options [¶](#options-58)

```
-hw-inline-empty      : Inline modules that are empty (only contain hw.output)
-hw-inline-no-outputs : Inline modules that have no output ports
-hw-inline-single-use : Inline modules that have only one use
-hw-inline-small      : Inline modules that are small (fewer than smallThreshold operations)
-hw-small-threshold   : Maximum number of operations for a module to be considered small
-hw-inline-with-state : Allow inlining of modules that contain state (seq.firreg operations)
-hw-inline-public     : Inline public modules as well as private ones
-hw-inline-all        : Inline all private modules regardless of heuristics (default behavior)
```

### `-hw-foo-wires` [¶](#-hw-foo-wires)

*Change all wires’ name to foo*.\_

Very basic pass that numbers all of the wires in a given module.
The wires’ names are then all converte to foo\_.

### `-hw-parameterize-constant-ports` [¶](#-hw-parameterize-constant-ports)

*Parametize constant ports on private modules*

This pass converts input ports on private modules into parameters when all
instances pass constant values (or parameter values) to those ports.

By converting constant ports to parameters, synthesis pipelines can
recognize these values as compile-time constants through local analysis
alone, without requiring inter-module analysis. This enables more precise
timing information and better optimization of each instance independently,
since the constant values are immediately visible at the module interface.

### `-hw-print-instance-graph` [¶](#-hw-print-instance-graph)

*Print a DOT graph of the module hierarchy.*

### `-hw-print-module-graph` [¶](#-hw-print-module-graph)

*Print a DOT graph of the HWModule’s within a top-level module.*

#### Options [¶](#options-59)

```
-verbose-edges : Print information on SSA edges (types, operand #, ...)
```

### `-hw-specialize` [¶](#-hw-specialize)

*Specializes instances of parametric hw.modules*

Any `hw.instance` operation instantiating a parametric `hw.module` will
trigger a specialization procedure which resolves all parametric types and
values within the module based on the set of provided parameters to the
`hw.instance` operation. This specialized module is created as a new
`hw.module` and the referring `hw.instance` operation is rewritten to
instantiate the newly specialized module.

### `-hw-verify-irn` [¶](#-hw-verify-irn)

*Verify InnerRefNamespaceLike operations, if not self-verifying.*

Kanagawa Dialect Passes [¶](#kanagawa-dialect-passes)
-----------------------------------------------------

### `-kanagawa-add-operator-library` [¶](#-kanagawa-add-operator-library)

*Injects the Kanagawa operator library into the IR*

Injects the Kanagawa operator library into the IR, which contains the
definitions of the Kanagawa operators.

### `-kanagawa-argify-blocks` [¶](#-kanagawa-argify-blocks)

*Add arguments to kanagawa blocks*

Analyses `kanagawa.sblock` operations and converts any SSA value defined outside
the `kanagawa.sblock` to a block argument. As a result, `kanagawa.sblock.isolated`
are produced.

### `-kanagawa-call-prep` [¶](#-kanagawa-call-prep)

*Convert kanagawa method calls to use `dc.value`*

### `-kanagawa-clean-selfdrivers` [¶](#-kanagawa-clean-selfdrivers)

*Kanagawa clean selfdrivers pass*

* Removes `kanagawa.port.input`s which are driven by operations within the same
  container.
* Removes reads of instance ports which are also written to within the same
  container.

### `-kanagawa-containerize` [¶](#-kanagawa-containerize)

*Kanagawa containerization pass*

Convert Kanagawa classes to containers, and outlines containers inside classes.

### `-kanagawa-convert-cf-to-handshake` [¶](#-kanagawa-convert-cf-to-handshake)

*Converts an `kanagawa.method` to `kanagawa.method.df`*

Converts an `kanagawa.method` from using `cf` operations and MLIR blocks to
an `kanagawa.method.df` operation, using the `handshake` dialect to represent
control flow through the `handshake` fine grained dataflow operations.

### `-kanagawa-convert-containers-to-hw` [¶](#-kanagawa-convert-containers-to-hw)

*Kanagawa containers to hw conversion pass*

Converts `kanagawa.container` ops to `hw.module` ops.

### `-kanagawa-convert-handshake-to-dc` [¶](#-kanagawa-convert-handshake-to-dc)

*Converts an `kanagawa.method.df` to use DC*

Converts an `kanagawa.method.df` from using `handshake` operations to
`dc` operations.

### `-kanagawa-convert-methods-to-containers` [¶](#-kanagawa-convert-methods-to-containers)

*Converts `kanagawa.method.df` to `kanagawa.container`s*

### `-kanagawa-eliminate-redundant-ops` [¶](#-kanagawa-eliminate-redundant-ops)

*Kanagawa eliminate redundant operations pass*

Eliminates redundant operations within Kanagawa containers to optimize the IR.
This pass analyzes operations within containers and removes unnecessary or
duplicate operations that do not affect the semantic behavior.

Redundant operations can (read: will) cause issues in other passes. So this
pass needs to be run after any pass which can introduce redundant
operations.

### `-kanagawa-inline-sblocks` [¶](#-kanagawa-inline-sblocks)

*Inlines `kanagawa.sblock` operations as MLIR blocks*

Inlines `kanagawa.sblock` operations, by creating MLIR blocks and `cf`
operations, while adding attributes to the parent operation about
`sblock`-specific attributes.

The parent attributes are located under the `kanagawa.blockinfo` identifier as
a dictionary attribute.
Each entry in the dictionary consists of:

* Key: an ID (numerical) string identifying the block.
* Value: a dictionary of attributes. As a minimum this will contain a
  `loc`-keyed attribute specifying the location of the block.

### `-kanagawa-lower-portrefs` [¶](#-kanagawa-lower-portrefs)

*Kanagawa portref lowering pass*

Lower `kanagawa.portref<portref <T>>` to T (i.e. portref resolution).

We do this by analyzing how a portref is used
inside a container, and then creating an in- or output port based on that.
That is:

* write to `portref<in portref<in, T>>` becomes `out T`
  i.e this container wants to write to an input of another container, hence
  it will produce an output value that will drive that input port.
* read from `portref<in portref<out, T>>` becomes `in T`
  i.e. this container wants to read from an output of another container,
  hence it will have an input port that will be driven by that output port.
* write to `portref<out portref<out, T>>` becomes `out T`
  i.e. a port reference inside the module will be driven by a value from
  the outside.
* read from `portref<out portref<in, T>>` becomes `in T`
  i.e. a port reference inside the module will be driven by a value from
  the outside.

A benefit of having portref lowering separate from portref tunneling is that
portref lowering can be done on an `kanagawa.container` granularity, allowing
for a bit of parallelism in the flow.

### `-kanagawa-prepare-scheduling` [¶](#-kanagawa-prepare-scheduling)

*Prepare `kanagawa.sblock` operations for scheduling*

Prepares `kanagawa.sblock` operations for scheduling by:

* creating an `kanagawa.pipleine.header` operation
* moving operations of an `kanagawa.sblock` into a `pipeline.unscheduled`
  operation, which is connected to the pipeline header.

### `-kanagawa-reblock` [¶](#-kanagawa-reblock)

*Recreates `kanagawa.sblock` operations from a CFG*

Recreates `kanagawa.sblock` operations from a CFG. Any `kanagawa.block.attributes`
operations at the parent operation will be added to the resulting blocks.

The IR is expected to be in maximal SSA form prior to this pass, given that
the pass will only inspect the terminator operation of a block for any
values that are generated within the block. Maximum SSA form thus ensures
that any value defined within the block is never used outside of the block.

It is expected that `kanagawa.call` operations have been isolated into
their own basic blocks before this pass is run. This implies that all
operations within a block (except for the terminator operation) can be
statically scheduled with each other.

e.g.

```
^bb_foo(%arg0 : i32, %arg1 : i32):
  %res = arith.addi %arg0, %arg1 : i32
  %v = ...
  cf.br ^bb_bar(%v : i32)
```

becomes

```
^bb_foo(%arg0 : i32, %arg1 : i32):
  %v_outer = kanagawa.sblock(%a0 : i32 = %arg0, %a1 : i32 = %arg1) -> (i32) {
    %res = arith.addi %arg0, %arg1 : i32
    %v = ...
    kanagawa.sblock.return %v : i32
  }
  cf.br ^bb_bar(%v_outer : i32)
```

### `-kanagawa-tunneling` [¶](#-kanagawa-tunneling)

*Kanagawa tunneling pass*

Tunnels relative `get_port` ops through the module hierarchy, based on
`kanagawa.path` ops. The result of this pass is that various new in- and output
ports of `!kanagawa.portref<...>` type are created.
After this pass, `get_port` ops should only exist at the same scope of
container instantiations.

The user may provide options for `readSuffix` and `writeSuffix`, respectively,
which is to be used to generate the name of the ports that are tunneled
through the hierarchy, with respect to how the port was requested to be accessed.
Suffixes must be provided in cases where a port is tunneled for both read and
write accesses, and the suffixes must be different (in this case, the suffixes
will be appended to the target port name, and thus de-alias the resulting ports).

#### Options [¶](#options-60)

```
-read-suffix  : Suffix to be used for the port when a port is tunneled for read access
-write-suffix : Suffix to be used for the port when a port is tunneled for write access
```

LLHD Dialect Passes [¶](#llhd-dialect-passes)
---------------------------------------------

### `-llhd-combine-drives` [¶](#-llhd-combine-drives)

*Combine scalar drives into aggregate drives*

If individual drives cover all of an aggregate signal’s fields, merge them
into a single drive of the whole aggregate value.

### `-llhd-deseq` [¶](#-llhd-deseq)

*Convert sequential processes to registers*

### `-llhd-hoist-signals` [¶](#-llhd-hoist-signals)

*Hoist probes and promote drives to process results*

### `-llhd-inline-calls` [¶](#-llhd-inline-calls)

*Inline all function calls in HW modules*

Inlines all `func.call` operations nested within `llhd.combinational`,
`llhd.process`, and `llhd.final` ops within `hw.module`s. The `func.func`
definitions are left untouched. After the inlining, MLIR’s symbol DCE pass
may be used to eliminate the function definitions where possible.

This pass expects the `llhd-wrap-procedural-ops` pass to have already run.
Otherwise call ops immediately in the module body, a graph region, cannot be
inlined at all since the graph region cannot accommodate the function’s
control flow.

#### Statistics [¶](#statistics-34)

```
calls-inlined : Number of call ops that were inlined
```

### `-llhd-lower-processes` [¶](#-llhd-lower-processes)

*Convert process ops to combinational ops where possible*

### `-llhd-mem2reg` [¶](#-llhd-mem2reg)

*Promotes memory and signal slots into values.*

### `-llhd-remove-control-flow` [¶](#-llhd-remove-control-flow)

*Remove acyclic control flow and replace block args with muxes*

Remove the control flow in `llhd.combinational` operations by merging all
blocks into the entry block and replacing block arguments with multiplexers.
This requires the control flow to be acyclic, for example by unrolling all
loops beforehand. Additionally, since this moves operations from
conditionally executed blocks into the unconditionally executed entry block,
all operations must be side-effect free.

### `-llhd-sig2reg` [¶](#-llhd-sig2reg)

*Promote LLHD signals to SSA values*

### `-llhd-unroll-loops` [¶](#-llhd-unroll-loops)

*Unroll control flow loops with static bounds*

Unroll loops in `llhd.combinational` operations by replicating the loop body
and replacing induction variables with constants. The loop bounds must be
known at compile time.

### `-llhd-wrap-procedural-ops` [¶](#-llhd-wrap-procedural-ops)

*Wrap procedural ops in modules to make them inlinable*

Operations such as `func.call` or `scf.if` may appear in an `hw.module` body
directly. They cannot be inlined into the module though, since the inlined
function or result of converting the SCF ops to the CF dialect may create
control-flow operations with multiple blocks. This pass wraps such
operations in `llhd.combinational` ops to give them an SSACFG region to
inline into.

#### Statistics [¶](#statistics-35)

```
ops-wrapped : Number of procedural ops wrapped
```

MSFT Dialect Passes [¶](#msft-dialect-passes)
---------------------------------------------

### `-msft-export-tcl` [¶](#-msft-export-tcl)

*Create tcl ops*

#### Options [¶](#options-61)

```
-tops     : List of top modules to export Tcl for
-tcl-file : File to output Tcl into
```

### `-msft-lower-constructs` [¶](#-msft-lower-constructs)

*Lower high-level constructs*

### `-msft-lower-instances` [¶](#-msft-lower-instances)

*Lower dynamic instances*

OM Dialect Passes [¶](#om-dialect-passes)
-----------------------------------------

### `-om-freeze-paths` [¶](#-om-freeze-paths)

*Hard code all path information*

Replaces paths to hardware with hard-coded string paths. This pass should
only run once the hierarchy will no longer change, and the final names for
objects have been decided.

### `-om-link-modules` [¶](#-om-link-modules)

*Link separated OM modules into a single module*

Flatten nested modules and resolve external classes.

### `-om-verify-object-fields` [¶](#-om-verify-object-fields)

*Verify fields of ObjectOp are valid*

Verify object fields are valid.

### `-strip-om` [¶](#-strip-om)

*Remove OM information*

Removes all OM operations from the IR. Useful as a prepass in pipelines that
have to discard OM.

Pipeline Dialect Passes [¶](#pipeline-dialect-passes)
-----------------------------------------------------

### `-pipeline-explicit-regs` [¶](#-pipeline-explicit-regs)

*Makes stage registers explicit.*

Makes all stage-crossing def-use chains into explicit registers.

### `-pipeline-schedule-linear` [¶](#-pipeline-schedule-linear)

*Schedules `pipeline.unscheduled` operations.*

Schedules `pipeline.unscheduled` operations based on operator latencies.

Seq Dialect Passes [¶](#seq-dialect-passes)
-------------------------------------------

### `-externalize-clock-gate` [¶](#-externalize-clock-gate)

*Convert seq.clock\_gate ops into hw.module.extern instances*

#### Options [¶](#options-62)

```
-name          : Name of the external clock gate module
-input         : Name of the clock input
-output        : Name of the gated clock output
-enable        : Name of the enable input
-test-enable   : Name of the optional test enable input
-instance-name : Name of the generated instances
```

#### Statistics [¶](#statistics-36)

```
num-clock-gates-converted : Number of clock gates converted to external module instances
```

### `-hw-memory-sim` [¶](#-hw-memory-sim)

*Implement FIRRTMMem memories nodes with simulation model*

This pass replaces generated module nodes of type FIRRTLMem with a model
suitable for simulation.

#### Options [¶](#options-63)

```
-disable-mem-randomization                                : Disable emission of memory randomization code
-disable-reg-randomization                                : Disable emission of register randomization code
-repl-seq-mem                                             : Prepare seq mems for macro replacement
-read-enable-mode                                         : specify the behaviour of the read enable signal
-add-mux-pragmas                                          : Add mux pragmas to memory reads
-add-vivado-ram-address-conflict-synthesis-bug-workaround : Add a vivado attribute to specify a ram style of an array register
```

### `-lower-seq-fifo` [¶](#-lower-seq-fifo)

*Lower seq.fifo ops*

### `-lower-seq-hlmem` [¶](#-lower-seq-hlmem)

*Lowers seq.hlmem operations.*

### `-lower-seq-shiftreg` [¶](#-lower-seq-shiftreg)

*Lower seq.shiftreg ops*

Default pass for lowering shift register operations. This will lower
into a chain of `seq.compreg.ce` operations.
Note that this is *not* guaranteed to be a performant implementation,
but instead a default, fallback lowering path which is guaranteed to
provide a semantically valid path to verilog emissions.
Users are expected to provide a custom lowering pass to maps `seq.shiftreg`
operations to target-specific primitives.

### `-seq-reg-of-vec-to-mem` [¶](#-seq-reg-of-vec-to-mem)

*Convert register arrays to FIRRTL memories*

This pass identifies register arrays that follow memory access patterns
and converts them to seq.firmem operations. It looks for patterns where:

1. A register array is updated via array\_inject operations
2. The array is read via array\_get operations
3. Updates are controlled by enable signals through mux operations
4. Read and write operations use the same clock

SSP Dialect Passes [¶](#ssp-dialect-passes)
-------------------------------------------

### `-ssp-print` [¶](#-ssp-print)

*Prints all SSP instances as DOT graphs.*

### `-ssp-roundtrip` [¶](#-ssp-roundtrip)

*Roundtrips all SSP instances via the scheduling infrastructure*

#### Options [¶](#options-64)

```
-check  : Check the problem's input constraints.
-verify : Verify the problem's solution constraints.
```

### `-ssp-schedule` [¶](#-ssp-schedule)

*Schedules all SSP instances.*

#### Options [¶](#options-65)

```
-scheduler : Scheduling algorithm to use.
-options   : Scheduler-specific options.
```

SV Dialect Passes [¶](#sv-dialect-passes)
-----------------------------------------

### `-hw-cleanup` [¶](#-hw-cleanup)

*Cleanup transformations for operations in hw.module bodies*

This pass merges sv.alwaysff operations with the same condition, sv.ifdef
nodes with the same condition, and perform other cleanups for the IR.
This is a good thing to run early in the HW/SV pass pipeline to expose
opportunities for other simpler passes (like canonicalize).

#### Options [¶](#options-66)

```
-merge-always-blocks : Allow always and always_ff blocks to be merged
```

### `-hw-eliminate-inout-ports` [¶](#-hw-eliminate-inout-ports)

*Raises usages of inout ports into explicit input and output ports*

This pass raises usages of inout ports into explicit in- and output ports.
This is done by analyzing the usage of the inout ports and creating new
input and output ports at the using module, and subsequently moving the
inout read- and writes to the instantiation site.

#### Options [¶](#options-67)

```
-read-suffix  : Suffix to be used for the port when the inout port has readers
-write-suffix : Suffix to be used for the port when the inout port has writers
```

### `-hw-export-module-hierarchy` [¶](#-hw-export-module-hierarchy)

*Export module and instance hierarchy information*

This pass exports the module and instance hierarchy tree for each module
with the firrtl.moduleHierarchyFile attribute. These are lowered to
sv.verbatim ops with the output\_file attribute.

### `-hw-generator-callout` [¶](#-hw-generator-callout)

*Lower Generator Schema to external module*

This pass calls an external program for all the hw.module.generated nodes,
following the description in the hw.generator.schema node.

#### Options [¶](#options-68)

```
-schema-name                    : Name of the schema to process
-generator-executable           : Generator program executable with optional full path
-generator-executable-arguments : Generator program arguments separated by ;
```

### `-hw-legalize-modules` [¶](#-hw-legalize-modules)

*Eliminate features marked unsupported in LoweringOptions*

This pass lowers away features in the SV/Comb/HW dialects that are
unsupported by some tools, e.g. multidimensional arrays. This pass is
run relatively late in the pipeline in preparation for emission. Any
passes run after this must be aware they cannot introduce new invalid
constructs.

### `-hw-stub-external-modules` [¶](#-hw-stub-external-modules)

*Transform external hw modules to empty hw modules*

This pass creates empty module bodies for external modules. This is
useful for linting to eliminate missing file errors.

### `-prettify-verilog` [¶](#-prettify-verilog)

*Transformations to improve quality of ExportVerilog output*

This pass contains elective transformations that improve the quality of
SystemVerilog generated by the ExportVerilog library. This pass is not
compulsory: things that are required for ExportVerilog to be correct
should be included as part of the ExportVerilog pass itself to make sure
it is self contained.

### `-sv-extract-test-code` [¶](#-sv-extract-test-code)

*Extract simulation only constructs to modules and bind*

This pass extracts cover, assume, assert operations to a module, along with
any ops feeding them only, to modules which are instantiated with a bind
statement.

#### Options [¶](#options-69)

```
-disable-instance-extraction : Disable extracting instances only that feed test code
-disable-register-extraction : Disable extracting registers only that feed test code
-disable-module-inlining     : Disable inlining modules that only feed test code
```

#### Statistics [¶](#statistics-37)

```
num-ops-extracted : Number of ops extracted
num-ops-erased    : Number of ops erased
```

### `-sv-trace-iverilog` [¶](#-sv-trace-iverilog)

*Add tracing to an iverilog simulated module*

This pass adds the necessary instrumentation to a HWModule to trigger
tracing in an iverilog simulation.

#### Options [¶](#options-70)

```
-top-only : If true, will only add tracing to the top-level module.
-module   : Module to trace. If not provided, will trace all modules
-dir-name : Directory to emit into
```

Synth Dialect Passes [¶](#synth-dialect-passes)
-----------------------------------------------

### `-synth-abc-runner` [¶](#-synth-abc-runner)

*Run ABC on AIGER files*

This pass runs ABC on AIGER files. It is a wrapper around AIGERRunner that
uses ABC as the external solver. It runs the following ABC commands:

* `read <inputFile>`: Read the AIGER file
* for each command in `abcCommands`, run `-q <command>`
* `write <outputFile>`: Write the AIGER file

#### Options [¶](#options-71)

```
-continue-on-failure : Don't fail even if the AIGER exporter, external solver, or AIGER importer fail
-abc-path            : Path to the ABC executable
-abc-commands        :
```

### `-synth-aiger-runner` [¶](#-synth-aiger-runner)

*Run external solver on AIGER files*

This pass runs an external solver on AIGER files. It exports the current
module to AIGER format, runs the external solver, and imports the result
back into the module.

#### Options [¶](#options-72)

```
-continue-on-failure : Don't fail even if the AIGER exporter, external solver, or AIGER importer fail
-aiger-path          : Path to the AIGER file
-solver-path         : Path to the external solver
-solver-args         :
```

### `-synth-generic-lut-mapper` [¶](#-synth-generic-lut-mapper)

*LUT mapping using generic K-input LUTs*

This pass performs technology mapping using generic K-input lookup tables
(LUTs). It converts combinational logic networks into implementations
using K-input LUTs (comb.truth\_table) with unit area cost and delay.

#### Options [¶](#options-73)

```
-max-cuts-per-root : Maximum number of cuts to maintain per node
-strategy          : Optimization strategy (area vs. timing)
-test              : Attach timing to IR for testing
-max-lut-size      : Maximum number of inputs per LUT
```

### `-synth-lower-variadic` [¶](#-synth-lower-variadic)

*Lower variadic operations to binary operations*

This pass lowers variadic operations to binary operations using a
delay-aware algorithm. For commutative operations, it builds a balanced
tree by combining values with the earliest arrival times first to minimize
the critical path.

#### Options [¶](#options-74)

```
-op-names     : Specify operation names to lower (empty means all)
-timing-aware : Lower operators with timing information
```

### `-synth-lower-word-to-bits` [¶](#-synth-lower-word-to-bits)

*Lower multi-bit AndInverter to single-bit ones*

#### Statistics [¶](#statistics-38)

```
num-lowered-bits      : Number of total bits lowered including constant
num-lowered-constants : Number of total constant bits lowered
num-lowered-ops       : Number of total operations lowered
```

### `-synth-maximum-and-cover` [¶](#-synth-maximum-and-cover)

*Maximum And Cover for And-Inverter*

This pass performs maximum AND-cover optimization by collapsing single-fanout
and-inverter ops into their users

### `-synth-print-longest-path-analysis` [¶](#-synth-print-longest-path-analysis)

*Print longest path analysis results with detailed timing statistics*

This pass performs longest path analysis on AIG circuits and outputs detailed
timing information including:

* Delay distribution statistics showing timing levels and path counts
* Critical path details for the top N end points
* Path history with intermediate debug points for detailed analysis

The analysis considers each AIG and-inverter operation to have unit delay and
computes maximum delays through combinational paths across module hierarchies.

#### Options [¶](#options-75)

```
-output-file        : Output file for analysis results (use '-' for stdout)
-test               : Emit longest paths as diagnostic remarks for testing
-show-top-k-percent : The size of the longest paths to show.
-emit-json          : Output analysis results in JSON format
-top-module-name    : Name of the top module to analyze (empty for automatic inference from instance graph)
```

### `-synth-sop-balancing` [¶](#-synth-sop-balancing)

*SOP (Sum-of-Products) balancing for delay optimization*

This pass optimizes delay by restructuring the logic network using balanced
sum-of-products expressions. It enumerates cuts, derives their SOP
representation, and selects the best cut based on delay and area.

The algorithm is based on “Delay Optimization Using SOP Balancing” by
Mishchenko et al. (ICCAD 2011).

#### Options [¶](#options-76)

```
-max-cuts-per-root  : Maximum number of cuts to maintain per node
-strategy           : Optimization strategy (area vs. timing)
-test               : Attach timing to IR for testing
-max-cut-input-size : Maximum number of inputs per cut
```

### `-synth-structural-hash` [¶](#-synth-structural-hash)

*Structural hashing (CSE) for Synth operations*

This pass performs aggressive structural hashing-based CSE for Synth dialect
operations (AIG/MIG), domain-specific to AIG/MIG operations to enable operand
reordering based on structural properties and inversion flag consideration
for canonicalization.

### `-synth-tech-mapper` [¶](#-synth-tech-mapper)

*Technology mapping using cut rewriting*

This pass performs technology mapping by converting logic network
(AIG etc) representations into technology-specific gate implementations.
It uses cut-based rewriting with priority cuts and NPN canonical forms for
efficient pattern matching.

The pass serves dual purposes: providing practical technology mapping
capabilities and acting as a test vehicle for the cut rewriting framework,
since testing cut enumeration and pattern matching algorithms directly
would otherwise be difficult without a concrete application.

Supports both area and timing optimization strategies.

#### Options [¶](#options-77)

```
-max-cuts-per-root : Maximum number of cuts to maintain per node
-strategy          : Optimization strategy (area vs. timing)
-test              : Attach timing to IR for testing
```

### `-synth-test-priority-cuts` [¶](#-synth-test-priority-cuts)

*Test priority cuts for synthesis*

#### Options [¶](#options-78)

```
-max-cuts-per-root  : Maximum number of cuts to maintain per node
-max-cut-input-size : Maximum number of cut inputs to consider
```

SystemC Dialect Passes [¶](#systemc-dialect-passes)
---------------------------------------------------

### `-systemc-lower-instance-interop` [¶](#-systemc-lower-instance-interop)

*Lower all SystemC instance interop operations.*

LEC (logical equivalence checking) Passes [¶](#lec-logical-equivalence-checking-passes)
---------------------------------------------------------------------------------------

### `-construct-lec` [¶](#-construct-lec)

*Lower CIRCTs core dialects to a LEC problem statement*

Takes two `hw.module` operations and lowers them to a `verif.lec` operation
inside a `func.func` matching the name of the first module.

This pass can also (optionally) insert a ‘main’ function that calls the
function with the `verif.lec` and adheres to the typical main function
signature. This is useful when compiling to a standalone binary rather than
using the JIT compiler.

This pass also inserts a few LLVM dialect operation to print the result of
the `verif.lec` using the `printf` function of the standard library.

#### Options [¶](#options-79)

```
-first-module  : Name of the first of the two modules to compare.
-second-module : Name of the second of the two modules to compare.
-insert-mode   : Select what additional code to add.
```

BMC (bounded model checking) Passes [¶](#bmc-bounded-model-checking-passes)
---------------------------------------------------------------------------

### `-externalize-registers` [¶](#-externalize-registers)

*Removes registers and adds corresponding input and output ports*

### `-lower-to-bmc` [¶](#-lower-to-bmc)

*Lower CIRCTs core dialects to a BMC problem statement*

#### Options [¶](#options-80)

```
-top-module           : Name of the top module to verify.
-bound                : Cycle bound.
-ignore-asserts-until : Specifies an initial window of cycles where assertions should be ignored (starting from 0).
-insert-main          : Whether a main function should be inserted for AOT compilation.
-rising-clocks-only   : Only consider the circuit and property on rising clock edges.
```

 [Prev - HLS in CIRCT](https://circt.llvm.org/docs/HLS/ "HLS in CIRCT")
[Next - Python CIRCT Design Entry (PyCDE)](https://circt.llvm.org/docs/PyCDE/ "Python CIRCT Design Entry (PyCDE)") 

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
  + [Python CIRCT Design Entry (PyCDE)+](https://circt.llvm.org/docs/PyCDE/)
    - [Compiling CIRCT and PyCDE](https://circt.llvm.org/docs/PyCDE/compiling/)
    - [PyCDE Basics](https://circt.llvm.org/docs/PyCDE/basics/)
  + [Static scheduling infrastructure](https://circt.llvm.org/docs/Scheduling/)
  + [Symbol and Inner Symbol Rationale](https://circt.llvm.org/docs/RationaleSymbols/)
  + [Using the Python Bindings](https://circt.llvm.org/docs/PythonBindings/)
  + [Verilog and SystemVerilog Generation](https://circt.llvm.org/docs/VerilogGeneration/)