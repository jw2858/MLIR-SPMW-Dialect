Verif Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

Verif Dialect
=============

This dialect provides a collection of operations to express various verification concerns, such as assertions and interacting with a piece of hardware for the sake of verifying its proper functioning.

* [Contracts](#contracts)
  + [Multiply-by-9 Example](#multiply-by-9-example)
  + [Carry-Save Compressor Example](#carry-save-compressor-example)
  + [Carry-Save Adder Example](#carry-save-adder-example)
  + [Multiplexer-based Shifter](#multiplexer-based-shifter)
* [Operations](#operations)
  + [`verif.assert` (circt::verif::AssertOp)](#verifassert-circtverifassertop)
  + [`verif.assume` (circt::verif::AssumeOp)](#verifassume-circtverifassumeop)
  + [`verif.bmc` (circt::verif::BoundedModelCheckingOp)](#verifbmc-circtverifboundedmodelcheckingop)
  + [`verif.clocked_assert` (circt::verif::ClockedAssertOp)](#verifclocked_assert-circtverifclockedassertop)
  + [`verif.clocked_assume` (circt::verif::ClockedAssumeOp)](#verifclocked_assume-circtverifclockedassumeop)
  + [`verif.clocked_cover` (circt::verif::ClockedCoverOp)](#verifclocked_cover-circtverifclockedcoverop)
  + [`verif.contract` (circt::verif::ContractOp)](#verifcontract-circtverifcontractop)
  + [`verif.cover` (circt::verif::CoverOp)](#verifcover-circtverifcoverop)
  + [`verif.ensure` (circt::verif::EnsureOp)](#verifensure-circtverifensureop)
  + [`verif.formal` (circt::verif::FormalOp)](#verifformal-circtverifformalop)
  + [`verif.format_verilog_string` (circt::verif::FormatVerilogStringOp)](#verifformat_verilog_string-circtverifformatverilogstringop)
  + [`verif.has_been_reset` (circt::verif::HasBeenResetOp)](#verifhas_been_reset-circtverifhasbeenresetop)
  + [`verif.lec` (circt::verif::LogicEquivalenceCheckingOp)](#veriflec-circtveriflogicequivalencecheckingop)
  + [`verif.print` (circt::verif::PrintOp)](#verifprint-circtverifprintop)
  + [`verif.refines` (circt::verif::RefinementCheckingOp)](#verifrefines-circtverifrefinementcheckingop)
  + [`verif.require` (circt::verif::RequireOp)](#verifrequire-circtverifrequireop)
  + [`verif.simulation` (circt::verif::SimulationOp)](#verifsimulation-circtverifsimulationop)
  + [`verif.symbolic_value` (circt::verif::SymbolicValueOp)](#verifsymbolic_value-circtverifsymbolicvalueop)
  + [`verif.yield` (circt::verif::YieldOp)](#verifyield-circtverifyieldop)

Contracts [¶](#contracts)
-------------------------

Formal contracts are a key building block of the Verif dialect to help make formal verification scale to larger designs and deep module hierarchies.
Contracts describe what a circuit expects from its inputs (`verif.require`) and what it guarantees its output to be (`verif.ensure`).
The `verif.contract` op can be inserted into an SSA edge similar to an `hw.wire`, where the contract simply passes its operands through to its results.
During formal verification the contract results are replaced by symbolic values that uphold the guarantees described in the contract’s body.
Conceptually, contracts are similar to Hoare triples used in software verification, where a piece of code is verified to produce a given set of postconditions when the given set of preconditions is met.

These contracts can then be used in two key ways:

1. A contract can be *checked* by turning `require`s into `assume`s and `ensure`s into `assert`s.
   Doing so verifies that a circuit upholds the contract by placing asserts on the values it produces, and by placing assumes on the input values the circuit sees.
   In a nutshell, this checks that, assuming the inputs to the circuit honor the contract, the output from the circuit also upholds the contract.
   This check can be done very efficiently by creating `verif.formal` ops to verify each contract.
2. A contract can be *applied* by turning `require`s into `assert`s and `ensure`s into `assume`s.
   Doing so verifies that the inputs fed into a circuit uphold the contract, such that the outputs can be assumed to have the promised values.
   In a nutshell, this checks that the inputs to a circuit honor the contract and therefore the circuit can be assumed to uphold the contract.
   Assuming a contract can often eliminate large parts of the circuit’s actual implementation, since the contracts tend to be a simpler description of a circuit’s functionality.

### Multiply-by-9 Example [¶](#multiply-by-9-example)

Consider the following example of a HW module that computes `9 * a` using a left-shift and an addition.
Note that this is using `verif.ensure_equal` as a standin for `verif.ensure(comb.icmp eq)`.
The module’s output `%z` is the result of a `verif.contract` operation.
Contracts can be completely ignored by simply passing through their operands, `%1` in this case, to their results.
This is their normal interpretation outside a formal verification flow, for example for synthesis.

```
hw.module @Mul9(in %a: i42, out z: i42) {
  // Compute 9*a as (a<<3)+a.
  %c3_i42 = hw.constant 3 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = comb.add %a, %0 : i42

  // Contract to check that the circuit actually produces 9*a.
  %z = verif.contract %1 : i42 {
    %c9_i42 = hw.constant 9 : i42
    %a9 = comb.mul %a, %c9_i42 : i42
    verif.ensure_equal %z, %a9
  }

  hw.output %z : i42
}
```

To check that the contract holds, it can be pulled out into a `verif.formal` op along with all ops in its fan-in cone.
The contract’s results are then replaced with its operands, the body is inlined, and `ensure` ops are replaced with `assert` and `require` ops are replaced with `assume`.
Running this formal test will check that the `(a<<3)+a` implemented by the module is indeed the same as the `9*a` promised by the contract.

```
verif.formal @Mul9_CheckContract {
  %a = verif.symbolic_value : i42

  %c3_i42 = hw.constant 3 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = comb.add %a, %0 : i42

  // Contract inlined with ensure -> assert, %z -> %1
  %c9_i42 = hw.constant 9 : i42
  %a9 = comb.mul %a, %c9_i42 : i42
  verif.assert_equal %1, %a9 : i42
}
```

Once the contract is checked, it can be assumed to hold everywhere.
Assuming it holds can be done by inlining it into its parent block and replacing its results with symbolic values.
At the same time, `ensure`s are replaced with `assume`s and `require`s with `assert`s.
Inlining the contract in this example and replacing the symbolic value with what it is assumed to be equal to will make the module produce the simple `9*a` term.
This means that the ops describing the original implementation become obsolete and will be DCEd.
Making bits of a module’s implementation unused is *the* key characteristic of contracts that makes them help formal verification scale.
If all of a module’s behavior can be described by one or more simpler contracts, its entire original implementation would simply disappear in favor of the simpler contracts.

```
hw.module @Mul9_ApplyContract(in %a: i42, out z: i42) {
  // The original implementation has become unused since the contract results
  // have been replaced with symbolic values. Dead code elimination will clean
  // this up.
  %c3_i42 = hw.constant 3 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = comb.add %a, %0 : i42

  // Results of the contract are replaced with symbolic values.
  // Contract inlined with ensure -> assume, %z -> %any
  // assume(eq(any, x)) can be canonicalized to an any -> x replacement.
  %any = verif.symbolic_value : i42
  %c9_i42 = hw.constant 9 : i42
  %a9 = comb.mul %a, %c9_i42 : i42
  verif.assume_equal %any, %a9 : i42

  hw.output %any : i42
}

// ----- after canonicalization, CSE, and DCE ----- //

hw.module @Mul9_ApplyContract_Simplified(in %a: i42, out z: i42) {
  %c9_i42 = hw.constant 9 : i42
  %a9 = comb.mul %a, %c9_i42 : i42
  hw.output %a9 : i42
}
```

These two constructs can coexist in the IR.
A contract can be turned into a `verif.formal` proof that it holds, and inlined everywhere else to leverage the fact that the contract holds.

### Carry-Save Compressor Example [¶](#carry-save-compressor-example)

Consider the following slightly more involved example of a compression stage you would find in a carry-save adder.
This module takes 3 input values and produces 2 output values that sum up to the same value as its inputs.
However, the contract in the module does not specify which *exact* output values the module produces.
Instead, it uses two symbolic values to express that the module can produce *any* output values that sum up to the inputs.
In a sense, how exactly the compressor combines 3 values into 2 is left as an implementation detail that you can’t know about, but the guarantee you can work with is that the sum will be correct.
This is different from the previous example, where the contract produced an exact replacement value for the module output.
Also, note how this uses an `assume` to constrain the sum of the symbolic values instead of an `ensure` or `require`.

```
// A module that takes 3 input values and produces 2 output values that sum up
// to the same value as the inputs. Instead of just using add it uses a
// bit-parallel full adder that takes each 3-tuple of bits in the 3 inputs, runs
// them through a full adder, and treats the resulting sum and carry as the 2
// corresponding bits for its 2 output values.
hw.module @CarrySaveCompress3to2(
  in %a0: i42, in %a1: i42, in %a2: i42,
  out z0: i42, out z1: i42
) {
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.xor %a0, %a1, %a2 : i42  // sum bits of FA (a0^a1^a2)
  %1 = comb.and %a0, %a1 : i42
  %2 = comb.or %a0, %a1 : i42
  %3 = comb.and %2, %a2 : i42
  %4 = comb.or %1, %3 : i42          // carry bits of FA (a0&a1 | a2&(a0|a1))
  %5 = comb.shl %4, %c1_i42 : i42    // %5 = carry << 1
  // At this point, %0+%5 is the same as %a0+%a1+%a2, but without creating a
  // long ripple-carry chain.

  // Contract to check that we output _some_ two numbers that sum up to the same
  // value as the sum of the three inputs. We don't say which exact numbers.
  %z0, %z1 = verif.contract %0, %5 {
    // The contract promises that its outputs will sum up to the same value as
    // the sum of the module inputs.
    %inputSum = comb.add %a0, %a1, %a2 : i42
    %outputSum = comb.add %z0, %z1 : i42
    verif.ensure_equal %inputSum, %outputSum : i42
  }

  hw.output %z0, %z1 : i42, i42
}
```

The contract can be checked by extracting it into a new `verif.formal` op alongside its entire fan-in cone.
Again we pretend that the contract doesn’t exist by passing its operands `%0` and `%5`, the actual implementation, through to its results `%z0` and `%z1`.
Replacing ensures with asserts then verifies that the module’s outputs do indeed sum up to the same value as the inputs.

```
verif.formal @CarrySaveCompress3to2_CheckContract {
  %a0 = verif.symbolic_value : i42
  %a1 = verif.symbolic_value : i42
  %a2 = verif.symbolic_value : i42

  %c1_i42 = hw.constant 1 : i42
  %0 = comb.xor %a0, %a1, %a2 : i42
  %1 = comb.and %a0, %a1 : i42
  %2 = comb.or %a0, %a1 : i42
  %3 = comb.and %2, %a2 : i42
  %4 = comb.or %1, %3 : i42
  %5 = comb.shl %4, %c1_i42 : i42

  // Contract inlined with ensure -> assert, (%z0, %z1) -> (%0, %5).
  %inputSum = comb.add %a0, %a1, %a2 : i42
  %outputSum = comb.add %0, %5 : i42
  verif.assert_equal %inputSum, %outputSum : i42
}
```

With the contract checked it can be assumed to hold by inlining it everywhere and replacing its results `%z0` and `%z1` with symbolic values.
This provides the symbolic values as a replacement for the actual implementation of the module, which causes the entire original implementation to be DCEd.

```
hw.module @CarrySaveCompress3to2_ApplyContract(
  in %a0: i42, in %a1: i42, in %a2: i42,
  out z0: i42, out z1: i42
) {
  // The original implementation has become unused since the contract results
  // have been replaced with symbolic values. Dead code elimination will clean
  // this up.
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.xor %a0, %a1, %a2 : i42
  %1 = comb.and %a0, %a1 : i42
  %2 = comb.or %a0, %a1 : i42
  %3 = comb.and %2, %a2 : i42
  %4 = comb.or %1, %3 : i42
  %5 = comb.shl %4, %c1_i42 : i42

  // Results of the contract are replaced with symbolic values.
  // Contract inlined with ensure -> assume, (%z0, %z1) -> (%any0, %any1).
  %any0 = verif.symbolic_value : i42
  %any1 = verif.symbolic_value : i42
  %inputSum = comb.add %a0, %a1, %a2 : i42
  %outputSum = comb.add %any0, %any1 : i42
  verif.assume_equal %inputSum, %outputSum : i42

  hw.output %any0, %any1 : i42, i42
}

// ----- after canonicalization, CSE, and DCE ----- //

hw.module @CarrySaveCompress3to2_ApplyContract_Simplified(
  in %a0: i42, in %a1: i42, in %a2: i42,
  out z0: i42, out z1: i42
) {
  %any0 = verif.symbolic_value : i42
  %any1 = verif.symbolic_value : i42
  %inputSum = comb.add %a0, %a1, %a2 : i42
  %outputSum = comb.add %any0, %any1 : i42
  verif.assume_equal %inputSum, %outputSum : i42
  hw.output %any0, %any1 : i42, i42
}
```

### Carry-Save Adder Example [¶](#carry-save-adder-example)

Consider the following carry-save adder built based on the compressor from the previous example.
It takes 5 input values and sums them up.
To do so, it uses multiple instances of the compressor to compress three of the input terms down to two iteratively, until only two terms are left.
The remaining two terms are then summed up with a plain old adder to get the final result.
This carry-save adder module has its own little contract which promises that the output is going to be the sum of all input terms.

```
// A module that takes 5 input values and sums them up using a carry save adder.
hw.module @CarrySaveAdder5(
  in %a0: i42, in %a1: i42, in %a2: i42, in %a3: i42, in %a4: i42,
  out z: i42
) {
  // Each stage takes 3 of the terms and compresses them to 2.
  // terms: [a0, a1, a2, a3, a4]
  %b0, %b1 = hw.instance "comp0" @CarrySaveCompress3to2(a0: %a0: i42, a1: %a1: i42, a2: %a2: i42) -> (z0: i42, z1: i42)
  // terms: [b0, b1, a3, a4]
  %c0, %c1 = hw.instance "comp1" @CarrySaveCompress3to2(a0: %b1: i42, a1: %a3: i42, a2: %a4: i42) -> (z0: i42, z1: i42)
  // terms: [b0, c0, c1]
  %d0, %d1 = hw.instance "comp2" @CarrySaveCompress3to2(a0: %b0: i42, a1: %c0: i42, a2: %c1: i42) -> (z0: i42, z1: i42)
  // terms: [d0, d1]
  %e = comb.add %d0, %d1 : i42
  // terms: [e]

  // Contract to check that the output is the sum of all inputs.
  %z = verif.contract %e {
    %inputSum = comb.add %a0, %a1, %a2, %a3, %a4, %a5 : i42
    verif.ensure_equal %z, %inputSum : i42
  }

  hw.output %z : i42
}
```

Checking the contract looks as follows.
Note how the formal proof can already assume that contracts inside the compressor submodule hold.
This is a neat example of recursive use of contracts, and how checking contracts in parent modules can benefit from contracts in child modules.
Instead of having to include the compressor module implementation in this test again, potentially complicating the proof, we can already use the simplified version described by the compressor’s contract.
This turns the compressor instances basically into a few additions among symbolic values, which formal solvers are very good at working with.

```
verif.formal @CarrySaveAdder5_CheckContract {
  %a0 = verif.symbolic_value : i42
  %a1 = verif.symbolic_value : i42
  %a2 = verif.symbolic_value : i42
  %a3 = verif.symbolic_value : i42
  %a4 = verif.symbolic_value : i42

  // The following instances are just two symbolic values each, constrained to
  // sum up to the instance inputs. This makes for a more trivial solve.
  %b0, %b1 = hw.instance "comp0" @CarrySaveCompress3to2_ApplyContract(a0: %a0: i42, a1: %a1: i42, a2: %a2: i42) -> (z0: i42, z1: i42)
  %c0, %c1 = hw.instance "comp1" @CarrySaveCompress3to2_ApplyContract(a0: %b1: i42, a1: %a3: i42, a2: %a4: i42) -> (z0: i42, z1: i42)
  %d0, %d1 = hw.instance "comp2" @CarrySaveCompress3to2_ApplyContract(a0: %b0: i42, a1: %c0: i42, a2: %c1: i42) -> (z0: i42, z1: i42)
  %e = comb.add %d0, %d1 : i42

  // Contract inlined with ensure -> assert, %z -> %e.
  %inputSum = comb.add %a0, %a1, %a2, %a3, %a4, %a5 : i42
  verif.assert_equal %e, %inputSum : i42
}
```

The contract can then be assumed to hold by inlining it into the carry-save adder as follows.

```
hw.module @CarrySaveAdder5_ApplyContract(
  in %a0: i42, in %a1: i42, in %a2: i42, in %a3: i42, in %a4: i42,
  out z: i42
) {
  // The original implementation has become unused since the contract results
  // have been replaced with symbolic values. Dead code elimination will clean
  // this up.
  %b0, %b1 = hw.instance "comp0" @CarrySaveCompress3to2_ApplyContract(a0: %a0: i42, a1: %a1: i42, a2: %a2: i42) -> (z0: i42, z1: i42)
  %c0, %c1 = hw.instance "comp1" @CarrySaveCompress3to2_ApplyContract(a0: %b1: i42, a1: %a3: i42, a2: %a4: i42) -> (z0: i42, z1: i42)
  %d0, %d1 = hw.instance "comp2" @CarrySaveCompress3to2_ApplyContract(a0: %b0: i42, a1: %c0: i42, a2: %c1: i42) -> (z0: i42, z1: i42)
  %e = comb.add %d0, %d1 : i42

  // Results of the contract are replaced with symbolic values.
  // Contract inlined with ensure -> assume, %z -> %any.
  %any = verif.symbolic_value : i42
  %inputSum = comb.add %a0, %a1, %a2, %a3, %a4, %a5 : i42
  // assume(eq(any, x)) can be canonicalized to an any -> x replacement.
  verif.assume_equal %any, %inputSum : i42

  hw.output %any : i42
}

// ----- after canonicalization, CSE, and DCE ----- //

hw.module @CarrySaveAdder5_ApplyContract_Simplified(
  in %a0: i42, in %a1: i42, in %a2: i42, in %a3: i42, in %a4: i42,
  out z: i42
) {
  %inputSum = comb.add %a0, %a1, %a2, %a3, %a4, %a5 : i42
  hw.output %inputSum : i42
}
```

### Multiplexer-based Shifter [¶](#multiplexer-based-shifter)

Consider the following module that left-shifts a value.
It uses a multiplexer tree to perform the shift, which cannot shift out the value completely.
Therefore, a `require` is placed in its contract to force the users of the shifter to never provide shift amounts outside the valid range.

```
hw.module @ShiftLeft(in %a: i8, in %b: i8, out z: i8) {
  %c4_i8 = hw.constant 4 : i8
  %c2_i8 = hw.constant 2 : i8
  %c1_i8 = hw.constant 1 : i8
  %b2 = comb.extract %b, 2 : i8 -> i1
  %b1 = comb.extract %b, 1 : i8 -> i1
  %b0 = comb.extract %b, 0 : i8 -> i1
  %0 = comb.shl %a, %c4_i8 : i8
  %1 = comb.mux %b2, %0, %a : i8
  %2 = comb.shl %1, %c2_i8 : i8
  %3 = comb.mux %b1, %2, %1 : i8
  %4 = comb.shl %3, %c1_i8 : i8
  %5 = comb.mux %b0, %4, %3 : i8

  // Contract to check that the multiplexers and constant shifts above indeed
  // produce the correct shift by 0 to 7 places, assuming the shift amount is
  // less than 8 (we can't shift a number out).
  %z = verif.contract %5 {
    // Shift amount must be less than 8.
    %c8_i8 = hw.constant 8 : i8
    %blt8 = comb.icmp ult %b, %c8_i8 : i8
    verif.require %blt8

    // In that case the mux tree computes the correct left-shift.
    %ashl = comb.shl %a, %b : i8
    verif.ensure_equal %z, %ashl : i42
  }

  hw.output %z : i8
}
```

The contract in the shifter can be checked as follows.
Note how the `require` is replaced by an `assume` in addition to the `ensure` being replaced by an `assert`.

```
verif.formal @ShiftLeft_CheckContract {
  %a = verif.symbolic_value : i8
  %b = verif.symbolic_value : i8

  %c4_i8 = hw.constant 4 : i8
  %c2_i8 = hw.constant 2 : i8
  %c1_i8 = hw.constant 1 : i8
  %b2 = comb.extract %b, 2 : i8 -> i1
  %b1 = comb.extract %b, 1 : i8 -> i1
  %b0 = comb.extract %b, 0 : i8 -> i1
  %0 = comb.shl %a, %c4_i8 : i8
  %1 = comb.mux %b2, %0, %a : i8
  %2 = comb.shl %1, %c2_i8 : i8
  %3 = comb.mux %b1, %2, %1 : i8
  %4 = comb.shl %3, %c1_i8 : i8
  %5 = comb.mux %b0, %4, %3 : i8

  // Contract inlined with ensure -> assert, require -> assume, %z -> %5.
  %c8_i8 = hw.constant 8 : i8
  %blt8 = comb.icmp ult %b, %c8_i8 : i8
  verif.assume %blt8
  %ashl = comb.shl %a, %b : i8
  verif.assert_equal %5, %ashl : i42
}
```

Once checked, the contract can be assumed to hold by inlining it into the shift-left module as follows.
Note how the value of input `b` is now asserted to be less than 8.
This causes the instantiation sites of this module to be checked to provide values for `b` that are less than 8, thus upholding the contract.
At the same time, it allows those sites to use the simplified `comb.shl %a, %b` implementation described in the contract.

```
hw.module @ShiftLeft_ApplyContract(in %a: i8, in %b: i8, out z: i8) {
  // The original implementation has become unused since the contract results
  // have been replaced with symbolic values. Dead code elimination will clean
  // this up.
  %c4_i8 = hw.constant 4 : i8
  %c2_i8 = hw.constant 2 : i8
  %c1_i8 = hw.constant 1 : i8
  %b2 = comb.extract %b, 2 : i8 -> i1
  %b1 = comb.extract %b, 1 : i8 -> i1
  %b0 = comb.extract %b, 0 : i8 -> i1
  %0 = comb.shl %a, %c4_i8 : i8
  %1 = comb.mux %b2, %0, %a : i8
  %2 = comb.shl %1, %c2_i8 : i8
  %3 = comb.mux %b1, %2, %1 : i8
  %4 = comb.shl %3, %c1_i8 : i8
  %5 = comb.mux %b0, %4, %3 : i8

  // Results of the contract are replaced with symbolic values.
  // Contract inlined with ensure -> assume, require -> assert, %z -> %any
  // assume(eq(any, x)) can be canonicalized to an any -> x replacement.
  %any = verif.symbolic_value : i8
  %c8_i8 = hw.constant 8 : i8
  %blt8 = comb.icmp ult %b, %c8_i8 : i8
  verif.assert %blt8
  %ashl = comb.shl %a, %b : i8
  verif.assume_equal %any, %ashl : i42

  hw.output %any : i8
}

// ----- after canonicalization, CSE, and DCE ----- //

hw.module @ShiftLeft_ApplyContract_Simplified(in %a: i8, in %b: i8, out z: i8) {
  %c8_i8 = hw.constant 8 : i8
  %blt8 = comb.icmp ult %b, %c8_i8 : i8
  verif.assert %blt8
  %ashl = comb.shl %a, %b : i8
  hw.output %ashl : i8
}
```

Operations [¶](#operations)
---------------------------

### `verif.assert` (circt::verif::AssertOp) [¶](#verifassert-circtverifassertop)

*Assert that a property holds.*

Syntax:

```
operation ::= `verif.assert` $property (`if` $enable^)? (`label` $label^)? attr-dict `:` type($property)
```

#### Attributes: [¶](#attributes)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands)

| Operand | Description |
| --- | --- |
| `property` | 1-bit signless integer or LTL sequence type or LTL property type |
| `enable` | 1-bit signless integer |

### `verif.assume` (circt::verif::AssumeOp) [¶](#verifassume-circtverifassumeop)

*Assume that a property holds.*

Syntax:

```
operation ::= `verif.assume` $property (`if` $enable^)? (`label` $label^)? attr-dict `:` type($property)
```

#### Attributes: [¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-1)

| Operand | Description |
| --- | --- |
| `property` | 1-bit signless integer or LTL sequence type or LTL property type |
| `enable` | 1-bit signless integer |

### `verif.bmc` (circt::verif::BoundedModelCheckingOp) [¶](#verifbmc-circtverifboundedmodelcheckingop)

*Perform a bounded model check*

Syntax:

```
operation ::= `verif.bmc` `bound` $bound `num_regs` $num_regs `initial_values` $initial_values attr-dict-with-keyword `init` $init `loop` $loop `circuit`
              $circuit
```

This operation represents a bounded model checking problem explicitly in
the IR. The `bound` attribute indicates how many times the `circuit` region
should be executed, and `num_regs` indicates the number of registers in the
design that have been externalized and appended to the region’s
inputs/outputs (these values are fed from each `circuit` region execution
to the next, as they represent register state, rather than being
overwritten with fresh variables like other inputs). `initial_values` is an
array containing the initial value of each register - where the register
has no initial value, a unit attribute is given. The `circuit` region
contains the circuit (alongside the `verif` property checking operations)
to be checked.

The `init` region contains the logic to initialize the clock signals, and
will be executed once before any other region - it cannot take any
arguments, and should return as many `!seq.clock` values as the `circuit`
region has `!seq.clock` arguments, followed by any initial arguments of
‘state’ arguments to be fed to the `loop` region (see below).

The `loop` region contains the logic to advance the clock signals, and will
be executed after each execution of the `circuit` region. It should take as
arguments as many `!seq.clock` values as the `circuit` region has, and
these can be followed by additional ‘state’ arguments to represent e.g.
which clock should be toggled next. The types yielded should be the same,
as this region yields the updated clock and state values (this should also
match the types yielded by the `init` region).

Traits: `IsolatedFromAbove`, `SingleBlockImplicitTerminator<verif::YieldOp>`, `SingleBlock`

Interfaces: `InferTypeOpInterface`

#### Attributes: [¶](#attributes-2)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `bound` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `num_regs` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `initial_values` | ::mlir::ArrayAttr | array attribute |

#### Results: [¶](#results)

| Result | Description |
| --- | --- |
| `result` | 1-bit signless integer |

### `verif.clocked_assert` (circt::verif::ClockedAssertOp) [¶](#verifclocked_assert-circtverifclockedassertop)

*Assert that a property holds, checked on a given clock’s
ticks and enabled if a given condition holds. Only supports
a single clock and a single disable.*

Syntax:

```
operation ::= `verif.clocked_assert` $property (`if` $enable^)? `,` $edge $clock
              (`label` $label^)? attr-dict `:` type($property)
```

#### Attributes: [¶](#attributes-3)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `edge` | circt::verif::ClockEdgeAttr | clock edge |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-2)

| Operand | Description |
| --- | --- |
| `property` | 1-bit signless integer or LTL sequence type or LTL property type |
| `clock` | 1-bit signless integer |
| `enable` | 1-bit signless integer |

### `verif.clocked_assume` (circt::verif::ClockedAssumeOp) [¶](#verifclocked_assume-circtverifclockedassumeop)

*Assume that a property holds, checked on a given clock’s
ticks and enabled if a given condition holds. Only supports
a single clock and a single disable.*

Syntax:

```
operation ::= `verif.clocked_assume` $property (`if` $enable^)? `,` $edge $clock
              (`label` $label^)? attr-dict `:` type($property)
```

#### Attributes: [¶](#attributes-4)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `edge` | circt::verif::ClockEdgeAttr | clock edge |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-3)

| Operand | Description |
| --- | --- |
| `property` | 1-bit signless integer or LTL sequence type or LTL property type |
| `clock` | 1-bit signless integer |
| `enable` | 1-bit signless integer |

### `verif.clocked_cover` (circt::verif::ClockedCoverOp) [¶](#verifclocked_cover-circtverifclockedcoverop)

*Cover on the holding of a property, checked on a given clock’s
ticks and enabled if a given condition holds. Only supports
a single clock and a single disable.*

Syntax:

```
operation ::= `verif.clocked_cover` $property (`if` $enable^)? `,` $edge $clock
              (`label` $label^)? attr-dict `:` type($property)
```

#### Attributes: [¶](#attributes-5)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `edge` | circt::verif::ClockEdgeAttr | clock edge |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-4)

| Operand | Description |
| --- | --- |
| `property` | 1-bit signless integer or LTL sequence type or LTL property type |
| `clock` | 1-bit signless integer |
| `enable` | 1-bit signless integer |

### `verif.contract` (circt::verif::ContractOp) [¶](#verifcontract-circtverifcontractop)

*A formal contract*

Syntax:

```
operation ::= `verif.contract` ($inputs^ `:` type($inputs))? attr-dict-with-keyword $body
```

This operation creates a new formal contract which can be used to locally
verify a part of the IR and provide simplifying substitutions. Contracts
contain `verif.require` ops to establish conditions that must hold for a
piece of IR to work properly, and `verif.ensure` ops to describe the
properties that the piece of IR must guarantees when the requirements hold.
Outside of formal verification, operands are simply passed through to the
results.

Contracts are checked by extracting them into their own `verif.formal` test
and replacing `require` with `assume` and `ensure` with `assert`. The
results of the contract are replaced with the operands of the contract.

Contracts are used as simplifications for other verification tasks by
inlining them and replacing `require` with `assert` and `ensure` with
`assume`. The results of the contract are replaced with symbolic values.

See the documentation of the Verif dialect for more details.

Traits: `NoRegionArguments`, `NoTerminator`, `SingleBlock`

Interfaces: `RegionKindInterface`

#### Operands: [¶](#operands-5)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

#### Results: [¶](#results-1)

| Result | Description |
| --- | --- |
| `outputs` | variadic of any type |

### `verif.cover` (circt::verif::CoverOp) [¶](#verifcover-circtverifcoverop)

*Ensure that a property can hold.*

Syntax:

```
operation ::= `verif.cover` $property (`if` $enable^)? (`label` $label^)? attr-dict `:` type($property)
```

#### Attributes: [¶](#attributes-6)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-6)

| Operand | Description |
| --- | --- |
| `property` | 1-bit signless integer or LTL sequence type or LTL property type |
| `enable` | 1-bit signless integer |

### `verif.ensure` (circt::verif::EnsureOp) [¶](#verifensure-circtverifensureop)

*A postcondition of a contract*

Syntax:

```
operation ::= `verif.ensure` $property
              (`if` $enable^)?
              (`label` $label^)?
              attr-dict `:` type($property)
```

This operation specifies a condition that is asserted when checking a
contract, and assumed when applying the contract as a simplification.

The `verif.ensure` op is commonly used to specify the conditions that output
values from a part of the IR are guaranteed to fulfill, under the condition
that all requirements are fulfilled.

Traits: `HasParent<verif::ContractOp>`

Interfaces: `RequireLike`

#### Attributes: [¶](#attributes-7)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-7)

| Operand | Description |
| --- | --- |
| `property` | 1-bit signless integer or LTL sequence type or LTL property type |
| `enable` | 1-bit signless integer |

### `verif.formal` (circt::verif::FormalOp) [¶](#verifformal-circtverifformalop)

*A formal unit test*

Syntax:

```
operation ::= `verif.formal` $sym_name $parameters attr-dict-with-keyword $body
```

This operation defines a formal unit test that can be automatically run by
various tools. To describe a test, the body of this op should contain the
hardware to be tested, alongside any asserts, assumes, and covers to be
formally verified. The body can contain instances of other modules, in which
case all asserts, assumes, and covers in those modules are also verified.

The `verif.symbolic_value` op can be used to create symbolic values to feed
into the hardware to be tested. Testing tools will then try to find concrete
values for them that violate any asserts or make any covers true.

#### Example [¶](#example)

```
verif.formal @AdderTest {myParam = 42, myTag = "hello"} {
  %0 = verif.symbolic_value : i42
  %1 = verif.symbolic_value : i42
  %2 = hw.instance "dut" @Adder(a: %0: i42, b: %1: i42) -> (c: i42)
  %3 = comb.add %0, %1 : i42
  %4 = comb.icmp eq %2, %3 : i42
  verif.assert %4 : i1
}
```

#### Parameters [¶](#parameters)

The following parameters have a predefined meaning and are interpreted by
tools such as `circt-test` to guide execution of tests:

* `ignore`: Indicates whether the test should be ignored and skipped. This
  can be useful for temporarily disabling tests without having to remove
  them from the input. Must be a *boolean* value.

  ```
  verif.formal @Foo {ignore = true}
  ```
* `require_runners`: A list of test runners that may be used to execute this
  test. This option may be used to force a test to run using one of a few
  known-good runners, acting like a whitelist. Must be an *array* of
  *strings*.

  ```
  verif.formal @Foo {require_runners = ["sby", "circt-bmc"]}
  ```
* `exclude_runners`: A list of test runners that must not be used to execute
  this test. This option may be used to exclude a few known-bad runners from
  executing this test, acting like a blacklist. Must be an *array* of
  *strings*.

  ```
  verif.formal @Foo {exclude_runners = ["sby", "circt-bmc"]}
  ```

Traits: `IsolatedFromAbove`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`

Interfaces: `RegionKindInterface`, `Symbol`

#### Attributes: [¶](#attributes-8)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `parameters` | ::mlir::DictionaryAttr | dictionary of named attribute values |

### `verif.format_verilog_string` (circt::verif::FormatVerilogStringOp) [¶](#verifformat_verilog_string-circtverifformatverilogstringop)

*Creates a formatted string.*

Syntax:

```
operation ::= `verif.format_verilog_string` $formatString `(` $substitutions `)` `:` type($substitutions) attr-dict
```

Creates a formatted string suitable for printing via the `verif.print` op.
The formatting syntax is expected to be identical to verilog string
formatting to keep things simple for emission.
If we in the future would like to be less tied to verilog formatting,
please ask your friendly neighbourhood compiler engineer to e.g. implement
a `FormatStringOp` which itself may lower to a `FormatVerilogStringOp`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-9)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `formatString` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-8)

| Operand | Description |
| --- | --- |
| `substitutions` | variadic of any type |

#### Results: [¶](#results-2)

| Result | Description |
| --- | --- |
| `str` | a HW string |

### `verif.has_been_reset` (circt::verif::HasBeenResetOp) [¶](#verifhas_been_reset-circtverifhasbeenresetop)

*Check that a proper reset has been seen.*

Syntax:

```
operation ::= `verif.has_been_reset` $clock `,` custom<KeywordBool>($async, "\"async\"", "\"sync\"")
              $reset attr-dict
```

The result of `verif.has_been_reset` reads as 0 immediately after simulation
startup and after each power-cycle in a power-aware simulation. The result
remains 0 before and during reset and only switches to 1 after the reset is
deasserted again.

This is a useful utility to disable the evaluation of assertions and other
verification constructs in the IR before the circuit being tested has been
properly reset. Verification failures due to uninitialized or randomized
initial state can thus be prevented.

Using the result of `verif.has_been_reset` to enable verification is more
powerful and proper than just disabling verification during reset. The
latter does not properly handle the period of time between simulation
startup or power-cycling and the start of reset. `verif.has_been_reset` is
guaranteed to produce a 0 value in that period, as well as during the reset.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-10)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `async` | ::mlir::BoolAttr | bool attribute |

#### Operands: [¶](#operands-9)

| Operand | Description |
| --- | --- |
| `clock` | 1-bit signless integer |
| `reset` | 1-bit signless integer |

#### Results: [¶](#results-3)

| Result | Description |
| --- | --- |
| `result` | 1-bit signless integer |

### `verif.lec` (circt::verif::LogicEquivalenceCheckingOp) [¶](#veriflec-circtveriflogicequivalencecheckingop)

*Represents a logical equivalence checking problem*

Syntax:

```
operation ::= `verif.lec` attr-dict (`:` type($isProven)^)?
              `first` $firstCircuit
              `second` $secondCircuit
```

This operation represents a logic equivalence checking problem explicitly in
the IR. There are several possibilities to perform logical equivalence
checking. For example, equivalence checking of combinational circuits can be
done by constructing a miter circuit and testing whether the result is
satisfiable (can be non-zero for some input), or two canonical BDDs could be
constructed and compared for identity, etc.

The number and types of the inputs and outputs of the two circuits (and thus
also the block arguments and yielded values of both regions) have to match.
Otherwise, the two should be considered trivially non-equivalent.

The operation can return a boolean result that is `true` iff equivalence
of the two circuits has been proven. The result can be omitted for use-cases
which do not allow further processing (e.g., SMT-LIB exporting).

Traits: `IsolatedFromAbove`, `SingleBlockImplicitTerminator<verif::YieldOp>`, `SingleBlock`

#### Results: [¶](#results-4)

| Result | Description |
| --- | --- |
| `isProven` | 1-bit signless integer |

### `verif.print` (circt::verif::PrintOp) [¶](#verifprint-circtverifprintop)

*Prints a message.*

Syntax:

```
operation ::= `verif.print` $string attr-dict
```

#### Operands: [¶](#operands-10)

| Operand | Description |
| --- | --- |
| `string` | a HW string |

### `verif.refines` (circt::verif::RefinementCheckingOp) [¶](#verifrefines-circtverifrefinementcheckingop)

*Check if the second module is a refinement of the first module*

Syntax:

```
operation ::= `verif.refines` attr-dict (`:` type($isProven)^)?
              `first` $firstCircuit
              `second` $secondCircuit
```

This operation represents a refinement checking problem explicitly in the
IR. Given two (purely combinational) circuits A and B with the same
signature, B refines A iff for all inputs the set of possible output
values of B is a subset of the possible output values of A given the
same input.

For strictly deterministic circuits the ‘refines’ relation is identical to
logical equivalence. Informally speaking, refining allows maintaining or
reducing the non-determinism of a circuit.

If the signatures of the circuits do not match, the second circuit is
trivially assumed to *not* be a refinement of the first circuit. Sequential
elements (i.e., registers and memories) are currently unsupported.

The operation can return a boolean result that is `true` iff the refinement
relation has been proven. The result can be omitted for use-cases which do
not allow further processing (e.g., SMT-LIB exporting).

Traits: `IsolatedFromAbove`, `SingleBlockImplicitTerminator<verif::YieldOp>`, `SingleBlock`

#### Results: [¶](#results-5)

| Result | Description |
| --- | --- |
| `isProven` | 1-bit signless integer |

### `verif.require` (circt::verif::RequireOp) [¶](#verifrequire-circtverifrequireop)

*A precondition of a contract*

Syntax:

```
operation ::= `verif.require` $property
              (`if` $enable^)?
              (`label` $label^)?
              attr-dict `:` type($property)
```

This operation specifies a condition that is assumed when checking against
the contract, and asserted when applying the contract as a simplification.

The `verif.require` op is commonly used to specify the conditions that input
values into a part of the IR must fulfill in order for the IR to work as
expected, i.e., as outlined in the contract.

Traits: `HasParent<verif::ContractOp>`

Interfaces: `RequireLike`

#### Attributes: [¶](#attributes-11)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `label` | ::mlir::StringAttr | string attribute |

#### Operands: [¶](#operands-11)

| Operand | Description |
| --- | --- |
| `property` | 1-bit signless integer or LTL sequence type or LTL property type |
| `enable` | 1-bit signless integer |

### `verif.simulation` (circt::verif::SimulationOp) [¶](#verifsimulation-circtverifsimulationop)

*A simulation unit test*

Syntax:

```
operation ::= `verif.simulation` $sym_name $parameters attr-dict-with-keyword $region
```

This operation defines a simulation unit test that can be automatically run
by various tools. To describe a test, the body of this op should contain the
hardware to be tested, alongside any necessary forms of stimulus generation.

#### Inputs [¶](#inputs)

The body has two block arguments as input values: a “clock” signal of type
`!seq.clock` and an “init” signal of type `i1`. The clock signal starts at 0
and continuously toggles between 0 and 1 throughout the simulation. The init
signal starts at 1, remains 1 during a single 0-to-1 transition of the
clock, and then drops to 0 for the remainder of the simulation.

#### Outputs [¶](#outputs)

The body must have a `verif.yield` terminator op with exactly two operands:

The first operand is a “done” signal of type `i1` which indicates the end of
the simulation. The simulation stops when the done signal is 1 during a
0-to-1 transition of the clock after the init signal has dropped to 0. No
additional clock toggles occur once done has been sampled as 1.

The second operand is a “success” signal of type `i1` which indicates the
success of a test as 1, or failure as 0. The signal is sampled at the same
time as the done signal. Simulators must signal failure to the operating
system through a non-zero exit code.

#### Schedule [¶](#schedule)

The clock and init values adhere to the following schedule during
simulation:

| Time | Clock | Init | Comment |
| --- | --- | --- | --- |
| t0 (>=0s) | undef | undef | Clock and init may initially be undefined. |
| t1 (>=t0) | 0 | 1 | Initialization code (e.g., `seq.initial`, Verilog `initial` procedures) may run before or after clock and init change to their initial value. |
| t2 (>t1) | 1 | 1 | Single rising clock edge occurs while init is high. |
| t3 (>t2) | 0 | 1 | Init may stay during the falling clock edge. |
| t4 (>=t3) | 0 | 0 | Init goes to 0 before second rising clock edge. |
| t5 (>t4) | 1 | 0 | Clock toggles continue indefinitely. |
| t6 (>t5) | 0 | 0 |  |

Simulation termination occurs when the done signal is 1 during a 0-to-1
transition of the clock.

#### Example [¶](#example-1)

```
verif.simulation @AdderTest {myParam = 42, myTag = "hello"} {
^bb0(%clock: !seq.clock, %init: i1):
  // Count the first 9001 simulation cycles.
  %c0_i19 = hw.constant 0 : i19
  %c1_i19 = hw.constant 1 : i19
  %c9001_i19 = hw.constant 9001 : i19
  %count = seq.compreg %0, %clock reset %init, %c0_i19 : i19
  %done = comb.icmp eq %count, %c9001_i19 : i19
  %0 = comb.add %count, %c1_i19 : i19

  // Generate inputs to the adder.
  %1, %2 = func.call @generateAdderInputs(%count) : (i19) -> (i42, i42)
  %3 = hw.instance "dut" @Adder(a: %1: i42, b: %2: i42) -> (c: i42)

  // Check results and track failures.
  %4 = comb.add %1, %2 : i42
  %5 = comb.icmp eq %3, %4 : i42
  %true = hw.constant true
  %success = seq.compreg %6, %clock reset %init, %true : i1
  %6 = comb.and %success, %5 : i1

  verif.yield %done, %success : i1, i1
}
```

Traits: `IsolatedFromAbove`, `SingleBlockImplicitTerminator<verif::YieldOp>`, `SingleBlock`

Interfaces: `OpAsmOpInterface`, `RegionKindInterface`, `Symbol`

#### Attributes: [¶](#attributes-12)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `parameters` | ::mlir::DictionaryAttr | dictionary of named attribute values |

### `verif.symbolic_value` (circt::verif::SymbolicValueOp) [¶](#verifsymbolic_value-circtverifsymbolicvalueop)

*Create a symbolic value for formal verification*

Syntax:

```
operation ::= `verif.symbolic_value` attr-dict `:` type($result)
```

This operation creates a new symbolic value that can be used to formally
verify designs. Verification tools will try to find concrete assignments for
symbolic values that violate asserts or make covers true. This value is not
fixed - the value taken can vary arbitrarily between timesteps.

Traits: `HasParent<verif::FormalOp, hw::HWModuleOp>`

#### Results: [¶](#results-6)

| Result | Description |
| --- | --- |
| `result` | any type |

### `verif.yield` (circt::verif::YieldOp) [¶](#verifyield-circtverifyieldop)

*Yields values from a region*

Syntax:

```
operation ::= `verif.yield` ($inputs^ `:` type($inputs))? attr-dict
```

Traits: `HasParent<verif::BoundedModelCheckingOp, verif::LogicEquivalenceCheckingOp, verif::RefinementCheckingOp, verif::SimulationOp>`, `Terminator`

#### Operands: [¶](#operands-12)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of any type |

 [Prev - SMT Dialect](https://circt.llvm.org/docs/Dialects/SMT/ "SMT Dialect")
[Next - EDA Tool Workarounds](https://circt.llvm.org/docs/ToolsWorkarounds/ "EDA Tool Workarounds") 

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