LTL Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

LTL Dialect
===========

This dialect provides operations and types to model
[Linear Temporal Logic](https://en.wikipedia.org/wiki/Linear_temporal_logic), sequences, and properties, which are useful for hardware verification.

* [Rationale](#rationale)
  + [Sequences and Properties](#sequences-and-properties)
* [Representing SVAs](#representing-svas)
  + [Sequence Concatenation and Cycle Delay](#sequence-concatenation-and-cycle-delay)
  + [Implication](#implication)
  + [Repetition](#repetition)
  + [Clocking](#clocking)
  + [Disable Iff](#disable-iff)
* [Mapping SVA to CIRCT dialects](#mapping-sva-to-circt-dialects)
* [Representing the LTL Formalism](#representing-the-ltl-formalism)
  + [Next / Delay](#next--delay)
  + [Until and Eventually](#until-and-eventually)
  + [Concatenation and Repetition](#concatenation-and-repetition)
* [Types](#types)
  + [Overview](#overview)
  + [PropertyType](#propertytype)
  + [SequenceType](#sequencetype)
* [Operations](#operations)
  + [`ltl.and` (circt::ltl::AndOp)](#ltland-circtltlandop)
  + [`ltl.boolean_constant` (circt::ltl::BooleanConstantOp)](#ltlboolean_constant-circtltlbooleanconstantop)
  + [`ltl.clock` (circt::ltl::ClockOp)](#ltlclock-circtltlclockop)
  + [`ltl.concat` (circt::ltl::ConcatOp)](#ltlconcat-circtltlconcatop)
  + [`ltl.delay` (circt::ltl::DelayOp)](#ltldelay-circtltldelayop)
  + [`ltl.eventually` (circt::ltl::EventuallyOp)](#ltleventually-circtltleventuallyop)
  + [`ltl.goto_repeat` (circt::ltl::GoToRepeatOp)](#ltlgoto_repeat-circtltlgotorepeatop)
  + [`ltl.implication` (circt::ltl::ImplicationOp)](#ltlimplication-circtltlimplicationop)
  + [`ltl.intersect` (circt::ltl::IntersectOp)](#ltlintersect-circtltlintersectop)
  + [`ltl.non_consecutive_repeat` (circt::ltl::NonConsecutiveRepeatOp)](#ltlnon_consecutive_repeat-circtltlnonconsecutiverepeatop)
  + [`ltl.not` (circt::ltl::NotOp)](#ltlnot-circtltlnotop)
  + [`ltl.or` (circt::ltl::OrOp)](#ltlor-circtltlorop)
  + [`ltl.past` (circt::ltl::PastOp)](#ltlpast-circtltlpastop)
  + [`ltl.repeat` (circt::ltl::RepeatOp)](#ltlrepeat-circtltlrepeatop)
  + [`ltl.until` (circt::ltl::UntilOp)](#ltluntil-circtltluntilop)

Rationale [¶](#rationale)
-------------------------

The main goal of the `ltl` dialect is to capture the core formalism underpinning SystemVerilog Assertions (SVAs), the de facto standard for describing temporal logic sequences and properties in hardware verification. (See IEEE 1800-2017 section 16 “Assertions”.) We expressly try *not* to model this dialect like an AST for SVAs, but instead try to strip away all the syntactic sugar and Verilog quirks, and distill out the core foundation as an IR. Within the CIRCT project, this dialect intends to enable emission of rich temporal assertions as part of the Verilog output, but also provide a foundation for formal tools built ontop of CIRCT.

### Sequences and Properties [¶](#sequences-and-properties)

The core building blocks for modeling temporal logic in the `ltl` dialect are *sequences* and *properties*. In a nutshell, sequences behave like regular expressions over time, whereas properties provide the quantifiers to express that sequences must be true under certain conditions.

**Sequences** describe boolean expressions at different points in time. They can be easily verified by a finite state automaton, similar to how regular expressions and languages have an equivalent automaton that recognizes the language. For example:

* The boolean `a` is a sequence. It holds if `a` is true in cycle 0 (the current cycle).
* The boolean expression `a & b` is also a sequence. It holds if `a & b` is true in cycle 0.
* `##1 a` checks that `a` is true in cycle 1 (the next cycle).
* `##[1:4] a` checks that `a` is true anywhere in cycle 1, 2, 3, or 4.
* `a ##1 b` checks that `a` holds in cycle 0 and `b` holds in cycle 1.
* `##1 (a ##1 b)` checks that `a` holds in cycle 1 and `b` holds in cycle 2.
* `(a ##1 b) ##5 (c ##1 d)` checks that the sequence `(a ##1 b)` holds and is followed by the sequence `(c ##1 d)` 5 or 6 cycles later. Concretely, this checks that `a` holds in cycle 0, `b` holds in cycle 1, `c` holds in cycle 6 (5 cycles after the first sequence ended in cycle 1), and `d` holds in cycle 7.

**Properties** describe concrete, testable propositions or claims built from sequences. While sequences can observe and match a certain behavior in a circuit at a specific point in time, properties allow you to express that these sequences hold in every cycle, or hold at some future point in time, or that one sequence is always followed by another. For example:

* `always s` checks that the sequence `s` holds in every cycle. This is often referred to as the **G** (or “globally”) operator in LTL.
* `eventually s` checks that the sequence `s` will hold at some cycle now or in the future. This is often referred to as the **F** (or “finally”) operator in LTL.
* `p until q` checks that the property `p` holds in every cycle before the first cycle `q` holds. This is often referred to as the **U** (or “until”) operator in LTL.
* `s implies t` checks that whenever the sequence `s` is observed, it is immediately followed by sequence `t`.

Traditional definitions of the LTL formalism do not make a distinction between sequences and properties. Most of their operators fall into the property category, for example, quantifiers like *globally*, *finally*, *release*, and *until*. The set of sequence operators is usually very small, since it is not necessary for academic treatment, consisting only of the *next* operator. The `ltl` dialect provides a richer set of operations to model sequences.

Representing SVAs [¶](#representing-svas)
-----------------------------------------

### Sequence Concatenation and Cycle Delay [¶](#sequence-concatenation-and-cycle-delay)

The primary building block for sequences in SVAs is the *concatenation* expression. Concatenation is always associated with a cycle delay, which indicates how many cycles pass between the end of the LHS sequence and the start of the RHS sequence. One, two, or more sequences can be concatenated at once, and the overall concatenation can have an initial cycle delay. For example:

```
a ##1 b ##1 c      // 1 cycle delay between a, b, and c
##2 a ##1 b ##1 c  // same, plus 2 cycles of initial delay before a
```

In the simplest form, a cycle delay can appear as a prefix of another sequence, e.g., `##1 a`. This is essentially a concatenation with only one sequence, `a`, and an initial cycle delay of the concatenation of `1`. The prefix delays map to the LTL dialect as follows:

* `##N seq`. **Fixed delay.** Sequence `seq` has to match exactly `N` cycles in the future. Equivalent to `ltl.delay %seq, N, 0`.
* `##[N:M] seq`. **Bounded range delay.** Sequence `seq` has to match anywhere between `N` and `M` cycles in the future, inclusive. Equivalent to `ltl.delay %seq, N, (M-N)`
* `##[N:$] seq`. **Unbounded range delay.** Sequence `seq` has to match anywhere at or beyond `N` cycles in the future, after a finite amount of cycles. Equivalent to `ltl.delay %seq, N`.
* `##[*] seq`. Shorthand for `##[0:$]`. Equivalent to `ltl.delay %seq, 0`.
* `##[+] seq`. Shorthand for `##[1:$]`. Equivalent to `ltl.delay %seq, 1`.

Concatenation of two sequences always involves a cycle delay specification in between them, e.g., `a ##1 b` where sequence `b` starts in the cycle after `a` ends. Zero-cycle delays can be specified, e.g., `a ##0 b` where `b` starts in the same cycle as `a` ends. If `a` and `b` are booleans, `a ##0 b` is equivalent to `a && b`.

The dialect separates concatenation and cycle delay into two orthogonal operations, `ltl.concat` and `ltl.delay`, respectively. The former models concatenation as `a ##0 b`, and the latter models delay as a prefix `##1 c`. The SVA concatenations with their infix delays map to the LTL dialect as follows:

* `seqA ##N seqB`. **Binary concatenation.** Sequence `seqB` follows `N` cycles after `seqA`. This can be represented as `seqA ##0 (##N seqB)`, which is equivalent to

  ```
  %0 = ltl.delay %seqB, N, 0
  ltl.concat %seqA, %0
  ```
* `seqA ##N seqB ##M seqC`. **Variadic concatenation.** Sequence `seqC` follows `M` cycles after `seqB`, which itself follows `N` cycles after `seqA`. This can be represented as `seqA ##0 (##N seqB) ##0 (##M seqC)`, which is equivalent to

  ```
  %0 = ltl.delay %seqB, N, 0
  %1 = ltl.delay %seqC, M, 0
  ltl.concat %seqA, %0, %1
  ```

  Since concatenation is associative, this is also equivalent to `seqA ##N (seqB ##M seqC)`:

  ```
  %0 = ltl.delay %seqC, M, 0
  %1 = ltl.concat %seqB, %0
  %2 = ltl.delay %1, N, 0
  ltl.concat %seqA, %2
  ```

  And also `(seqA ##N seqB) ##M seqC`:

  ```
  %0 = ltl.delay %seqB, N, 0
  %1 = ltl.concat %seqA, %0
  %2 = ltl.delay %seqC, M, 0
  ltl.concat %1, %2
  ```
* `##N seqA ##M seqB`. **Initial delay.** Sequence `seqB` follows `M` cycles afer `seqA`, which itself starts `N` cycles in the future. This is equivalent to a delay on `seqA` within the concatenation:

  ```
  %0 = ltl.delay %seqA, N, 0
  %1 = ltl.delay %seqB, M, 0
  ltl.concat %0, %1
  ```

  Alternatively, the delay can also be placed on the entire concatenation:

  ```
  %0 = ltl.delay %seqB, M, 0
  %1 = ltl.concat %seqA, %0
  ltl.delay %1, N, 0
  ```
* Only the fixed delay `##N` is shown here for simplicity, but the examples extend to the other delay flavors `##[N:M]`, `##[N:$]`, `##[*]`, and `##[+]`.

### Implication [¶](#implication)

```
seq |-> prop
seq |=> prop
```

The overlapping `|->` and non-overlapping `|=>` implication operators of SVA, which only check a property after a precondition sequence matches, map to the `ltl.implication` operation. When the sequence matches in the overlapping case `|->`, the property check starts at the same time the matched sequence ended. In the non-overlapping case `|=>`, the property check starts *at the clock tick after the* end of the matched sequence, unless the matched sequence was empty, in which special rules apply. (See IEEE 1800-2017 section 16.12.7 “Implication”.) The non-overlapping operator can be expressed in terms of the overlapping operator:

```
seq |=> prop
```

is equivalent to

```
(seq ##1 true) |-> prop
```

The `ltl.implication` op implements the overlapping case `|->`, such that the two SVA operator flavors map to the `ltl` dialect as follows:

* `seq |-> prop`. **Overlapping implication.** Equivalent to `ltl.implication %seq, %prop`.
* `seq |=> prop`. **Non-overlapping implication.** Equivalent to

  ```
  %true = hw.constant true
  %0 = ltl.delay %true, 1, 0
  %1 = ltl.concat %seq, %0
  ltl.implication %1, %prop
  ```

An important benefit of only modeling the overlapping `|->` implication operator is that it does not interact with a clock. The end point of the left-hand sequence is the starting point of the right-hand sequence. There is no notion of delay between the end of the left and the start of the right sequence. Compare this to the `|=>` operator in SVA, which implies that the right-hand sequence happens at “strictly the next clock tick”, which requires the operator to have a notion of time and clocking. As described above, it is still possible to model this using an explicit `ltl.delay` op, which already has an established interaction with a clock.

### Repetition [¶](#repetition)

Consecutive repetition repeats the sequence by a number of times. For example, `s[*3]` repeats the sequence `s` three times, which is equivalent to `s ##1 s ##1 s`. This also applies when the sequence `s` matches different traces with different lengths. For example `(##[0:3] a)[*2]` is equivalent to the disjunction of all the combinations such as `a ##1 a`, `a ##1 (##3 a)`, `(##3 a) ##1 (##2 a)`. However, the repetition with unbounded range cannot be expanded to the concatenations as it produces an infinite formula.

The definition of `ltl.repeat` is similar to that of `ltl.delay`. The mapping from SVA’s consecutive repetition to the LTL dialect is as follows:

* `seq[*N]`. **Fixed repeat.** Repeats `N` times. Equivalent to `ltl.repeat %seq, N, 0`.
* `seq[*N:M]`. **Bounded range repeat.** Repeats `N` to `M` times. Equivalent to `ltl.repeat %seq, N, (M-N)`.
* `seq[*N:$]`. **Unbounded range repeat.** Repeats `N` to an indefinite finite number of times. Equivalent to `ltl.repeat %seq, N`.
* `seq[*]`. Shorthand for `seq[*0:$]`. Equivalent to `ltl.repeat %seq, 0`.
* `seq[+]`. Shorthand for `seq[*1:$]`. Equivalent to `ltl.repeat %seq, 1`.

#### Non-Consecutive Repetition [¶](#non-consecutive-repetition)

Non-consecutive repetition checks that a sequence holds a certain number of times within an arbitrary repetition of the sequence. There are two ways of expressing non-consecutive repetition, either by including the last iteration in the count or not. If the last iteration is included, then this is called a “go-to” style non-consecutive repetition and can be defined using the `ltl.goto_repeat <input>, <N>, <window>` operation, e.g. `a !b b b !b !b b c` is a valid observation of `ltl.goto_repeat %b, 1, 2`, but `a !b b b !b !b b !b !b c` is not. If we omit the constraint of having the last iteration hold, then this is simply called a non-consecutive repetition, and can be defined using the `ltl.non_consecutive_repeat <input, <N>, <window>` operation, e.g. both `a !b b b !b !b b c` and `a !b b b !b !b b !b !b c` are valid observations of `ltl.non_consecutive_repeat %b, 1, 2`. The SVA mapping of these operations is as follows:

* `seq[->n:m]`: **Go-To Style Repetition**, equivalent to `ltl.goto_repeat %seq, n, (m-n)`.
* `seq[=n:m]` : **Non-Consecutive Repetition** equivalent to `ltl.non_consecutive_repeat %seq, n, (m-n)`.

### Clocking [¶](#clocking)

Sequence and property expressions in SVAs can specify a clock with respect to which all cycle delays are expressed. (See IEEE 1800-2017 section 16.16 “Clock resolution”.) These map to the `ltl.clock` operation.

* `@(posedge clk) seqOrProp`. **Trigger on low-to-high clock edge.** Equivalent to `ltl.clock %seqOrProp, posedge %clk`.
* `@(negedge clk) seqOrProp`. **Trigger on high-to-low clock edge.** Equivalent to `ltl.clock %seqOrProp, negedge %clk`.
* `@(edge clk) seqOrProp`. **Trigger on any clock edge.** Equivalent to `ltl.clock %seqOrProp, edge %clk`.

### Disable Iff [¶](#disable-iff)

Properties in SVA can have a disable condition attached, which allows for preemptive resets to be expressed. If the disable condition is true at any time during the evaluation of a property, the property is considered disabled. (See IEEE 1800-2017 end of section 16.12 “Declaring properties”.) This maps to the `ltl.disable` operation.

* `disable iff (expr) prop`. **Disable condition.** Equivalent to `ltl.disable %prop if %expr`.

Note that SVAs only allow for entire properties to be disabled, at the point at which they are passed to an assert, assume, or cover statement. It is explicitly forbidden to define a property with a `disable iff` clause and then using it within another property. For example, the following is forbidden:

```
property p0; disable iff (cond) a |-> b; endproperty
property p1; eventually p0; endproperty
```

In this example, `p1` refers to property `p0`, which is illegal in SVA since `p0` itself defines a disable condition.

In contrast, the LTL dialect explicitly allows for properties to be disabled at arbitrary points, and disabled properties to be used in other properties. Since a disabled nested property also disables the parent property, the IR can always be rewritten into a form where there is only one `disable iff` condition at the root of a property expression.

Mapping SVA to CIRCT dialects [¶](#mapping-sva-to-circt-dialects)
-----------------------------------------------------------------

Knowing how to map SVA constructs to CIRCT is important to allow these to expressed correctly in any front-end. Here you will find a non-exhaustive list of conversions from SVA to CIRCT dialects.

* **properties**: `!ltl.property`.
* **sequences**: `!ltl.sequence`.
* **`disable iff (cond)`**: `ltl.disable %prop if %cond`
* **local variables**: Not currently possible to encode.
* **`$rose(a)`**:

```
%1 = ltl.compreg %a, %clock : i1
%rose = comb.icmp bin ult %1, %a : i1
```

* **`$fell(a)`**:

```
%1 = ltl.compreg %a, %clock : i1
%fell = comb.icmp bin ugt %a, %1 : i1
```

* **`$stable(a)`**:

```
%1 = ltl.compreg %a, %clock : i1
%rose = comb.icmp bin eq %a, %1 : i1
```

* **`$past(a, n)`**:

```
%zero = hw.constant 0 : i1
%true = hw.constant 1 : i1
%1 = seq.shiftreg n, %a, %clk, %true, powerOn %zero : i1
```

> The following functions are not yet supported by CIRCT:
>
> * **`$onehot(a)`**
> * **`$onehot0(a)`**
> * **`$isunknown(a)`**
> * **`$countones(a)`**

* **`a ##n b`**:

```
%a_n = ltl.delay %a, n, 0 : i1
%anb = ltl.concat %a_n, %b : !ltl.sequence
```

* **`a ##[n:m] b`**:

```
%a_n = ltl.delay %a, n, (m-n) : i1
%anb = ltl.concat %a_n, %b : !ltl.sequence
```

* **`s [*n]`**:

```
%repsn = ltl.repeat %s, n, 0 : !ltl.sequence
```

* **`s [*n:m]`**:

```
%repsnm = ltl.repeat %s, n, (m-n) : !ltl.sequence
```

* **`s[*n:$]`**:

```
%repsninf = ltl.repeat %s, n : !ltl.sequence
```

* **`s[->n:m]`**:

```
%1 = ltl.goto_repeat %s, n, (m-n) : !ltl.sequence
```

* **`s[=n:m]`**:

```
%1 = ltl.non_consecutive_repeat %s, n, (m-n) : !ltl.sequence
```

* **`s1 ##[+] s2`**:

```
%ds1 = ltl.delay %s1, 1
%s1s2 = ltl.concat %ds1, %s2 : !ltl.sequence
```

* **`s1 ##[*] s2`**:

```
%ds1 = ltl.delay %s1, 0
%s1s2 = ltl.concat %ds1, %s2 : !ltl.sequence
```

* **`s1 and s2`**:

```
ltl.and %s1, %s2 : !ltl.sequence
```

* **`s1 intersect s2`**:

```
ltl.intersect %s1, %s2 : !ltl.sequence
```

* **`s1 or s2`**:

```
ltl.or %s1, %s2 : !ltl.sequence
```

* **`not s`**:

```
ltl.not %s1 : !ltl.sequence
```

* **`first_match(s)`**: Not possible to encode yet.
* **`expr throughout s`**:

```
%repexpr = ltl.repeat %expr, 0 : !ltl.sequence
%res = ltl.intersect %repexpr, %s : !ltl.sequence
```

* **`s1 within s2`**:

```
%c1 = hw.constant 1 : i1
%rep1 = ltl.repeat %c1, 0 : !ltl.sequence
%drep1 = ltl.delay %rep1, 1, 0 : !ltl.sequence
%ds1 = ltl.delay %s1, 1, 0 : !ltl.sequence
%evs1 = ltl.concat %drep1, %ds1, %c1 : !ltl.sequence
%res = ltl.intersect %evs1, %s2 : !ltl.sequence
```

* **`s |-> p`**:

```
%1 = ltl.implication %s, %p : !ltl.property
```

* **`s |=> p`**:

```
%c1 = hw.constant 1 : i1
%ds = ltl.delay %s, 1, 0 : i1
%antecedent = ltl.concat %ds, %c1 : !ltl.sequence
%impl = ltl.implication %antecedent, %p : !ltl.property
```

* **`p1 implies p2`**:

```
%np1 = ltl.not %p1 : !ltl:property
%impl = ltl.or %np1, %p2 : !ltl.property
```

* **`p1 iff p2`**: equivalent to `(not (p1 or p2)) or (p1 and p2)`

```
%p1orp2 = ltl.or %p1, %p2 : !ltl.property
%lhs = ltl.not %p1orp2 : !ltl.property
%rhs = ltl.and %p1, %p2 : !ltl.property
%iff = ltl.or %lhs, %rhs : !ltl.property
```

* **`s #-# p`**:

```
%np = ltl.not %p : !ltl.property
%impl = ltl.implication %s, %np : !ltl.property
%res = ltl.not %impl : !ltl.property
```

* **`s #=# p`**:

```
%np = ltl.not %p : !ltl.property	
%ds = ltl.delay %s, 1, 0 : !ltl.sequence
%c1 = hw.constant 1 : i1
%ant = ltl.concat %ds, c1 : !ltl.sequence 
%impl = ltl.implication %ant, %np : !ltl.property
%res = ltl.not %impl : !ltl.property
```

* **`strong(s)`**: default for coverpoints, not supported in other cases.
* **`weak(s)`**: default for assert and assume, not supported for cover.
* **`nexttime p`**:

```
ltl.delay %p, 1, 0 : !ltl.sequence
```

* **`nexttime[n] p`**:

```
ltl.delay %p, n, 0 : !ltl.sequence
```

* **`s_nexttime p`**: not really distinguishable from the weak version in CIRCT.
* **`s_nexttime[n] p`**: not really distinguishable from the weak version in CIRCT.
* **`always p`**:

```
ltl.repeat %p, 0 : !ltl.sequence
```

* **`always[n:m] p`**:

```
ltl.repeat %p, n, m : !ltl.sequence
```

* **`s_always[n:m] p`**: not really distinguishable in CIRCT
* **`s_eventually p`**:

```
ltl.eventually %p : !ltl.property
```

* **`eventually[n:m] p`**: not yet encodable in CIRCT.
* **`s_eventually[n:m] p`**: not yet encodable in CIRCT.
* **`p1 until p2`**:

```
%1 = ltl.until %p1, %p2 : !ltl.sequence
```

* **`p1 s_until p2`**: not really distinguishable from the weak version in CIRCT.
* **`p1 until_with p2`**: Equivalent to `(p1 until p2) implies (p1 and p2)`

```
%1 = ltl.until %p1, %p2 : !ltl.sequence
%2 = ltl.and %p1, %p2 : !ltl:property
%n1 = ltl.not %1 : !ltl.property
%res = ltl.or %n1, %2 : !ltl.property
```

* **`p1 s_until_with p2`**: not really distinguishable from the weak version in CIRCT.

> We don’t yet support abort constructs but might in the future.
>
> * **`accept_on ( expr ) p`**
> * **`reject_on ( expr ) p`**
> * **`sync_accept_on ( expr ) p`**
> * **`sync_reject_on ( expr ) p`**

Representing the LTL Formalism [¶](#representing-the-ltl-formalism)
-------------------------------------------------------------------

### Next / Delay [¶](#next--delay)

The `ltl.delay` sequence operation represents various shorthands for the *next*/**X** operator in LTL:

| Operation | LTL Formula |
| --- | --- |
| `ltl.delay %a, 0, 0` | a |
| `ltl.delay %a, 1, 0` | **X**a |
| `ltl.delay %a, 3, 0` | **XXX**a |
| `ltl.delay %a, 0, 2` | a ∨ **X**a ∨ **XX**a |
| `ltl.delay %a, 1, 2` | **X**(a ∨ **X**a ∨ **XX**a) |
| `ltl.delay %a, 0` | **F**a |
| `ltl.delay %a, 2` | **XXF**a |

### Until and Eventually [¶](#until-and-eventually)

`ltl.until` is *weak*, meaning the property will hold even if the trace does not contain enough clock cycles to evaluate the property. `ltl.eventually` is *strong*, where `ltl.eventually %p` means `p` must hold at some point in the trace.

### Concatenation and Repetition [¶](#concatenation-and-repetition)

The `ltl.concat` sequence operation does not have a direct equivalent in LTL. It builds a longer sequence by composing multiple shorter sequences one after another. LTL has no concept of concatenation, or a *“v happens after u”*, where the point in time at which v starts is dependent on how long the sequence u was.

For a sequence u with a fixed length of 2, concatenation can be represented as *"(u happens) and (v happens 2 cycles in the future)"*, u ∧ **XX**v. If u has a dynamic length though, for example a delay between 1 and 2, `ltl.delay %u, 1, 1` or **X**u ∨ **XX**u in LTL, there is no fixed number of cycles by which the sequence v can be delayed to make it start after u. Instead, all different-length variants of sequence u have to be enumerated and combined with a copy of sequence v delayed by the appropriate amount: (**X**u ∧ **XX**v) ∨ (**XX**u ∧ **XXX**v). This is basically saying “u delayed by 1 to 2 cycles followed by v” is the same as either *“u delayed by 1 cycle and v delayed by 2 cycles”*, or *“u delayed by 2 cycles and v delayed by 3 cycles”*.

The *“v happens after u”* relationship is crucial to express sequences efficiently, which is why the LTL dialect has the `ltl.concat` op. If sequences are thought of as regular expressions over time, for example, `a(b|cd)` or *“a followed by either (b) or (c followed by d)”*, the importance of having a concatenation operation as temporal connective becomes apparent. Why LTL formalisms tend to not include such an operator is unclear.

As for `ltl.repeat`, it also relies on the semantics of *v happens after u* to compose the repeated sequences. Unlike `ltl.concat`, which can be expanded by LTL operators within a finite formula size, unbounded repetition cannot be expanded by listing all cases. This means unbounded repetition imports semantics that LTL cannot describe. The LTL dialect has this operation because it is necessary and useful for regular expressions and SVA.

Types [¶](#types)
-----------------

### Overview [¶](#overview)

The `ltl` dialect operations defines two main types returned by its operations: sequences and properties. These types form a hierarchy together with the boolean type `i1`:

* a boolean `i1` is also a valid sequence
* a sequence `!ltl.sequence` is also a valid property

```
i1 <: ltl.sequence <: ltl.property
```

The two type constraints `AnySequenceType` and `AnyPropertyType` are provided to implement this hierarchy. Operations use these constraints for their operands, such that they can properly accept `i1` as a sequence, `i1` or a sequence as a property. The return type is an explicit `!ltl.sequence` or `!ltl.property`.

### PropertyType [¶](#propertytype)

*LTL property type*

Syntax: `!ltl.property`

The `ltl.property` type represents a verifiable property built from linear
temporal logic sequences and quantifiers, for example, *“if you see sequence
A, eventually you will see sequence B”*.

Note that this type explicitly identifies a *property*. However, a boolean
value (`i1`) or a sequence (`ltl.sequence`) is also a valid property.
Operations that accept a property as an operand will use the `AnyProperty`
constraint, which also accepts `ltl.sequence` and `i1`.

### SequenceType [¶](#sequencetype)

*LTL sequence type*

Syntax: `!ltl.sequence`

The `ltl.sequence` type represents a sequence of linear temporal logic, for
example, *“A is true two cycles after B is true”*.

Note that this type explicitly identifies a *sequence*. However, a boolean
value (`i1`) is also a valid sequence. Operations that accept a sequence as
an operand will use the `AnySequence` constraint, which also accepts `i1`.

Operations [¶](#operations)
---------------------------

### `ltl.and` (circt::ltl::AndOp) [¶](#ltland-circtltlandop)

*A conjunction of booleans, sequences, or properties.*

Syntax:

```
operation ::= `ltl.and` $inputs attr-dict `:` type($inputs)
```

If any of the `$inputs` is of type `!ltl.property`, the result of the op is
an `!ltl.property`. Otherwise it is an `!ltl.sequence`.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of 1-bit signless integer or LTL sequence type or LTL property type |

#### Results: [¶](#results)

| Result | Description |
| --- | --- |
| `result` | 1-bit signless integer or LTL sequence type or LTL property type |

### `ltl.boolean_constant` (circt::ltl::BooleanConstantOp) [¶](#ltlboolean_constant-circtltlbooleanconstantop)

*A constant boolean property value.*

Syntax:

```
operation ::= `ltl.boolean_constant` $value attr-dict
```

Represents a constant boolean value as an LTL property. This operation
takes a boolean attribute and produces an `!ltl.property` result. This is
useful for representing constant property values in canonicalization patterns.

Example:

```
%prop = ltl.boolean_constant true
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::BoolAttr | bool attribute |

#### Results: [¶](#results-1)

| Result | Description |
| --- | --- |
| `result` | LTL property type |

### `ltl.clock` (circt::ltl::ClockOp) [¶](#ltlclock-circtltlclockop)

*Specify the clock for a property or sequence.*

Syntax:

```
operation ::= `ltl.clock` $input `,` $edge $clock attr-dict `:` type($input)
```

Specifies the `$edge` on a given `$clock` to be the clock for an `$input`
property or sequence. All cycle delays in the `$input` implicitly refer to a
clock that advances the state to the next cycle. The `ltl.clock` operation
provides that clock. The clock applies to the entire property or sequence
expression tree below `$input`, up to any other nested `ltl.clock`
operations.

The operation returns a property if the `$input` is a property, and a
sequence otherwise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `edge` | circt::ltl::ClockEdgeAttr | clock edge |

#### Operands: [¶](#operands-1)

| Operand | Description |
| --- | --- |
| `input` | 1-bit signless integer or LTL sequence type or LTL property type |
| `clock` | 1-bit signless integer |

#### Results: [¶](#results-2)

| Result | Description |
| --- | --- |
| `result` | LTL sequence type or LTL property type |

### `ltl.concat` (circt::ltl::ConcatOp) [¶](#ltlconcat-circtltlconcatop)

*Concatenate sequences into a longer sequence.*

Syntax:

```
operation ::= `ltl.concat` $inputs attr-dict `:` type($inputs)
```

Concatenates all of the `$inputs` sequences one after another into one
longer sequence. The sequences are arranged such that the end time of the
previous sequences coincides with the start time of the next sequence. This
means there is no implicit cycle of delay between the concatenated
sequences, which may be counterintuitive.

If a sequence should follow in the cycle after another sequence finishes,
that cycle of delay needs to be explicit. For example, *“u followed by v in
next cycle”* (`u ##1 v` in SVA) is represented as
`concat(u, delay(v, 1, 0))`:

```
%0 = ltl.delay %v, 1, 0 : i1
ltl.concat %u, %v : !ltl.sequence, !ltl.sequence
```

The resulting sequence checks for `u` in the first cycle and `v` in the
second, `[u, v]` in short.

Without this explicit delay, the previous sequence’s end overlaps with the
next sequence’s start. For example, consider the two sequences `u = a ##1 b`
and `v = c ##1 d`, which check for `a` and `c` in the first, and `b` and `d`
in the second cycle. When these two sequences are concatenated,
`concat(u, v)`, the end time of the first sequence coincides with the start
time of the second. As a result, the check for `b` at the end of the first
sequence will coincide with the check for `c` at the start of the second
sequence: `concat(u, v) = a ##1 (b && c) ##1 d`. The resulting sequence
checks for `a` in the first cycle, `b` and `c` in the second, and `d` in the
third, `[a, (b && c), d]` in short.

By making the delay between concatenated sequences explicit, the `concat`
operation behaves nicely in the presence of zero-length sequences. An empty,
zero-length sequence in a concatenation behaves as if the sequence wasn’t
present at all. Compare this to SVAs which struggle with empty sequences.
For example, `x ##1 y ##1 z` would become `x ##2 z` if `y` was empty.
Similarly, expressing zero or more repetitions of a sequence, `w ##[*]`, is
challenging in SVA since concatenation always implies a cycle of delay, but
trivial if the delay is made explicit. This is related to the handling of
empty rules in a parser’s grammar.

Note that concatenating two boolean values *a* and *b* is equivalent to
computing the logical AND of them. Booleans are sequences that check if the
boolean is true in the current cycle, which means that the sequence starts
and ends in the same cycle. Since concatenation aligns the sequences such
that end time of *a* and start time of *b* coincide, the resulting sequence
checks if *a* and *b* both are true in the current cycle, which is an AND
operation.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-2)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of 1-bit signless integer or LTL sequence type |

#### Results: [¶](#results-3)

| Result | Description |
| --- | --- |
| `result` | LTL sequence type |

### `ltl.delay` (circt::ltl::DelayOp) [¶](#ltldelay-circtltldelayop)

*Delay a sequence by a number of cycles.*

Syntax:

```
operation ::= `ltl.delay` $input `,` $delay (`,` $length^)? attr-dict `:` type($input)
```

Delays the `$input` sequence by the number of cycles specified by `$delay`.
The delay must be greater than or equal to zero. The optional `$length`
specifies during how many cycles after the initial delay the sequence can
match. Omitting `$length` indicates an unbounded but finite delay. For
example:

* `ltl.delay %seq, 2, 0` delays `%seq` by exactly 2 cycles. The resulting
  sequence matches if `%seq` matches exactly 2 cycles in the future.
* `ltl.delay %seq, 2, 2` delays `%seq` by 2, 3, or 4 cycles. The resulting
  sequence matches if `%seq` matches 2, 3, or 4 cycles in the future.
* `ltl.delay %seq, 2` delays `%seq` by 2 or more cycles. The number of
  cycles is unbounded but finite, which means that `%seq` *has* to match at
  some point, instead of effectively never occuring by being delayed an
  infinite number of cycles.
* `ltl.delay %seq, 0, 0` is equivalent to just `%seq`.

#### Clocking [¶](#clocking-1)

The cycle delay specified on the operation refers to a clocking event. This
event is not directly specified by the delay operation itself. Instead, the
[`ltl.clock`](#ltlclock-circtltlclockop) operation can be used to associate
all delays within a sequence with a clock.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-2)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `delay` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `length` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands: [¶](#operands-3)

| Operand | Description |
| --- | --- |
| `input` | 1-bit signless integer or LTL sequence type |

#### Results: [¶](#results-4)

| Result | Description |
| --- | --- |
| `result` | LTL sequence type |

### `ltl.eventually` (circt::ltl::EventuallyOp) [¶](#ltleventually-circtltleventuallyop)

*Ensure that a property will hold at some time in the future.*

Syntax:

```
operation ::= `ltl.eventually` $input attr-dict `:` type($input)
```

Checks that the `$input` property will hold at a future time. This operator
is strong: it requires that the `$input` holds after a *finite* number of
cycles. The operator does *not* hold if the `$input` can’t hold in the
future.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-4)

| Operand | Description |
| --- | --- |
| `input` | 1-bit signless integer or LTL sequence type or LTL property type |

#### Results: [¶](#results-5)

| Result | Description |
| --- | --- |
| `result` | LTL property type |

### `ltl.goto_repeat` (circt::ltl::GoToRepeatOp) [¶](#ltlgoto_repeat-circtltlgotorepeatop)

*`goto`-style non-consecutively repeating sequence.*

Syntax:

```
operation ::= `ltl.goto_repeat` $input `,` $base `,` $more attr-dict `:` type($input)
```

Non-consecutive repetition of the `$input` sequence. This must hold between `$base`
and `$base + $more` times in a finite number of evaluations. The final evaluation
in the sequence has to match. The `$base` must be greater than or equal to zero
and the range `$more` can’t be omitted. For example, a !b b b !b !b b c represents
a matching observation of `ltl.goto_repeat %b, 1, 2`, but a !b b b !b !b b !b c doesn’t.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-3)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `base` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `more` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands: [¶](#operands-5)

| Operand | Description |
| --- | --- |
| `input` | 1-bit signless integer or LTL sequence type |

#### Results: [¶](#results-6)

| Result | Description |
| --- | --- |
| `result` | LTL sequence type |

### `ltl.implication` (circt::ltl::ImplicationOp) [¶](#ltlimplication-circtltlimplicationop)

*Only check a property after a sequence matched.*

Syntax:

```
operation ::= `ltl.implication` operands attr-dict `:` type(operands)
```

Preconditions the checking of the `$consequent` property on the
`$antecedent` sequence. In a nutshell, if the `$antecedent` sequence matches
at a given point in time, the `$consequent` property is checked starting at
the point in time at which the matched sequence ends. The result property of
the `ltl.implication` holds if the `$consequent` holds. Conversely, if the
`$antecedent` does *not* match at a given point in time, the result property
trivially holds. This is conceptually identical to the implication operator
→, but with additional temporal semantics.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-6)

| Operand | Description |
| --- | --- |
| `antecedent` | 1-bit signless integer or LTL sequence type |
| `consequent` | 1-bit signless integer or LTL sequence type or LTL property type |

#### Results: [¶](#results-7)

| Result | Description |
| --- | --- |
| `result` | LTL property type |

### `ltl.intersect` (circt::ltl::IntersectOp) [¶](#ltlintersect-circtltlintersectop)

*The intersection of booleans sequences or properties.*

Syntax:

```
operation ::= `ltl.intersect` $inputs attr-dict `:` type($inputs)
```

The intersection of two properties. This checks that both properties both hold
and have the same start and end times. This differs from `ltl.and` which doesn’t
consider the timings of each operand, only their results.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-7)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of 1-bit signless integer or LTL sequence type or LTL property type |

#### Results: [¶](#results-8)

| Result | Description |
| --- | --- |
| `result` | 1-bit signless integer or LTL sequence type or LTL property type |

### `ltl.non_consecutive_repeat` (circt::ltl::NonConsecutiveRepeatOp) [¶](#ltlnon_consecutive_repeat-circtltlnonconsecutiverepeatop)

*`goto`-style non-consecutively repeating sequence.*

Syntax:

```
operation ::= `ltl.non_consecutive_repeat` $input `,` $base `,` $more attr-dict `:` type($input)
```

Non-consecutive repetition of the `$input` sequence. This must hold between `$base`
and `$base + $more` times in a finite number of evaluations. The final evaluation
in the sequence does not have to match. The `$base` must be greater than or equal to zero
and the range `$more` can’t be omitted. For example, both a !b b b !b !b b c and
a !b b b !b !b b !b c represent matching observations of
`ltl.non_consecutive_repeat %b, 1, 2`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-4)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `base` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `more` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands: [¶](#operands-8)

| Operand | Description |
| --- | --- |
| `input` | 1-bit signless integer or LTL sequence type |

#### Results: [¶](#results-9)

| Result | Description |
| --- | --- |
| `result` | LTL sequence type |

### `ltl.not` (circt::ltl::NotOp) [¶](#ltlnot-circtltlnotop)

*A negation of a property.*

Syntax:

```
operation ::= `ltl.not` $input attr-dict `:` type($input)
```

Negates the `$input` property. The resulting property evaluates to true if
`$input` evaluates to false, and it evaluates to false if `$input` evaluates
to true.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-9)

| Operand | Description |
| --- | --- |
| `input` | 1-bit signless integer or LTL sequence type or LTL property type |

#### Results: [¶](#results-10)

| Result | Description |
| --- | --- |
| `result` | LTL property type |

### `ltl.or` (circt::ltl::OrOp) [¶](#ltlor-circtltlorop)

*A disjunction of booleans, sequences, or properties.*

Syntax:

```
operation ::= `ltl.or` $inputs attr-dict `:` type($inputs)
```

If any of the `$inputs` is of type `!ltl.property`, the result of the op is
an `!ltl.property`. Otherwise it is an `!ltl.sequence`.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-10)

| Operand | Description |
| --- | --- |
| `inputs` | variadic of 1-bit signless integer or LTL sequence type or LTL property type |

#### Results: [¶](#results-11)

| Result | Description |
| --- | --- |
| `result` | 1-bit signless integer or LTL sequence type or LTL property type |

### `ltl.past` (circt::ltl::PastOp) [¶](#ltlpast-circtltlpastop)

*Observe the sequence $delay cycles earlier.*

Syntax:

```
operation ::= `ltl.past` $input `,` $delay attr-dict `:` type($input)
```

Semantically works like `ltl.delay %seq, -$delay, 0`

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-5)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `delay` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands: [¶](#operands-11)

| Operand | Description |
| --- | --- |
| `input` | a signless integer bitvector |

#### Results: [¶](#results-12)

| Result | Description |
| --- | --- |
| `result` | a signless integer bitvector |

### `ltl.repeat` (circt::ltl::RepeatOp) [¶](#ltlrepeat-circtltlrepeatop)

*Repeats a sequence by a number of times.*

Syntax:

```
operation ::= `ltl.repeat` $input `,` $base (`,` $more^)? attr-dict `:` type($input)
```

Repeat the `$input` sequence at least `$base` times, at most `$base` +
`$more` times. The number must be greater than or equal to zero. Omitting
`$more` indicates an unbounded but finite repetition. For example:

* `ltl.repeat %seq, 2, 0` repeats `%seq` exactly 2 times.
* `ltl.repeat %seq, 2, 2` repeats `%seq` 2, 3, or 4 times.
* `ltl.repeat %seq, 2` repeats `%seq` 2 or more times. The number of times
  is unbounded but finite.
* `ltl.repeat %seq, 0, 0` represents an empty sequence.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes-6)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `base` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `more` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands: [¶](#operands-12)

| Operand | Description |
| --- | --- |
| `input` | 1-bit signless integer or LTL sequence type |

#### Results: [¶](#results-13)

| Result | Description |
| --- | --- |
| `result` | LTL sequence type |

### `ltl.until` (circt::ltl::UntilOp) [¶](#ltluntil-circtltluntilop)

*Property always holds until another property holds.*

Syntax:

```
operation ::= `ltl.until` operands attr-dict `:` type(operands)
```

Checks that the `$input` property always holds until the `$condition`
property holds once. This operator is weak: the property will hold even if
`$input` always holds and `$condition` never holds. This operator is
nonoverlapping: `$input` does not have to hold when `$condition` holds.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands-13)

| Operand | Description |
| --- | --- |
| `input` | 1-bit signless integer or LTL sequence type or LTL property type |
| `condition` | 1-bit signless integer or LTL sequence type or LTL property type |

#### Results: [¶](#results-14)

| Result | Description |
| --- | --- |
| `result` | LTL property type |

 [Prev - LLHD Dialect](https://circt.llvm.org/docs/Dialects/LLHD/ "LLHD Dialect")
[Next - Moore Dialect](https://circt.llvm.org/docs/Dialects/Moore/ "Moore Dialect") 

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