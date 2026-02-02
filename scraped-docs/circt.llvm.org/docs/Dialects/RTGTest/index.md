'rtgtest' Dialect - CIRCT

CIRCT
=====

Circuit IR Compilers and Tools

* Community
  + [Forums](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/40)
  + [Chat](https://discord.gg/xS7Z362)
* [Source](https://github.com/llvm/circt/tree/main/)
  + [Doxygen](/doxygen/)
  + [GitHub](https://github.com/llvm/circt/tree/main/)

'rtgtest' Dialect
=================

*Types and operations for random test generation testing*

This dialect defines the `rtgtest` dialect, which provides a set of
operation definitions to test the RTG dialect.

* [Operations](#operations)
  + [`rtgtest.constant_test` (::circt::rtgtest::ConstantTestOp)](#rtgtestconstant_test-circtrtgtestconstanttestop)
  + [`rtgtest.get_hartid` (::circt::rtgtest::GetHartIdOp)](#rtgtestget_hartid-circtrtgtestgethartidop)
  + [`rtgtest.implicit_constraint_op` (::circt::rtgtest::ImplicitConstraintTestOp)](#rtgtestimplicit_constraint_op-circtrtgtestimplicitconstrainttestop)
  + [`rtgtest.rv32i.add` (::circt::rtgtest::ADD)](#rtgtestrv32iadd-circtrtgtestadd)
  + [`rtgtest.rv32i.addi` (::circt::rtgtest::ADDI)](#rtgtestrv32iaddi-circtrtgtestaddi)
  + [`rtgtest.rv32i.and` (::circt::rtgtest::AND)](#rtgtestrv32iand-circtrtgtestand)
  + [`rtgtest.rv32i.andi` (::circt::rtgtest::ANDI)](#rtgtestrv32iandi-circtrtgtestandi)
  + [`rtgtest.rv32i.auipc` (::circt::rtgtest::AUIPC)](#rtgtestrv32iauipc-circtrtgtestauipc)
  + [`rtgtest.rv32i.beq` (::circt::rtgtest::BEQ)](#rtgtestrv32ibeq-circtrtgtestbeq)
  + [`rtgtest.rv32i.bge` (::circt::rtgtest::BGE)](#rtgtestrv32ibge-circtrtgtestbge)
  + [`rtgtest.rv32i.bgeu` (::circt::rtgtest::BGEU)](#rtgtestrv32ibgeu-circtrtgtestbgeu)
  + [`rtgtest.rv32i.blt` (::circt::rtgtest::BLT)](#rtgtestrv32iblt-circtrtgtestblt)
  + [`rtgtest.rv32i.bltu` (::circt::rtgtest::BLTU)](#rtgtestrv32ibltu-circtrtgtestbltu)
  + [`rtgtest.rv32i.bne` (::circt::rtgtest::BNE)](#rtgtestrv32ibne-circtrtgtestbne)
  + [`rtgtest.rv32i.ebreak` (::circt::rtgtest::EBREAKOp)](#rtgtestrv32iebreak-circtrtgtestebreakop)
  + [`rtgtest.rv32i.ecall` (::circt::rtgtest::ECALLOp)](#rtgtestrv32iecall-circtrtgtestecallop)
  + [`rtgtest.rv32i.jal` (::circt::rtgtest::JAL)](#rtgtestrv32ijal-circtrtgtestjal)
  + [`rtgtest.rv32i.jalr` (::circt::rtgtest::JALROp)](#rtgtestrv32ijalr-circtrtgtestjalrop)
  + [`rtgtest.rv32i.la` (::circt::rtgtest::LA)](#rtgtestrv32ila-circtrtgtestla)
  + [`rtgtest.rv32i.lb` (::circt::rtgtest::LBOp)](#rtgtestrv32ilb-circtrtgtestlbop)
  + [`rtgtest.rv32i.lbu` (::circt::rtgtest::LBUOp)](#rtgtestrv32ilbu-circtrtgtestlbuop)
  + [`rtgtest.rv32i.lh` (::circt::rtgtest::LHOp)](#rtgtestrv32ilh-circtrtgtestlhop)
  + [`rtgtest.rv32i.lhu` (::circt::rtgtest::LHUOp)](#rtgtestrv32ilhu-circtrtgtestlhuop)
  + [`rtgtest.rv32i.lui` (::circt::rtgtest::LUI)](#rtgtestrv32ilui-circtrtgtestlui)
  + [`rtgtest.rv32i.lw` (::circt::rtgtest::LWOp)](#rtgtestrv32ilw-circtrtgtestlwop)
  + [`rtgtest.rv32i.or` (::circt::rtgtest::OR)](#rtgtestrv32ior-circtrtgtestor)
  + [`rtgtest.rv32i.ori` (::circt::rtgtest::ORI)](#rtgtestrv32iori-circtrtgtestori)
  + [`rtgtest.rv32i.sb` (::circt::rtgtest::SB)](#rtgtestrv32isb-circtrtgtestsb)
  + [`rtgtest.rv32i.sh` (::circt::rtgtest::SH)](#rtgtestrv32ish-circtrtgtestsh)
  + [`rtgtest.rv32i.sll` (::circt::rtgtest::SLL)](#rtgtestrv32isll-circtrtgtestsll)
  + [`rtgtest.rv32i.slli` (::circt::rtgtest::SLLI)](#rtgtestrv32islli-circtrtgtestslli)
  + [`rtgtest.rv32i.slt` (::circt::rtgtest::SLT)](#rtgtestrv32islt-circtrtgtestslt)
  + [`rtgtest.rv32i.slti` (::circt::rtgtest::SLTI)](#rtgtestrv32islti-circtrtgtestslti)
  + [`rtgtest.rv32i.sltiu` (::circt::rtgtest::SLTIU)](#rtgtestrv32isltiu-circtrtgtestsltiu)
  + [`rtgtest.rv32i.sltu` (::circt::rtgtest::SLTU)](#rtgtestrv32isltu-circtrtgtestsltu)
  + [`rtgtest.rv32i.sra` (::circt::rtgtest::SRA)](#rtgtestrv32isra-circtrtgtestsra)
  + [`rtgtest.rv32i.srai` (::circt::rtgtest::SRAI)](#rtgtestrv32israi-circtrtgtestsrai)
  + [`rtgtest.rv32i.srl` (::circt::rtgtest::SRL)](#rtgtestrv32isrl-circtrtgtestsrl)
  + [`rtgtest.rv32i.srli` (::circt::rtgtest::SRLI)](#rtgtestrv32isrli-circtrtgtestsrli)
  + [`rtgtest.rv32i.sub` (::circt::rtgtest::SUB)](#rtgtestrv32isub-circtrtgtestsub)
  + [`rtgtest.rv32i.sw` (::circt::rtgtest::SW)](#rtgtestrv32isw-circtrtgtestsw)
  + [`rtgtest.rv32i.xor` (::circt::rtgtest::XOR)](#rtgtestrv32ixor-circtrtgtestxor)
  + [`rtgtest.rv32i.xori` (::circt::rtgtest::XORI)](#rtgtestrv32ixori-circtrtgtestxori)
* [Attributes](#attributes-2)
  + [CPUAttr](#cpuattr)
  + [RegA0Attr](#rega0attr)
  + [RegA1Attr](#rega1attr)
  + [RegA2Attr](#rega2attr)
  + [RegA3Attr](#rega3attr)
  + [RegA4Attr](#rega4attr)
  + [RegA5Attr](#rega5attr)
  + [RegA6Attr](#rega6attr)
  + [RegA7Attr](#rega7attr)
  + [RegF0Attr](#regf0attr)
  + [RegGpAttr](#reggpattr)
  + [RegRaAttr](#regraattr)
  + [RegS0Attr](#regs0attr)
  + [RegS1Attr](#regs1attr)
  + [RegS2Attr](#regs2attr)
  + [RegS3Attr](#regs3attr)
  + [RegS4Attr](#regs4attr)
  + [RegS5Attr](#regs5attr)
  + [RegS6Attr](#regs6attr)
  + [RegS7Attr](#regs7attr)
  + [RegS8Attr](#regs8attr)
  + [RegS9Attr](#regs9attr)
  + [RegS10Attr](#regs10attr)
  + [RegS11Attr](#regs11attr)
  + [RegSpAttr](#regspattr)
  + [RegT0Attr](#regt0attr)
  + [RegT1Attr](#regt1attr)
  + [RegT2Attr](#regt2attr)
  + [RegT3Attr](#regt3attr)
  + [RegT4Attr](#regt4attr)
  + [RegT5Attr](#regt5attr)
  + [RegT6Attr](#regt6attr)
  + [RegTpAttr](#regtpattr)
  + [RegZeroAttr](#regzeroattr)
* [Types](#types)
  + [CPUType](#cputype)
  + [FloatRegisterType](#floatregistertype)
  + [IntegerRegisterType](#integerregistertype)

Operations [¶](#operations)
---------------------------

### `rtgtest.constant_test` (::circt::rtgtest::ConstantTestOp) [¶](#rtgtestconstant_test-circtrtgtestconstanttestop)

Syntax:

```
operation ::= `rtgtest.constant_test` type($result) attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](#attributes)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::Attribute | any attribute |

#### Results: [¶](#results)

| Result | Description |
| --- | --- |
| `result` | any type |

### `rtgtest.get_hartid` (::circt::rtgtest::GetHartIdOp) [¶](#rtgtestget_hartid-circtrtgtestgethartidop)

Syntax:

```
operation ::= `rtgtest.get_hartid` $cpu attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](#operands)

| Operand | Description |
| --- | --- |
| `cpu` | handle to a specific CPU |

#### Results: [¶](#results-1)

| Result | Description |
| --- | --- |
| `hartid` | index |

### `rtgtest.implicit_constraint_op` (::circt::rtgtest::ImplicitConstraintTestOp) [¶](#rtgtestimplicit_constraint_op-circtrtgtestimplicitconstrainttestop)

Syntax:

```
operation ::= `rtgtest.implicit_constraint_op` (`implicit_constraint` $implicitConstraint^)? attr-dict
```

Interfaces: `ImplicitConstraintOpInterface`

#### Attributes: [¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `implicitConstraint` | ::mlir::UnitAttr | unit attribute |

### `rtgtest.rv32i.add` (::circt::rtgtest::ADD) [¶](#rtgtestrv32iadd-circtrtgtestadd)

Syntax:

```
operation ::= `rtgtest.rv32i.add` $rd `,` $rs1 `,` $rs2 attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-1)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |

### `rtgtest.rv32i.addi` (::circt::rtgtest::ADDI) [¶](#rtgtestrv32iaddi-circtrtgtestaddi)

Syntax:

```
operation ::= `rtgtest.rv32i.addi` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-2)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.and` (::circt::rtgtest::AND) [¶](#rtgtestrv32iand-circtrtgtestand)

Syntax:

```
operation ::= `rtgtest.rv32i.and` $rd `,` $rs1 `,` $rs2 attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-3)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |

### `rtgtest.rv32i.andi` (::circt::rtgtest::ANDI) [¶](#rtgtestrv32iandi-circtrtgtestandi)

Syntax:

```
operation ::= `rtgtest.rv32i.andi` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-4)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.auipc` (::circt::rtgtest::AUIPC) [¶](#rtgtestrv32iauipc-circtrtgtestauipc)

Syntax:

```
operation ::= `rtgtest.rv32i.auipc` $rd `,` $imm `:` type($imm) attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-5)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `imm` | a 32-bit immediate or a reference to a label |

### `rtgtest.rv32i.beq` (::circt::rtgtest::BEQ) [¶](#rtgtestrv32ibeq-circtrtgtestbeq)

Syntax:

```
operation ::= `rtgtest.rv32i.beq` $rs1 `,` $rs2 `,` $imm `:` qualified(type($imm)) attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-6)

| Operand | Description |
| --- | --- |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |
| `imm` | a 13-bit immediate or a reference to a label |

### `rtgtest.rv32i.bge` (::circt::rtgtest::BGE) [¶](#rtgtestrv32ibge-circtrtgtestbge)

Syntax:

```
operation ::= `rtgtest.rv32i.bge` $rs1 `,` $rs2 `,` $imm `:` qualified(type($imm)) attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-7)

| Operand | Description |
| --- | --- |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |
| `imm` | a 13-bit immediate or a reference to a label |

### `rtgtest.rv32i.bgeu` (::circt::rtgtest::BGEU) [¶](#rtgtestrv32ibgeu-circtrtgtestbgeu)

Syntax:

```
operation ::= `rtgtest.rv32i.bgeu` $rs1 `,` $rs2 `,` $imm `:` qualified(type($imm)) attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-8)

| Operand | Description |
| --- | --- |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |
| `imm` | a 13-bit immediate or a reference to a label |

### `rtgtest.rv32i.blt` (::circt::rtgtest::BLT) [¶](#rtgtestrv32iblt-circtrtgtestblt)

Syntax:

```
operation ::= `rtgtest.rv32i.blt` $rs1 `,` $rs2 `,` $imm `:` qualified(type($imm)) attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-9)

| Operand | Description |
| --- | --- |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |
| `imm` | a 13-bit immediate or a reference to a label |

### `rtgtest.rv32i.bltu` (::circt::rtgtest::BLTU) [¶](#rtgtestrv32ibltu-circtrtgtestbltu)

Syntax:

```
operation ::= `rtgtest.rv32i.bltu` $rs1 `,` $rs2 `,` $imm `:` qualified(type($imm)) attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-10)

| Operand | Description |
| --- | --- |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |
| `imm` | a 13-bit immediate or a reference to a label |

### `rtgtest.rv32i.bne` (::circt::rtgtest::BNE) [¶](#rtgtestrv32ibne-circtrtgtestbne)

Syntax:

```
operation ::= `rtgtest.rv32i.bne` $rs1 `,` $rs2 `,` $imm `:` qualified(type($imm)) attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-11)

| Operand | Description |
| --- | --- |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |
| `imm` | a 13-bit immediate or a reference to a label |

### `rtgtest.rv32i.ebreak` (::circt::rtgtest::EBREAKOp) [¶](#rtgtestrv32iebreak-circtrtgtestebreakop)

Syntax:

```
operation ::= `rtgtest.rv32i.ebreak` attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

### `rtgtest.rv32i.ecall` (::circt::rtgtest::ECALLOp) [¶](#rtgtestrv32iecall-circtrtgtestecallop)

Syntax:

```
operation ::= `rtgtest.rv32i.ecall` attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

### `rtgtest.rv32i.jal` (::circt::rtgtest::JAL) [¶](#rtgtestrv32ijal-circtrtgtestjal)

Syntax:

```
operation ::= `rtgtest.rv32i.jal` $rd `,` $imm `:` type($imm) attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-12)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `imm` | a 21-bit immediate or a reference to a label |

### `rtgtest.rv32i.jalr` (::circt::rtgtest::JALROp) [¶](#rtgtestrv32ijalr-circtrtgtestjalrop)

Syntax:

```
operation ::= `rtgtest.rv32i.jalr` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-13)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.la` (::circt::rtgtest::LA) [¶](#rtgtestrv32ila-circtrtgtestla)

Syntax:

```
operation ::= `rtgtest.rv32i.la` $rd `,` $mem `:` type($mem) attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-14)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `mem` | handle to a memory or an ISA immediate or a reference to a label |

### `rtgtest.rv32i.lb` (::circt::rtgtest::LBOp) [¶](#rtgtestrv32ilb-circtrtgtestlbop)

Syntax:

```
operation ::= `rtgtest.rv32i.lb` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-15)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.lbu` (::circt::rtgtest::LBUOp) [¶](#rtgtestrv32ilbu-circtrtgtestlbuop)

Syntax:

```
operation ::= `rtgtest.rv32i.lbu` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-16)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.lh` (::circt::rtgtest::LHOp) [¶](#rtgtestrv32ilh-circtrtgtestlhop)

Syntax:

```
operation ::= `rtgtest.rv32i.lh` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-17)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.lhu` (::circt::rtgtest::LHUOp) [¶](#rtgtestrv32ilhu-circtrtgtestlhuop)

Syntax:

```
operation ::= `rtgtest.rv32i.lhu` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-18)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.lui` (::circt::rtgtest::LUI) [¶](#rtgtestrv32ilui-circtrtgtestlui)

Syntax:

```
operation ::= `rtgtest.rv32i.lui` $rd `,` $imm `:` type($imm) attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-19)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `imm` | a 32-bit immediate or a reference to a label |

### `rtgtest.rv32i.lw` (::circt::rtgtest::LWOp) [¶](#rtgtestrv32ilw-circtrtgtestlwop)

Syntax:

```
operation ::= `rtgtest.rv32i.lw` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-20)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.or` (::circt::rtgtest::OR) [¶](#rtgtestrv32ior-circtrtgtestor)

Syntax:

```
operation ::= `rtgtest.rv32i.or` $rd `,` $rs1 `,` $rs2 attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-21)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |

### `rtgtest.rv32i.ori` (::circt::rtgtest::ORI) [¶](#rtgtestrv32iori-circtrtgtestori)

Syntax:

```
operation ::= `rtgtest.rv32i.ori` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-22)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.sb` (::circt::rtgtest::SB) [¶](#rtgtestrv32isb-circtrtgtestsb)

Syntax:

```
operation ::= `rtgtest.rv32i.sb` $rs1 `,` $rs2 `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-23)

| Operand | Description |
| --- | --- |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.sh` (::circt::rtgtest::SH) [¶](#rtgtestrv32ish-circtrtgtestsh)

Syntax:

```
operation ::= `rtgtest.rv32i.sh` $rs1 `,` $rs2 `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-24)

| Operand | Description |
| --- | --- |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.sll` (::circt::rtgtest::SLL) [¶](#rtgtestrv32isll-circtrtgtestsll)

Syntax:

```
operation ::= `rtgtest.rv32i.sll` $rd `,` $rs1 `,` $rs2 attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-25)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |

### `rtgtest.rv32i.slli` (::circt::rtgtest::SLLI) [¶](#rtgtestrv32islli-circtrtgtestslli)

Syntax:

```
operation ::= `rtgtest.rv32i.slli` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-26)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 5-bit immediate |

### `rtgtest.rv32i.slt` (::circt::rtgtest::SLT) [¶](#rtgtestrv32islt-circtrtgtestslt)

Syntax:

```
operation ::= `rtgtest.rv32i.slt` $rd `,` $rs1 `,` $rs2 attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-27)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |

### `rtgtest.rv32i.slti` (::circt::rtgtest::SLTI) [¶](#rtgtestrv32islti-circtrtgtestslti)

Syntax:

```
operation ::= `rtgtest.rv32i.slti` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-28)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.sltiu` (::circt::rtgtest::SLTIU) [¶](#rtgtestrv32isltiu-circtrtgtestsltiu)

Syntax:

```
operation ::= `rtgtest.rv32i.sltiu` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-29)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.sltu` (::circt::rtgtest::SLTU) [¶](#rtgtestrv32isltu-circtrtgtestsltu)

Syntax:

```
operation ::= `rtgtest.rv32i.sltu` $rd `,` $rs1 `,` $rs2 attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-30)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |

### `rtgtest.rv32i.sra` (::circt::rtgtest::SRA) [¶](#rtgtestrv32isra-circtrtgtestsra)

Syntax:

```
operation ::= `rtgtest.rv32i.sra` $rd `,` $rs1 `,` $rs2 attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-31)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |

### `rtgtest.rv32i.srai` (::circt::rtgtest::SRAI) [¶](#rtgtestrv32israi-circtrtgtestsrai)

Syntax:

```
operation ::= `rtgtest.rv32i.srai` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-32)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 5-bit immediate |

### `rtgtest.rv32i.srl` (::circt::rtgtest::SRL) [¶](#rtgtestrv32isrl-circtrtgtestsrl)

Syntax:

```
operation ::= `rtgtest.rv32i.srl` $rd `,` $rs1 `,` $rs2 attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-33)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |

### `rtgtest.rv32i.srli` (::circt::rtgtest::SRLI) [¶](#rtgtestrv32isrli-circtrtgtestsrli)

Syntax:

```
operation ::= `rtgtest.rv32i.srli` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-34)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 5-bit immediate |

### `rtgtest.rv32i.sub` (::circt::rtgtest::SUB) [¶](#rtgtestrv32isub-circtrtgtestsub)

Syntax:

```
operation ::= `rtgtest.rv32i.sub` $rd `,` $rs1 `,` $rs2 attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-35)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |

### `rtgtest.rv32i.sw` (::circt::rtgtest::SW) [¶](#rtgtestrv32isw-circtrtgtestsw)

Syntax:

```
operation ::= `rtgtest.rv32i.sw` $rs1 `,` $rs2 `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-36)

| Operand | Description |
| --- | --- |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |
| `imm` | a 12-bit immediate |

### `rtgtest.rv32i.xor` (::circt::rtgtest::XOR) [¶](#rtgtestrv32ixor-circtrtgtestxor)

Syntax:

```
operation ::= `rtgtest.rv32i.xor` $rd `,` $rs1 `,` $rs2 attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-37)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs1` | represents an integer register |
| `rs2` | represents an integer register |

### `rtgtest.rv32i.xori` (::circt::rtgtest::XORI) [¶](#rtgtestrv32ixori-circtrtgtestxori)

Syntax:

```
operation ::= `rtgtest.rv32i.xori` $rd `,` $rs `,` $imm attr-dict
```

Traits: `InstructionOpAdaptorTrait`

Interfaces: `InstructionOpInterface`

#### Operands: [¶](#operands-38)

| Operand | Description |
| --- | --- |
| `rd` | represents an integer register |
| `rs` | represents an integer register |
| `imm` | a 12-bit immediate |

Attributes [¶](#attributes-2)
-----------------------------

### CPUAttr [¶](#cpuattr)

*This attribute represents a CPU referred to by the core ID*

Syntax:

```
#rtgtest.cpu<
  size_t   # id
>
```

#### Parameters: [¶](#parameters)

| Parameter | C++ type | Description |
| --- | --- | --- |
| id | `size_t` |  |

### RegA0Attr [¶](#rega0attr)

Syntax: `#rtgtest.a0`

### RegA1Attr [¶](#rega1attr)

Syntax: `#rtgtest.a1`

### RegA2Attr [¶](#rega2attr)

Syntax: `#rtgtest.a2`

### RegA3Attr [¶](#rega3attr)

Syntax: `#rtgtest.a3`

### RegA4Attr [¶](#rega4attr)

Syntax: `#rtgtest.a4`

### RegA5Attr [¶](#rega5attr)

Syntax: `#rtgtest.a5`

### RegA6Attr [¶](#rega6attr)

Syntax: `#rtgtest.a6`

### RegA7Attr [¶](#rega7attr)

Syntax: `#rtgtest.a7`

### RegF0Attr [¶](#regf0attr)

Syntax: `#rtgtest.f0`

### RegGpAttr [¶](#reggpattr)

Syntax: `#rtgtest.gp`

### RegRaAttr [¶](#regraattr)

Syntax: `#rtgtest.ra`

### RegS0Attr [¶](#regs0attr)

Syntax: `#rtgtest.s0`

### RegS1Attr [¶](#regs1attr)

Syntax: `#rtgtest.s1`

### RegS2Attr [¶](#regs2attr)

Syntax: `#rtgtest.s2`

### RegS3Attr [¶](#regs3attr)

Syntax: `#rtgtest.s3`

### RegS4Attr [¶](#regs4attr)

Syntax: `#rtgtest.s4`

### RegS5Attr [¶](#regs5attr)

Syntax: `#rtgtest.s5`

### RegS6Attr [¶](#regs6attr)

Syntax: `#rtgtest.s6`

### RegS7Attr [¶](#regs7attr)

Syntax: `#rtgtest.s7`

### RegS8Attr [¶](#regs8attr)

Syntax: `#rtgtest.s8`

### RegS9Attr [¶](#regs9attr)

Syntax: `#rtgtest.s9`

### RegS10Attr [¶](#regs10attr)

Syntax: `#rtgtest.s10`

### RegS11Attr [¶](#regs11attr)

Syntax: `#rtgtest.s11`

### RegSpAttr [¶](#regspattr)

Syntax: `#rtgtest.sp`

### RegT0Attr [¶](#regt0attr)

Syntax: `#rtgtest.t0`

### RegT1Attr [¶](#regt1attr)

Syntax: `#rtgtest.t1`

### RegT2Attr [¶](#regt2attr)

Syntax: `#rtgtest.t2`

### RegT3Attr [¶](#regt3attr)

Syntax: `#rtgtest.t3`

### RegT4Attr [¶](#regt4attr)

Syntax: `#rtgtest.t4`

### RegT5Attr [¶](#regt5attr)

Syntax: `#rtgtest.t5`

### RegT6Attr [¶](#regt6attr)

Syntax: `#rtgtest.t6`

### RegTpAttr [¶](#regtpattr)

Syntax: `#rtgtest.tp`

### RegZeroAttr [¶](#regzeroattr)

Syntax: `#rtgtest.zero`

Types [¶](#types)
-----------------

### CPUType [¶](#cputype)

*Handle to a specific CPU*

Syntax: `!rtgtest.cpu`

This type implements a specific context resource to test RTG operations
taking context resources as operands (such as `on_context`) and other things
requiring a concrete instance of a `ContextResourceTypeInterface`.

### FloatRegisterType [¶](#floatregistertype)

*Represents a floating-point register*

Syntax: `!rtgtest.freg`

### IntegerRegisterType [¶](#integerregistertype)

*Represents an integer register*

Syntax: `!rtgtest.ireg`

 [Prev - Pipeline Dialect Rationale](https://circt.llvm.org/docs/Dialects/Pipeline/RationalePipeline/ "Pipeline Dialect Rationale")
[Next - 'seq' Dialect](https://circt.llvm.org/docs/Dialects/Seq/ "'seq' Dialect") 

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