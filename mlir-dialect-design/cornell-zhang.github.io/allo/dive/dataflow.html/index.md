Dataflow Programming — Allo Documentation

[|||](# "Toggle sidebar")

[Allo Documentation](../index.html "Go to homepage")
====================================================




### Quick search

Getting Started

* [Installation](../setup/index.html)

Tutorials

* [Getting Started](../gallery/tutorial_01_get_started.html)
* [Vivado/Vitis HLS Backend](../gallery/tutorial_02_vhls.html)

Deep Dive

* [Frontend Syntax Guide](frontend_syntax.html)
* [Dataflow Programming](#)
* [Data Types and Type Casting](../gallery/dive_01_data_types.html)
* [Template Kernels](../gallery/dive_02_template.html)
* [Kernel Composition](../gallery/dive_03_composition.html)
* [IP Integration](ip.html)
* [PyTorch Integration](pytorch.html)
* [Equivalence Checking](../gallery/dive_05_verifier.html)
* [Other Features](../gallery/dive_04_features.html)

Backends

* [LLVM (CPU)](../backends/llvm.html)
* [AMD Vitis HLS (FPGA)](../backends/vitis.html)
* [Siemens Catapult HLS (FPGA)](../backends/catapult.html)
* [RapidStream TAPA (FPGA)](../backends/tapa.html)
* [Google XLS (ASIC)](../backends/xls.html)
* [Multi-Threaded Simulator (CPU)](../backends/simulator.html)
* [AMD MLIR-AIE (AI Engine)](../backends/aie/index.html)

Developer Guide

* [Developer Setup](../developer/index.html)
* [IR Builder Walkthrough](../gallery/developer_01_ir_builder.html)
* [MLIR Translation Guide](../gallery/developer_02_mlir.html)

Python API

* [Schedule Primitives](../api/index.html)
* [Data Types](../api/index.html#data-types)

Dataflow Programming[¶](#dataflow-programming "Link to this heading")
=====================================================================

This document describes the dataflow programming model in Allo, which enables
spatial parallelism through explicit kernel decomposition and stream-based
communication.

Table of Contents

* [Overview](#overview)

  + [Key Concepts](#key-concepts)
* [Getting Started](#getting-started)

  + [Import the Dataflow Module](#import-the-dataflow-module)
  + [Basic Structure](#basic-structure)
* [Regions](#regions)

  + [Region Declaration](#region-declaration)
  + [Parameterized Regions](#parameterized-regions)
* [Kernels](#kernels)

  + [Kernel Declaration](#kernel-declaration)
  + [Single Kernel Instance](#single-kernel-instance)
  + [Multi-Dimensional Kernel Grid](#multi-dimensional-kernel-grid)
  + [Getting Kernel Position](#getting-kernel-position)
* [Streams](#streams)

  + [Stream Declaration](#stream-declaration)
  + [Stream Arrays](#stream-arrays)
  + [Stream of Blocks](#stream-of-blocks)
  + [Stream Operations](#stream-operations)
* [Systolic Array Patterns](#systolic-array-patterns)

  + [Basic Systolic Structure](#basic-systolic-structure)
* [Tiled Computation](#tiled-computation)
* [Hierarchical Regions](#hierarchical-regions)
* [Building and Execution](#building-and-execution)

  + [Build for Simulation](#build-for-simulation)
  + [Build for LLVM](#build-for-llvm)
  + [Build for HLS](#build-for-hls)
  + [Customization (Schedule Primitives)](#customization-schedule-primitives)
* [Complete Example: Producer-Consumer](#complete-example-producer-consumer)
* [Tensor Layouts](#tensor-layouts)

  + [Layout Concepts](#layout-concepts)
  + [Import and Setup](#import-and-setup)
  + [Applying Layouts to Kernel Arguments](#applying-layouts-to-kernel-arguments)
  + [Layout Patterns](#layout-patterns)
  + [Understanding Mesh Axis Mapping](#understanding-mesh-axis-mapping)
  + [How Layouts Compute Local Shapes](#how-layouts-compute-local-shapes)
* [Best Practices](#best-practices)

[Overview](#id1)[¶](#overview "Link to this heading")
-----------------------------------------------------

Modern deep learning workloads are increasingly **memory-bound**, with many kernels
stalling on data movement rather than computation. While dataflow accelerators
(FPGAs, NPUs) offer on-chip streaming to mitigate off-chip bandwidth limitations,
existing programming models fall into two extremes:

* **Low-level interfaces** provide fine-grained control but impose significant
  development overhead and poor portability
* **High-level tile-based languages** abstract away communication, which restricts
  optimization of complex dataflow patterns and forces compilers to reconstruct
  the intended dataflow

The Allo dataflow programming model provides a **middle ground** by elevating
data communication and sharding to **first-class constructs**. Developers express
programs as a graph of **tasks (kernels)** connected via explicit **stream types**,
enabling efficient on-chip data reuse without spilling back to DRAM.

### [Key Concepts](#id2)[¶](#key-concepts "Link to this heading")

The dataflow model in Allo consists of three core abstractions:

* **Regions**: Define the dataflow graph with inputs, outputs, and streams
* **Kernels**: Units of computation mapped onto processing elements (PEs)
* **Streams**: FIFO-based communication channels between kernels

[Getting Started](#id3)[¶](#getting-started "Link to this heading")
-------------------------------------------------------------------

### [Import the Dataflow Module](#id4)[¶](#import-the-dataflow-module "Link to this heading")

```
import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df

# For AIE targets with tensor distribution:
from allo.memory import Layout
S = Layout.Shard       # Shorthand for sharding
R = Layout.Replicate   # Shorthand for replication
```

### [Basic Structure](#id5)[¶](#basic-structure "Link to this heading")

A dataflow program consists of:

1. A **region** decorated with `@df.region()` that defines inputs/outputs
2. One or more **kernels** decorated with `@df.kernel()` inside the region
3. **Streams** declared in the region for inter-kernel communication

```
@df.region()
def top(A: float32[M, N], B: float32[M, N]):
    pipe: Stream[float32, 4]  # Stream with depth 4

    @df.kernel(mapping=[1], args=[A])
    def producer(local_A: float32[M, N]):
        # Kernel logic
        pass

    @df.kernel(mapping=[1], args=[B])
    def consumer(local_B: float32[M, N]):
        # Kernel logic
        pass
```

[Regions](#id6)[¶](#regions "Link to this heading")
---------------------------------------------------

A region defines the top-level dataflow graph with its inputs, outputs, and
internal communication channels.

### [Region Declaration](#id7)[¶](#region-declaration "Link to this heading")

```
@df.region()
def top(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    # Stream declarations
    # Kernel definitions
    pass
```

The region signature specifies the external memory interfaces (inputs and outputs).

Important

Inside a `@df.region()`, only **stream variable declarations** and
**``@df.kernel`` definitions** are allowed. No other function calls or
computations should appear directly in the region body. All computation
logic must be placed inside kernels.

### [Parameterized Regions](#id8)[¶](#parameterized-regions "Link to this heading")

Regions can be parameterized with type parameters for reusability:

```
@df.region()
def inner[P0, P1](A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def gemm(local_A: float32[M, K], local_B: float32[K, N], local_C: float32[M, N]):
        pi, pj = df.get_pid()
        # Tiled computation using pi, pj
        pass

# Instantiate with different parallelism:
inner[2, 2](A, B, C1)  # 2x2 grid
inner[4, 4](A, B, C2)  # 4x4 grid
```

[Kernels](#id9)[¶](#kernels "Link to this heading")
---------------------------------------------------

Kernels are the basic units of computation in a dataflow graph.

### [Kernel Declaration](#id10)[¶](#kernel-declaration "Link to this heading")

```
@df.kernel(mapping=[P0, P1], args=[A, B, C])
def kernel_name(local_A: Ty[...], local_B: Ty[...], local_C: Ty[...]):
    # Kernel body
    pass
```

**Parameters:**

* `mapping`: A list specifying the kernel’s parallel instantiation shape
* `args`: A list of region-level arguments accessible by this kernel.
  The element at index `i` in the `args` list is passed as the `i`-th argument to the kernel function (so the number and argument type must match).
  `args` is optional and can be omitted if the kernel takes no arguments.

### [Single Kernel Instance](#id11)[¶](#single-kernel-instance "Link to this heading")

For a single kernel instance, use `mapping=[1]`:

```
@df.kernel(mapping=[1], args=[A])
def producer(local_A: float32[M, N]):
    for i, j in allo.grid(M, N):
        pipe.put(local_A[i, j])
```

### [Multi-Dimensional Kernel Grid](#id12)[¶](#multi-dimensional-kernel-grid "Link to this heading")

For multi-kernel arrays (e.g., systolic arrays), specify multiple dimensions:

```
@df.kernel(mapping=[P0, P1], args=[A, B, C])
def gemm(local_A: float32[M, K], local_B: float32[K, N], local_C: float32[M, N]):
    pi, pj = df.get_pid()  # Get kernel's position in the grid
    # Kernel executes at position (pi, pj)
    pass
```

### [Getting Kernel Position](#id13)[¶](#getting-kernel-position "Link to this heading")

Use `df.get_pid()` to retrieve the kernel’s position in the mapping grid, which returns a per-kernel compile-time constant variable:

```
# For 1D mapping
i = df.get_pid()

# For 2D mapping
pi, pj = df.get_pid()

# For 3D mapping
pi, pj, pk = df.get_pid()
```

[Streams](#id14)[¶](#streams "Link to this heading")
----------------------------------------------------

Streams provide FIFO-based communication between kernels.

### [Stream Declaration](#id15)[¶](#stream-declaration "Link to this heading")

Declare streams inside a region with their element type and buffer depth:

```
# Basic scalar stream
pipe: Stream[float32, 4]  # Stream of float32 with depth 4

# Stream of unsigned integers
stream: Stream[UInt(16), 4]
```

### [Stream Arrays](#id16)[¶](#stream-arrays "Link to this heading")

For multi-kernel designs, declare arrays of streams:

```
# 2D array of streams for systolic array communication
fifo_A: Stream[float32, 4][P0, P1]
fifo_B: Stream[float32, 4][P0, P1]
```

### [Stream of Blocks](#id17)[¶](#stream-of-blocks "Link to this heading")

Streams can carry tensor blocks (for coarse-grained data movement):

```
# Stream where each element is a 4x4 block
pipe: Stream[int16[M, N], 4]

# Stream of 1D blocks
pipe: Stream[float32[M], 4]

# Array of block streams
pipe: Stream[float32[Mt, Nt], 2][P0, P1]
```

### [Stream Operations](#id18)[¶](#stream-operations "Link to this heading")

**Put (Send):**

```
# Send a scalar value
pipe.put(value)

# Send to a specific stream in an array
fifo_A[i, j + 1].put(a)

# Send a block
block: float32[M, N] = 0
for m, n in allo.grid(M, N):
    block[m, n] = local_A[i * M + m, n]
pipe.put(block)
```

**Get (Receive):**

```
# Receive a scalar value
data = pipe.get()

# Receive from a specific stream in an array
a: float32 = fifo_A[i, j].get()

# Receive a block
block: float32[M, N] = pipe.get()
```

[Systolic Array Patterns](#id19)[¶](#systolic-array-patterns "Link to this heading")
------------------------------------------------------------------------------------

Systolic arrays are a common design pattern in dataflow programming.

### [Basic Systolic Structure](#id20)[¶](#basic-systolic-structure "Link to this heading")

A typical systolic array has:

* **Loader kernels**: Feed data from external memory to the edge of the array
* **Compute kernels**: Perform the main computation while passing data to neighbors
* **Drain kernels**: Remove data that has passed through the array

```
@df.region()
def top(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    fifo_A: Stream[float32, 4][P0, P1]
    fifo_B: Stream[float32, 4][P0, P1]

    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def gemm(local_A: float32[M, K], local_B: float32[K, N], local_C: float32[M, N]):
        i, j = df.get_pid()

        # Corner: skip
        with allo.meta_if(i in {0, M + 1} and j in {0, N + 1}):
            pass

        # Left edge: load A
        with allo.meta_elif(j == 0):
            for k in range(K):
                fifo_A[i, j + 1].put(local_A[i - 1, k])

        # Top edge: load B
        with allo.meta_elif(i == 0):
            for k in range(K):
                fifo_B[i + 1, j].put(local_B[k, j - 1])

        # Right edge: drain A
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(K):
                a: float32 = fifo_A[i, j].get()

        # Bottom edge: drain B
        with allo.meta_elif(i == M + 1 and j > 0):
            for k in range(K):
                b: float32 = fifo_B[i, j].get()

        # Interior: compute and forward
        with allo.meta_else():
            c: float32 = 0
            for k in range(K):
                a: float32 = fifo_A[i, j].get()
                b: float32 = fifo_B[i, j].get()
                c += a * b
                fifo_A[i, j + 1].put(a)  # Forward to right neighbor
                fifo_B[i + 1, j].put(b)  # Forward to bottom neighbor
            local_C[i - 1, j - 1] = c
```

[Tiled Computation](#id21)[¶](#tiled-computation "Link to this heading")
------------------------------------------------------------------------

For large computations, tiles the work across multiple kernels:

```
M, N, K = 32, 32, 32
P0, P1 = 2, 2
Mt, Nt = M // P0, N // P1

@df.region()
def top(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def gemm(local_A: float32[M, K], local_B: float32[K, N], local_C: float32[M, N]):
        pi, pj = df.get_pid()
        # Each kernel computes its tile
        for i in range(pi * Mt, (pi + 1) * Mt):
            for j in range(pj * Nt, (pj + 1) * Nt):
                for k in range(K):
                    local_C[i, j] += local_A[i, k] * local_B[k, j]
```

[Hierarchical Regions](#id22)[¶](#hierarchical-regions "Link to this heading")
------------------------------------------------------------------------------

Regions can call other regions for hierarchical decomposition:

```
@df.region()
def inner[P0, P1](A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def gemm(local_A: float32[M, K], local_B: float32[K, N], local_C: float32[M, N]):
        pi, pj = df.get_pid()
        Mt: ConstExpr[int32] = M // P0
        Nt: ConstExpr[int32] = N // P1
        for i in range(pi * Mt, (pi + 1) * Mt):
            for j in range(pj * Nt, (pj + 1) * Nt):
                for k in range(K):
                    local_C[i, j] += local_A[i, k] * local_B[k, j]

@df.region()
def top(A: float32[M, K], B: float32[K, N], C1: float32[M, N], C2: float32[M, N]):
    @df.kernel(mapping=[2], args=[A, B, C1, C2])
    def wrapper(local_A: float32[M, K], local_B: float32[K, N],
               local_C1: float32[M, N], local_C2: float32[M, N]):
        i = df.get_pid()
        with allo.meta_if(i == 0):
            inner[2, 2](local_A, local_B, local_C1)  # 2x2 grid
        with allo.meta_if(i == 1):
            inner[4, 4](local_A, local_B, local_C2)  # 4x4 grid
```

[Building and Execution](#id23)[¶](#building-and-execution "Link to this heading")
----------------------------------------------------------------------------------

### [Build for Simulation](#id24)[¶](#build-for-simulation "Link to this heading")

Use the simulator target for functional verification:

```
sim_mod = df.build(top, target="simulator")
sim_mod(A, B, C)
np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
```

### [Build for LLVM](#id25)[¶](#build-for-llvm "Link to this heading")

Build for CPU execution (mainly for debugging):

```
mod = df.build(top)  # Default LLVM target
mod(A, B, C)
```

### [Build for HLS](#id26)[¶](#build-for-hls "Link to this heading")

Generate Vitis HLS code:

```
# Generate HLS code only
mod = df.build(top, target="vhls")
print(mod.hls_code)

# Build and run HLS C-simulation
mod = df.build(top, target="vitis_hls", mode="csim", project="myproject.prj")
mod(A, B, C)

# Build for hardware emulation
mod = df.build(top, target="vitis_hls", mode="hw_emu", project="myproject.prj")
mod(A, B, C)
```

### [Customization (Schedule Primitives)](#id27)[¶](#customization-schedule-primitives "Link to this heading")

Apply schedule primitives before building:

```
s = df.customize(top)
s.partition("top:A", dim=1, factor=2)
s.partition("top:B", dim=2, factor=2)
s.partition("top:C", dim=0, factor=2)
mod = s.build(target="vitis_hls", mode="hw_emu", project="myproject.prj")
```

[Complete Example: Producer-Consumer](#id28)[¶](#complete-example-producer-consumer "Link to this heading")
-----------------------------------------------------------------------------------------------------------

A complete example showing a simple producer-consumer pattern:

```
import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df
import numpy as np

M, N = 16, 16

@df.region()
def top(A: float32[M, N], B: float32[M, N]):
    pipe: Stream[float32, 4]

    @df.kernel(mapping=[1], args=[A])
    def producer(local_A: float32[M, N]):
        for i, j in allo.grid(M, N):
            out: float32 = local_A[i, j]
            pipe.put(out)

    @df.kernel(mapping=[1], args=[B])
    def consumer(local_B: float32[M, N]):
        for i, j in allo.grid(M, N):
            data = pipe.get()
            local_B[i, j] = data + 1

# Run with simulator
A = np.random.rand(M, N).astype(np.float32)
B = np.zeros((M, N), dtype=np.float32)
sim_mod = df.build(top, target="simulator")
sim_mod(A, B)
np.testing.assert_allclose(B, A + 1)
print("Passed!")
```

[Tensor Layouts](#id29)[¶](#tensor-layouts "Link to this heading")
------------------------------------------------------------------

When targeting hardware accelerators like AIE (AI Engine), tensors need to be
distributed across multiple processing elements (PEs). The `Layout` class
provides a declarative way to specify how global tensors are partitioned and
mapped to local PE memories.

### [Layout Concepts](#id30)[¶](#layout-concepts "Link to this heading")

The `Layout` class encodes a **partitioning scheme** for a tensor. For each
*tensor dimension*, you specify either:

* **Shard(axis)**: Partition this dimension across the specified *mesh axis*
* **Replicate**: Keep this dimension fully replicated across all PEs

### [Import and Setup](#id31)[¶](#import-and-setup "Link to this heading")

```
from allo.memory import Layout

S = Layout.Shard   # Shorthand for sharding
R = Layout.Replicate  # Shorthand for replication
```

### [Applying Layouts to Kernel Arguments](#id32)[¶](#applying-layouts-to-kernel-arguments "Link to this heading")

Use the `@` operator to annotate kernel arguments with their layout. If no layout is specified, the default is **replicated** for all dimensions:

```
LyA = [S(0), R]  # Shard first dim on mesh axis 0, replicate second dim

@df.kernel(mapping=[P0, P1], args=[A, B])
def kernel(local_A: Ty[M, N] @ LyA, local_B: Ty[M, N]):
    # local_A is automatically partitioned according to LyA
    # local_B has no layout annotation (defaults to Replicate)
    pass
```

### [Layout Patterns](#id33)[¶](#layout-patterns "Link to this heading")

**1D Tensor Sharding:**

```
# Vector of size M, sharded across 4 PEs
Ly = [S(0)]  # Shard on mesh axis 0

@df.kernel(mapping=[4], args=[A, B])
def core(local_A: int32[M] @ Ly, local_B: int32[M] @ Ly):
    # Each PE gets M/4 elements
    local_B[:] = allo.add(local_A, 1)
```

**2D Tensor Row Sharding:**

```
# Matrix [M, N], shard rows across mesh axis 0
LyA = [S(0), R]

@df.kernel(mapping=[4], args=[A])
def kernel(local_A: Ty[M, N] @ LyA):
    # Each PE gets M/4 rows (full columns)
    pass
```

**2D Tensor Column Sharding:**

```
# Matrix [M, N], shard columns across mesh axis 0
LyB = [R, S(0)]

@df.kernel(mapping=[4], args=[B])
def kernel(local_B: Ty[M, N] @ LyB):
    # Each PE gets full rows, N/4 columns
    pass
```

**2D Tensor Full Sharding (GEMM Example):**

For matrix multiplication `C = A @ B` with 2D PE grid:

```
M, N, K = 64, 64, 64
P0, P1 = 2, 2  # 2x2 PE grid

# A[M, K]: shard M on axis 1 (for row parallelism), replicate K
LyA = [S(1), R]
# B[K, N]: replicate K, shard N on axis 0 (for column parallelism)
LyB = [R, S(0)]
# C[M, N]: shard both dimensions
LyC = [S(1), S(0)]

@df.region()
def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def gemm(
        local_A: Ty[M, K] @ LyA,
        local_B: Ty[K, N] @ LyB,
        local_C: Ty[M, N] @ LyC
    ):
        # Each PE computes a tile of C
        local_C[:, :] = allo.matmul(local_A, local_B)
```

**3D Sharding for Temporal Reduction (Split-K GEMM):**

```
M, N, K = 64, 64, 128
Pk, Pm, Pn = 4, 2, 2  # 4x2x2 PE grid

# A[M, K]: shard M on axis 1, shard K on axis 0
LyA = [S(1), S(0)]
# B[K, N]: shard K on axis 0, shard N on axis 2
LyB = [S(0), S(2)]
# C[M, N]: shard M on axis 1, shard N on axis 2
LyC = [S(1), S(2)]

@df.kernel(mapping=[Pk, Pm, Pn], args=[A, B, C])
def gemm(
    local_A: Ty[M, K] @ LyA,
    local_B: Ty[K, N] @ LyB,
    local_C: Ty[M, N] @ LyC
):
    pk, pm, pn = df.get_pid()
    # Each PE along the K dimension computes a partial sum
```

### [Understanding Mesh Axis Mapping](#id34)[¶](#understanding-mesh-axis-mapping "Link to this heading")

The `Shard(axis)` parameter refers to the **mesh dimension** (from the kernel’s
`mapping` parameter), not the tensor dimension:

```
Kernel mapping: [P0, P1, P2] -> 3D mesh with axes 0, 1, 2
                 |   |   |
                 0   1   2  <- mesh axes

Layout [S(1), R] means:
  - Tensor dim 0: sharded across mesh axis 1 (P1 partitions)
  - Tensor dim 1: replicated (each PE has full dim)
```

### [How Layouts Compute Local Shapes](#id35)[¶](#how-layouts-compute-local-shapes "Link to this heading")

The local tensor shape at each PE is automatically computed:

```
# Global tensor: [64, 64]
# Layout: [S(0), S(1)]
# Mapping: [4, 2]  -> 4x2 PE grid
#
# Local shape per PE: [64/4, 64/2] = [16, 32]
```

For replicated dimensions, the local shape equals the global shape for that dimension.

[Best Practices](#id36)[¶](#best-practices "Link to this heading")
------------------------------------------------------------------

1. **Start with the simulator**: Use `target="simulator"` for initial debugging
   before moving to HLS.
2. **Match stream types carefully**: Ensure `put` and `get` operations match
   the declared stream element type.
3. **Use meta conditionals for PE differentiation**: In systolic arrays, use
   `allo.meta_if/meta_elif/meta_else` to generate different code for different
   PE positions.
4. **Consider stream depth**: Set appropriate FIFO depths based on the
   producer-consumer latency mismatch.
5. **Use ConstExpr for tile sizes**: When computing tile boundaries, use
   `ConstExpr` for values that should be computed at compile time.

[<Page contents](#)

[>Page contents:](#)

* [Dataflow Programming](#)
  + [Overview](#overview)
    - [Key Concepts](#key-concepts)
  + [Getting Started](#getting-started)
    - [Import the Dataflow Module](#import-the-dataflow-module)
    - [Basic Structure](#basic-structure)
  + [Regions](#regions)
    - [Region Declaration](#region-declaration)
    - [Parameterized Regions](#parameterized-regions)
  + [Kernels](#kernels)
    - [Kernel Declaration](#kernel-declaration)
    - [Single Kernel Instance](#single-kernel-instance)
    - [Multi-Dimensional Kernel Grid](#multi-dimensional-kernel-grid)
    - [Getting Kernel Position](#getting-kernel-position)
  + [Streams](#streams)
    - [Stream Declaration](#stream-declaration)
    - [Stream Arrays](#stream-arrays)
    - [Stream of Blocks](#stream-of-blocks)
    - [Stream Operations](#stream-operations)
  + [Systolic Array Patterns](#systolic-array-patterns)
    - [Basic Systolic Structure](#basic-systolic-structure)
  + [Tiled Computation](#tiled-computation)
  + [Hierarchical Regions](#hierarchical-regions)
  + [Building and Execution](#building-and-execution)
    - [Build for Simulation](#build-for-simulation)
    - [Build for LLVM](#build-for-llvm)
    - [Build for HLS](#build-for-hls)
    - [Customization (Schedule Primitives)](#customization-schedule-primitives)
  + [Complete Example: Producer-Consumer](#complete-example-producer-consumer)
  + [Tensor Layouts](#tensor-layouts)
    - [Layout Concepts](#layout-concepts)
    - [Import and Setup](#import-and-setup)
    - [Applying Layouts to Kernel Arguments](#applying-layouts-to-kernel-arguments)
    - [Layout Patterns](#layout-patterns)
    - [Understanding Mesh Axis Mapping](#understanding-mesh-axis-mapping)
    - [How Layouts Compute Local Shapes](#how-layouts-compute-local-shapes)
  + [Best Practices](#best-practices)

[<Frontend Syntax Guide](frontend_syntax.html)

[Data Types and Type Casting>](../gallery/dive_01_data_types.html)

© Copyright 2025, Allo Authors.
Created using [Sphinx](https://www.sphinx-doc.org/) 8.2.3.

Styled using the [Piccolo Theme](https://github.com/piccolo-orm/piccolo_theme)