# Runtime Overview

## Overview

A typical runtime consists of the following parts:

- [Compiled](#compiled)
- [Allocator](#allocator)
- [Program](#program)
- [Compiler](#compiler)

### Compiled

The `Compiled` class is responsible for initializing and managing a device.

::: tinygrad.device.Compiled
    options:
        members: [
            "synchronize"
        ]
        show_source: false

### Allocator

The `Allocator` class manages memory on the device and caches allocated buffers for reuse.

`Buffer(device, nbytes)` owns untyped storage. Its length and view offsets are in bytes:
`buffer.view(nbytes, offset)` shares a byte range of the original allocation. Allocators may reserve additional padding internally.
The graph retains element counts and dtypes; `UOp.from_buffer(buffer, dtype)` gives storage a typed interpretation without converting its contents.
Use `buffer.numpy(dtype)` for typed host reads and `buffer.copy_from(source)` for byte copies.

::: tinygrad.device.Allocator
    options:
        members: true
        show_source: false

### Program

The `Program` class is created for each loaded program. It is responsible for executing the program on the device. As an example, here is a `CPUProgram` implementation which loads program and runs it.

::: tinygrad.runtime.ops_cpu.CPUProgram
    options:
        members: true

### Compiler

The `Compiler` class compiles the output from the `Renderer` and produces it in a device-specific format.

::: tinygrad.device.Compiler
    options:
        members: true
