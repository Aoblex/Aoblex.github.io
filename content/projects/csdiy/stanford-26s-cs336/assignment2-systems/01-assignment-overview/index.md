+++
title = 'Section 1: Assignment Overview'
date = '2026-08-19T15:26:51+08:00'
summary = 'Measure, optimize, and scale Transformer training from one GPU to many.'
weight = 10
draft = false
+++

CS336 Assignment 2 shifts the focus from defining a Transformer to executing it efficiently. Starting from the language model built in Assignment 1, we study where training time and memory go, reduce avoidable overhead on a single GPU, and then distribute the remaining work across multiple GPUs.

The central question is not simply whether an implementation is correct, but how its computation, memory use, and communication determine end-to-end performance. Every optimization is therefore paired with measurement: wall-clock benchmarks establish the outcome, while compute and memory profiles explain it.

---

# What We Will Implement

The assignment follows the path from single-GPU measurement to distributed training:

1. **Benchmarking and profiling** measure the forward pass, backward pass, and optimizer step, then attribute their runtime and memory use to individual operations.
2. **Activation checkpointing** trades additional computation for a lower activation-memory footprint by recomputing selected intermediate values during the backward pass.
3. **FlashAttention-2** reorganizes attention into tiled GPU kernels so that intermediate attention matrices do not need to be materialized in high-bandwidth memory.
4. **Distributed data parallel training** replicates the model across workers and synchronizes gradients, first directly and then with communication overlapped with backpropagation.
5. **Optimizer state sharding** partitions optimizer state across workers instead of storing a complete copy on every GPU.
6. **Fully sharded data parallel training** extends sharding to parameters and gradients, gathering each layer only when its computation requires it.

Together, these components turn the model from Assignment 1 into a training system whose bottlenecks can be measured and whose memory use can scale beyond a single device.

---

# What We Will Measure

The experiments examine three resources that constrain large-model training:

- **Compute time** is measured with synchronized wall-clock benchmarks and NVIDIA Nsight Systems traces. Warm-up iterations, asynchronous CUDA execution, mixed precision, and kernel fusion all affect what a timing result means.
- **Memory** is separated into parameters, gradients, optimizer state, and saved activations. Memory snapshots and analytical accounting show which terms dominate and how checkpointing or sharding changes them.
- **Communication** is characterized by collective operations, transferred bytes, and the degree of overlap with useful computation. These measurements distinguish nominal parallelism from an actual end-to-end speedup.

The final analysis compares data parallelism, fully sharded data parallelism, tensor parallelism, and two-dimensional combinations of these strategies. The objective is to understand when each strategy is limited by memory capacity, compute throughput, or interconnect bandwidth rather than to treat any one method as universally optimal.
