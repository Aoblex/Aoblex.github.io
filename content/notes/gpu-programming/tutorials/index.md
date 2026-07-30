+++
title = 'Tutorials'
date = '2026-04-30T13:47:58+08:00'
summary = 'A collection of helpful GPU programming tutorials. 📖'
weight = 10
+++

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/): this is the very first document that I read when I started learning CUDA from scratch. I think the most important thing is to get a big picture of [**the CUDA programming model**](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model).

{{< figure
  src="figures/grid-of-thread-blocks.png"
  link="https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-blocks-and-grids"
  alt="Grid of Thread Blocks"
  caption="Grid of Thread Blocks"
  width="400"
  align="center"
>}}

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): White paper covering the most common issues related to NVIDIA GPUs.

{{< figure
    src="figures/computing-row-of-tile.png"
    link="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#shared-memory-in-matrix-multiplication-c-ab"
    alt="Computing Row of Tile"
    caption="Computing Row of Tile"
    width="200"
    align="center"
>}}

- [GPU glossary](https://modal.com/gpu-glossary): this glossary **connects the dots**. It really helped me a lot in understanding all kinds of GPU related terminologies.

{{< figure
    src="figures/terminal-gh100-sm.svg"
    link="https://modal.com/gpu-glossary/device-hardware/core"
    alt="The internal architecture of an H100 GPU's Streaming Multiprocessors."
    caption="The internal architecture of an H100 GPU's Streaming Multiprocessors."
    width="400"
    align="center"
>}}
