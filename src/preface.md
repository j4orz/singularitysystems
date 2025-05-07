![](./au197.jpg)

> *Dreams. Each man longs to pursue his dream.* — Guts
# Preface: au-197
**This chapter introduces the golden age of systems ml and provides a course overview**

## Golden Age — Software 2.0 Infrastructure Buildout

Throughout the past decade, modern day AI infrastructure has rapidly evolved
to meet the needs of deep neural networks — most notably with the throughput
performance of GPUs moving from `TFLOPS` to `PFLOPS`. Datacenter
computing now has the goal of a machine with `EFLOPS` speed, now that that
the throughput of the fastest (non-distributed) supercomputers on TOP500 are just
reaching `EFLOP` levels.

---

### 1 second version
SOTA semiconductor physics is forcing both languages and machines to be redesigned
from frist principles. This creates a golden age for compiler engineers and
computer architects because who need to software-hardware codesign the machines
of tomorrow.

*Welcome to the golden age of Systems ML!*

---

### 1 minute version
Although the brain is an existence proof of physics powering `20PFLOP` machines
with `20W`, the problem with SOTA semiconductor physics is two-fold:
1. instruction-level parallelism from out of order, superscalar pipelines is providing diminishing returns
2. frequency scaling is hitting against [Dennard scaling](https://en.wikipedia.org/wiki/Dennard_scaling)'s power wall

and so this free-single-thread-performance-lunch aspect to [Moore's law](https://en.wikipedia.org/wiki/Moore%27s_law)
that transitioned us from minis to micros and from micros to mobile is "over".
Now, the complexity of extracting program performance is leaking from the hardware,
and the programmer is now exposed to thread-level parallelism with multi-core and
many-core machines also known as vector processors.

The problem with the vector processing of multi-core/many-core machines is two-fold:
1. programming model: compiler engineers wrote [sufficiently smart compilers](https://wiki.c2.com/?SufficientlySmartCompiler) with [autovectorization](https://pharr.org/matt/blog/2018/04/18/ispc-origins)
2. execution model: program speedups were bound by [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law) — their serial portions

But the industry sidestepped these problems by changing the programming model to
SIMT on SIMD (CUDA) and finding domains whose execution models had more parallelism
(deep neural networks). The challenge (and golden age) of compiler engineers and
chip architects face is the creation of new programming languages and machines
while minimizing the accidental complexity that naturally builds up along the way:

- [The Golden Age of Compiler Design (Lattner)](https://www.youtube.com/watch?v=4HgShra-KnY)
- [A New Golden Age for Computer Architecture (Hennessy and Patterson)](https://www.youtube.com/watch?v=3LVeEjsn8Ts)

*Welcome to the golden age of Systems ML!*


---

### 10 minute version

In this course, we will convert [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) micrograd
into [Singularity Systems: Zero to Hero](./syllabus.md) picograd (line by line, from scratch).

The difference between the two is that:
- [micrograd](https://github.com/karpathy/micrograd) is a toy backpropagation engine on scalars that is extremely helpful for researchers to understand the leaky abstraction that backpropagation is (gradient activations, normalizations, etc.)
- [picograd](https://github.com/j4orz/picograd) is a pedagogical deep learning framework with full modern-day tensor-support for both pytorch1 interpretation and pytorch2 compilation.

While picograd is more oriented towards low-level system programmers and
performance engineers, this framework is still *pedagogical* and remains a
point-wise solution meaning that we will support 1 model (llama), 1 framework
(pytorch api) and 1 hardware (riscv, nvidia).


### 10 hour version
This course!

### 100 hour - version




accidental complexity ==> FLOPS/W or FLOPS/$




Deep learning (and graphics) are enjoying 10-100x more perf(throughput) of GPUs/TPUs,
but it's very expensive so can we make it more cheaper (efficient) `FLOP/W` or `FLOP/$` for existing AI/graphics workloads?
-> which might have downstream effects of unlocking FLOP/S to any compute that
can be expressed as a graph.

programming model implemented by interpreter/compiler. "only" achieving 50% utilization
execution model implemented by chip: graph programs on vector processors

CONCLUSION: lots of accidental complexity arisen (genius workaround at first but now ducktaping)

SOLUTIONs today: graph compilers, kernel authoring: CUBLAS, CUDNN,
SOLUTIONS tmr: mojo/apl/futhark (metaprogramming with comptime), tenstorrent/cell (dataflow computing)

we will build a poitn-wise (1 model, 1 framework, 1 hardware) solution picograd
built that were hillclimbed from yesterdays solutions:

*Welcome to the golden age of Systems ML!*

### Golden Age (10 minute version)

programming an M1 and a whatever1 feel relative the same.
the hardware/software contract is pretty clean for scalar computing
and does not leak complexity up the stack.

the same cannot be said for unlocking the performance of modern day for
vector processors and tensorprocessors. you have to know about (come back here:
TMA, X, Y, Z)

- moores law. scaling laws.
- bells law
1. A New Golden Age for Computer Architecture[0] — Hennessy and Patterson
2. The Golden Age of Compiler Design[1] — Chris Lattner

This is because
the "soul of the machine" is constructed by the dance that happens between the
compiler and chip — together through **software-hardware codesign** they bridge
the semantic gap between humans and electrons.
All of the original assumptions held in the design of
the compilers and chips that make up the soul of the machine are being reassessed
if the dominant workload is evaluating matrix multiplications for stochastic
distributions instead of fetching data for discrete algorithms.


Many toy autograds exist — the software 2.0[2] version of calculator interpreters
capable of evaluating arithmetic. These are excellent in providing intuition for
backpropagation (calculus on a computational graph), a very important abstraction
to understand considering how leaky it is (think gradient initialization and
normalization). However, there are zero resources that dive deeper into cutting
your teeth on the advanced capabilities of PyTorch such as torch.compile and
torch distributed. Linux has xv6[3], clang has chibicc[4], and llvm has qbe[5],
but pytorch is missing it's teaching compiler. This is the gap that the course
fills.

1. shriram krishnamurthi:
- high level blog post. or read the source code.
- this course is the bridge between blog posts and phd theses. gaps are good for bridging.
- common semantic core. taxonification of tf vs pt is useless. link to soumith's tweet
-> implementation limited to
   - RISCV (references to x86-64, AMD)
   - CUDA (references to AMD)
-> getting production-level performance, portability, and usability is for you
   to go forward and read source code.