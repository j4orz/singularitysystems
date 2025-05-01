# Singularity Systems: Zero to Hero
**The Hacker's Guide to Tensor Compilers**

## Course Description
Singularity Systems: Zero to Hero follows up from Neural Networks: Zero to Hero.
We turn the toy backpropagation engine micrograd into picograd: a deep learning
framework with .compile(model) support.

Singularity Systems bridges the missing gap in the ecosystem for low level system
programmers and performance engineers interested in the internals of deep learning
frameworks. The course provides the software 2.0 equivalent of SICP (interpreters)
and dragon book (compilers), where you will build a PT1 "eager" interpreter
throughout chapters 1-2 and a PT2 "graph" compiler throughout chapters 3-4, line
by line from scratch. Along the way, we compare and contrast the design decisions
and tradeoffs PyTorch has made versus other frameworks such as Jax and Tinygrad.
By the end of Singularity Systems, you will be better equipped to contribute to
production-level deep learning frameworks.

### Prereqs
- solid deep learning (sd && llama)
- solid systems programming (C || C++ || Rust || Python*)
  * python is ok (following pure-python systems like PyTorch 2.0 & CUTLASS 4.0)

### Syllabus
0. **au197**: introduces the golden age of systems ml.
1. **dfdx(nd)**: implements an interpreter for neural networks (pytorch1).
2. **tiles**: implements a compiler for a tiling language (triton).
3. **pt2**: implements a compiler for neural networks (pytorch2)
4. **turing**: implements a compiler that maps C to RISCV.

### Citation
```
@article{j4orz2025singsys,
  author  = "j4orz",
  title   = "Singularity Systems: Zero to Hero",
  year    = "2025",
  url     = "https://github.com/j4orz/singularitysystems"
}
```