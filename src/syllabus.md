![](./flammarion.webp)
# Singularity Systems: Zero to Hero
*Dragon Book 2.0: The Hacker's Guide to Tensor Compilers*

## Course Description
Singularity Systems: Zero to Hero follows up from
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html). We convert
- [**micrograd**](https://github.com/karpathy/micrograd): toy backpropagation engine *into...*
- [**picograd**](https://github.com/j4orz/picograd): modern deep learning framework

While micrograd is good for research scientists to further understand the leaky
abstraction of backpropagation, picograd is intended for systems programmers and
performance engineers looking to further understand the compilers and chips of
deep learning.

### Prerequisites
- solid deep learning (sd && llama)
- solid systems programming (C || C++ || Rust)

### Syllabus
**Core**: *Deep Learning Compilers of Today*

1. **dfdx(nd)**: implements an interpreter for neural networks (pytorch1).
2. **brrr**: accelerates the interpreter with vector processing (pytorch1).
3. **pt2**: constructs a compiler for neural networks (pytorch2).

**Appendix**: *Parallel Computing of Yesterday and Tomorrow*
-  **A: backus&brooks**: speedruns an optimizing compiler (c -> riscv).
-  **B: allen&cray**: tourguides the unsolved problem of parallel computing (mojo, tenstorrent).

### Citation
```
@article{j4orz2025singsys,
  author  = "j4orz",
  title   = "Singularity Systems: Zero to Hero",
  year    = "2025",
  url     = "https://github.com/j4orz/singularitysystems"
}
```