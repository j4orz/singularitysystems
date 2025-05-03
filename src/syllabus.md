![](./flammarion.webp)
# Singularity Systems: Zero to Hero
The Hacker's Guide to Tensor Compilers: *"Talk is cheap. Show me the code"*

## Course Description
Singularity Systems: Zero to Hero follows up from
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html). We convert
- [**micrograd**](https://github.com/karpathy/micrograd): toy backpropagation engine *into...*
- [**picograd**](https://github.com/j4orz/picograd): modern deep learning framework

While micrograd is good for research scientists to further understand the leaky
abstraction of backpropagation (think activation gradients and normalization),
picograd is intended for systems programmers and performance engineers looking
to further understand the compilers and chips of deep learning.

### Prerequisites
- solid deep learning (sd && llama)
- solid systems programming (C || C++ || Rust* || Python**)
  * *rust is currently the only supported language
  * **python is ok (following pure-python hackable PyTorch 2, CUTLASS 4)

### Syllabus
**Core**: *Deep Learning Compilers of Today*
1. **au197**: introduces the golden age of systems ml and course overview.
2. **dfdx(nd)**: implements an interpreter for neural networks (pytorch1).
3. **pt2**: constructs a compiler for neural networks (pytorch2).
4. **tiles**: constructs a compiler for a tiling language (triton).

**Appendix**: *Parallel Computing of Yesterday and Tomorrow*
1. **turing**: reviews a compiler that maps C to RISCV.
2. **richard**:: explores heterogenous parallel languages and machines.

### Citation
```
@article{j4orz2025singsys,
  author  = "j4orz",
  title   = "Singularity Systems: Zero to Hero",
  year    = "2025",
  url     = "https://github.com/j4orz/singularitysystems"
}
```