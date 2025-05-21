![](./flammarion.webp)
# Singularity Systems: Zero to Hero
*Dragon Book 2.0: The Hacker's Guide to Tensor Compilers*

## Course Description
Singularity Systems: Zero to Hero follows up from
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html). We convert
- [**micrograd**](https://github.com/karpathy/micrograd): toy backpropagation engine *into...*
- [**picograd**](https://github.com/j4orz/picograd): modern deep learning framework

While micrograd helps research scientists to understand the leaky abstraction of
backpropagation, picograd is intended for systems programmers and performance
engineers looking to further understand the compilers and chips of deep learning.
Try it out with:
```
pip install picograd
```

### Prerequisites
- solid deep learning (llama)
- solid systems programming (C || C++ || Rust)

### Syllabus
**Core**: *Deep Learning Compilers*

1. **dfdx(nd)**: implements an interpreter for neural networks (HIPS/autograd)
2. **brrr**: accelerates the interpreter with vector processing (pytorch1)
3. **pt2**: constructs a compiler for neural networks (pytorch2)
4. **az1use1**: 3d parallelism

**Appendix**: *Speedrunning Prerequisites*
-  **A: s2.0**: speedruns deep neural networks (ffn -> rnn -> lstm -> gpt)
-  **B: soul**: speedruns optimizing compilers and scalar processors (c -> riscv)

### Citation
```
@article{j4orz2025singsys,
  author  = "j4orz",
  title   = "Singularity Systems: Zero to Hero",
  year    = "2025",
  url     = "https://j4orz.ai/zero-to-hero/"
}
```