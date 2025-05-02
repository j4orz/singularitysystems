# Chapter 1: au-197
**This chapter introduces the golden age of systems ml**

## Golden Age — Software 2.0 Infrastructure Buildout
Welcome to the second golden age of computing. The first one took place in the
1940s when we discovered how to program sequences of instructions on
latency-oriented machines with scalar processing. We collectively refer to this
period as the *information revolution*. The second one is taking place today in
the 2020s where we are discovering how to compress sequences with stochastic
distributions on throughput-oriented machines with vector processing. People are
describing this period as the beginning of the *intelligence revolution*.

What we see in both periods is the occurence of positive-sum-game-generating
socioeconomic feedback loops dubbed "scaling laws" between software and hardware.
That is, the more demand generated from applications, the more supply from
infrastructure, which unlocks better applications, which generates more demand,
which increases the supply, and so on. The information revolution gave rise
to Moore's Law, an exponential trend between transistor density and TODO: X,
while the intelligence revolution today gives rise to Scaling Law's, another
exponential between compression and parameter count.

This textbook is all about the latter. More specifically we focus our attention
on the **compilers and chips** that are powering the massive AI infrastructure
buildout that's happening today. What's exciting from the perspective of a
compiler engineer or chip architect is that all of the assumptions held in the
design of our computers need to be reassessed and reevaluated. This is because
the "soul of the machine" is constructed by the dance that happens between the
compiler and chip — *together* through software-hardware codesign they bridge
the semantic gap between humans and electrons.

The original assumption held in the computers designed from the information
revolution was [Dennard Scaling]() (also referred to as MOSFET Scaling) which
states that as the size of transistors decreases, the power density stays
constant.









TODO: von neumann prog/exec model. 
The von neumann

What's interesting to systems programmers and performance engineers is that the
field is witnessing the rhyming of history. The feedback loop between the
information revolution's software applications and sequential hardware
(OOO, pipelining, TODO) creates this sociocultural phenomena dubbed
"Moore's Law"[X]. The same loop is happening with the intelligence revolution's
models and parallel hardware (tensor cores, TMA, TODO) which is now giving rise
to the "Scaling Laws"[X]. All of the original assumptions held in the design of
the compilers and chips that make up the soul of the machine are being reassessed
if the dominant workload is evaluating matrix multiplications for stochastic
distributions instead of fetching data for discrete algorithms.

In transportation technology is when humanity wants to go to the moon, engineers
reassess the design of vehicles from first principles. Given the workload of
space travel, it's economically feasiable to start from fundamentals and design
a rocket rather than a car or bus. As a result, the engineers behind the massive
infrastructure buildout are dubbing this period of
history as a new golden age:

1. A New Golden Age for Computer Architecture[0] — Hennessy and Patterson
2. The Golden Age of Compiler Design[1] — Chris Lattner

Many toy autograds exist — the software 2.0[2] version of calculator interpreters
capable of evaluating arithmetic. These are excellent in providing intuition for
backpropagation (calculus on a computational graph), a very important abstraction
to understand considering how leaky it is (think gradient initialization and
normalization). However, there are zero resources that dive deeper into cutting
your teeth on the advanced capabilities of PyTorch such as torch.compile and
torch distributed. Linux has xv6[3], clang has chibicc[4], and llvm has qbe[5],
but pytorch is missing it's teaching compiler. This is the gap that the course
fills.

The field of artificial intelligence is no stranger to pedagogical gaps.
Researchers talk about "research debt"[6] with respect to model development, but
what they forget to talk about is that the debt also applies to the underlying
infrastructure that powers the training and inference of deep neural networks.
This is largely in part because it's only been a decade since deep learning
has entered the zeitgeist of academia[7] and the public[8].

To give you a better idea of how nascent system ML as a field:
  1. the leading conferences for AI (NeurIPS) and Hardware (HOTCHIPS) have both
     been running for ~30 years[9][10]. Meanwhile, the conference for the
     intersection of the two (mlsys) has only been running for 6[11].
  2. the lingua franca deep learning framework was released[12] less than a
     decade, and the 2.0 release[13] was released only 2 years ago

Welcome to the golden age of Systems ML!

References
----------
0. [https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
0. [https://www.nature.com/articles/nature14539](https://www.nature.com/articles/nature14539)
0. [https://www.youtube.com/watch?v=3LVeEjsn8Ts](https://www.youtube.com/watch?v=3LVeEjsn8Ts)
0. [https://www.youtube.com/watch?v=4HgShra-KnY](https://www.youtube.com/watch?v=4HgShra-KnY)
0. [https://karpathy.medium.com/software-2-0-a64152b37c35](https://karpathy.medium.com/software-2-0-a64152b37c35)
0. [https://github.com/mit-pdos/xv6-public](https://github.com/mit-pdos/xv6-public)
0. [https://github.com/rui314/chibicc](https://github.com/rui314/chibicc)
0. [https://c9x.me/compile/](https://c9x.me/compile/)
0. [https://distill.pub/2017/research-debt/](https://distill.pub/2017/research-debt/)
0. [https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
0. [https://www.nature.com/articles/nature14539](https://www.nature.com/articles/nature14539)
0. [https://papers.nips.cc/](https://papers.nips.cc/)
0. [https://hotchips.org/archives/](https://hotchips.org/archives/)
0. [https://proceedings.mlsys.org/](https://proceedings.mlsys.org/)
0. [https://soumith.ch/blog/2023-12-17-pytorch-design-origins.md.html](https://soumith.ch/blog/2023-12-17-pytorch-design-origins.md.html)
0. [https://pytorch.org/assets/pytorch2-2.pdf](https://pytorch.org/assets/pytorch2-2.pdf)
