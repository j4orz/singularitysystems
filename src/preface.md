# Preface: au-197
**This chapter introduces the golden age of systems ml and provides a course overview**


<!-- ## Revolutions: Industrial, Information, Intelligence

### Industrial Revolution
During the industrial revolution we naturalized *energy* ("ability to do work")
with *thermodynamics* by defining it as product of power and time
`E(J)=P(W)*T(s)`. As a result we moved from horsepower to W(J/s), which lets us
answer questions such as:
- Q: how much energy is required to heat water from 20°C to 100°C?
- Q: how much energy is required to cool a fridge from X to Y?
- Q: how much energy is required to heat a steam engine from X to y?

![](./energy.svg)


### Information Revolution
During the information revolution we naturalized *computation* ("execution of
a sequence of instructions) with *informatics* by defining it as turing-complete
`L,M,N ::= x | (λx.N) | (L M)`. As a result we moved from human computers to
mechanical ones, which lets us answer questions such as:
- Q: is Dijkstra's algorithm computable? A: yes.
- Q: how compute is required for Dijkstra's algorithm? A: O(x) time, O(x) space
- Q: is the Traveling Salesman problem computable? A: ??
- Q: how compute is required for Dijkstra's algorithm? A: idk.
- Q: is the Halting Problem computable? A: no.

![](./computation.svg)


### Industrial Revolution
Today, we are at the dawn of an intelligence revolution which is naturalizing
*intelligence* ("achieving goals") with *entropics*, defining intelligence as
compression `I=FOO`. As a result we will move from our intuitive personyears to
petaFLOP/s

- Q: how much intelligence is required for 2000 ELO in chess? A:
- Q: how much intelligence is required for 2000 ELO in go? A:
- Q: how much intelligence is required for 2000 ELO in codeforces? A:
- Q: how much intelligence is required to prove fermat's last theorem? A: idk. andrew wyles.

![](./intelligence.svg) -->

## Golden Age — Software 2.0 Infrastructure Buildout

Throughout the past decade, modern day AI infrastructure has extremely rapidly
to meet the needs of training deep neural networks, most notably with the
throughput performance of datacenter GPUs moving from TFLOPS(1e12) to
PFLOPS(1e15). And now with supercomputing reaching (non-distributed) EFLOP(1e15)
performance, datacenter computing will follow.

### Golden Age (1 minute version)
Working backwards from physics, Dennard scaling (also known as MOSFET scaling)
describes how power stays constant even though transistor size shrinks.
Since `TODO: P proportional Load? * V^2 * Hz`, the semiconductor industry is
described as the "power wall". While there's still plenty of room at the bottom
(brains are existence proof of PFLOP machines powered by 20W), the strategy to
solve the problem today is to tapeout specialized processors over generalized
ones.

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

**Welcome to the golden age of Systems ML!**

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

2. karpathy
- community building (build the ramp)
- a good first contribution to any company is docs. so in some way this course
- is my version of that for the community.

**Welcome to the golden age of Systems ML!**