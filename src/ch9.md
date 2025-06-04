<!-- ![](./brrr.jpg) -->
<!-- > *Will and existence. The rest is mere data — Lda237, Nietzsche.*  -->
# 9. Vector Processors

good at kernel authoring.
phillipe can write performant cuda kernels. so he's a good person for writing triton compiler.
phillipe knows the optimizations he needs to make.
-> autovectorization is straightforward.
-> thread ___ is straightforward.
-> all that's difficult is the block-level foo bar baz.

references
- python -> c -> parallel loops -> memory optimization -> SIMD
1 -> 63,000 times speedup

compiler writers kill for factors of 2 (2x). 2x. 5x. 10x.
https://www.microsoft.com/en-us/research/wp-content/uploads/2020/11/Leiserson-et-al-Theres-plenty-of-room-at-the-top.pdf


conclude:
why are we spending most of our runtime in the slow ops that are memory bound?
we're not getting peak theoretical FLOP utilization
-> thus, we need PT2 (chapter 3)

# Section 2: Speed
- does the current cpu ops vectorize? godbolt.

CPU, GPU (they look similar. see tim sweeney's talk)
prog model: autov11n -> intrinsics -> ??
exec model: SIMD -> SIMT on SIMD. `AVX512`, `AMX`, `NEON`

## Part 5 CPU: `RVV`
-> numpy simd
-> we will focus on RISC.
-> fearless SIMD with rust.

## Part 6 GPU: `RVV`
