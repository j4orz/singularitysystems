![](./brrr.jpg)
> *Will and existence. The rest is mere data — Lda237, Nietzsche.* 
# Chapter 2: brrr

references
- python -> c -> parallel loops -> memory optimization -> SIMD
1 -> 63,000 times speedup

compiler writers kill for factors of 2 (2x). 2x. 5x. 10x.
https://www.microsoft.com/en-us/research/wp-content/uploads/2020/11/Leiserson-et-al-Theres-plenty-of-room-at-the-top.pdf


conclude:
why are we spending most of our runtime in the slow ops that are memory bound?
we're not getting peak theoretical FLOP utilization
-> thus, we need PT2 (chapter 3)