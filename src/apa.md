![](./apa.gif)
> *A famous colleague once sent an actually very well-written paper he was quite proud of to a famous complexity theorist. His answer: “I can’t find a theorem in the paper. I have no idea what this paper is about.”*
# Appendix A: 2.0

*This appendix speedruns deep learning prerequisites by using language modelling as a running example.*
By the end of the appendix you should be comfortable with state of the art
open source models such as [llama]() and [r1]().

**Contents**
1. [probability: language modeling]()
2. [statistical learning: linear ->  non-linear](#part-1--statistical-learning-linear-to-non-linear)
3. [deep learning: from ffn to gpt](#part-2--deep-learning-from-ffn-to-gpt)
4. [return of the rl]()

**References**
1. [Piech 2024](https://chrispiech.github.io/probabilityForComputerScientists/en/)
1. [Hardt, Recht 2022](https://mlstory.org/)
1. [Recht 2023](https://www.argmin.net/p/patterns-predictions-and-actions)
1. [Bach 2024](https://www.di.ens.fr/~fbach/ltfp_book.pdf)
1. [Cho 2015](https://arxiv.org/abs/1511.07916)
1. [Jurafsky, Martin 2025](https://web.stanford.edu/~jurafsky/slp3/)

## Part 1 — probability: language modeling
**Probability** measures the uncertainty around stochastic (non-deterministic)
phenomena.

Even though the world may be deterministic probability theory provides methods
to measure the degrees of subjective belief. Perhaps knowing the position of
every water molecule lets you deterministically calculate and predict tomorrow's
weather, but in practice, this is intractable.

Formally, a probability space is formalized as a triplet
$(\Omega, \mathcal{E}, P)$ that comprises a sample space $\Omega$, an
event space $\mathcal{E}$, and a measure $P: \mathcal{E} \to [0,1]$. A
probability space distributes truth across a weighted set of events.

---
**Example 1**

Modeling the sample space and event space for language looks like:
1. the sample space $\Omega$ is the *set* of all possible *outcomes* (**words**)
2. the event space $\mathcal{E}$ is the *power set* $2^\Omega$ of all possible *subsets* (**sentences**)

and so if the *experiment* is to sample a single word from the english corpus,
then the:
- $\Omega=\{a, aardvark, ... zygote\}$
- and some event $E$ where you sample a word that starts with the letter j is the subset $E=\{j, jaguar, ...\}$.

on the other hand if the *experiment* is to sample two words from the english
corppus, then:
- the sample space $\Omega=\{\{a,a\}, \{a, aardvark\}, ... \{zygote,zebra\}, \{zygote, zygote\},\}$
- and some event $F$ where you sample two words that start with a letter in the first half of the alphabet is the subset $F=$
---

What can be counterintuitive at first is that the outcomes in the sample space
can be sets themselves, which formalizes an experiment with more than one "object".
Moreover, you can conceptualize an event as some semantic invariant placed on the
sample space which constructs a subset of the sample space. But to complete our
formalization of language as a probability space triplet $(\Omega, \mathcal{E}, P)$, we need:

3. the measurer $P: \mathcal{E} \to [0,1]$ is the *function* that produces the size of any *subset* $E$ relative to the *powerset* $\mathcal{E}$ (**chance of a sentence ocurring**)

what can be counter intuitive at first is that the domain of function $P$ is
$\mathcal{E}$ and not $\Omega$. Formalizing $P$ as $P(E) = \lim_{n\to\infty} \frac{|E|}{n}$
places the foundations of probability on top of set theory with the following
three axioms:

1. $0 \leq P(E) \leq 1$
2. $P(\Omega) = 1$
3. $P(A \cup B ) = P(A) + P(B)$ if events A and B are mutually exclusive

So evaluating the probability of a single outcome translates to measuring the
size of a subset with that single outcome, relative to the powerset. More
coloquially however, the sample space is the **vocabulary**, the event space is
the **language** and the measurer is the **language model**.

So for example, if a mathematically-inclined rapper wanted to ensure that the
chances of their lyrics were "fresh" and "original", they should hope that
the calculated is probability low. That is, the size of the event relative to the
event space should be evaluated.

```
P("
  i got an empire of emotion,
  a monopoly on lotion,
  in a state of commotion,
  i gesture at the notion.
")

P("
  i got an empire of emotion,
  squad see me cruisin', cruisin' in my go kart,
  i'm war, ho, i'm warhol,
  i'm wario when i'm in mario kart.
")
```

then intuitively we expect $P(A)$ is higher than $P(B)$ and would conclude that
the second set of lyrics are more "fresh". But how do we access $P$, allowing us
to reduce our common sense into calculation?
$X is a random variable$
$Let the event A be X="empire"$ apply a boolean operator to the RV.
$Let the event B be X>="e"$
$P(X="empire")$ is the probability of an event. remember P is only defined on event space -> [0,1]
$P(X=k)$ is the probability mass function. all events the random variable can take on is summarized with this single function.
if continuous then probability density function.

functions can be represented with
- equations
- code
- charts (2D. 3D)
any way of representing gives you the relationship between the events and measures.
language model is the function $P(X=k)$.

% First thought is to construct the sample space, a subset for the event we care
% about, and then *count*.

% ```python
% # data
% import torch
% import torch.nn.functional as F
% import matplotlib.pyplot as plt
% %matplotlib inline
% g = torch.Generator().manual_seed(2147483647)

% words = open('./data/names.txt', 'r').read().splitlines()
% v = sorted(list(set(''.join(words))))
% encode = { c:i+1 for i,c in enumerate(v) }
% encode['.'] = 0
% decode = { i:c for c,i in encode.items() }

% X, Y = [], []
% for w in words[:1]:
%     cs = ['.'] + list(w) + ['.']
%     for c1, c2 in zip(cs, cs[1:]):
%         x, y = encode[c1], encode[c2]
%         X.append(x)
%         Y.append(y)
% X, Y = torch.tensor(X), torch.tensor(Y)

% # model + training
% # 1. freq counts
% C = torch.zeros((27,27), dtype=torch.int32)
% for w in words:
%     cs = ['.'] + list(w) + ['.']
%     for c1, c2 in zip(cs, cs[1:]):
%         C[encode[c1], encode[c2]] += 1  # model: counts
% smoothing = 1
% P = (C.float()+smoothing) / C.float().sum(1, keepdim=True) # todo, understand broadcasting and shapes.
%                                                # keepdim=False ==> normalizing cols instead of rows
% plt.imshow(P)

% nll, n = 0.0, 0
% for w in words:
%     cs = ['.'] + list(w) + ['.']
%     for c1, c2 in zip(cs, cs[1:]):
%         i, j = encode[c1], encode[c2]
%         nll -= torch.log(P[i, j])
%         n += 1
% anll = nll/n
% print(f'{anll=}')

% # 2.  y_hat = softmax(Wx) (5,27) @ (27,27) -> (5,27)
% for k in range(10):
%     # -forward
%     Xe = F.one_hot(X, num_classes=27).float() # .one_hot does not support d_type=torch.int32
%     W = torch.randn((27,27), generator=g, requires_grad=True)
%     logits = (Xe @ W) # R
%     y_hat = logits.exp() / logits.exp().sum(1, keepdims=True) # softmax = R+/normalize
%     loss = -y_hat[torch.arange(5), Y].log().mean() # todo: pluck out probs?
%     # @, .exp(), / and .sum() are all differentiable
%     print(loss.item())

%     # -backward
%     W.grad = None
%     loss.backward()

%     # -update
%     W.data += -0.1 * W.grad
% ```

% ngrams suffer from curse of dimensionality. it's intractable to get long context lengths.
% but somehow neural networks today have millions of context length.

Returning to the probability space, the sample space is the set of all words,
the event space is the powerset of words (all possible sentence combinations),
so we can formalize the inquiry as

<!-- $$
\begin{align*}
&P(Y=🌭|\textbf{X}=\textbf{x};\theta) := \sigma(\theta^{\top}\textbf{x}) \underset{total\ law}{\implies} P(Y=¬🌭|\textbf{X}=\textbf{x};\theta) = 1 - \sigma(\theta^{\top}\textbf{x}) \\
\implies &P(Y=c|\textbf{X}=\textbf{x};θ) = \hat{y}^y (1-\hat{y})^{1-y} = \sigma(\theta^{\top}\textbf{x})^y [1-\sigma(\theta^{\top}\textbf{x})]^{1-y} \tag*{[continuous]}\\
\end{align*}
$$ -->

<!-- In order to define notions such as expectation, variance, and other computable
estimators, we map the sample space to the real number line using a random variable. -->

Random variables. Probability mass function.
Probability distributions:
Distributions are characterized by their parameters.

running example is autoregressive language modelling.
ngram model becomes intractable. time O(?). space O(?)
In the case of language modelling, ex... probability space is token space...

```python
message = "Hello, world!"

print(message)
```


**Statistics** on the other hand formalizes the recovery of parameters which
characterize the underlying probability distributions that generate the observed
data. Once recovered, we can sample from the distribution it in order to generate
predictions.

*Linear parametric models*:
*Non-linear parametric models*:

show manim animation

## Part 2 — deep learning: from ffn to gpt
```python
"""
Dimension key:
B: batch size
T: sequence length
V: vocabulary size
E: embedding dimension (E != D)
D: model dimension
"""

import picograd
# from jaxtyping import

# *********************MODEL*********************
B, T = 32, 3
V, E, D = 27, 10, 200

class Linear:
  def __init__(self, D_in, D_out, bias=True):
    self.W_DiDo = picograd.randn((D_in, D_out)) * 0.01
    self.b_Do = picograd.zeros(D_out) if bias else None

  def __call__(self, X_Di):
    self.X_Do = X_Di @ self.W_DiDo
    if self.b_Do is not None: self.X_Do += self.b_Do
    self.out = self.X_Do
    return self.X_Do

  def parameters(self):
    return [self.W_DiDo] + ([] if self.b_Do is None else [self.b_Do])

class Tanh:
  def __call__(self, X_BD):
    self.X_BD = picograd.tanh(X_BD)
    self.out = self.X_BD
    return self.X_BD
  
  def parameters(self):
    return []

model = [
  Linear(T * E, D, bias=False), Tanh(),
  Linear(D, D, bias=False), Tanh(),
  Linear(D, V, bias=False)
]

C_VE = picograd.randn((V,E)) #, generator=g)
params = [C_VE] + [p for l in model for p in l.parameters()]
for p in params:
    p.requires_grad = True

print("model loaded to cpu")


# *********************INFERENCE LOOP*********************
for _ in range(20): # 20 samples
  output, context = [], [0] * T
  while True:
    X_1T = picograd.tensor([context]) # B=1 for inference, T=3, in [0..27] (set to 0 for init)
    X_1TE = C_VE[X_1T] # using 0..27 as indices into C_VE for each B=1 example of context length T
    X_1cTE = X_1TE.view(-1, T*E) # B=1, TE
    X = X_1cTE

    for h in model:
      X = h(X)

    y_hat = F.softmax(X, dim=1)

    # sample and autoregressively update context
    token = picograd.multinomial(y_hat, num_samples=1, replacement=True).item()#, generator=g).item()
    context = context[1:] + [token]
    output.append(decode[token])
    if token == 0:
        break
  print(''.join(output))
```







<!-- 0. [ffn]()
1. [rnn]()
2. [lstm]()
3. [gpt]()
4. [beyond gpt]() -->

<!-- ## Part 0: non-linear parametric models: `nn.Linear()`, `nn.ReLU()` -->
<!-- Before jumping into the implementation of our deep learning framework's
multidimensional array with autodifferentiation capability, let's review the
mathematics of neural networks. We will incrementally construct a family of
functions from logistic regression, multiclass regression, feedforward
neural networks, attention and chain of thought variants, all for the
classification setting where Y⊆ℕ.

Recall that the logistic regression model for binary classification recovers the
bernouilli distribution $\mathbb{P}: \mathbb{R}^d \to [0,1]$ by assuming -->

<!-- $$
\begin{align*}
\implies \mathbb{P}(Y=y|\textbf{X}=\textbf{x};θ) &= \hat{y}^y (1-\hat{y})^{1-y} \\
                 &= \sigma(\textbf{w}^{\top}\textbf{x})^y [1-\sigma(\textbf{w}^{\top}\textbf{x})]^{1-y}
\end{align*}
$$ -->