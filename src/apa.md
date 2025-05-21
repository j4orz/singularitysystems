![](./apa.gif)
> *A famous colleague once sent an actually very well-written paper he was quite proud of to a famous complexity theorist. His answer: “I can’t find a theorem in the paper. I have no idea what this paper is about.”*
# Appendix A: 2.0

*This appendix speedruns deep learning prerequisites by using language modeling as a running example.*
By the end of the appendix you should be comfortable with state of the art open
source models such as [llama](https://arxiv.org/abs/2407.21783) and [r1](https://arxiv.org/abs/2501.12948).

**Contents**
1. [preliminaries: probability, algebra, and analysis]()
2. [machine learning: energy](#part-1--statistical-learning-linear-to-non-linear)
3. [deep learning: from ffn to gpt](#part-2--deep-learning-from-ffn-to-gpt)
4. [attention is all you need]()
5. [return of the rl]()

[References](#references)

## Part 1 — preliminaries: probability, algebra, analysis

- [Measure Theory]()
- [Probability Theory]()
- [Linear Algebra]()
- [Matrix Calculus]()

### Measure Theory

By default, mathematical reasoning is understood to be deterministic where
a **statement** $S$ is either true (holds) or false (does not hold). Any
**variable** $x$ can only take on one specific value at a time. However there are
other times where what's desirable is describing phenomena that is in fact
non-deterministic (while still remaining precise). Even if base reality turns
out to be deterministic, in practice there are many scenarios where carrying
out calculation is intractable. i.e, predicting tomorrow's weather with the
position of every water molecule.

The most widely adopted mathematical language for formalizing our intuitions
around non-deterministic stochastic phenomena is probability theory (as opposed
to alternative frameworks such as [probabilistic logic](https://en.wikipedia.org/wiki/Probabilistic_logic) or
[uncertainty quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification)).
In probability theory  **statements** $S$ are neither true nor false. Rather,
truth is distributed across a weighted set of **random events** $E$. Similarly,
**random variables** $X$ do not take on a definite value but rather a set of
values.

The sample space $\Omega$ is the **countably discrete** set all possible
**outcomes** $\omega$ in an experiment. Subsets of this sample space are
**events** which are constructed by placing a boolean operator (corresponding
to some semantic invariant) on the sample space. Functions $X: \Omega \to R$
are **random variables**, where most of the time we use real random variables
so $R=\mathbb{R}$.

In the appendix' running example, consider describing a rappers vocabulary as
some sample space $\Omega=\{ acid, balls, chop, death, grind, mind, poppin, pills, razor, shallow, tequila, zoey \}$. Let $X: \Omega \to \mathbb{R} $ be the r.v that maps words to their
lengths, and let $Y: \Omega \to \mathbb{R}, $ be the r.v that maps the word
$\omega$ to the ordinal-based unicode number of the first letter. So the event
$X=5$ constructs the subset $A=\{balls, death, grind, pills, razor\}$. The event
$Y=112$ constructs the subset $B=\{poppin, pills\}$. The event where $X=5$ and
$Y=112$ constructs the subset $C=A\cup B=\{pills\}$. With sample spaces that are
countably discrete like the rappers vocabulary, we can assign probabilities to
each outcome $\omega \in \Omega$ with $p: \Omega \to [0,1]$ where p needs to
satisfiy *non-negativity*: $\forall \omega \in \Omega, p(\omega) > 0.$
*additivity:*, and *normalization*: $\sum_{\omega \in \Omega}P(\omega) = 1$.

The problem with this model of probability is when the sample space $\Omega$
is **uncountably continuous**. There is no sensible way of adding an uncountable
set of outcomes where $p(\omega)=0$. Since defining probabilities on outcomes
$\omega \in \Omega$ is nonsensical when $\Omega$ is uncountable, the next
approach is to define probabilities on events with $P: \mathcal{F} \to [0,1]$ 
for some class $\mathcal{F}$ of subsets $E \subset \Omega$.

The question that arises naturally for the theoretician is how should
$\mathcal{F}$ be axiomatized, and what properties should $P$ defined on
$\mathcal{F}$ satisfy? We point the reader to explore other resources which
delve deeper into measure-theoretic foundations proposed by Kolmogrov in 1933.

### Probability Theory
Measure theory provides an abstract model for **random events** $E$ and
**random variables** $X$ are measurable subsets and measurable functions of a
measurable space $\Omega$.


Moving forward, we let the probability space fade into the background and focus
on the probabilistic concepts invariant to sample space extensions. Recall that


independance, joint, mutual exclusion, independance, etc.


## Part 2 — machine learning: energy

The goal of **machine learning** in general is to compress regularity found in
data in order to make predictions.

a probabilistic model is a special case of an energy model
more general way than f: X->Y
factors: F: XxY -> R which measure compatibility between input and output

### Energy Models
graphical models: factor graphs are bipartite graphs
bipartite: two different nodes: G=({Variable, Factor}, E)
factor graphs are generally interpreted as probabilistic models. lecun isn't going to do that.















### Decision Theory
Particularly with **supervised learning**,
provided a $d=\{(x^{(i)},y^{(i)})\}_{i=0}^n$ and given previously unseen
input $x \in \mathcal{X}$, the goal is to predict output $y \in \mathcal{Y}$.
Most often $X=\mathbb{R}^d$, whereas $Y=\{0, 1, \ldots, k\} \subseteq \mathbb{N}$
(referred to as **classification**) or $Y=\mathbb{R}$ (referred to as **regression**).
The primary formalization of supervised learning is the probabilistic formulation,
where $(x^{(i)},y^{(i)}) \overset{\text{i.i.d.}}{\sim} (\mathcal{X}\times\mathcal{Y})^n$
is assumed (the joint distribution will be referred to with $P$), and the machine
learning algorithm is then $A: (\mathcal{X}\times\mathcal{Y})^n \to (\mathcal{X} \to \mathcal{Y})$
where $A(d) \mapsto f: \mathcal{X} \to \mathcal{Y}$.

The next component of the probabilistic formulation is the **loss** function
and its expectation (referred to ask **risk**). The loss function is
$l: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{+}$ which quantifies the error
when a mismatch occurs between the **actual predicted output** and the
**expected labeled output**. along with its expectation (refered to
as **risk**).

- loss function
- risk
- bayes predictor

### Statistical Learning Theory


### Example 1:  "fresh" lyrics crodie

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

### Sequence Modeling (Autoregressive Models)

### Diffusion Modeling
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


## Part 3 — attention is all you need

Within the GPTs, we saw the rise of [sparse attention](https://arxiv.org/abs/1904.10509).
BERT variants showed us how [layer normalization](https://arxiv.org/abs/1607.06450) evolved into [RMSNorm](https://arxiv.org/abs/1910.07467).
LLaMA iterations demonstrated the progression from standard attention to [grouped-query attention](https://arxiv.org/abs/1910.07467).
DeepSeek’s releases, particularly constrained by hardware limitations, showed algorithms that enabled frontier performance without frontier compute.




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



## References

*Probability Theory*
1. [Varadhan 2001](https://www.ams.org/books/cln/007/cln007-endmatter.pdf)
1. [Chan 2021](https://probability4datascience.com/)
1. [Piech 2024](https://chrispiech.github.io/probabilityForComputerScientists/en/)
1. [Harchol-Balter 2024](https://www.cs.cmu.edu/~harchol/Probability/book.html)

*Statistical Learning*
1. [Hardt, Recht 2022](https://mlstory.org/)
1. [Recht 2023](https://www.argmin.net/p/patterns-predictions-and-actions)
1. [Bach 2024](https://www.di.ens.fr/~fbach/ltfp_book.pdf)

*Matrix Calculus*
1. [Kang, Cho 2024](https://kyunghyuncho.me/linear-algebra-for-data-science/)
1. [Scardapane 2024](https://www.sscardapane.it/alice-book/)
1. [Bright, Edelman, Johnson 2025](https://arxiv.org/abs/2501.14787)

*Machine Learning*
1. [Ng, Ma 2023](https://cs229.stanford.edu/main_notes.pdf)
1. [Ermon, Grover 2023](https://deepgenerativemodels.github.io/notes/index.html)
1. [Cho 2025](https://arxiv.org/abs/2505.03861)

*Deep Learning*
1. [Cho 2015](https://arxiv.org/abs/1511.07916)
1. [Goodfellow et al. 2016](https://www.deeplearningbook.org/)
1. [Manning 2019](https://web.stanford.edu/class/cs224n/readings/)
1. [Jurafsky, Martin 2025](https://web.stanford.edu/~jurafsky/slp3/)
1. [Prince 2025](https://udlbook.github.io/udlbook/)