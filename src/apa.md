
# A. Posteriors and Parameters
> *God does not play dice.*

<!-- *This appendix covers machine learning prerequisites by using language modeling as a running example.*
By the end of the appendix you should be comfortable with state of the art open
source models such as [llama](https://arxiv.org/abs/2407.21783) and [r1](https://arxiv.org/abs/2501.12948). -->

**Contents**
- [A.1 Posterior Updates](#probability-theory)
    - [Probability Theory]()
        - [Probability Space: Measurable Space + Measure](#probability-space)
        - [Discrete Random Variables: $Bin(n,p)$, $Poi(\lambda)$, $Geo(p)$]()
        - [Continuous Random Variables: $Uni(\alpha,\beta)$, $Exp(\lambda)$, $Nor(\mu, \sigma^2)$]()
        - [Joint Random Variables: Product Rule, Sum Rule, Bayes' Rule](#rulez-rule-product-rule-sum-rule-bayes-rule)
    - [Linear Algebra]()
- [A.2 Parameter Estimation](#statistics)
    - [Statistics]()
        - [Maximum Likelihood Estimation (MLE)](#maximum-likelihood-estimation-mle)
        - [Maximum a Posterior Estimation (MAP)](#maximum-a-posterior-estimation-map)
        - [Empirical Risk Minimization (ERM)](#empirical-risk-estimation-erm)
    - [Matrix Calculus](#matrix-calculus)
        - [Derivatives, Gradients, and Jacobians](#derivatives-gradients-and-jacobians)
        - [Automatic Diffrentiation: Sum Rule, Product Rule, Chain Rule](#automatic-differentiation)
    - [Optimization](#optimization)
        - [Gradient Descent]()
        - [Stochastic Gradient Descent]()
        - [Adam]()
        - [Muon]()

# A.1 Posterior Updates
## Probability Theory

By default, mathematical reasoning is understood to be deterministic where
a **statement** $S$ is either true (holds) or false (does not hold). Any
**variable** $x$ can only take on one specific value at a time. However there are
other times where what's desirable is describing phenomena that is in fact
non-deterministic (while still remaining precise). Even if base reality is
fundamentally deterministic, in practice there are many scenarios where carrying
out calculation is intractable. i.e, predicting tomorrow's weather with the
position of every water molecule.

The most widely adopted mathematical language for formalizing our intuitions
around non-deterministic stochastic phenomena is probability theory (as opposed
to alternative frameworks such as [probabilistic logic](https://en.wikipedia.org/wiki/Probabilistic_logic) or
[uncertainty quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification)).
In probability theory  **statements** $S$ are neither true nor false. Rather,
truth is distributed across a weighted set of **random events** $E$. Similarly,
**random variables** $X$ and their **probability distributions** $p$ do not take
on definite values but rather sets of values. Probability theory is the
generalization of aristotelian logic.


### Probability Space: Measurable Space + Measure
---

These three **concrete implementations** create the toolbox which serve as
workhorses for probability theory and require an **abstract interface**. The
Kolmogorov's 1933 formulation of probability theory in
[Grundbegriffe der Wahrscheinlichkeitsrechnung](https://archive.org/details/kolmogoroff-1933-grundbegriffe-der-wahrscheinlichkeitsrechnung/page/10/mode/2up)
is the universally accepted model for random events and random variables.
The guiding light in the axiomaticization are the first two out of three properties
that $p$ should satisfy: *non-negativity* and *normalization*, which formalize
*impossible (surely false)* and *guaranteed (surely true)* statements. They
ensure that probabilities are never less than 0 nor greater than 1.

The foundations start with an ambient sample space $\Omega$ which is the set of all
possible **outcomes** $\omega$ in an experiment. It's natural to expect the
probability function $p$ to be defined on $\Omega$, which is acceptable when
the sample space is **countable**. That is, when $|\Omega|<\aleph_1$,
the first two properties are possible to satisfy:
- **non-negativity:** $\forall \omega \in \Omega, P(\omega) > 0.$
- **normalization:** $\sum\limits_{i=0}^{\infty} P(\omega_i)=1.$

but when the sample space is **infinitely uncountable** then normalization is not possible.
That is, when $|\Omega| = \aleph_1$, TODO... The next approach then is to define
probabilities for the powerset $\mathcal{F}=2^\Omega$ with $p: \mathcal{F} \to [0,1]$.
The powerset $F$ is the set of all subsets, and is referred to as the
**event space**. The event space must be a **field**. That is, it must be
**closed** under union, intersection, and complement, satisfying:
- $\varnothing \in \mathcal{F} \land \Omega \in \mathcal{F}$.
- $A \in \mathcal{F} \implies \Omega \setminus A\in \mathcal{F}$.
- $A \in \mathcal{F} \land B \in \mathcal{F} \implies A \cup B \in \mathcal{F} \land A \cap B \in \mathcal{F} $.

This formulation of $p$ is now defined on $E \subset \Omega$ but still does not
support sample spaces where $|\Omega|=\aleph_1$ let alone $|\Omega|=\aleph_0$.
The property of *closure* of union and intersection need to be extended to support
sample spaces that are countable and uncountable infinite. Starting with the
former, a field $F$ that satisfies
- $E_1, E_2, \ldots \in \mathcal{F} \implies \bigcup\limits_{i=1}^{\infty}E_i \in \mathcal{F} \land \bigcup\limits_{i=1}^{\infty} \in \mathcal{F}$.

is then a **$\sigma$-field** and is defined for $|\Omega|=\aleph_0$. This 
**$\sigma$-field** with the sample space $\Omega$ form a **measurable space**.
Finally, the original goal that motivated the need to redefine $p$'s domain from
$\Omega$ to $\mathcal{F}=2^\Omega$ was to support uncountable infinities where
$|\Omega|=\aleph_1$. This motivates the need to add additional properties the
event space must satisfy which is...
this is a borel sigma field.

Returning to the probability function $p$, the original first two properties are
satisified, in addition to a third and final one:
- **non-negativity:** $\forall \omega \in \Omega, P(\omega) > 0.$
- **normalization:** $\sum\limits_{i=0}^{\infty} P(\omega_i)=1.$
- **additivity:**

A measurable sapce $(\Omega, \mathcal{F})$ with a measurable function $p: \mathcal{F} \to [0,1]$
is the abstract model for a **probability space**.

<!-- going back to $p$, this lets us normalize on sample spaces that are uncountably infinte?
we add a third property here?:
 whereas the third property — arguably
the most crucial — ensures we can add probabilities when two statements don't
"overlap".

Subsets of this sample space are **events** which are constructed by placing a boolean operator
(corresponding to some semantic invariant) on the sample space. Functions
$X: \Omega \to \mathbb{R}$ are **random variables**.. -->

##### Example 1: finite probability space
---

In the notebook's running example, consider describing a rappers vocabulary as
some sample space $\Omega=\{ acid, balls, chop, death, grind, mind, poppin, pills, razor, shallow, tequila, zoey \}$,
and the following probability assignment (take a moment to confierm that they
are all *non-negative* and *normalize* to 1).
- $P(\{acid\})=0.04$
- $P(\{balls\})=0.01$
- $P(\{chop\})=0.09$
- $P(\{death\})=0.08$
- $P(\{grind\})=0.08$
- $P(\{mind\})=0.05$
- $P(\{poppin\})=0.04$
- $P(\{pills\})=0.19$
- $P(\{razor\})=0.03$
- $P(\{shallow\})=0.22$
- $P(\{tequila\})=0.15$
- $P(\{zoey\})=0.02$

Let $X: \Omega \to \mathbb{R} $ be the r.v that maps words to their
lengths, and let $Y: \Omega \to \mathbb{R}, $ be the r.v that maps the word
$\omega$ to the ordinal-based unicode number of the first letter

- the rv taking on value $X=5$ constructs the event modeled by the subset $A=\{balls, death, grind, pills, razor\}$ and the p
- the rv taking on value $Y=112$ constructs the event modeled by the subset $B=\{poppin, pills\}$


The event where $X=5$ and
$Y=112$ constructs the subset $C=A\cup B=\{pills\}$. The event where $X=5$ or
$Y=112$ constructs the subset $C=A\cap B=\{balls, death, grind, poppin, pills, razor\}$


##### Example 2: continuous probability space
---

### Discrete Random Variables

### Continuous Random Variables

### Rulez Rule: Product Rule, Sum Rule, Bayes' Rule

With an axiomaticization in place where **random events** $E$ and **random variables**
$X$ are modeled as measurable subsets and measurable functions of a measurable space $\Omega$,
the focus is now on the core — invariant to sample space modifications — probabilistic
concepts that comprise the notion of updating beliefs.

It all starts with of **conditional** probability, which is the probability of some
event $A$ **given** that event $B$ was **observed**. If the Kolmogorov
axiomaticization admits modeling stochastic events $A$ and $B$ as a venn diagram
within an ambient sample space, then **conditional** probability changes the
fidelity of description so that $B$ is the new (and narrowed) "sample space"
being considered. That is, probability of $A$ given $B$ is denoted with
$P(A\mid B)$ and is defined as

$$P(A\mid B)\stackrel{\triangle}{=}\frac{P(A\cap B)}{P(B)}$$

and can itself be proved a valid probability space. In other scenarios
$P(A\mid B)$ is given, but $P(A\cap B)$ is not. This is
amenable by rearranging the definition and solving for the latter with
$P(A\cap B) = P(B)P(A\mid B)$. and is referred to as the **product rule**
(emphasis on the product) or **chain rule** (emphasis on the factors).

In other scenarios $P(A\mid B)$ is given (and now $P(A \cap B)$ by the product rule),
but $P(A)$ is not. This too is amenable by using event $B$ as a "background event",
where assuming events $B_1, B_2, \ldots B_n$ are mutually exclusive, then
$P(A) = P(A \cap B_1) +  P(A \cap B_2) + \ldots + P(A\cap B_N)$. This is referred
to as the **sum rule**, and can be composed with the product rule to further
decompose the terms of products into terms of conditionals.

Finally, in other scenarios where $P(A\mid B)$ is given but $P(B\mid A)$ is not, this
is amenable by rewriting the numerator and denominator of conditional probability
using the product rule and sum rule:

$$
\begin{aligned}
P(A\mid B) &= \frac{P(A\cap B)}{P(B)}                         
           && \text{[by definition of conditional]}\\
           &= \frac{P(A)\,P(B\mid A)}{P(B)}                    
           && \text{[by product rule on joint]}\\
        %    &= \frac{P(A)\,P(B\mid A)}
        %           {P(B\cap A)+P(B\cap\neg A)}
        %    && \text{[by sum rule on marginal]}\\
        %    &= \frac{P(A)\,P(B\mid A)}
        %           {P(A)P(B|A)+P(\neg A)P(B|\neg A)}
        %    && \text{[by product rule]}
\end{aligned}
$$

$$
\underbrace{p(x \mid y)}_{\text{posterior}}
\;=\;
\frac{
    \overbrace{\,p(y \mid x)\,}^{\text{likelihood}}
    \;
    \overbrace{p(x)}^{\text{prior}}
}{
    \underbrace{\displaystyle\int p(y \mid x)\,p(x)\,dx}_{\text{evidence}}
}\;.
$$

and is referred to as **bayes' rule**. The scenario where
$P(A\mid B)$ is accessible but $P(B\mid A)$ is not comes up many times practice,
where $A$ is some **latent** event and $B$ is an **observable** event. This
happens in science (why bayes rule is also referred to as the *logic of science*)
where $P(A\mid B)$ is used to model **hypothesis** and **evidence** where the
**posterior** $P(H\mid E)$ (latent given observed) is updated by
evaluating the product of the **prior** $P(H)$ and the **likelihood**
(observed given latent) $P(E\mid H)$ normalized by the **evidence** $P(E)$.


##### Example 3: conditional probability
---
Let $\Omega$ be the sample space used from the previous examples,
$Y: \Omega \to \mathbb{R}$ once again be the r.v that maps the word
$\omega$ to the ordinal-based unicode number of the first letter, and let
$Z: \Omega \to \mathbb{R}$ be a new identity r.v $Z$ that maps the word
to it's hash. What's $P(Z=hash("pills")|Y=112)$? The answer by definition
of conditional probability is $\frac{0.19}{0.04+0.19}=0.83$, where
$P(Y=112)=0.04+0.19$ because of *additivity*.


todo: examples 4/5/6: product rule/sum rule/bayes rule


### Inference: Joint Random Variables

```python
from jax import numpy as jnp
from jaxtyping import Array, Float

def bayes(joint: Float[Array, "dx dy"]) -> Float[Array, "dx dy"]:
    prior = 0
    likelihood = 0
    evidence = 0

    posterior = 0
    return prior, likelihood, evidence, posterior
```

# A.2 Parameter Estimation
## Statistics
In probabilistic models distributions of latent hypotheses are usually unknown
and their posteriors are inferred with bayes rule by taking the product of the
prior $p(h)$, likelihood $p(e,h)$, and normalizing by the evidence $p(e)$. When
these latter distributions (more specifically, the parameters $\theta$ that
characterize them), the parameters need to be recovered from data via
**parameter estimation**.

### Empirical Risk Minimization (ERM)
Risk is a quantity where the objective is to minimize.
Reward is a quantity where the objective is to maximize.

loss functions and risk.
- theta_likelihood from L_likelihood (likelihood from the probabilistic perspective)
- theta_0-1 from L_01
- theta_absolute from L_abs

### Unbiased Estimators

### Maximum Likelihood Estimation (MLE)

Maximizing the likelihood uses the probability model itself as the loss function.
Following the notion of minimizing risk, the maximization corresponds to
minimizing the negative likelihood $-p(x; \theta)$. One more transformation
is applied by mapping the likelihood to log space which simplifies the
$\operatorname{argmin}$ routine by decomposing the joint probability of the data
$p(x_1, x_2, \ldots, x_n; \theta)$ into a product of likelihoods with the
*chain rule* (assuming $X_1, X_2, \ldots, X_N\overset{\text{iid}}{\sim}p$). The
parameters which minimize negative log likelihood are the same that minimize
negative likelihood since logarithms perserve **monotonicity**. All together,
the optimization problem is:

$$
\begin{aligned}
\hat{\boldsymbol{\theta}} &\in \underset{\boldsymbol{\theta} \in \Theta}{\operatorname{argmax}}\ p(x;\theta) \\
                          &= \operatorname{argmax}\ \log p(x_1,x_2, \ldots, x_n;\theta)
                          && \text{[by monotonicity of log]}\\
                          &= \operatorname{argmin}\ -\log p(x_1,x_2, \ldots, x_n;\theta) \\
                          &= \operatorname{argmin}\ -\log \prod_{i=1}^{n}p(x_i;\theta)
                          && \text{[by $X_i \overset{\text{iid}}{\sim}p$]} \\
                          &= \operatorname{argmin}\ - \sum_{i=1}^{n}\log p(x_i;\theta)
                          && \text{[by log laws]} \\
\end{aligned}
$$

In the following examples $\operatorname{argmin}$ is implemented symbolically by
solving for $\frac{d}{d\theta}=0$.

##### Example 1: maximizing likelihood with $Poi(\lambda)$
---
Assume $X_1, X_2, \ldots, X_n \overset{\text{iid}}{\sim} Poi(\lambda).$
Then using negative log likelihood as the loss function corresponds to the
following optimization:

$$
\begin{aligned}
\hat{\lambda}_{MLE} &\in \operatorname{argmin} \mathcal{L(\lambda)} \\
                          &= \operatorname{argmin} - \sum_{i=1}^{n} \log \frac{e^{-\lambda}\lambda^{x_i}}{x_i!} \\
                          &= \operatorname{argmin} - \sum_{i=1}^{n} -\lambda + x_i\log\lambda - \log x_i!
\end{aligned}
$$

and solving for the arguments $\lambda$ which minimize the expression corresponds
to taking the derivative with respect to $\lambda$ and solve for those that equal 0:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \lambda} &= 0 \\
    - \sum_{i=1}^{n} -1 + \frac{x_i}{\lambda} &= 0 \\
     n - \frac{1}{\lambda} \sum_{i=1}^{n} x_i &= 0 \\
               \frac{1}{n} \sum_{i=1}^{n} x_i &= \lambda_{MLE}

\end{aligned}
$$

##### Example 2: maximizing likelihood with $Ber(p)$
---
Assume $X_1, X_2, \ldots, X_n \overset{\text{iid}}{\sim} Ber(p).$
Then using negative log likelihood as the loss function corresponds to the
following optimization:

$$
\begin{aligned}
\hat{p}_{MLE} &\in \operatorname{argmin} \mathcal{L(p)} \\
                          &= \operatorname{argmin} - \sum_{i=1}^{n} \log[p^{x_i}(1-p)^{1-x_i}] \\
                          &= \operatorname{argmin} - \sum_{i=1}^{n} x_i\log p + (1-x_i)\log(1-p) \\
\end{aligned}
$$

and solving for the arguments $p$ which minimize the expression corresponds
to taking the derivative with respect to $p$ and solve for those that equal 0:

$$
\begin{aligned}
                                                  \frac{\partial \mathcal{L}}{\partial p} &= 0 \\
    \frac{\partial \mathcal{L}}{\partial p} - \sum_{i=1}^{n} x_i\log p + (1-x_i)\log(1-p) &= 0 \\
                    \frac{\partial \mathcal{L}}{\partial p} -[X\log p + (n - X)\log(1-p)] &= 0 && \text{[where $X = \sum_{i=1}^{n}x_i$]} \\
                                                           \frac{-X}{p} + \frac{n-X}{1-p} &= 0 \\
                                                                         -X + Xp + np -Xp &= 0 \\
                                                                         np &= X \\
                                                                          p_{MLE} &= \frac{1}{n}\sum_{i=1}^{n}x_i \\


\end{aligned}
$$



##### Example 3: maximizing likelihood with $Nor(n, \sigma^2)$
---
Assume $X_1, X_2, \ldots, X_n \overset{\text{iid}}{\sim} Nor(\mu, \sigma^2).$
Then using negative log likelihood as the loss function corresponds to the
following optimization:

$$
\begin{aligned}
\hat{\theta}_{MLE} &\in \operatorname{argmin} \mathcal{L(\theta)} && \text{[where $\theta=(\mu, \sigma^2)$]}\\
                          &= \operatorname{argmin} - \sum_{i=1}^{n} \log[p^{x_i}(1-p)^{1-x_i}] \\
\end{aligned}
$$

and solving for the arguments $p$ which minimize the expression corresponds
to taking the derivative with respect to $p$ and solve for those that equal 0:

$$
\begin{aligned}
                                                  \frac{\partial \mathcal{L}}{\partial p} &= 0 \\
    \frac{\partial \mathcal{L}}{\partial p} - \sum_{i=1}^{n} x_i\log p + (1-x_i)\log(1-p) &= 0 \\
                    \frac{\partial \mathcal{L}}{\partial p} -[X\log p + (n - X)\log(1-p)] &= 0 && \text{[where $X = \sum_{i=1}^{n}x_i$]} \\
                                                           \frac{-X}{p} + \frac{n-X}{1-p} &= 0 \\
                                                                         -X + Xp + np -Xp &= 0 \\
                                                                         np &= X \\
                                                                          p_{MLE} &= \frac{1}{n}\sum_{i=1}^{n}x_i \\


\end{aligned}
$$


### Maximum a Posterior Estimation (MAP)

## Matrix Calculus
Recall that for some scalar function $f: \mathbb{R} \to \mathbb{R}$ that the
derivative is usually defined as $\frac{df}{dx} = f'(x)$ where the **derivative**
$f'(x)$ is the **ratio** between the **differentials** (**inifinitesmal differences**)
of input $dx$ and output $df$. While $/: \mathbb{R} \to \mathbb{R}$ is defined
on $\mathbb{R}$, this is not always the case for higher dimensional euclidean
spaces.

Moreover, the problem with the formulation of derivatives as $\frac{df}{dx}$ is
not limited to denotational semantic lawyering — there's also a problem with the
derivative **rules** themselves. Consider the function
$f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n}$, $f(\mathbf{X}) := \mathbf{X}^2$
which squares a matrix. The power rule of scalar functions does not generalize here,
because $f'(x) \neq 2\mathbf{X}$.

Defining $f'(x)$ as a ratio $f'(x) = \frac{df}{dx}$ does not generalize. Instead, the
correct approach is to define the derivative as **linearization** which defines
the derivative as the linear operator you apply to a change in output to receive
a change in output. That is, $df = f'(x)dx$. This formulation of the derivative
generalizes **scalar** and **vector** differential calculus over
$\mathbb{R}$ and $\mathbb{R}^n$ into higher dimensional vector spaces for
**matrix** and **tensor** calculus over $\mathbb{R}^{n\times m}$ and
$\mathbb{R}^{d_0 \times d_1 \cdots \times d_n}$. In essence, linearization
is locally approximating complex surfaces in vector spaces $\mathcal{V}$
(more specifically, **Banach spaces** equipped with norm $||x||$) with
**linear operators** $L: \mathcal{V} \to \mathcal{V}$ ($L[v_1+v_2] = L[v_1] + L[v_2]$) so that $\Delta \text{out} = L[\Delta\text{in}]$.

### Derivatives, Gradients, and Jacobians
- for some function $f: \mathbb{R} \to \mathbb{R}$, $f'(x)$ is the term such that
$df = f'(x)[dx]$. For this expression to be defined,
$dx \in \mathbb{R} \land df \in \mathbb{R} \implies f'(x) \in \mathbb{R} \land [\bullet]:= *$ where $*: \mathbb{R} \to \mathbb{R}$.
- for some function $f: \mathbb{R}^n \to \mathbb{R}$, $f'(\mathbf{x})$ is the term such that
$df = f'(\mathbf{x})[d\mathbf{x}]$. For this expression to be defined,
$d\mathbf{x} \in \mathbb{R}^n \land df \in \mathbb{R} \implies f'(\mathbf{x}) \in \mathbb{R}^n \land [\bullet]:= \cdot$ where $\cdot: \mathbb{R}^n, \mathbb{R}^n \to \mathbb{R}$. In this case, $f'(\mathbf{x})$ is often referred to as the **gradient** of $f$ and denoted by $\nabla f$ 
- for some function $f: \mathbb{R}^n \to \mathbb{R}^m$, $f'(\mathbf{x})$ is the term such that
$d\mathbf{f} = f'(\mathbf{x})[d\mathbf{x}]$. For this expression to be defined,
$d\mathbf{x} \in \mathbb{R}^n \land d\mathbf{f} \in \mathbb{R}^m \implies f'(\mathbf{x}) \in \mathbb{R}^{m\times n} \land [\bullet]:= @$ where $@: \mathbb{R}^n \to \mathbb{R}^m$. In this case, $f'(\mathbf{x})$ is often referred to as the **jacobian** of $f$ and denoted by $Jf$ 

numerical example.

### Automatic Differentiation
as opposed to symbolic or numerical.

## Optimization

### Gradient Descent

### Stochastic Gradient Descent

### Adam

### Muon



and more generally we define the derivative *as* the linear operator L on vector
space V which you apply to a change in input in order to obtain the change in
output. That is, Δout = (lin.op)[Δin]. which generalizes to higher dimensions.
While defining f'x ≜ df/dx is legal when f: ℝ -> ℝ, it's usually not clear what
df/dx means in higher dimensions, since /: ℝ^n, R^m -> ? is not defined.

f: ℝ^d -> ℝ
-----------
Consider f'(x) when f: ℝ^d -> ℝ. Then, given that dx ∈ ℝ and df ∈ ℝ^d, the linear
operator f'(x) *must* (formalized by the Riecz-Frchét Representation Theorem) be
the dot product with some column vector, (multiplication with the column vector
transposed). Let's call this column vector the "gradient" of f, where the
component-wise definition with indices is:
```
    ∇f ∈ ℝ^d, f'(x) = ∇fᵀx
    ∇f_i ≜ ∂f/∂x_i ∀ i ∈ [0..d]
and all together, we use f'(x) to linearize df:
    f(x+dx)-f(x) = df
                 = f'(x)dx
                 = ∇fᵀdx
```

f: ℝ^m -> ℝ^n
-------------
Moving on, consider f'(x) where f: ℝ^m -> ℝ^n. Then given that dx ∈ ℝ^m and
df ∈ ℝ^n, the linear operator f'(x) *must* (formalized by the Riecz-Frchét
Representation Theorem once again) expressable as a matrix multiplication with
some matrix A ∈ ℝ^(mxn). Let's call this matrix the "jacobian", where the
component-wise definition with indices is the following:
```
    Jf ∈ ℝ^(nxm), f'(x) = Jdx
    Jf_ij ≜ ∂fi/∂xj ∀ i ∈ [0..n], j ∈ [0..m]
and all together, we use f'(x) to linearize df:
    f(x+dx)-f(x) = df
                 = f'(x)dx
                 = Jdx
```

f: ℝ^m, ℝ^n -> ℝ^p [f: ℝ^(nxm), ℝ^(mxp) -> ℝ^(nxp)]
---------------------------------------------------
For example, given f: ℝ^m -> ℝ^n, f(x) := Ax
```
==> df = f(x+dx)-f(x)
       = (Ax+dx)-Ax
       = Adx
==> f'(x) = A = J
```

While it's very elegant that the derivative of Ax turns out to be A, training
neural networks requires that is A is not just constant, but variable with
respect to f. To make the deep learning context clearer, we will use W instead
of A:
```
  fi(W,x) ≜ ΣWij*xj
    f(x) := [f1(x), f2(x), ..., fn(x)]
but we can just promote x to be a matrix too. Finally, consider f'(x) when
f: ℝ^(nxm), ℝ^(mxp) -> ℝ^(nxp)

ℒ: ℝ^d0 -> ℝ ∋ ℒ(Z) where Z := X @ Y
==> dℒ = dℒ/dZ : dZ                             [by Riecz. Representation Theorem]
       = dℒ/dZ : (dX@Y + X@dY)                  [by product rule on dZ]
       = dℒ/dZ : (dX@Y) + dℒ/dZ : (X@dY)        [: linear]
       = (dℒ/dZ@Yᵀ) : dX + (Xᵀ@dℒ/dZ) : dY      [by TODO...circularity of trace]                              
==> dℒ/dX = dℒ/dZ@Yᵀ ∧ dℒ/dY = Xᵀ@dℒ/dZ
```






Stepping back, defining the derivative this way generalizes the rise over run with
scalars, the gradient with vectors, and the jacobian with matrices into the same
notion: they are the linear operators defined on vector spaces that you apply to
a change in input in order to obtain the change in output, and the types/shapes
are defined the way they are out of necessity, rather than by fiat. Moreover,
this also sharpens the semantics of the denotation in higher dimensions given
that df/dx is often ill-defined. i.e. while df/dx makes sense when df, dx ∈ ℝ^n,
dividing tensors with mismatched shapes becomes nonsensical.

This also means that this definition extends to any vector space where a linear
operator is defined, for instance, the space of matrices (vectorization of
matrices), and the space of functions (calculus of variations).

While training deep neural networks do not involve functions that map
matrices to matrices or functions to functions, defined this way it's clear
that many of the "grads" we take in pytorch are actually jacobians that all
compose into one single gradient for the entire expression graph which represents
some function f: ℝ^n -> ℝ.

2b — Derivative rules: backward methods
---------------------------------------
defining rules for any f: V->V

```
Sum rule
--------
f: V -> V
f(x) := g(x) + h(x)
==>      df = dg + dh
    f'(x)dx = g'(x)dx + h'(x)dx   [by def]
      f'(x) = g'(x) + h'(x)       [/ dx]

Product rule
------------
f: V -> V
f(x) := g(x)h(x)
==> df = f(x+dx)-f(x)
       = g(x+dx)h(x+dx)-g(x)h(x)                        [by def of f]
       = [g(x)+g'(x)dx][h(x)+h'(x)dx]-g(x)h(x)          [by def of i(x+dx)=i(x)+i'(x)]
       = g(x)h(x)                                       [expand product]
         + g(x)h'(x)dx
         + h(x)g'(x)dx
         + g'(x)h'(x)(dx)^2
       - g(x)h(x)
       = g(x)h'(x)dx + h(x)g'(x)dx + g'(x)h'(x)(dx)^2   [g(x)h(x) cancels]
       = g(x)h'(x)dx + h(x)g'(x)dx                      [g'(x)h'(x)(dx)^2 -> 0]
       = [g(x)h'(x) + h(x)g'(x)]dx
       = f'(x)dx

Chain rule
----------
f: V -> V
f(x) := g[h(x)]
==> df = f(x+dx)-f(x)
       = g[h(x+dx)]-g[h(x)]
       = g[h(x)+h'(x)dx]-g[h(x)]                  [by def of h(x+dx)=h(x)+h'(x)]
       = g'[h(x)]h'(x)dx                          [g(x+dx)-g(x) = dg = g'(x)dx
                                                   where x=h'(x)dx]
```

2c — Automatic differentiation: calculus on computational graph
---------------------------------------------------------------
Recall that we need to evaluate the gradient of a neural network's loss
function in order to train it with parameter estimation. The method used to
compute derivatives is automatic differentiation — an algorithmic technique as
opposed to symbolic techniques that are inefficient or numeric ones that are
inaccurate. Given some expression graph that represents a neural network
f: ℝ^d_0 -> ℝ, autodiff will recursively apply the chain rule for each
subexpression h: ℝ^d_i -> ℝ^d_j.

```
c = a*b (1)
d = e+f (2)
g = c*d (3)

 .-.
( a <---+
 `-'    |   * .-. (1)
 .-.    +----( c <---+
( b )<--+     `-'    |
 `-'                 |    * .-. (3)
 .-.                 +-----( g )
( e )<--+   / .-. (2)|      `-'
 `-'    +----( d )<--+
 .-.    |     `-'
( f )<--+
 `-'
Fig 4. expression graph
```

Notice in figure 4 that even though evaluation of the expression is denoted from
left to right, the edges of the graph form a directed acyclic graph (tree) rooted
at the output of the expression which simplifies graph modification.

Interpreting each step of computation (1, 2, 3) is referred to as the
"forward pass" and constructs the graph in memory which either creates a new tree
amongst a forest (2) or roots the forest into a single tree (3).

Evaluating the derivative of the output with respect to all inputs is referred
to as the "backward pass" — as opposed to "second forward pass" because deep
learning frameworks implement reverse-mode differentiation as opposed to
forward-mode.

The difference between forward-mode and reverse-mode differentiation is the
direction you step through the graph which can end up influencing speed depending
on the shape of the function. Stepping from inputs to outputs is considered a
forward traversal, and the opposite is considered to be a backward traversal.

For instance, consider the applicaiton of the chain rule when evaluating the
f'(x) for f(x) := cos(x^2). Which subexpression do you derive first? Evaluating
d/dx[cos(x)] = sinx first is reverse mode, and evaluating d/dx[x^2] = 2x first
is forward mode. With neural networks, most expression graphs represent functions
of the form f: ℝ^n -> ℝ, and so the default traversal for all deep learning
frameworks is reverse mode.

However, regardless of the order of traversal, the effective computation is the
same: a summation over all possible products of paths. The base case starts
with ∂f/∂f = 1, propagating the local dout/din, and recursively applies the chain
rule for each subexpression. This is why reverse-mode differentiation is also
referred to as "backpropagation". Stepping through the backward for the same
function f as above where d denotes the global derivative and ∂ denotes the local
derivative (overloading ∂'s classic usage for partial derivative)

```
---forward
c = a*b (1)
d = e+f (2)
g = c*d (3)

---backward
∂g/∂g = 1 [base case] (4)

dg/dc = ∂g/∂g*∂g/∂c           [chain rule]
      = ∂g/∂g*d               [product rule]
      = d                     [∂g/∂g cached] (5)

dg/dd = ∂g/∂g*∂g/∂d           [chain rule]
      = ∂g/∂g*c               [product rule]
      = c                     [∂g/∂g cached] (6)

dg/da = ∂g/∂g*∂g/∂c*∂c/∂a     [chain rule]
      = ∂g/∂g*∂g/∂c*b         [product rule]
      = d*b                   [∂g/∂g*∂g/∂c cached] (7)

dg/db = ∂g/∂g*∂g/∂c*∂c/∂b     [chain rule]
      = ∂g/∂g*∂g/∂c*a         [product rule]
      = d*a                   [∂g/∂g*∂g/∂c cached] (8)

dg/de = ∂g/∂g*∂g/∂d*∂d/∂e     [chain rule]
      = ∂g/∂g*∂g/∂d*1         [sum rule]
      = c                     [∂g/∂g*∂g/∂d cached] (9)

dg/df = ∂g/∂g*∂g/∂d*∂d/∂f     [chain rule]
      = ∂g/∂g*∂g/∂d*1         [sum rule]
      = c                     [∂g/∂g*∂g/∂d cached] (10)

 .-. (7)
( a <---+
 `-'    |   * .-. (1) (5)
 .-.    +----( c <---+
( b )<--+     `-'    |
 `-' (8)             |    * .-. (4)
 .-. (9)             +-----( g )
( e )<--+   + .-. (2)|(6)   `-'
 `-'    +----( d )<--+
 .-.    |     `-'
( f )<--+
 `-'(10)
 ```

What's delightful from a user experience point of view is that autodifferentiation
allows the user to implicitly construct the expression graph by specifying the
forward pass, and the backward pass is abstracted away with an output.backward().

TODO: show expression graph for a neural network.
TODO: let's now revisit the definition of the derivative (2b) and rederive a few derivative rules (2c)
- +(tensor, tensor)
- @(tensor, tensor)
- tanh(tensor)


2d — Advanced AD: jacobian-vector and vector-jacobian products
--------------------------------------------------------------

2e — Gradient descent: non-convex optimization
----------------------------------------------
Finally, recall evaluating the derivative of the computational graph's output
with respect to all inputs is used to implement argmin ℒ(Θ) by iteratively
adjusting θ̂ via gradient descent:
    θ^(t+1) := θ^t - α∇ℒ(Θ)
             = θ^t - α∇-ΣlogP(y^i|x^i;Θ)