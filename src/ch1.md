
# 1. Mathematics
> *God does not play dice.*

<!-- *This appendix covers machine learning prerequisites by using language modeling as a running example.*
By the end of the appendix you should be comfortable with state of the art open
source models such as [llama](https://arxiv.org/abs/2407.21783) and [r1](https://arxiv.org/abs/2501.12948). -->

**Contents**
- [Probability Theory](#probability-theory)
    - [Probability Space: Measurable Space + Measure](#probability-space)
    - [Discrete Random Variables: $Bin(n,p)$, $Poi(\lambda)$, $Geo(p)$]()
    - [Continuous Random Variables: $Uni(\alpha,\beta)$, $Exp(\lambda)$, $Nor(\mu, \sigma^2)$]()
    - [Joint Random Variables: Product Rule, Sum Rule, Bayes' Rule](#rulez-rule-product-rule-sum-rule-bayes-rule)
- [Statistics](#statistics)
    - [Maximum Likelihood Estimation (MLE)](#maximum-likelihood-estimation-mle)
    - [Maximum a Posterior Estimation (MAP)](#maximum-a-posterior-estimation-map)
    - [Empirical Risk Minimization (ERM)](#empirical-risk-estimation-erm)
- [Decision Theory](#decision-theory)
- [Linear Algebra](#linear-algebra)
- [Matrix Calculus](#matrix-calculus)

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

## Decision Theory
## Linear Algebra
## Matrix Calculus