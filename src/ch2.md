# 2. Prediction

> *A famous colleague once sent an actually very well-written paper he was quite proud of to a famous complexity theorist. His answer: “I can’t find a theorem in the paper. I have no idea what this paper is about.”*

**Contents**
- [Prediction Overview](#overview)
- [Parametric Models]()
    - [Function Approximation via Parameter Estimation (MLE/MAP)]()
    - [Generalized Linear Models (GLMs)]()
    - [Deep Neural Networks (DNNs)]()
- [Non-parametric Models]()
    - [Function Approximation via Posterior Updates (Bayes' Rule)]()
    - [Bayesian Neural Networks (BNNs)]()
    - [Gaussian Processes (GPs)]()
    - [Support Vector Machines (SVMs)]()

## Prediction Overview
The primary goal of machine learning is to leverage patterns from stochastic
phenomena to **predict** quantities of interest without enumerating an entire
population. In simpler cases the population can be modeled with closed form
analytic solutions such as single random variables and their distributions
($Bin(n,p)$, $Nor(\mu, \sigma)$). In more complex cases this is intractable, and
so the underlying distribution must be recovered from a random sample.

This is often the case with machine learning, where given dataset
$D=\{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)}): (\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\overset{\text{iid}}{\sim} \mathcal{X} \times \mathcal{Y} \}_{i=0}^{n}$
with $\mathcal{X} = \mathbb{R}^{d_{in}}$, $p(\mathbf{y}|\mathbf{x})$ needs to be
recovered. The assumption of **independence** is the statistical
questionably axiomatic counterpart to the probabilistic one  of **totality**. While
the assumption is philosophically and theoretically interesting to question with
alternative probabilistic frameworks and statistical modeling, the iid assumption
is a *useful* one to make progress in modeling. Returning to the problem setup,
when $\mathcal{Y} \subseteq \mathbb{N}$, the task is
**classification**, and when $\mathcal{Y} = \mathbb{R}^{d_{out}}$, the task is
**regression**.

There are a variety classes of functions that can be used to approximate
$p(\mathbf{y}|\mathbf{x})$, but the primary distinction is between **parametric**
and **non-parametric** models. The former have a *fixed* number of parameters
with respect to the size of the dataset where **differentiation** is
used to compute gradients in order to **optimize loss functions**. This stands in
contrast to the latter which has *variable* number of parameters where
**integration** is used to compute expectations in order to **update posteriors**.
Before the start of the deep learning "revolution" in 2012 with the ImageNet
"moment" the parametric/non-parametric distinction was more well known. There
were primarily three "schools of thought":

- the **"kernelists"**: proponents of **support vector machines (SVMs)**
- the **"bayesians"**: proponents of **gaussian processes (GPs)**
- the **"connectionists"**: proponents of **neural networks (NNs)**

where SVMs and GPs are non-parametric, and NNs are parametric. This chapter will
cover all three, highlighting their differences and similarities, and — as usual
in the artificial sciences — how this taxonification is in fact falsely trichotomous.
That is, by the end of this chapter the reader should understand how SVMs/GPs
are effectively **shallow networks** whereas NNs are **deep networks**,
and how the former is much better understood *theoretically* while the latter
is more successful *experimentally*.

todo: murphy's global/local latent distinction (with cho EBMs?)


## Parametric Models
### Function Approximation via Parameter Estimation (MLE/MAP)
Optimization and Differentiation

The primary method in selecting the most accurate predictor from the space of
parameterized functions is to select the parameters $\theta \in \Theta$ to
**maximize the likelihood of the data**.

todo: consider just 1 random variable. X^(1), X^(2), ... X^(n)

 That is,
$$
\begin{aligned}
\hat{\boldsymbol{\theta}} &\in \underset{\boldsymbol{\theta} \in \Theta}{\operatorname{argmax}}\ p(x;\theta) \\
                          &= \underset{\boldsymbol{\theta} \in \Theta}{\operatorname{argmax}}\ \log p(x;\theta)
                          && \text{[by monotonicity of log]}\\
\end{aligned}
$$

where $\operatorname{argmin}$ for simple cases can be solved symbolically.
but for more complex cases need to be solved algorithmically with automatic differentiation/gradient descent.


Optimization is the primary method used to  selecting some function from the space of functions parameterized
by
is **optimization**. Namely,

### Generalized Linear Models
the first models with two random variables which model the input space and output space.

##### Linear Regression
##### ~~Logistic Regression~~ Sigmoidal Classification

### Deep Neural Networks

## Non-Parametric Models
### Bayesian Neural Networks
### Gaussian Processes

## Energy Perspective: Parameterization, Learning, Inference.
The energy perspective to machine learning not only provides a unifying
foundation, but also relates the discipline and it's toolbox to that of the one
used by physicists dating back to the 19th century, which is
describing systems in terms of **minimizing free energy**. The free energy
$e: \mathcal{O} \times \mathcal{Z} \times \Theta \to \mathbb{R}$
is the incomptability score between an observed/unobserved instance $(x,z)$ and is
parameterized by $\theta$. TODO: rubber band?

TODO: if we do not observe z, we cannot fully determine how preferable x is?
Partitioning the observation space into input/output space $\mathcal{X}$/$\mathcal{Y}$,
and given unseen input $x'$ with no unobserved $z=\varnothing$, the goal
of **supervised learning** is to search for the the $y$ that minimizes the free
energy with:
$$
\hat{y} := \argmin_{y\in \mathcal{Y}}e([x',y], \varnothing, \theta)
$$

where the countable discrete case $\mathcal{Y}\subseteq\mathbb{N}$ is **classification**
and the uncountable continuous case $\mathcal{Y}=\mathbb{R}$ is **regression**. With respect
to the unobserved space $\mathcal{Z}$, the goal is to search for the $z$ that
minimizes the free enrgy with:
$$
\hat{z} := \argmin_{z\in \mathcal{Z}}e([x,y], z, \theta)
$$

where the discrete is **clustering**, the continuous case is **representation learning**,
and noting that $x$ is not primed.

From this perspective, all machine learning paradigms are given some
**partial observation** and must **infer** the **unobserved**. This is done by
via **optimization** with respect to some subset of the energy function's domain.
Inference defined this way more general than forward pass evaluation of the
learned model — inference is now expanded to **search**. This is why the
discipline has overloaded the term *inference*, because the goal
is to search for a prediction about *the* future, not *the* model used to predict
the future.

TODO
- learning (backprop. sgd) of $\theta$
- parameterization (networks) of $e$