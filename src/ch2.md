# 1. Prediction

> *A famous colleague once sent an actually very well-written paper he was quite proud of to a famous complexity theorist. His answer: “I can’t find a theorem in the paper. I have no idea what this paper is about.”*

**Contents**
- [1.0 Prediction Overview](#overview)
- [1.1 Non-parametric Models with Posterior Updates]()
    - [Support Vector Machines (SVMs)]()
    - [Gaussian Processes (GPs)]()
    - [Bayesian Neural Networks (BNNs)]()
- [1.2 Parametric Models with Parameter Estimation]()
    - [Generalized Linear Models (GLMs)](#generalized-linear-models)
    - [Deep Neural Networks (DNNs)](#deep-neural-networks)
    - [Recurrent Neural Networks (RNNs)]()
    - [Long Short-Term Memory Networks (LSTMs)]()
    - [Generative Pretrained Transformers (GPTs)]()


## 1.0 Prediction Overview
The primary goal of machine learning is to leverage patterns from stochastic
phenomena to **predict** quantities of interest without enumerating an entire
population. That is, to recover the underlying distribution from a random sample.
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

---

## 1.1 Non-Parametric Models with Posterior Updates

### Gaussian Processes
### Bayesian Logistic Regression
### Bayesian Neural Networks

---

## 1.2 Parametric Models with Parameter Estimation

Autoregressive language models.
logistic regression -> FFN -> RNN -> LSTM -> Transformer.

in chapter 3. generation, other generative DNN variants will be covered.
- diffusion
- ebm
- etc.

### Generalized Linear Models (GLMs)
A natural inductive bias to make is that the input and output
spaces $\mathcal{X}$, $\mathcal{Y}$ are **affinely** related. We will
see these models are all exponential...

**Naive Bayes**

**Logistic ~~Regression~~ Classification**

When the output space $\mathcal{Y}=\{-1, 1\}$ is a **binary** encoding of positive/negative
outcomes, the distribution used for the **discriminative** model $p(y|\mathbf{x}; \theta)$ is $Ber(p)$.
That is,
$$
\begin{aligned}
&p(Y=1|\mathbf{X}=\mathbf{x}) := \sigma(w^{\mathsf{T}}\mathbf{x})\\
\implies &p(Y=-1|\mathbf{X}=\mathbf{x}) = 1 - \sigma(w^{\mathsf{T}}\mathbf{x}) \\
\end{aligned}
$$

where $p=p(Y=1|\mathbf{X}=\mathbf{x})$ and $\sigma: \mathbb{R} \to [0,1]$, $\sigma := \frac{1}{1+\exp(-z)}$.
The closed form *continuous* and *differentiable* mass function for $Ber(p)$ whose parameter $p=\sigma(w^{\mathsf{T}}\mathbf{x})$
needs to be estimated from the data $D=\{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)}): (\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\overset{\text{iid}}{\sim} \mathcal{X} \times \mathcal{Y} \}_{i=0}^{n}$ is then

$$
\begin{aligned}
p(Y=c|\mathbf{X}=\mathbf{x}) &= p^c(1-p)^{1-c} \\
                             &= \sigma(w^{\mathsf{T}}\mathbf{x})^c(1-\sigma(w^{\mathsf{T}}\mathbf{x}))^{1-c}
\end{aligned}
$$

Before moving on to the estimation of parameter $p$, take a moment to convince
yourself what the assumption $\sigma(w^{\mathsf{T}}\mathbf{x})$ is *really* doing.

*Multinomial Classification*

**Linear Regression**

### Deep Neural Networks (DNNs)

---

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