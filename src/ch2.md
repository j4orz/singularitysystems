# 1. Prediction

> *A famous colleague once sent an actually very well-written paper he was quite proud of to a famous complexity theorist. His answer: “I can’t find a theorem in the paper. I have no idea what this paper is about.”*

**Contents**
- [1.0 Prediction Overview](#overview)
- [1.1 Non-parametric Inference with Posterior Updates]()
    - [Support Vector Machines (SVMs)]()
    - [Gaussian Processes (GPs)]()
    - [Bayesian Neural Networks (BNNs)]()
- [1.2 Parametric Inference with Parameter Estimation](#12-parametric-models-with-parameter-estimation)
    - [Generalized Linear Models (GLMs)](#generalized-linear-models-glms)
    - [Deep Neural Networks (DNNs)](#deep-neural-networks-dnns)

# 1.0 Prediction Overview
The primary goal of supervised machine learning is to leverage patterns from stochastic
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

# 1.1 Non-Parametric Inference with Posterior Updates

## Gaussian Processes
## Bayesian Logistic Regression
## Bayesian Neural Networks

---

# 1.2 Parametric Inference with Parameter Estimation

## Generalized Linear Models (GLMs)

### Regression
Recall that the supervised learning problem is considered regreession when
$\mathcal{Y} = \mathbb{R}$.

x: youtube views y: spotify streams


### Classification
Recall that the supervised learning problem is considered classification
when $\mathcal{Y} \subseteq \mathbb{N}$. The simple case when the output space is
$\mathcal{Y}=\{-1, 1\}$ is referred to as **binary classification**,
and in the case when the output space is $\mathcal{Y}=\{0,1, \ldots, k\}$ is
referred to as **multi-class classification**.

Many predictive values of interest that can be posed as answers to yes/no questions
are suitable to be modeled as binary classifiers. For instance, benign or malignant
in health care, innocent or guilty in law, real or spam in engineering. The running
example used for the next two models will be the domain of music — to classifying
a set of rap lyrics as a praise or diss. The set of lyrics $x^{(i)}$ that is of
particular interest is Kendrick's featured verse in Kanye's "No More Parties in LA":

```
Hey baby you forgot your Ray Bans
And my sheets still orange from your spray tan
It was more than soft porn for the K-man
She remember my Sprinter, said "I was in the grape van"
Uhm, well cutie, I like your bougie booty
Come Erykah Badu me, well, let's make a movie
Hell, you know my repertoire is like a wrestler
I show you the ropes, connect the dots
A country girl that love Hollywood
Mama used to cook red beans and rice
Now it's Denny's, 4 in the morning, spoil your appetite
Liquor pouring and niggas swarming your section with erection
Smoke in every direction, middle finger pedestrians
R&B singers and lesbians, rappers and managers
Music and iPhone cameras
This shit unanimous for you, it's damaging for you, I think
That pussy should only be holding exclusive rights to me, I mean
He flew you in this motherfucker on first class
Even went out his way so you could check in an extra bag
Now you wanna divide the yam like it equate the math?
That shit don't add up, you're making him mad as fuck
She said she came out here to find an A-list rapper
I said baby, spin that round and say the alphabet backwards
You're dealing with malpractice, don't kill a good nigga's confidence
Just cause he a nerd and you don't know what a condom is
The head still good though, the head still good though
Make me say "Nam Myoho Renge Kyo"
Make a nigga say big words and act lyrical
Make me get spiritual
Make me believe in miracles, Buddhist monks and Cap'n Crunch cereal
Lord have mercy, thou will not hurt me
Five buddies all herded up on a Thursday
Bottle service, head service, I came in first place
The opportunity, the proper top of breast and booty cheek
The pop community, I mean these bitches come with union fee
And I want two of these, moving units through consumer streets
Then my shoe released, she was kicking in gratuity
And yeah G, I was all for it
She said K Lamar, you kind of dumb to be a poet
I'mma put you on game for the lames that don't know they're a rookie
Instagram is the best way to promote some pussy
```

The sentiment of "No More Parties in LA" is extremely difficult for a machine to
classify correctly due to Kendrick's wordplay, double entendres, and insinuations.
Consider the positive/negative dictionaries where
positive={baby, cutie, love, good, spiritual, miracles, buddhist, buddies, opportunity, promote}
and negative={shit, spoil, damaging, motherfucker, mad, fuck, malpractice, kill, mercy, hurt, fee, dumb}.
Then a possible feature extractor $\phi: \mathcal{X} \to \mathbb{R}^3$ which maps
raw input to dimensioned feature representation is defined by:

| Dim             | Feature             | Value     |
|-----------------|-----------------------|------------------|
| $\phi(x)_1$ | $|\{ \forall w \in x : w \in positive\}|$                              | 10   |
| $\phi(x)_2$ | $|\{ \forall w \in x : w \in negative\}|$                          | 12    |
| $\phi(x)_3$ | $$\begin{cases} 1 & \text{if "lamar"} \in \text{x} \\ 0 & \text{otherwise} \end{cases}$$                          | 1 |

Now that the input space $\mathcal{X}$, feature extractor $\phi:\mathcal{X}\to \mathbb{R}^3$,
and output space $\mathcal{Y}=\{-1, 1\}$ is defined, the goal is to select some
function $f: \mathcal{X} \to \mathcal{Y}$.

**Naive Bayes**

**Logistic Regression**

The logistic regression model is a **discriminative** $p(Y=y|\mathbf{X}=\mathbf{x}; \theta)\sim Ber(p)$
with a **decision boundary** $k \in [0,1]$ (in the case of $Ber(p)$, $k$ is usually set to $0.5$ so
as to divide the mass by 2) that assumes that the parameter $p$ is affine with respect to $x$. That is,
$$
\begin{aligned}
p(Y=1 \mid \mathbf X=\mathbf x) &= p := \sigma\bigl(\mathbf w^{\mathsf T}\phi(x)\bigr) 
                                 &&\text{[$\phi(x)_0 = 1$]} \\[4pt]
\implies\; p(Y=-1 \mid \mathbf X=\mathbf x) &= 1-p = 1-\sigma\bigl(\mathbf w^{\mathsf T}\phi(x)\bigr) \\[4pt]
\implies\; p(Y=y \mid \mathbf X=\mathbf x) &= p^{y}(1-p)^{1-y} \\[4pt]
                                           &= \sigma\bigl(\mathbf w^{\mathsf T}\phi(x)\bigr)^{y} \bigl[1-\sigma\bigl(\mathbf w^{\mathsf T}\phi(x)\bigr)\bigr]^{1-y}

\end{aligned}
$$

where $\sigma: \mathbb{R} \to [0,1]$ is a non-linear function
$\sigma := \frac{1}{1+\exp(-z)}$, and where $\mathbf{w^{\mathsf T} \phi(x)}$ is
referred to as the **logit** since the inverse $\sigma^{-1}:= \log \frac{p}{1-p}$
is defined as the log odds ratio.

todo: remark 2.1.1 in course notes

With the model now defined, the parameter $w$ needs to be estimated from the data $D=\{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)}): (\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\overset{\text{iid}}{\sim} \mathcal{X} \times \mathcal{Y} \}_{i=0}^{n}$.
This is done by using the negatve log likelihood as the loss function $\mathcal{L}: \mathbb{R}^n \to \mathbb{R}$
to **minimize** so that $\mathcal{L(\mathbf{w})} := -\log \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)})$
is fixed with respect to the data $(x^{(i)}, y^{(i)})$

$$
\begin{aligned}
\hat{\mathbf{w}}_{MLE} &\in \operatorname{argmin} \mathcal{L(\mathbf{w})} \\
                          &= \operatorname{argmin} - \sum_{i=1}^{n} \log \sigma\bigl(\mathbf w^{\mathsf T}\phi(x^{(i)})\bigr)^{y^{(i)}} \bigl[1-\sigma\bigl(\mathbf w^{\mathsf T}\phi(x^{(i)})\bigr)\bigr]^{1-y^{(i)}} \\
                          &= \operatorname{argmin} - \sum_{i=1}^{n} y^{(i)}\log  \sigma\bigl(\mathbf w^{\mathsf T}\phi(x^{(i)})\bigr) + (1-y^{(i)})\log  \sigma\bigl(\mathbf w^{\mathsf T}\phi(x^{(i)})\bigr) \\
\end{aligned}
$$

where todo kl->ce.

where $\operatorname{argmin}$ is implemented by first evaluating the **gradient**
$\nabla \mathcal{L(\mathbf{w})}$ and then iteratively applying **gradient descent**
for each time step $t$, $\mathbf{w}^{(t+1)} := \mathbf{w}^{t} -\alpha \nabla \mathcal{L(\mathbf{w})}$.

First, to evaluate the gradient, $\nabla \mathcal{L}(\mathbf{w})$,
the negative log likelihood as loss function is simplified by defining
$\hat{y}^{(i)} := \sigma(\mathbf w^{\mathsf T}\phi(x^{(i)}))$ so that
$\nabla \mathcal{L}(\mathbf{w}) = - \sum_{i=1}^{n} y^{(i)}\log  \hat{y}^{(i)} + (1-y^{(i)})\log \hat{y}^{(i)}$.
Note that $\hat{y}$ is not the target label but the probability of the target label.
Then, since the derivative is linear with the derivative of the sum is the sum
of the derivatives where
$\frac{\partial}{\partial\mathbf{w}}\sum_{i=1}^{n}f(\mathbf{w}) = \sum_{i=1}^{n}\frac{\partial}{\partial\mathbf{w}}f(\mathbf{w})$, taking the derivative for a *single* example $i \in [1, n]$ where $\mathcal{L}(\mathbf{w}) = y\log  \hat{y} + (1-y)\log \hat{y}$ looks like

$$
\begin{aligned}
\frac{\partial \mathcal{L(\mathbf{w})}}{\partial\mathbf{w}_j} &= \frac{\partial\mathcal{L}(\mathbf{w})}{\partial\hat{y}} \frac{\partial \hat{y}}{\partial \mathbf{w}_j}
&& \text{[by chain rule]}\\
&= \frac{\partial\mathcal{L}(\mathbf{w})}{\partial\hat{y}} \hat{y}(1-\hat{y})\phi(x)_j
&& \text{[$= \sigma(1-\sigma)$]}\\
&= \bigl[\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}\bigr] \hat{y}(1-\hat{y})\phi(x)_j
&& \text{[by $\frac{d\log(x)}{dx} = \frac{1}{x}$]}\\
&= \bigl[\frac{y(1-\hat{y}) - \hat{y}(1-y) }{\hat{y}(1-\hat{y})}\bigr] \hat{y}(1-\hat{y})\phi(x)_j \\
&= \bigl[y(1-\hat{y}) - \hat{y}(1-y)\bigr]\phi(x)_j\\
&= \bigl(y -\hat{y})\phi(x)_j\\
\end{aligned}
$$

and so the evaluating the derivative of all examples
$\nabla \mathcal{L}(\mathbf{w}) = - \sum_{i=1}^{n} y^{(i)}\log  \hat{y}^{(i)} + (1-y^{(i)})\log \hat{y}^{(i)}$
where $\hat{y}^{(i)} := \sigma(\mathbf w^{\mathsf T}\phi(x^{(i)}))$ looks like

$$
\begin{aligned}
\frac{\partial \mathcal{L(\mathbf{w})}}{\partial\mathbf{w}_j} &= - \sum_{i=1}^{n} \frac{\partial}{\partial \mathbf{w}_j}\bigl[y^{(i)}\log  \hat{y}^{(i)} + (1-y^{(i)})\log \hat{y}^{(i)}\bigr] \\
&= - \sum_{i=1}^{n} \bigl(y^{(i)} -\hat{y^{(i)}})\phi(x^{(i)})_j \\
&= - \sum_{i=1}^{n} \bigl(y^{(i)} -\sigma(\mathbf w^{\mathsf T}\phi(x^{(i)}))\bigr]\phi(x^{(i)})_j \\
\end{aligned}
$$

Recall the second step in implementing $\operatorname{argmin}$ after taking
$\nabla \mathcal{L}(\mathbf{w})$ is to then iteratively apply gradient descent
for each time step $t$, $\mathbf{w}^{(t+1)} := \mathbf{w}^{t} -\alpha \nabla \mathcal{L(\mathbf{w})}$.

**Multinomial Logistic Regression**

### GLM: $\mathcal{N}(\mu, \sigma^2)$ and $Ber(p)$ is $Exp()$

## Deep Neural Networks (DNNs)

| Feature                                                                                     | Dimension             | Value     |
|-----------------|-----------------------|------------------|
| $|\{ \forall w \in x : w \in positive\}|$                                                                    | $\phi(x)_1$                              | 12   |
| $|\{ \forall w \in x : w \in negative\}|$                                                                    | $\phi(x)_2$                          | 11    |
| $$\begin{cases} 1 & \text{if "lamar"} \in \text{x} \\ 0 & \text{otherwise} \end{cases}$$              | $\phi(x)_3$                          | 1 |

which apriori seems difficult to learn a useful mapping for, given that the
first and second dimension of the feature representation which respectively encode
positive/negative sentiment are roughly tied. For instance, what is the feature
extractor that maps the following set of lyrics (with subtle insinuations)
into a *useful* feature representation for a sentiment classifier?

```
A country girl that love Hollywood
Mama used to cook red beans and rice
Now it's Denny's, 4 in the morning, spoil your appetite
Liquor pouring and niggas swarming your section with erection
```

The short answer is that it's difficult because the concept and negative sentiment
becomes apparent after reading the entire paragraph rather than any specific word.
This is the motivation for representation learning.

<!-- ## Energy Perspective: Parameterization, Learning, Inference.
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
- parameterization (networks) of $e$ -->