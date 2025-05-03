# Chapter 2: dfdx(nd)
**This chapter implements an interpreter for neural networks (pytorch1 "eager" mode).**
As stated in the syllabus, solid deep learning and systems programming are
prerequisites. Section 1 on correctness assumes mathematical maturity with
statistics and matrix calculus. Section 2 assumes familiarity with out of order
and superscalar processors. By the end of this chapter, you will have an
accelerated implementation of the multidimensional array with autodifferentiation
capability capable of interpreting llama and sd.

## Table of Contents
[**Section 1: Correctness**]()

0. [non-linear parametric models: `nn.Linear()`, `nn.ReLU()`]()
1. [multidimensional array: `model.forward()`]()
2. [gradient descent: `loss.backward()`, `opt.step()`]()
3. [attention: a(q,k,v)]()
4. [thought:]()

[**Section 2: Speed**]()

5. [CPU: `RVV`]()
6. [GPU:]()
7. [GPU Tensor core (TPU)]()


# Section 1: Correctness

## Part 0: non-linear parametric models: `nn.Linear()`, `nn.ReLU()`
Before jumping into the implementation of our deep learning framework's
multidimensional array with autodifferentiation capability, let's review the
mathematics of neural networks. We will incrementally construct a family of
functions from logistic regression, multiclass regression, feedforward
neural networks, attention and chain of thought variants, all for the
classification setting where Y⊆ℕ.

Recall that the logistic regression model for binary classification recovers the
bernouilli distribution:
```
    P: ℝ^d -> [0,1] by assuming
    P(Y=1|X=x;θ={w,b}) := σ(wᵀx) ==> P(Y=0|X=x;θ) = 1 - σ(wᵀx)
==> P(Y=y|X=x;θ) = ŷ^y (1-ŷ)^(1-y) [continuous form]
                 = σ(wᵀx)^y [1-σ(wᵀx)]^(1-y)

+---+             
| x1+---+         
| x2+--+|w_0      
|   |  ||         
|   |w_1|         
| . |  |+---->-+. 
| . |  +-->(wᵀ |σ)
| . |  +--->`--+' 
|   |  |          
|   |  |w_d       
|   |  |          
| xd+--+          
+---+
Fig 1. logistic regression
```

where we can omit the bias by prepending column vector x with x_0 := 1, using
using w_0 as b. From now on we will implicitly assume that the hyperparameter
d is d_old+1 so there is an extra row for the bias terms. And since we are
training from a dataset D={(x_i, y_i)} (i=1..n), we can evaluate the function
P: ℝ^d -> [0,1], P(x^i;θ={w,b}) := σ(wᵀx^(i)) for all n input-output pairs with
a single matrix vector multiplication:
    ŷ = σ(Xw, dim=0)
where:
    X ∈ ℝ^{nxd}
    w ∈ ℝ^d

We can also extend this model to multi-class classification by generalizing
logistic regression to softmax regression by replacing σ(z) with softmax(z):
    σ: ℝ -> [0,1], σ(z) := 1/[1+exp(-z)]
    softmax: ℝ^k -> [0,1]^k, softmax(z_i) := exp(z_i) / Σ exp(z_j)
                                                        j=1..k

recalling that 1. ∫softmax(z) = 1 and 2. that z is referred to as the logits
since sigmoid and softmax map ℝ to log odds (ln[p/(1-p)]). Generalizing sigmoid
to softmax allows us to recover the multinomial distribution:
```
    P: ℝ^d -> [0,1]^k by assuming
    P(Y=i|X=x) = exp(w_iᵀx) / Σ exp(w_jᵀx)
                              j=1..k
==> P(Y=y|X=x) = softmax(Wx), where W ∈ ℝ^{kxd}, x ∈ ℝ^d : x_0 = 1

+---++---------------++----+                          +------------------------+
| x1||               ||w1ᵀσ|                          |exp(w1ᵀx) / Σ exp(w_jᵀx)|
| x2||               ||w2ᵀσ|                          |exp(w2ᵀx) / Σ exp(w_jᵀx)|
|   ||               ||    |                          |                        |
|   ||               ||    |                          |                        |
| . ||               || .  |    +----------------+    | .                      |
| . ||       W       || .  |--->|g(z):=softmax(z)|--->| .                      |
| . ||               || .  |    +----------------+    | .                      |
|   ||               ||    |                          |                        |
|   ||               ||    |                          |                        |
|   ||               ||    |                          |                        |
| xd||               ||wdᵀσ|                          |exp(wdᵀx) / Σ exp(w_jᵀx)|
+---++---------------++----+                          +------------------------+
Fig 2. softmax regression
```

And to evaluate P: ℝ^d -> [0,1]^k, P(x^i;θ) := softmax(Wx) for all n input-output
pairs with a single matrix multiplication:
    ŷ = softmax(XW, dim=0)
where:
    X ∈ ℝ^{nxd}
    W ∈ ℝ^{dxk}

Finally, we can extend the multinomial logistic regression by introducing
intermediate stages of computation to project the representation of inputs
into a basis that is more tractable when mapping to a distribution:
```
    P: ℝ^d -> [0,1]^k
    P(x;θ={w,b}) := (softmax ◦ W_L ◦ (φ ◦ W_{L-1}) ◦ ... ◦ (φ ◦ W_1))(x)

and each pair of linearity W and non-linearity φ is a "hidden layer" so
    h^(1) := φ(W^1x+b_1)
    h^(2) := φ(W^2h^(1)+b_2)
    .
    .
    .
    h^(L) := φ(W^Lh^(L-1)+b_L)

                                                                               +------------------------+
                                                                               |exp(w1ᵀx) / Σ exp(w_jᵀx)|
                                                                               |exp(w2ᵀx) / Σ exp(w_jᵀx)|
                                                                               |                        |
                      +---++---------------++-------+                          |                        |
                      | a1||               ||φ(w1ᵀσ)|    +----------------+    | .                      |
                      | a2||               ||φ(w2ᵀσ)|--->|g(z):=softmax(z)|--->| .                      |
                      |   ||               ||       |    +----------------+    | .                      |
  +-------------------+---++----+          ||       |                          |                        |
 ++----------------------------+|          || .     |                          |                        |
++--++---------------++-------+||  WL      || .     |                          |                        |
| x1||               ||φ(w1ᵀσ)|||          || .     |                          |exp(wdᵀx) / Σ exp(w_jᵀx)|
| x2||               ||φ(w2ᵀσ)|||          ||       |                          +------------------------+
|   ||               ||       |||  *       ||       |
|   ||               ||       ||| *        ||       |
| . ||               || .     |||*         ||φ(wdᵀσ)|
| . ||       W1      || .     ||+----------++-------+
| . ||               || .     |||
|   ||               ||       |||
|   ||               ||       |||
|   ||               ||       |++
| xd++               ++φ(wdᵀσ)++
+---++---------------++-------+
Fig 3. neural network
```

Recall that neural networks are the family of functions that are non-linear
and parametric. The non-linearities that act as feature extractors differentiate
this class from the family of linear functions such as linear and logistic
regression, and the parametric weights and biases differentiate the class from
other non-linear functions such as gaussian processes and kernel methods. Favoring
the mathematical specification over the biological inspiration, neural networks,
reductively put, are a lot of logistic regressions (weighted sums with
non-linearities) stacked together.

```python
"""
model: Neural Language Models (Bengio et al. 2003) URL: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

Dimension key:
# windows
B: batch size
T: sequence length

# input/output
V: vocabulary size
E: embedding dimension (E != D in paper)
D: model dimension
"""
import picograd
B, T = 32, 3
V, E, D = 27, 10, 200

# *********************MODEL*********************
class Linear:
  def __init__(self, D_in, D_out, bias=True):
    self.W_DiDo = picograd.randn((D_in, D_out)) * (5/3)/D_in**0.5 # kaiming init (He et al. 2015)
    self.b_Do = picograd.zeros(D_out) if bias else None

  def __call__(self, X_Di):
    self.X_Do = X_Di @ self.W_DiDo
    if self.b_Do is not None:
        self.X_Do += self.b_Do
    self.out = self.X_Do
    return self.X_Do

  def parameters(self):
    return [self.W_DiDo] + ([] if self.b_Do is None else [self.b_Do])

class Tanh:
  def __call__(self, X_BD):
    self.X_BD = picograd.tanh(X_BD)
    # plt.hist(self.X_BD.view(-1).tolist(), 50); # distribution of weights
    # plt.imshow(self.X_BD.abs() > 0.99, cmap='gray', interpolation='nearest') # vanishing gradients
    self.out = self.X_BD
    return self.X_BD
  
  def parameters(self):
    return []

model = [
  Linear(T * E, D, bias=False), Tanh(),
  Linear(D, D, bias=False), Tanh(),
  Linear(D, V, bias=False)
]

C_VE = picograd.randn((V,E))

# *********************INFERENCE LOOP*********************
for _ in range(20): # 20 samples
  output, context = [], [0] * T
  while True:
    X_1T = picograd.tensor([context]) # B=1 for inference, T=3, in [0..27] (set to 0 for init)
    X_1TE = C_VE[X_1T] # using 0..27 as indices into C_VE for each B=1 example of context length T
    print(X_1TE)
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





## Part 1: multidimensional array: `model.forward()`
Let's start with the multidimensional array and follow pytorch's `tensor` naming
convention. Consider the following product type for tensor:

struct Tensor {
    ndim: i32
    shape: Vec<usize>
    stride: Vec<usize>
    storage: Vec<f32>
}

where ndim is an integer, shape as well as stride are lists of integers, and
finally, storage is list of floats. Defining the tensor this way (rather than
naively using nested arrays) makes the aliasing of underlying data buffers quite
natural, since  physical storage can be logically interpreted using shape and
stride. This is desirable in deep learning since many tensor operations only
involve modifying the shape and stride, thereby updating the "view" of the tensor.
Consider example 1 and 2 which illustrate that operations such as reshape and
transpose fundamentally do not add or remove elements from the tensor — they only
change the interpretation of data:

Example 1:
01 02 03   reshape      01 02
04 05 06     ==>        03 04

07 08 09                05 06
10 11 12                07 08

                        09 10
                        11 12

Example 2:
1 2 3    transpose    1 4 7
4 5 6      ==>        2 5 8
7 8 9                 3 6 9

TODO:
- binops (zip), uops(map), reduceops(), viewops, matmul.

We have built an abstraction for the multidimensional array, which makes the
current framework more or less on par with the capabilities of numpy. But in
order to train neural networks we need to recover the most likely parameters
that characterize the underlying generating distribution:
```
    θ̂ ∈ argmin ℒ(Θ)
      = argmin -P(y1|x1,...yn|xn;Θ)
      = argmin -ΠP(y^i|x^i;Θ) [assuming (X^i,Y^i)~iid]
      = argmin -logΠP(y^i|x^i;Θ) [log monotonicity]
      = argmin -ΣlogP(y^i|x^i;Θ) [log laws]

where argmin is implemented iteratively via gradient descent:
    θ^(t+1) := θ^t - α∇ℒ(Θ)
             = θ^t - α∇-ΣlogP(y^i|x^i;Θ)

and so in the case of binary classification with logistic regression:
    P: ℝ^d -> [0,1]
    P(y^i|x^i;θ={w,b}) := ŷ^y (1-ŷ)^(1-y) = σ(wᵀx)^y [1-σ(wᵀx)]^(1-y)
==> θ̂ ∈ argmin -Σlog[σ(wᵀx)^y [1-σ(wᵀx)]^(1-y)] [by def]
      = argmin ylogσ(wᵀx) - (1-y)log[1-σ(wᵀx)]
==> θ^(t+1) := θ^t - α∇[ ylogσ(wᵀx) - (1-y)log[1-σ(wᵀx)] ]

    ∂ℒ/∂wi = ∂/wi [ ylogσ(wᵀx) - (1-y)log[1-σ(wᵀx)] ]
            = - ∂/wi[ylogσ(wᵀx)] + ∂/∂wi[(1-y)log[1-σ(wᵀx)]] [∂ linear]
            = -y/σ(wᵀx)*∂/wi[σ(wᵀx)] + -(1-y)/[1-σ(wᵀx)]*∂/∂wi[1-σ(wᵀx)] [chain rule]
            = ∂/wi[σ(wᵀx)] * -[y/σ(wᵀx) - (1-y)/[1-σ(wᵀx)]]
            = TODO: ...
            = [σ(wᵀx)-y]xi

and in the case of multiclass classification with softmax regression:
    P: ℝ^d -> [0,1]^k
    P(y^i|x^i) = softmax(Wx), where W ∈ ℝ^{kxd}, x ∈ ℝ^d : x_0 = 1
==> θ̂ ∈ argmin -Σlog[softmax(Wx)] [by def]
==> θ^(t+1) := θ^t - α∇[ -Σlog[softmax(Wx)] ]

and in the case of multiclass classification with neural networks:
    P: ℝ^d -> [0,1]^k
    P(x;θ={w,b}) := (softmax ◦ W_L ◦ (φ ◦ W_{L-1}) ◦ ... ◦ (φ ◦ W_1))(x)
==> θ̂ ∈ argmin -Σlog[(softmax ◦ W_L ◦ (φ ◦ W_{L-1}) ◦ ... ◦ (φ ◦ W_1))(x)] [by def]
==> θ^(t+1) := θ^t - α∇[-Σlog[(softmax ◦ W_L ◦ (φ ◦ W_{L-1}) ◦ ... ◦ (φ ◦ W_1))(x)]]
```

but since |θ| where θ={w,b} are reaching the billions and trillions, deriving
the gradient of the loss function with respect to weights and biases becomes
intractable. So in part 2 and part 3 we will add autodifferentiation and
gradient descent support to our tensor library enabling the training of deep
neural networks where deriving the gradient and optimizing the network is
abstracted for the user with a loss.backward() and an optim.step(). The framework
will be able to interpret the following training loop for the FFN from above:

```
# *********************TRAINING LOOP*********************
losses, steps = [], []
for step in range(100): #200000:
    # 1. forward
    # minibatch: X_NT -> X_BT
    i_B = torch.randint(0, X_NT.shape[0], (B,))
    X_BT, Y_B = X_NT[i_B], Y_N[i_B]

    # embed: X_BT -> X_BTE
    X_BTE = C_VE[X_BT] # embed the B examples with T tokens range that span [0..27]
                       # using 0..27 as indices into C_VE
    X_BcTE = X_BTE.view(-1, T * E) #. concat
    X = X_BcTE

    # X_BcTE -> X_BD -> X_BV (y_hat: logits)
    for h in model:
        X = h(X)
    loss = F.cross_entropy(X, Y_B) # 5. picograd.cross_entropy

    # 2. backward
    for layer in model:
        layer.out.retain_grad() # 6 .retain_grad()
    for p in params:
        p.grad = None
    loss.backward()

    # 3. update
    for p in params:
        p.data += -0.01 * p.grad

    steps.append(step)
    losses.append(loss.log10().item())
    if step % 10000 == 0:
        print(f"step: {step}/{200000}, loss {loss.item()}")

plt.plot(steps, losses)
```





## Part 2: gradient descent: `model.backward()`, `opt.step()`
  2a — Derivative: approximation via local linearization
  2b — Derivative rules: backward methods
  2c — Automatic differentiation: calculus on computational graph
  2d — Advanced AD: jacobian-vector and vector-jacobian products
  2e — Gradient descent: non-convex optimization

2a — Derivative: approximation via local linearization
------------------------------------------------------
The first place to start is to clarify and disambiguate the notion of a derivative
in order to generalize the denotation we use for higher dimensions to hold a more
precise semantics. At the end of part 2a you will have a clear understanding
why the gradient and jacobian are defined the way they are, and how what most
people refer to as "gradients" with pytorch are really jacobians.

By redefining the derivative as a linear operator L defined on some vector space
V that you apply to a change in input to obtain a change in output (sometimes
referred to as the Frechét derivative), it becomes very clear why the gradient
and jacobian are defined the way they are. Clarity that is not achievable when
defining them by fiat. (i.e. "let's *just* rearrange the partials in a vector/matrix").

Let's begin by recalling that for some f: ℝ -> ℝ the difference of a function at
a point x and x+δx is f(x+δx)-f(x) = δf which can be approximated by f'(x)δx plus
some error terms. That is,

```
    f(x+δx)-f(x) = δf
                 = f'(x)δx + o(δx)
                 ≈ f'(x)δx

and if we take the difference to be an infinitesimal then it becomes a differential
    f(x+dx)-f(x) = df
                 = f'(x)dx
```

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

<!-- ## Part 3: attention — a(q,k,v)
Since the discovery of the attention operator in 2017 by (Vaswani et al. 2017),
language, vision, audio and any domain that can be expressed as an autoregressive
sequence of tokens (text/pixels/waves) — that is, an n-dimensional joint distribution:
  p(X_1=x^1,...,X_n=x^m;θ) = Πp(x^i|x^<i)
                             i=1..n
are all converging onto a single network architecture: transformers and their
attention variants. Later in part four we'll reproduce Llama using our deep
learning framework, but for now, we'll prototype with a more simpler feed-forward
neural network (FFN). Here is the model definiton and inference loop:



## Part 4: assistant — RLHF
reinforcement learning from human rewards



## Part 5: reasoner — GRPO
reinforcement learning from automated rewards -->
















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




## Section 1 References
0. [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)
1. [https://arxiv.org/pdf/1912.01703](https://arxiv.org/pdf/1912.01703)
2. [http://blog.ezyang.com/2019/05/pytorch-internals/](http://blog.ezyang.com/2019/05/pytorch-internals/)
3. [https://docs.jax.dev/en/latest/autodidax.html](https://docs.jax.dev/en/latest/autodidax.html)
4. [https://numpy.org/doc/stable/dev/internals.html](https://numpy.org/doc/stable/dev/internals.html)
5. [https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray)
6. [https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/)
7. [https://indico.ijclab.in2p3.fr/event/2914/contributions/6483/subcontributions/180/attachments/6060/7185/automl-short.pdf](https://indico.ijclab.in2p3.fr/event/2914/contributions/6483/subcontributions/180/attachments/6060/7185/automl-short.pdf)
8. [https://blog.x.com/engineering/en_us/topics/infrastructure/2015/autograd-for-torch](https://blog.x.com/engineering/en_us/topics/infrastructure/2015/autograd-for-torch)
9. [https://openreview.net/pdf?id=BJJsrmfCZ](https://openreview.net/pdf?id=BJJsrmfCZ)
10. [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf)
11. [https://arxiv.org/pdf/2312.16730](https://arxiv.org/pdf/2312.16730)
12. [https://arxiv.org/abs/2412.05265](https://arxiv.org/abs/2412.05265)
13. [https://spinningup.openai.com/en/latest/](https://spinningup.openai.com/en/latest/)
14. [https://rlhfbook.com/](https://rlhfbook.com/)

## Section 2 References