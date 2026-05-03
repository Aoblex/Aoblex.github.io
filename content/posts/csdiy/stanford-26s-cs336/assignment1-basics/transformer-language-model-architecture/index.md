+++
title = 'Section 2: Transformer Language Model Architecture'
date = '2026-04-28T23:49:45+08:00'
summary = 'Build a Transformer model from scratch! 🥸'
weight = 20
draft = false
+++

# Introduction

In [the last section](../byte-pair-encoding/), we worked hard to convert input text into token ids.
Before using these token ids, we need to build a model that takes these ids as input:
$$
\text{input: (batch, seqlen)} \xrightarrow{\text{model}} \text{output: (batch, seqlen, vocab)}
$$

Formally, the input and output spaces can be described as:

$$
\text{input} \in \mathbb{N}^{\text{batch} \times \text{seqlen}}, \quad
\text{output} \in \mathbb{R}^{\text{batch} \times \text{seqlen} \times \text{vocab}}
$$

In this part of the assignment, we focus on the **Transformer architecture** and we will build one from scratch!

---

# Transformer LM

The structure of a modern transformer LM is given in the first figure below, and the internal structure of a transformer block is shown in the second figure.

{{< gallery cols="2" gap="16px" align="center" >}}
{{< gallery-item src="figures/transformer-overview.png" alt="An overview of transformer model" caption="An overview of transformer model" >}}
{{< gallery-item src="figures/transformer-block.png" alt="The structure of a transformer block" caption="The structure of a transformer block" >}}
{{< /gallery >}}

Next we will look into the details of each component and implement them from scratch.

> [!NOTE] Parameter Initialization
> Since this assignment is already long, the course staff will save the details of parameter initialization for assignment 3. Here they specified such initialization configuration:
>
> - Linear weights:
>   $\mathcal{N}\left(\mu = 0, \sigma^2 = \frac{2}{d_{in} + d_{out}}\right)$ truncated at $[-3\sigma, 3\sigma]$
>
> - Embedding:
>   $\mathcal{N}(\mu = 0, \sigma^2 = 1)$ truncated at $[-3, 3]$
>
> - RMSNorm:
>   $1$

---

# Modules

## Linear Module

Mathematical notation:

$$
y = W x
$$

The forward function:

```python
class Linear(nn.Module):
    def forward(
        self,
        x: Float[torch.Tensor, "... d_in"],
    ) -> Float[torch.Tensor, "... d_out"]: ...
```

Note that we do not include a bias term, following most modern LLMs.

## Embedding Module

Mathematical notation:

$$
y = \text{Embeddings[}x\text{]}
$$

The forward function:

```python
class Embedding(nn.Module):
    def forward(
        self,
        token_ids: Int[torch.LongTensor, "batch seqlen"],
    ) -> Float[torch.Tensor, "batch seqlen d_model"]: ...
```

## RMSNorm: Root Mean Square Layer Normalization

Mathematical notation:

$$
\begin{align*}
\text{RMSNorm}(a_i) &= \frac{a_i}{\text{RMS}(a)} g_i \\
\text{RMS}(a) &= \sqrt{\left(\frac{1}{d_{model}}\sum_{i=1}^{d_{model}} a_i^2\right) + \varepsilon}
\end{align*}
$$

where $a \in \mathbb{R}^{d_{model}}$ is the input activations, and $g \in \mathbb{R}^{d_{model}}$ is a vector of learnable "gain" parameters.

The forward function:

```python
class RMSNorm(nn.Module):
    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]: ...
```

## SwiGLU: Sigmoid Linear Unit + Gated Linear Unit

We will implement the position-wise feed-forward network in a transformer block with a [**SwiGLU**](https://github.com/huggingface/transformers/blob/c6c7e189846ccaa3bb410569bfc6556d193c638d/src/transformers/models/qwen3/modeling_qwen3.py#L70) activation function, which combines a [SiLU](https://arxiv.org/pdf/1702.03118) and a [GLU](https://arxiv.org/abs/2002.05202):

$$
\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = \underbrace{W_2}_{\text{down proj}} (\underbrace{\text{SiLU}(W_1 x)}_{\text{gate proj}} \odot \underbrace{W_3 x}_{\text{up proj}})
$$

where $x \in \mathbb{R}^{d_{\text{model}}}$,
$W_1, W_3 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$,
$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$,
$\text{FFN}(x) \in \mathbb{R}^{d_{\text{model}}}$,
and canonically, $d_{\text{ff}} = \frac{8}{3} d_{\text{model}}$.

Some heuristic arguments from the writeup:

- GLUs are suggested to "reduce the vanishing gradient problem for deep architectures by providing **a linear path** for the gradients while retaining non-linear capabilities". To be specific:

    $$
    g := \sigma(W_1x), v := W_2x \\
    y=g\odot v
    $$

    Then:

    $$
    \frac{\partial y}{\partial x}= \frac{\partial y}{\partial g} \frac{\partial g}{\partial x} + \frac{\partial y}{\partial v} \frac{\partial v}{\partial x} = \underbrace{\operatorname{diag}(v) \sigma^\prime(W_1 x)}_{g \text{ channel}} + \underbrace{\operatorname{diag}(g) W_2}_{v \text{ channel}}
    $$

    In which $\sigma^\prime (W_1 x) = \sigma (W_1 x) \odot (1 - \sigma(W_1 x)) \le 0.25$ elementwisely when $\sigma(x) = \frac{1}{1 + e^{-x}}$. So the $v$ channel provides a linear path for gradients.

- It is fine to round the dimensions to a nearby multiple of 64 for hardware efficiency.

- Keep an empirical perspective.

```python
class SwiGLU(nn.Module):
    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]:
```

## RoPE: Relative Positional Embeddings

The idea behind [RoPE](https://kexue.fm/archives/8265) is to **implement relative positional encoding via absolute positional encoding.**

Suppose that we have a query vector $q$ at position $m$ and a key vector $k$ at position $n$. Our positional embedding function is $f: \mathbb{R}^{d_{\text{model}}} \times \mathbb{N} \rightarrow \mathbb{R}^{d_{\text{model}}}$:

$$
\tilde{q}_m = f(q, m), \tilde{k}_n = f(k, n)
$$

Then the absolution position information is injected into $\tilde{q}_m$ and $\tilde{k}_n$. But we hope that we can get the **relative** position information after the inner product in attention layer:

$$
\langle \tilde{q}_m, \tilde{k}_n \rangle = \langle f(q, m), f(k, n) \rangle = g(q, k, m - n)
$$

When $d_{\text{model}} = 2$ and we interpret $q$ as a complex number, a [solution](https://kexue.fm/archives/8265#%E7%BC%96%E7%A0%81%E5%BD%A2%E5%BC%8F) is:

$$
f(q, m) = q e^{i m \theta} =
\begin{pmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{pmatrix}
\begin{pmatrix}
q_0 \\
q_1
\end{pmatrix},
$$

which rotates the vector $q$ counterclockwise by an angle $m\theta$.

If $d_{\text{model}}$ is a multiple of $2$, then we can stack this 2D rotation to get the full positional embedding:

$$
f(q, m) =
\underbrace{
\begin{pmatrix}
\cos m\theta_0 & -\sin m\theta_0 & \cdots & 0 & 0 \\
\sin m\theta_0 & \cos m\theta_0 & \cdots & 0 & 0 \\
\vdots & \vdots  & \ddots & \vdots & \vdots \\
0 & 0  & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\
0 & 0  & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1}
\end{pmatrix}
}_{\mathcal{R}_m}
\begin{pmatrix}
q_0 \\
q_1 \\
\vdots \\
q_{d-2} \\
q_{d-1}
\end{pmatrix}
$$

Then the relative position information is encoded in attention weights:

$$
(\mathcal{R}_m q)^\top (\mathcal{R}_n k)
= q^\top \mathcal{R}_m^\top \mathcal{R}_n k
= q^\top \mathcal{R}_{n-m} k
$$

```python
class RoPE(nn.Module):
    def forward(
        self,
        x: Float[torch.Tensor, "... seqlen d_k"],
        token_positions: Int[torch.Tensor, "... seqlen"],
    ) -> Float[torch.Tensor, "... seqlen d_k"]: ...
```

## SDPA: Scaled Dot-Product Attention

Scaled dot-product attention is defined as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

where $Q \in \mathbb{R}^{n \times d_k}, K \in \mathbb{R}^{m \times d_k}, V \in \mathbb{R}^{m \times d_v}$.

```python
class ScaledDotProductAttention(nn.Module):
    def forward(
        self,
        Q: Float[torch.Tensor, "... queries d_k"],
        K: Float[torch.Tensor, "... keys d_k"],
        V: Float[torch.Tensor, "... queries d_v"],
        mask: Bool[torch.Tensor, "... queries keys"] | None = None,
    ) -> Float[torch.Tensor, "... queries d_v"]: ...
```

## MHA: Causal Multi-Head Self-Attention With RoPE

Mathematically, multi-head attention is defined as follows:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \\
\text{for } \text{head}_i = \text{Attention}(Q_i, K_i, V_i)
$$

And multi-head self-attention is defined as:

$$
\text{MultiHeadSelfAttention}(x) = W_O \text{MultiHeadAttention}(W_Qx, W_Kx, W_Vx) \\
$$

Some implementation details:

- Causal: add a lower triangle matrix `mask` to prevent current query from attending future tokens. (1: attend, 0: ignore)

- RoPE: add positional information to **queries** and **keys** with RoPE. Each head is applied independently. **Do not** apply RoPE to value vectors.

```python
class MultiheadSelfAttentionWithRoPE(nn.Module):
    def forward(
        self,
        x: Float[torch.Tensor, "... seqlen d_model"],
        token_positions: Int[torch.Tensor, "... seqlen"],
        mask: Bool[torch.Tensor, "... seqlen seqlen"] | None = None,
    ) -> Float[torch.Tensor, "... seqlen d_model"]:
```
