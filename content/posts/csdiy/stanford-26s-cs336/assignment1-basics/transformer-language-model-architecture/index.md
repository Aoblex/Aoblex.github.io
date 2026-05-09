+++
title = 'Section 2: Transformer Language Model Architecture'
date = '2026-04-28T23:49:45+08:00'
summary = 'Build a Transformer model from scratch! 🥸'
weight = 20
draft = false
+++

# Introduction

In [the last section](../byte-pair-encoding/), we converted input text into token IDs.
Before using these token IDs, we need to build a model that takes them as input:
$$
\text{input: (batch, sequence length)} \xrightarrow{\text{model}} \text{output: (batch, sequence length, vocab)}
$$

Formally, the input and output spaces can be described as:

$$
\text{input} \in \mathbb{N}^{\text{batch} \times \text{sequence length}}, \quad
\text{output} \in \mathbb{R}^{\text{batch} \times \text{sequence length} \times \text{vocab}}
$$

In this part of the assignment, we focus on the **Transformer architecture** and build one from scratch!

---

# Transformer LM

The first figure below shows the structure of a modern Transformer LM, and the second figure shows the internal structure of a Transformer block.

{{< gallery cols="2" gap="16px" align="center" >}}
{{< gallery-item src="figures/transformer-overview.png" alt="An overview of a Transformer model" caption="An overview of a Transformer model" >}}
{{< gallery-item src="figures/transformer-block.png" alt="The structure of a Transformer block" caption="The structure of a Transformer block" >}}
{{< /gallery >}}

Next, we will look at each component in detail and implement it from scratch.

> [!NOTE] Parameter Initialization
> Since this assignment is already long, the course staff defers the details of parameter initialization to assignment 3. For this assignment, they specify the following initialization configuration:
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

---

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
        token_ids: Int[torch.Tensor, "batch seqlen"],
    ) -> Float[torch.Tensor, "batch seqlen d_model"]:
```

---

## RMSNorm: Root Mean Square Layer Normalization

Mathematical notation:

$$
\begin{align*}
\text{RMSNorm}(a_i) &= \frac{a_i}{\text{RMS}(a)} g_i \\
\text{RMS}(a) &= \sqrt{\left(\frac{1}{d_{model}}\sum_{i=1}^{d_{model}} a_i^2\right) + \varepsilon}
\end{align*}
$$

where $a \in \mathbb{R}^{d_{model}}$ is the input activation vector, and $g \in \mathbb{R}^{d_{model}}$ is a vector of learnable "gain" parameters.

The forward function:

```python
class RMSNorm(nn.Module):
    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]: ...
```

---

## SwiGLU: Sigmoid Linear Unit + Gated Linear Unit

We will implement the position-wise feed-forward network in a Transformer block with a [**SwiGLU**](https://github.com/huggingface/transformers/blob/c6c7e189846ccaa3bb410569bfc6556d193c638d/src/transformers/models/qwen3/modeling_qwen3.py#L70) activation function, which combines [SiLU](https://arxiv.org/pdf/1702.03118) and [GLU](https://arxiv.org/abs/2002.05202):

$$
\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = \underbrace{W_2}_{\text{down proj}} (\underbrace{\text{SiLU}(W_1 x)}_{\text{gate proj}} \odot \underbrace{W_3 x}_{\text{up proj}})
$$

where $x \in \mathbb{R}^{d_{\text{model}}}$,
$W_1, W_3 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$,
$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$,
$\text{FFN}(x) \in \mathbb{R}^{d_{\text{model}}}$,
and canonically, $d_{\text{ff}} = \frac{8}{3} d_{\text{model}}$.

Some heuristic arguments from the writeup:

- GLUs are suggested to "reduce the vanishing gradient problem for deep architectures by providing **a linear path** for the gradients while retaining non-linear capabilities." To be specific:

    $$
    g := \sigma(W_gx), v := W_vx \\
    y=g\odot v
    $$

    Then:

    $$
    \frac{\partial y}{\partial x}= \frac{\partial y}{\partial g} \frac{\partial g}{\partial x} + \frac{\partial y}{\partial v} \frac{\partial v}{\partial x} = \underbrace{\operatorname{diag}(v) \sigma^\prime(W_g x) W_g}_{g \text{ channel}} + \underbrace{\operatorname{diag}(g) W_v}_{v \text{ channel}}
    $$

    Here, $\sigma^\prime (W_g x) = \sigma (W_g x) \odot (1 - \sigma(W_g x)) \le 0.25$ elementwise when $\sigma(x) = \frac{1}{1 + e^{-x}}$. Therefore, the $v$ channel provides a linear path for gradients.

- It is fine to round the dimensions to a nearby multiple of 64 for hardware efficiency.

- Keep an empirical perspective: the exact dimension choice is ultimately an implementation and performance tradeoff.

```python
class SwiGLU(nn.Module):
    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]: ...
```

---

## RoPE: Rotary Positional Embeddings

The idea behind [RoPE](https://kexue.fm/archives/8265) is to **implement relative positional encoding via absolute positional encoding.**

Suppose that we have a query vector $q$ at position $m$ and a key vector $k$ at position $n$. Our positional embedding function is $f: \mathbb{R}^{d_{\text{model}}} \times \mathbb{N} \rightarrow \mathbb{R}^{d_{\text{model}}}$:

$$
\tilde{q}_m = f(q, m), \tilde{k}_n = f(k, n)
$$

Then absolute position information is injected into $\tilde{q}_m$ and $\tilde{k}_n$. However, we want the inner product in the attention layer to depend on **relative** position information:

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

Then relative position information is encoded in the attention weights:

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

---

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
        V: Float[torch.Tensor, "... keys d_v"],
        mask: Bool[torch.Tensor, "... queries keys"] | None = None,
    ) -> Float[torch.Tensor, "... queries d_v"]: ...
```

---

## MHA: Causal Multi-Head Self-Attention With RoPE

Mathematically, multi-head attention is defined as follows:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \\
\text{for } \text{head}_i = \text{Attention}(Q_i, K_i, V_i)
$$

Multi-head self-attention is then defined as:

$$
\text{MultiHeadSelfAttention}(x) = W_O \text{MultiHeadAttention}(W_Qx, W_Kx, W_Vx) \\
$$

Some implementation details:

- Causal: apply a lower triangular `mask` to prevent the current query from attending to future tokens. (1: attend, 0: ignore)

- RoPE: add positional information to **queries** and **keys** with RoPE. Each head is processed independently. **Do not** apply RoPE to value vectors.

```python
class MultiheadSelfAttentionWithRoPE(nn.Module):
    def forward(
        self,
        x: Float[torch.Tensor, "... seqlen d_model"],
        token_positions: Int[torch.Tensor, "... seqlen"],
    ) -> Float[torch.Tensor, "... seqlen d_model"]:
```

---

## Transformer Block

The structure of a Transformer block is shown [here](#transformer-lm).

```python
class TransformerBlock(nn.Module):
    def forward(
        self,
        x: Float[torch.Tensor, "... seqlen d_model"],
        token_positions: Int[torch.Tensor, "... seqlen"],
        mask: Bool[torch.Tensor, "... seqlen seqlen"] | None = None,
    ) -> Float[torch.Tensor, "... seqlen d_model"]:
        x = x + self.attn(self.ln1(x), token_positions, mask)
        x = x + self.ffn(self.ln2(x))
        return x
```

---

## Transformer LM

Putting all these components together gives us the full Transformer model:

```python
class Transformer(nn.Module):
    def forward(
        self,
        x: Int[torch.Tensor, "batch seqlen"],
        token_positions: Int[torch.Tensor, "batch seqlen"],
    ) -> Float[torch.Tensor, "batch seqlen vocab"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
```

---

# Resource Accounting

Here, we do a comprehensive resource accounting for the entire Transformer model, including **compute** and **memory**.
For now, we ignore the $\text{batch}$ dimension and assume that the input has length $\text{n}$.

Notation:

- $n$: input sequence length;

- $d$: dimension of the model;

- $d_{\text{ff}}$: dimension of the FFN up projection;

- $h$: number of attention heads;

- $d^{\prime} = d / h$: dimension per attention head;

- $v$: vocabulary size;

- $L$: number of layers;

## Compute

{{< collapse summary="Transformer" >}}

{{< collapse summary="Embedding Layer" >}}
This operation consists only of indexing (lookup), so it is dominated by memory access rather than floating-point arithmetic.

Total: **0 FLOPs**.
{{< /collapse >}}

{{< collapse summary="Transformer Block" >}}

```python
x = x + self.attn(self.ln1(x), token_positions)
x = x + self.ffn(self.ln2(x))
```

{{< collapse summary="Attention" >}}

- $x_{\text{ln1}} = \text{ln1}(x)$:
    - $A = \sqrt{\frac{1}{d}\sum_i (x_i)^2 + \varepsilon} \approx 2nd$ FLOPs;
    - ${(x_{\text{ln1}})}_{i} = \frac{x_i g_i}{A} \approx 2nd$ FLOPs;
    - Total: $4nd$ FLOPs;
- $x_{\text{attn}} = x + \text{attn}(x_{\text{ln1}}, \text{token\\_positions})$:
    - $Q/K/V = W_{Q/K/V} x_{\text{ln1}} \approx 3 \times 2nd^2 = 6 nd^2$ FLOPs;
    - Reshape $Q/K/V: (n, d) \rightarrow (h, n, d^\prime) = 0$ FLOPs;
    - $Q/K = \text{RoPE}(Q/K, \text{token\\_positions}) \approx h \times 4nd^{\prime} = 4nd$ FLOPs;
    - $x_{\text{scores}} = QK^T / \sqrt{d'} \approx 2 h n^2 d^{\prime} = 2 n^2 d$ FLOPs;
    - $x_{\text{weights}} = \text{softmax}(x_{\text{scores}}) \approx 3 h n^2$ FLOPs;
    - $x_{\text{out}} = (\text{reshape}(x_{\text{weights}} V)) O \approx 2hn^2d^{\prime} + 2nd^2 = 2n^2d + 2nd^2$ FLOPs;
    - $x_{\text{attn}} = x + x_{\text{out}} \approx nd$ FLOPs;
    - Total: $(4d+3h)n^2+(9d+8d^2)n$ FLOPs;

{{< /collapse >}}

{{< collapse summary="FFN" >}}

- $x_{\text{ln2}} = \text{ln2}(x)$:
    - Total: $4nd$ FLOPs;
- $x_{\text{ffn}} = x + \text{ffn}(x_{\text{ln2}})$:
    - $x_{\text{act}} = W_3 x_{\text{ln2}} \approx 2ndd_{\text{ff}}$ FLOPs;
    - $x_{\text{gates}} = \text{SiLU}(W_1 x_{\text{ln2}}) \approx 2ndd_{\text{ff}} + 4 n d_{\text{ff}}$ FLOPs;
    - $x_{\text{up}} = x_{\text{act}} \odot x_{\text{gates}} \approx nd_{\text{ff}}$ FLOPs;
    - $x_{\text{out}} = W_2 x_{\text{up}} \approx 2nd_{\text{ff}}d$ FLOPs;
    - $x_{\text{ffn}} = x + x_{\text{out}} \approx nd$ FLOPs;
    - Total: $(6d_{\text{ff}}d + 5d_{\text{ff}} + 5d)n$ FLOPs;

{{< /collapse >}}

{{< /collapse >}}

{{< collapse summary="Out Norm" >}}

- $x_{\text{ln\\_final}} = \text{ln\\_final}(x)$
    - Total: $4nd$ FLOPs;

{{< /collapse >}}

{{< collapse summary="LM Head" >}}
- $x_{\text{lm}} = W_{\text{lm}}(x)$
    - Total: $2ndv$ FLOPs;
{{< /collapse >}}

{{< /collapse >}}

Summing them gives:

$$
\begin{align*}
\text{FLOPs} = &L\left\{ \left[ (4d+3h)n^2 + (9d+8d^2)n \right] + (6d_{\text{ff}}d+5d_{\text{ff}}+5d)n \right\} \\
               &+ 4nd + 2ndv
\end{align*}
$$


## Memory

{{< collapse summary="Transformer" >}}

{{< collapse summary="Embedding Layer" >}}

- embedding matrix: $\text{Float}[v, d]$;
- size: $vd \times 4 \text{ bytes} = 4vd \text{ bytes}$;

{{< /collapse >}}

{{< collapse summary="Transformer Block" >}}

- $\text{ln1/2}$: $\text{Float}[d]$;
- size: $8d \text{ bytes}$;

{{< collapse summary="Attention" >}}

- $W_{Q/K/V/O}$: $\text{Float}[d, d]$;
- $\text{mask}$: $\text{Bool}[n, n]$;
- $\text{RoPE}$:
    - $\cos$: $\text{Float}[n, \frac{d^{\prime}}{2}]$;
    - $\sin$: $\text{Float}[n, \frac{d^{\prime}}{2}]$;
- size: $16d^2 + n^2 + 4nd^{\prime} \text{ bytes}$;

{{< /collapse >}}

{{< collapse summary="FFN" >}}

- $W_{1/3}$: $\text{Float}[d_{\text{ff}}, d]$;
- $W_{2}$: $\text{Float}[d, d_{\text{ff}}]$;
- size: $12 d d_{\text{ff}} \text{ bytes}$;

{{< /collapse >}}

{{< /collapse >}}

{{< collapse summary="Out Norm" >}}

- $\text{ln\\_final}$: $\text{Float}[d]$;
- size: $4d \text{ bytes}$;

{{< /collapse >}}

{{< collapse summary="LM Head" >}}

- $W_{\text{lm}}$: $\text{Float}[v, d]$;
- size: $4dv \text{ bytes}$;

{{< /collapse >}}

{{< /collapse >}}

Summing them gives:

$$
\begin{align*}
\text{size} = 4vd + L(8d+16d^2+n^2+4nd^{\prime}+12dd_{\text{ff}})+4d+4vd \text{ bytes}
\end{align*}
$$
