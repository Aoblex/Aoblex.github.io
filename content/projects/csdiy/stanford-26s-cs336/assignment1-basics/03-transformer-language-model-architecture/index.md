+++
title = 'Section 3: Transformer Language Model Architecture'
date = '2026-04-28T23:49:45+08:00'
summary = 'Build a Transformer model from scratch.'
weight = 30
draft = false
+++

# Introduction

A Transformer language model maps a sequence of token IDs to next-token logits. The model first represents each token as a vector, repeatedly mixes information across positions and transforms each position independently, then projects the final hidden states back into the vocabulary space:

$$
X \in \{0,\ldots,v-1\}^{b\times n}
\longmapsto
H \in \mathbb{R}^{b\times n\times d}
\longmapsto
Z \in \mathbb{R}^{b\times n\times v}.
$$

The architecture is built from a small set of reusable operations. Their significance lies not in the individual modules, but in how attention, position-wise transformation, normalization, and residual connections cooperate to preserve and refine a shared residual stream.

---

# Transformer Language Model

{{< gallery cols="2" gap="16px" align="center" >}}
{{< gallery-item src="figures/transformer-overview.png" alt="An overview of a Transformer language model" caption="Transformer language model" >}}
{{< gallery-item src="figures/transformer-block.png" alt="The structure of a pre-norm Transformer block" caption="Pre-norm Transformer block" >}}
{{< /gallery >}}

Given token IDs $X$, the model computes

$$
H^{(0)} = E[X],
\qquad
H^{(\ell)} = \operatorname{Block}_{\ell}\!\left(H^{(\ell-1)}\right),
\qquad
Z = \operatorname{RMSNorm}\!\left(H^{(L)}\right)W_{\mathrm{LM}}^{\top},
$$

where $E\in\mathbb{R}^{v\times d}$ is the token embedding, $L$ is the number of Transformer blocks, and $Z$ contains one vocabulary-sized logit vector for every input position.

The model itself returns logits. Softmax is applied later when the logits are used to compute a loss or sample the next token.

## Tensor Conventions

For operations that act only on the final feature dimension, all preceding dimensions are batch-like and carry independent applications of the same transformation. For example,

$$
X\in\mathbb{R}^{\ldots\times d_{\mathrm{in}}},
\qquad
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
\qquad
Y=XW^{\top}\in\mathbb{R}^{\ldots\times d_{\mathrm{out}}}.
$$

Mathematical notation often writes the same transformation as $y=Wx$ using column vectors. Keeping the feature axes explicit makes the batching dimensions—including attention heads—largely mechanical.

---

# Basic Transformations

## Parameter Initialization

The assignment uses a simple initialization shared by the modules below:

- Linear weights: $\mathcal{N}(0,\sigma^2)$ with $\sigma^2=\frac{2}{d_{\mathrm{in}}+d_{\mathrm{out}}}$, truncated to $[-3\sigma,3\sigma]$.
- Token embeddings: $\mathcal{N}(0,1)$, truncated to $[-3,3]$.
- RMSNorm gains: initialized to $1$.

## Linear Projection

A bias-free linear layer transforms only the final feature dimension:

$$
y=Wx,
\qquad
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}.
$$

For a batched row-major tensor, this becomes $Y=XW^{\top}$. The same operation is used for attention projections, feed-forward projections, and the final language-model head.

## Token Embedding

An embedding layer is a lookup table rather than a matrix multiplication. For a token ID $x_i$,

$$
h_i=E[x_i],
\qquad
E\in\mathbb{R}^{v\times d}.
$$

Applied to $X\in\{0,\ldots,v-1\}^{b\times n}$, it produces $H^{(0)}\in\mathbb{R}^{b\times n\times d}$.

---

# Pre-Norm Transformer Block

A Transformer block contains two residual sublayers. Normalization is applied before each transformation, leaving an uninterrupted residual path through the block:

$$
U=X+\operatorname{MHA}(\operatorname{RMSNorm}(X)),
$$

$$
Y=U+\operatorname{SwiGLU}(\operatorname{RMSNorm}(U)).
$$

Self-attention mixes information across sequence positions, whereas SwiGLU transforms each position independently.

## RMSNorm

For $x\in\mathbb{R}^{d}$ and a learned gain $g\in\mathbb{R}^{d}$,

$$
\operatorname{RMSNorm}(x)=g\odot\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\varepsilon}}.
$$

RMSNorm controls the scale of the residual stream without subtracting its mean. The squared mean and normalization are computed in `float32` to avoid overflow, then converted back to the input dtype.

## SwiGLU

The position-wise feed-forward network uses two input projections whose outputs interact through a gate:

$$
\operatorname{SwiGLU}(x)=W_2\left(\operatorname{SiLU}(W_1x)\odot W_3x\right),
$$

where

$$
\operatorname{SiLU}(z)=z\sigma(z),
\qquad
W_1,W_3\in\mathbb{R}^{d_{\mathrm{ff}}\times d},
\qquad
W_2\in\mathbb{R}^{d\times d_{\mathrm{ff}}}.
$$

The assignment chooses $d_{\mathrm{ff}}\approx\frac{8}{3}d$, rounded to a nearby multiple of $64$. The factor compensates for SwiGLU's three projection matrices while retaining a parameter scale comparable to a conventional two-layer FFN with width $4d$.

## Rotary Positional Embeddings

RoPE injects position by rotating adjacent pairs of query and key coordinates. For pair index $k$ and position $m$, define

$$
\omega_k=\Theta^{-2k/d_k},
\qquad
R_{m,k}=
\begin{pmatrix}
\cos(m\omega_k)&-\sin(m\omega_k)\\
\sin(m\omega_k)&\cos(m\omega_k)
\end{pmatrix}.
$$

Applying the block-diagonal rotation $R_m$ to a vector encodes its absolute position. The attention score depends only on the relative displacement because

$$
\begin{aligned}
(R_mq)^{\top}(R_nk)
&=q^{\top}R_m^{\top}R_nk\\
&=q^{\top}R_{n-m}k.
\end{aligned}
$$

The sine and cosine values are fixed rather than learned, so they can be precomputed and reused. RoPE is applied to queries and keys, not values.

## Stable Softmax

Softmax converts scores into a probability distribution. Subtracting the maximum leaves the result unchanged while preventing exponentials from overflowing:

$$
\operatorname{softmax}(x)_i=\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)},
\qquad
m=\max_jx_j.
$$

## Scaled Dot-Product Attention

For queries, keys, and values

$$
Q\in\mathbb{R}^{\ldots\times n_q\times d_k},
\quad
K\in\mathbb{R}^{\ldots\times n_k\times d_k},
\quad
V\in\mathbb{R}^{\ldots\times n_k\times d_v},
$$

scaled dot-product attention is

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V.
$$

Scaling keeps the score magnitude controlled as $d_k$ grows. A boolean mask is applied before softmax by replacing disallowed scores with $-\infty$, which gives those positions zero probability.

## Causal Multi-Head Self-Attention

Self-attention derives queries, keys, and values from the same hidden states. With $h$ heads and $d_h=d/h$, the projections are reshaped from

$$
(b,n,d)
\longrightarrow
(b,h,n,d_h).
$$

RoPE is applied independently to the query and key of every head. A causal mask

$$
M_{ij}=[j\leq i]
$$

allows position $i$ to use only itself and earlier positions. The head outputs are concatenated and passed through an output projection:

$$
\operatorname{MHA}(X)=W_O\operatorname{Concat}\!\left(
\operatorname{Attention}(Q_1,K_1,V_1),\ldots,
\operatorname{Attention}(Q_h,K_h,V_h)
\right).
$$

---

# Full Transformer LM

The complete model preserves the sequence length throughout its hidden layers:

$$
X:(b,n)
\xrightarrow{\text{embedding}}
H^{(0)}:(b,n,d)
\xrightarrow{L\ \text{blocks}}
H^{(L)}:(b,n,d)
\xrightarrow{\text{final norm + LM head}}
Z:(b,n,v).
$$

The causal mask ensures that $Z_{:,i,:}$ depends only on input positions up to $i$. Each row of logits can therefore be trained to predict the following token without leaking future information.

---

# Resource Accounting

Matrix multiplications dominate the arithmetic cost of a Transformer. For

$$
A\in\mathbb{R}^{m\times k},
\qquad
B\in\mathbb{R}^{k\times p},
$$

the product $AB$ costs approximately $2mkp$ FLOPs.

Let $n$ be sequence length, $d$ model width, $d_{\mathrm{ff}}$ feed-forward width, $v$ vocabulary size, and $l$ the number of layers. Ignoring the batch dimension, the trainable parameters are:

| Component | Parameters |
| --- | ---: |
| Token embedding | $dv$ |
| Attention projections | $4d^2l$ |
| SwiGLU projections | $3dd_{\mathrm{ff}}l$ |
| RMSNorm gains | $2dl+d$ |
| LM head | $dv$ |
| **Total** | **$4d^2l+3dd_{\mathrm{ff}}l+2dl+2dv+d$** |

For one forward pass over a single sequence, the matrix-multiplication FLOPs are:

| Component | FLOPs |
| --- | ---: |
| Attention projections | $8nd^2l$ |
| Attention scores and value mixing | $4n^2dl$ |
| SwiGLU projections | $6ndd_{\mathrm{ff}}l$ |
| LM head | $2ndv$ |
| **Total** | **$8nd^2l+4n^2dl+6ndd_{\mathrm{ff}}l+2ndv$** |

At moderate context lengths, the FFN and projection terms usually dominate because they scale with model width. As $n$ grows, the quadratic attention term eventually becomes the main cost. Parameter memory follows directly from the parameter count, but excludes activations, gradients, optimizer states, temporary buffers, and KV cache.

---

# Solutions

> [!note]- Problem (`linear`): Implementing the Linear Module (1 point)
> > [!question]- Deliverable: Implement a bias-free linear transformation
> > Implement a `Linear` module with `in_features`, `out_features`, `device`, and `dtype`. Store $W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$ as an `nn.Parameter`, use the prescribed truncated-normal initialization, and do not use `nn.Linear` or `nn.functional.linear`. The implementation is evaluated through `[adapters.run_linear]`.
> >
> > **Answer**: The module stores the weight in mathematical orientation and applies it to arbitrary leading batch dimensions as $XW^{\top}$. It contains no bias term.

> [!note]- Problem (`embedding`): Implementing the Embedding Module (1 point)
> > [!question]- Deliverable: Implement token embedding lookup
> > Implement an `Embedding` module with `num_embeddings`, `embedding_dim`, `device`, and `dtype`. Store the matrix as an `nn.Parameter` of shape $(v,d)$, initialize it with the prescribed truncated normal, and do not use `nn.Embedding` or `nn.functional.embedding`. The implementation is evaluated through `[adapters.run_embedding]`.
> >
> > **Answer**: Integer token IDs index rows of the embedding matrix, preserving every input dimension and appending $d$ as the final feature dimension.

> [!note]- Problem (`rmsnorm`): Root Mean Square Layer Normalization (1 point)
> > [!question]- Deliverable: Implement RMSNorm
> > Implement `RMSNorm(d_model, eps=1e-5, device=None, dtype=None)` as an `nn.Module`. It must accept tensors ending in $d_{\mathrm{model}}$, normalize along that dimension, upcast the computation to `float32`, and return the original dtype. The implementation is evaluated through `[adapters.run_rmsnorm]`.
> >
> > **Answer**: RMSNorm divides each vector by its root-mean-square magnitude and applies a learned element-wise gain. Upcasting the reduction prevents overflow when low-precision activations are squared.

> [!note]- Problem (`positionwise_feedforward`): Position-Wise Feed-Forward Network (2 points)
> > [!question]- Deliverable: Implement SwiGLU
> > Implement the SwiGLU feed-forward network using SiLU and a gated linear unit. Choose $d_{\mathrm{ff}}\approx\frac{8}{3}d_{\mathrm{model}}$ and round it to a nearby multiple of $64$. The implementation is evaluated through `[adapters.run_swiglu]`.
> >
> > **Answer**: Two projections form the gate and value branches, their element-wise product is projected back to $d_{\mathrm{model}}$, and the operation is applied independently at every sequence position.

> [!note]- Problem (`rope`): Rotary Positional Embeddings (2 points)
> > [!question]- Deliverable: Implement RoPE
> > Implement `RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=None)`. It must accept $x$ of shape $(\ldots,n,d_k)$ and token positions of shape $(\ldots,n)$, support arbitrary leading batch dimensions, and return the same shape. Use token positions to select any precomputed sine and cosine values. The implementation is evaluated through `[adapters.run_rope]`.
> >
> > **Answer**: RoPE rotates adjacent coordinate pairs by position-dependent angles. The sine and cosine tables are fixed buffers, and broadcasting applies the same rotation rule across all leading dimensions.

> [!note]- Problem (`softmax`): Numerically Stable Softmax (1 point)
> > [!question]- Deliverable: Implement softmax along a selected dimension
> > Implement softmax for an input tensor and a specified dimension. The output must preserve the input shape and normalize along that dimension. Subtract the maximum value before exponentiation for numerical stability. The implementation is evaluated through `[adapters.run_softmax]`.
> >
> > **Answer**: Subtracting the maximum changes neither the normalized distribution nor its shape, but ensures that every exponent has a non-positive argument.

> [!note]- Problem (`scaled_dot_product_attention`): Scaled Dot-Product Attention (5 points)
> > [!question]- Deliverable: Implement scaled dot-product attention
> > Accept queries and keys of shape $(\text{batch},\ldots,n,d_k)$ and values of shape $(\text{batch},\ldots,n,d_v)$, allowing arbitrary batch-like dimensions. Return shape $(\text{batch},\ldots,n,d_v)$ and support an optional boolean mask whose `True` entries are the positions that may receive probability mass. The implementation is evaluated through `[adapters.run_scaled_dot_product_attention]`.
> >
> > **Answer**: Attention forms scaled query-key scores, replaces masked entries with $-\infty$, normalizes across keys, and uses the resulting probabilities to mix the values.

> [!note]- Problem (`multihead_self_attention`): Causal Multi-Head Self-Attention (5 points)
> > [!question]- Deliverable: Implement causal multi-head self-attention
> > Implement an `nn.Module` accepting at least `d_model` and `num_heads`, with $d_k=d_v=d_{\mathrm{model}}/h$. Compute query, key, and value projections, apply RoPE to queries and keys, enforce causal attention, combine the heads, and apply the output projection. The implementation is evaluated through `[adapters.run_multihead_self_attention]`.
> >
> > **Answer**: The head dimension acts as an additional batch dimension. A lower-triangular mask enforces causality, and only queries and keys receive positional rotations.

> [!note]- Problem (`transformer_block`): Pre-Norm Transformer Block (3 points)
> > [!question]- Deliverable: Implement the Transformer block
> > Implement the pre-norm block with `d_model`, `num_heads`, and `d_ff`, using causal multi-head self-attention and a SwiGLU feed-forward network. Each sublayer must be preceded by RMSNorm and wrapped in a residual connection. The implementation is evaluated through `[adapters.run_transformer_block]`.
> >
> > **Answer**: The block performs normalized attention and normalized feed-forward transformations while both residual additions preserve an identity path through the network.

> [!note]- Problem (`transformer_lm`): Transformer Language Model (3 points)
> > [!question]- Deliverable: Implement the complete Transformer LM
> > Assemble token embeddings, `num_layers` Transformer blocks, a final RMSNorm, and an LM head. In addition to the block parameters, accept `vocab_size`, `context_length`, and `num_layers`. The output must contain next-token logits for every input position. The implementation is evaluated through `[adapters.run_transformer_lm]`.
> >
> > **Answer**: The model maps token IDs to hidden states, refines them through a stack of causal blocks, and projects the normalized final states to logits of shape $(b,n,v)$.

> [!note]- Problem (`transformer_accounting`): Transformer LM Resource Accounting (5 points)
> > [!question]- (a) Count GPT-2 XL parameters and parameter memory
> > Consider the assignment architecture with `vocab_size=50,257`, `context_length=1,024`, `num_layers=48`, `d_model=1,600`, `num_heads=25`, and `d_ff=4,288`. How many trainable parameters does it contain, and how much memory is required when every parameter uses single-precision floating point?
> >
> > **Deliverable:** A one-to-two sentence response.
> >
> > **Answer**:
> >
> > | Component | Parameters |
> > | --- | ---: |
> > | Token embedding | 80,411,200 |
> > | Attention projections | 491,520,000 |
> > | SwiGLU FFN | 987,955,200 |
> > | RMSNorm | 155,200 |
> > | LM head | 80,411,200 |
> > | **Total** | **1,640,452,800** |
> >
> > The model has approximately **1.64B parameters**. At four bytes per parameter, the weights occupy **6,561,811,200 bytes**, or approximately **6.11 GiB**.
> >
> > > [!example]- Reproduce the parameter count
> > > ```bash
> > > uv run python scripts/resource_accounting.py parameters
> > > ```
>
> > [!question]- (b) Count the matrix-multiplication FLOPs in one forward pass
> > Identify the required matrix multiplications and compute their total FLOPs for one sequence of `context_length=1,024` tokens.
> >
> > **Deliverable:** A list of the matrix multiplications and the total number of FLOPs.
> >
> > **Answer**:
> >
> > | Operation | Multiplications | FLOPs | Share |
> > | --- | ---: | ---: | ---: |
> > | QKV projections | $3nd^2l$ | $6nd^2l=7.5497\times10^{11}$ | 21.47% |
> > | Output projection | $nd^2l$ | $2nd^2l=2.5166\times10^{11}$ | 7.16% |
> > | Attention scores | $n^2dl$ | $2n^2dl=1.6106\times10^{11}$ | 4.58% |
> > | Attention value mixing | $n^2dl$ | $2n^2dl=1.6106\times10^{11}$ | 4.58% |
> > | SwiGLU input projections | $2ndd_{\mathrm{ff}}l$ | $4ndd_{\mathrm{ff}}l=1.3489\times10^{12}$ | 38.36% |
> > | SwiGLU output projection | $ndd_{\mathrm{ff}}l$ | $2ndd_{\mathrm{ff}}l=6.7444\times10^{11}$ | 19.18% |
> > | LM head | $ndv$ | $2ndv=1.6468\times10^{11}$ | 4.68% |
> > | **Total** |  | **$3.5168\times10^{12}$** | **100.00%** |
> >
> > One forward pass costs approximately **3.52T FLOPs** for a single sequence.
> >
> > > [!example]- Reproduce the forward FLOPs
> > > ```bash
> > > uv run python scripts/resource_accounting.py flops
> > > ```
>
> > [!question]- (c) Identify the dominant FLOP terms
> > Based on the accounting above, which parts of the model require the most FLOPs?
> >
> > **Deliverable:** A one-to-two sentence response.
> >
> > **Answer**: At $n=1{,}024$, SwiGLU accounts for **57.53%** of forward FLOPs and the attention projections account for **28.62%**. The two quadratic attention operations contribute only **9.16%** because the context length remains smaller than the model width.
>
> > [!question]- (d) Compare GPT-2 Small, Medium, Large, and XL
> > Repeat the analysis for GPT-2 Small (`12` layers, `d_model=768`, `12` heads), Medium (`24`, `1,024`, `16`), and Large (`36`, `1,280`, `20`). As model size increases at fixed context length, which components occupy proportionally more or less of the total FLOPs?
> >
> > **Deliverable:** A one-to-two sentence response.
> >
> > **Answer**: Using $d_{\mathrm{ff}}=2{,}048$, $2{,}752$, $3{,}456$, and $4{,}288$:
> >
> > | Model | Total FLOPs | Attention projections | Quadratic attention | SwiGLU FFN | LM head |
> > | --- | ---: | ---: | ---: | ---: | ---: |
> > | Small | $2.9165\times10^{11}$ | 19.88% | 13.25% | 39.76% | 27.10% |
> > | Medium | $8.3017\times10^{11}$ | 24.83% | 12.42% | 50.05% | 12.70% |
> > | Large | $1.7867\times10^{12}$ | 27.04% | 10.82% | 54.76% | 7.37% |
> > | XL | $3.5168\times10^{12}$ | 28.62% | 9.16% | 57.53% | 4.68% |
> >
> > The per-layer projection and FFN terms gain share as the models become wider and deeper, while the one-time LM head and fixed-context quadratic attention terms lose share.
> >
> > > [!example]- Reproduce the model-size comparison
> > > ```bash
> > > uv run python scripts/resource_accounting.py model-scaling
> > > ```
>
> > [!question]- (e) Increase GPT-2 XL context length to 16,384
> > How does the total forward-pass FLOP count change, and how do the relative contributions of the components change?
> >
> > **Deliverable:** A one-to-two sentence response.
> >
> > **Answer**:
> >
> > | Component | FLOPs | Share |
> > | --- | ---: | ---: |
> > | Attention projections | $1.6106\times10^{13}$ | 12.06% |
> > | Attention scores | $4.1232\times10^{13}$ | 30.87% |
> > | Attention value mixing | $4.1232\times10^{13}$ | 30.87% |
> > | SwiGLU FFN | $3.2373\times10^{13}$ | 24.24% |
> > | LM head | $2.6349\times10^{12}$ | 1.97% |
> > | **Total** | **$1.3358\times10^{14}$** | **100.00%** |
> >
> > The longer context raises the cost to approximately **133.58T FLOPs**, about **38 times** the $1{,}024$-token case. The quadratic attention operations now dominate with a combined **61.74%** share.
> >
> > > [!example]- Reproduce the long-context comparison
> > > ```bash
> > > uv run python scripts/resource_accounting.py long-context
> > > ```
