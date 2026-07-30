+++
title = 'Understanding the KV Cache'
date = '2026-07-23T22:01:12+08:00'
summary = 'A derivation of key-value caching from causal self-attention, with explicit assumptions, correctness arguments, and complexity analysis.'
draft = false
+++

> [!info] Overview
> A **key-value (KV) cache** is an inference-time data structure that stores the key and value states produced by every self-attention layer. The cache is initialized during prefill and extended during [incremental decoding](https://arxiv.org/abs/1911.02150), allowing previously computed states to be reused rather than recomputed.

The validity of this reuse follows directly from the [causal masking constraint](https://arxiv.org/abs/1706.03762): when a sequence is extended on the right, earlier positions cannot attend to the appended tokens. Consequently, under the assumptions stated below, the hidden states—and therefore the key and value states—of earlier positions remain unchanged.

## Notation

| Symbol | Definition |
| --- | --- |
| $t$ | Number of tokens generated so far |
| $N_t$ | Sequence length at step $t$, where $N_0$ is the prompt length |
| $L$ | Number of Transformer blocks |
| $d$ | Hidden-state dimension; under the single-head assumption, also the attention-head dimension |
| $\mathcal{V}$ | Finite vocabulary of token IDs |
| $\ell$ | Layer index, with the embedding output indexed by $\ell = 0$ |
| $\mathbf{Y}_0$ | Token IDs of the prompt |
| $\mathbf{Y}_t$ | Token sequence after $t$ tokens have been generated |
| $y_t$ | The $t$-th generated token |
| $\mathbf{X}_t^{(\ell)}$ | Hidden states after layer $\ell$ at decoding step $t$ |
| $\mathbf{x}_t^{(\ell)}$ | Last row of $\mathbf{X}_t^{(\ell)}$ |

## Scope and Assumptions

The derivation considers a standard decoder-only Transformer evaluated under the following conditions:

- **Single sequence and single attention head.** We set `batch_size = 1` and `num_heads = 1` to suppress indices that do not affect the argument.
- **Deterministic inference.** Model parameters are fixed and stochastic training-time operations such as dropout are disabled.
- **Stable positions.** Appending a token does not change the position IDs assigned to existing tokens.
- **Causal sequence mixing.** Causal self-attention is the only operation that mixes information across sequence positions.
- **Position-wise sublayers are omitted from the notation.** Q/K/V projections are retained because their outputs define the cache. Output projections, feed-forward networks, normalization layers, residual connections, projection biases, and position-dependent transformations such as RoPE act independently at each position. Given unchanged inputs and position IDs, they preserve equality at earlier positions and therefore do not alter the argument.

Under these assumptions, each Transformer block can be represented by Q/K/V projections followed by causal self-attention. The same reasoning extends directly to multi-head attention by restoring the head dimension.

> [!warning] Applicability
> The argument applies to conventional causal decoding with an unchanged prefix. Bidirectional attention, masks that expose earlier positions to appended tokens, sequence-wise operations outside causal attention, or changes to existing position IDs require a separate analysis. Sliding-window and eviction-based caches additionally restrict which past states are retained.

## Naïve Autoregressive Decoding

### Repeated Full-Sequence Evaluation

Without a cache, a naïve decoder evaluates the complete sequence at every generation step:

$$
\begin{aligned}
\mathbf{Y}_t \in \mathcal{V}^{N_t}
&\xrightarrow{\operatorname{embedding}}
\mathbf{X}_t^{(0)}
\in \mathbb{R}^{N_t \times d}, \qquad N_t = N_0 + t
\\
&\xrightarrow{L\ \text{transformer blocks}}
\mathbf{X}_t^{(L)}
\in \mathbb{R}^{N_t \times d}
\\
&\xrightarrow{\text{take last state}}
\mathbf{x}_{t}^{(L)}
\in \mathbb{R}^{1 \times d}
\\
&\xrightarrow{\text{output stage}}
y_{t+1}
\in \mathcal{V}.
\end{aligned}
$$

Suppose a forward pass on $\mathbf{Y}_{t}$ has produced the next token $y_{t+1}$:

$$
\mathbf{Y}_t
\in \mathcal{V}^{N_t}
\xrightarrow{\text{model}}
y_{t+1}
\in \mathcal{V}.
$$

The decoder then appends $y_{t+1}$ and performs another full forward pass:

$$
\mathbf{Y}_{t+1} =
\begin{bmatrix}
\mathbf{Y}_t \\
y_{t+1}
\end{bmatrix}
\in \mathcal{V}^{N_{t+1}}
\xrightarrow{\text{model}}
y_{t+2}
\in \mathcal{V}.
$$

The sequence grows by one token per step until a stopping criterion is met. This procedure is mathematically correct but computationally wasteful because it repeatedly reconstructs states associated with the unchanged prefix.

> [!tip] The Prefill Phase
> In actual inference, we begin at $t = 0$:
>
> $$
> \mathbf{Y}_0 \in \mathcal{V}^{N_0} \xrightarrow{\text{model}} y_{1}.
> $$
>
> This first full-sequence evaluation is the **prefill phase**. It produces the first generated token and, in a cached implementation, initializes the per-layer key and value states used by subsequent decoding steps.

### Causal Masking Leaves Earlier Token States Unchanged

The key fact required by KV caching is a direct consequence of causal masking rather than a separately named architectural property.

> [!info] Proposition
> For every decoding step $t \geq 0$ and every layer $\ell \in \{0, 1, \ldots, L\}$, appending $y_{t+1}$ leaves the hidden states of the existing $N_t$ positions unchanged:
>
> $$
> \mathbf{X}_{t+1,\,1:N_t}^{(\ell)} =
> \mathbf{X}_{t}^{(\ell)}.
> $$
>
> Equivalently,
>
> $$
> \mathbf{X}_{t+1}^{(\ell)} =
> \begin{bmatrix}
> \mathbf{X}_{t}^{(\ell)} \\
> \mathbf{x}_{t+1}^{(\ell)}
> \end{bmatrix}.
> $$

The proposition is proved by induction over the layer index:

1. At $\ell=0$, embedding the extended token sequence preserves the states of existing positions.
2. For $\ell \in \{0, 1, \ldots, L-1\}$, if the input to block $\ell+1$ is unchanged at existing positions, causal self-attention produces unchanged outputs at those positions.

#### Base Case: Embedding the Extended Sequence

As shown in the [full-sequence evaluation](#repeated-full-sequence-evaluation), $\mathbf{Y}_{t+1} = \begin{bmatrix} \mathbf{Y}_{t} \\ y_{t+1} \end{bmatrix}$ extends $\mathbf{Y}_{t}$ without modifying its first $N_t$ token IDs. Token embedding is position-wise, and the stable-position assumption ensures that any positional transformation applied to an existing token is also unchanged. Therefore:

$$
\mathbf{X}_{t+1}^{(0)} =
\operatorname{embedding}\left(\mathbf{Y}_{t+1}\right) =
\operatorname{embedding}\left(\begin{bmatrix} \mathbf{Y}_{t} \\ y_{t+1} \end{bmatrix}\right) =
\begin{bmatrix}
\mathbf{X}_{t}^{(0)} \\
\mathbf{x}_{t+1}^{(0)}
\end{bmatrix}.
$$

#### Inductive Step: One Transformer Block

Assume that the proposition holds at layer $\ell$:

$$
\mathbf{X}_{t+1}^{(\ell)} =
\begin{bmatrix}
\mathbf{X}_{t}^{(\ell)} \\
\mathbf{x}_{t+1}^{(\ell)}
\end{bmatrix}.
$$

Under the simplified representation of a Transformer block:

$$
\begin{aligned}
\mathbf{X}_{t+1}^{(\ell+1)}
&=
\operatorname{TransformerBlock}_{\ell+1}
\left(
\mathbf{X}_{t+1}^{(\ell)}
\right) \\
&=
\operatorname{CausalSelfAttention}_{\ell+1}
\left(
\begin{bmatrix}
\mathbf{X}_{t}^{(\ell)} \\
\mathbf{x}_{t+1}^{(\ell)}
\end{bmatrix}
\right)
\end{aligned}.
$$

The displayed block contains only Q/K/V projections and causal attention. Reintroducing the omitted position-wise sublayers would not alter the proof: equal inputs at an existing position produce equal outputs at that position.

**Q/K/V projections.** Because each projection acts independently on every position:

$$
\mathbf{Q}_{t+1}^{(\ell+1)} =
\mathbf{X}_{t+1}^{(\ell)}
\mathbf{W}_Q^{(\ell+1)} =
\begin{bmatrix}
\mathbf{X}_{t}^{(\ell)} \\
\mathbf{x}_{t+1}^{(\ell)}
\end{bmatrix}
\mathbf{W}_Q^{(\ell+1)} =
\begin{bmatrix}
\mathbf{Q}_t^{(\ell+1)}\\
\mathbf{q}_{t+1}^{(\ell+1)}
\end{bmatrix}.
$$

Likewise, $\mathbf{K}$ and $\mathbf{V}$ satisfy:

$$
\mathbf{K}_{t+1}^{(\ell+1)} =
\begin{bmatrix}
\mathbf{K}_t^{(\ell+1)}\\
\mathbf{k}_{t+1}^{(\ell+1)}
\end{bmatrix},
\qquad
\mathbf{V}_{t+1}^{(\ell+1)} =
\begin{bmatrix}
\mathbf{V}_t^{(\ell+1)}\\
\mathbf{v}_{t+1}^{(\ell+1)}
\end{bmatrix}.
$$

**Causal attention.** Let $\mathbf{A}_{t+1}^{(\ell+1)}$ denote the row-normalized attention-weight matrix:

$$
\mathbf{A}_{t+1}^{(\ell+1)} =
\operatorname{softmax}_{\mathrm{row}}
\left(
\frac{
\mathbf{Q}_{t+1}^{(\ell+1)}
\left(\mathbf{K}_{t+1}^{(\ell+1)}\right)^\top
}{
\sqrt{d}
}
+
\mathbf{M}_{t+1}
\right),
$$

where $\mathbf{M}_{t+1}$ is the additive causal mask:

$$
(\mathbf{M}_{t+1})_{ij} =
\begin{cases}
0, & j \leq i,\\
-\infty, & j > i.
\end{cases}
$$

Substituting the block decompositions of $\mathbf{Q}_{t+1}^{(\ell+1)}$ and $\mathbf{K}_{t+1}^{(\ell+1)}$ gives:

$$
\begin{aligned}
\mathbf{A}_{t+1}^{(\ell+1)}
&=
\operatorname{softmax}_{\mathrm{row}}
\left(
\frac{1}{\sqrt{d}}
\begin{bmatrix}
\mathbf{Q}_t^{(\ell+1)}\\
\mathbf{q}_{t+1}^{(\ell+1)}
\end{bmatrix}
\begin{bmatrix}
\left(\mathbf{K}_t^{(\ell+1)}\right)^\top
&
\left(\mathbf{k}_{t+1}^{(\ell+1)}\right)^\top
\end{bmatrix}
+
\begin{bmatrix}
\mathbf{M}_t & -\infty\,\mathbf{1}_{N_t \times 1} \\
\mathbf{0}_{1 \times N_t} & 0
\end{bmatrix}
\right)
\\
&=
\operatorname{softmax}_{\mathrm{row}}
\left(
\begin{bmatrix}
\dfrac{
\mathbf{Q}_t^{(\ell+1)}
\left(\mathbf{K}_t^{(\ell+1)}\right)^\top
}{
\sqrt{d}
}
+
\mathbf{M}_t
&
-\infty\,\mathbf{1}_{N_t \times 1}
\\
\dfrac{
\mathbf{q}_{t+1}^{(\ell+1)}
\left(\mathbf{K}_t^{(\ell+1)}\right)^\top
}{
\sqrt{d}
}
&
\dfrac{
\mathbf{q}_{t+1}^{(\ell+1)}
\left(\mathbf{k}_{t+1}^{(\ell+1)}\right)^\top
}{
\sqrt{d}
}
\end{bmatrix}
\right) \\
&=
\begin{bmatrix}
\mathbf{A}_{t}^{(\ell+1)} & \mathbf{0}_{N_t \times 1} \\
\mathbf{a}_{t+1,\mathrm{past}}^{(\ell+1)} & a_{t+1,\mathrm{self}}^{(\ell+1)}
\end{bmatrix}
\end{aligned}.
$$

> [!important] Consequence of the Causal Mask
> For each of the first $N_t$ query positions, the appended key receives logit $-\infty$ and therefore attention weight zero. The remaining logits are identical to those at step $t$. Row-wise softmax consequently leaves every earlier attention row unchanged.

**Attention output.** Multiplying the attention weights by the value states gives:

$$
\begin{aligned}
\mathbf{O}_{t+1}^{(\ell+1)}
&=
\mathbf{A}_{t+1}^{(\ell+1)}
\mathbf{V}_{t+1}^{(\ell+1)}
\\
&=
\begin{bmatrix}
\mathbf{A}_t^{(\ell+1)}
&
\mathbf{0}_{N_t \times 1}
\\
\mathbf{a}_{t+1,\mathrm{past}}^{(\ell+1)}
&
a_{t+1,\mathrm{self}}^{(\ell+1)}
\end{bmatrix}
\begin{bmatrix}
\mathbf{V}_t^{(\ell+1)}
\\
\mathbf{v}_{t+1}^{(\ell+1)}
\end{bmatrix}
\\
&=
\begin{bmatrix}
\mathbf{A}_t^{(\ell+1)}
\mathbf{V}_t^{(\ell+1)}
\\
\mathbf{a}_{t+1,\mathrm{past}}^{(\ell+1)}
\mathbf{V}_t^{(\ell+1)}
+
a_{t+1,\mathrm{self}}^{(\ell+1)}
\mathbf{v}_{t+1}^{(\ell+1)}
\end{bmatrix}
\\
&=
\begin{bmatrix}
\mathbf{O}_t^{(\ell+1)}
\\
\mathbf{o}_{t+1}^{(\ell+1)}
\end{bmatrix}.
\end{aligned}
$$

In the simplified model, the attention output is also the block output. Therefore:

$$
\mathbf{X}_{t+1}^{(\ell+1)} =
\mathbf{O}_{t+1}^{(\ell+1)} =
\begin{bmatrix}
\mathbf{O}_t^{(\ell+1)}
\\
\mathbf{o}_{t+1}^{(\ell+1)}
\end{bmatrix} =
\begin{bmatrix}
\mathbf{X}_t^{(\ell+1)}
\\
\mathbf{x}_{t+1}^{(\ell+1)}
\end{bmatrix}.
$$

The first $N_t$ rows of $\mathbf{X}_{t+1}^{(\ell+1)}$ therefore equal $\mathbf{X}_{t}^{(\ell+1)}$. This establishes the inductive step and completes the proof for all layers.

### From Causal Masking to KV Reuse

The proposition identifies which computations are redundant. Consider the full model evaluation at step $t+1$:

$$
\begin{aligned}
\mathbf{Y}_{t+1} \in \mathcal{V}^{N_{t+1}}
&\xrightarrow{\operatorname{embedding}}
\mathbf{X}_{t+1}^{(0)}
\in \mathbb{R}^{N_{t+1} \times d}
\\
&\xrightarrow{L\ \text{transformer blocks}}
\mathbf{X}_{t+1}^{(L)}
\in \mathbb{R}^{N_{t+1} \times d}
\\
&\xrightarrow{\text{take last state}}
\mathbf{x}_{t+1}^{(L)}
\in \mathbb{R}^{1 \times d}
\\
&\xrightarrow{\text{output stage}}
y_{t+2}
\in \mathcal{V}.
\end{aligned}
$$

Only the final-position state $\mathbf{x}_{t+1}^{(L)}$ is required to predict $y_{t+2}$. Under the simplified attention-only representation:

$$
\begin{aligned}
\mathbf{x}_{t+1}^{(L)}
&= \mathbf{a}_{t+1}^{(L)} \mathbf{V}_{t+1}^{(L)} \\
&=
\operatorname{softmax}_{\mathrm{row}}
\left(
\frac{
\mathbf{q}_{t+1}^{(L)}
\left(\mathbf{K}_{t+1}^{(L)}\right)^\top
}{
\sqrt{d}
}
\right)
\mathbf{V}_{t+1}^{(L)}
\end{aligned}.
$$

Here, $\mathbf{a}_{t+1}^{(L)} \in \mathbb{R}^{1 \times N_{t+1}}$ is the final row of $\mathbf{A}_{t+1}^{(L)} \in \mathbb{R}^{N_{t+1} \times N_{t+1}}$.

Because $\mathbf{q}_{t+1}^{(L)}$ corresponds to the last position in the current sequence, every available key lies at or before that position. The final attention row therefore requires no explicit mask in this single-token expression.

This expression makes the attention dependencies explicit:

$$
\mathbf{x}_{t+1}^{(L)}
\xleftarrow{\text{attention}}
\mathbf{q}_{t+1}^{(L)},
\mathbf{K}_{t+1}^{(L)},
\mathbf{V}_{t+1}^{(L)}.
$$

The [result above](#causal-masking-leaves-earlier-token-states-unchanged) shows that $\mathbf{X}_{t+1}^{(\ell)}=\begin{bmatrix}\mathbf{X}_{t}^{(\ell)} \\ \mathbf{x}_{t+1}^{(\ell)} \end{bmatrix}$. Because the K/V projections act position-wise, the key and value states admit the same decomposition:

$$
\begin{aligned}
\mathbf{K}_{t+1}^{(\ell)}
&=
\mathbf{X}_{t+1}^{(\ell-1)} \mathbf{W}_K^{(\ell)} =
\begin{bmatrix}
\mathbf{X}_{t}^{(\ell-1)} \\
\mathbf{x}_{t+1}^{(\ell-1)}
\end{bmatrix}
\mathbf{W}_K^{(\ell)} =
\begin{bmatrix}
\mathbf{K}_t^{(\ell)} \\
\mathbf{k}_{t+1}^{(\ell)}
\end{bmatrix},
\\
\mathbf{V}_{t+1}^{(\ell)}
&=
\mathbf{X}_{t+1}^{(\ell-1)} \mathbf{W}_V^{(\ell)} =
\begin{bmatrix}
\mathbf{X}_{t}^{(\ell-1)} \\
\mathbf{x}_{t+1}^{(\ell-1)}
\end{bmatrix}
\mathbf{W}_V^{(\ell)} =
\begin{bmatrix}
\mathbf{V}_t^{(\ell)} \\
\mathbf{v}_{t+1}^{(\ell)}
\end{bmatrix},
\end{aligned}
\qquad
\forall \ell \in \{1, 2, \ldots, L\},
\quad
\forall t \in \{0, 1, \ldots\}.
$$

For $t=0$, $\mathbf{K}_{0}^{(\ell)}$ and $\mathbf{V}_{0}^{(\ell)}$ are produced during prefill. For $t\ge1$, $\mathbf{K}_{t}^{(\ell)}$ and $\mathbf{V}_{t}^{(\ell)}$ are available from the preceding decoding step. Since their rows for existing positions are mathematically unchanged, storing and reusing them is exact under the stated assumptions.

> [!note] What Is—and Is Not—Cached
> Standard KV caching stores the past key and value states at every layer. It does not store past queries or attention-weight matrices. A new query is required for the appended token, and its attention weights must be computed against the updated set of keys at each step.

At layer $L$, only the query, key, and value vectors for the appended token must be projected:

$$
\begin{aligned}
\mathbf{q}_{t+1}^{(L)} &= \mathbf{x}_{t+1}^{(L-1)} \mathbf{W}_Q^{(L)}, \\
\mathbf{k}_{t+1}^{(L)} &= \mathbf{x}_{t+1}^{(L-1)} \mathbf{W}_K^{(L)}, \\
\mathbf{v}_{t+1}^{(L)} &= \mathbf{x}_{t+1}^{(L-1)} \mathbf{W}_V^{(L)}.
\end{aligned}
$$

Computing these vectors requires only $\mathbf{x}_{t+1}^{(L-1)}$. The same dependency pattern holds at layer $L-1$:

$$
\mathbf{x}_{t+1}^{(L-1)}
\xleftarrow{\text{attention}}
\mathbf{q}_{t+1}^{(L-1)},
\mathbf{K}_{t+1}^{(L-1)},
\mathbf{V}_{t+1}^{(L-1)}.
$$

Thus, $\mathbf{K}_{t}^{(L-1)}$ and $\mathbf{V}_{t}^{(L-1)}$ can also be reused. Applying the argument recursively to every layer yields the standard per-layer KV cache.

## Incremental Decoding with KV Cache

### Cached Decoding Pipeline

The uncached full-sequence evaluation can now be replaced by an incremental computation that consumes only the newly appended token and the cache from the preceding step.

Assume that $t$ decoding steps have completed and that $\mathbf{K}_{t}^{(\ell)}$ and $\mathbf{V}_{t}^{(\ell)}$ are available for every $\ell \in \{1, 2, \ldots, L\}$. Given the newly appended token $y_{t+1}$, the next step is:

$$
\begin{aligned}
y_{t+1} \in \mathcal{V}
&\xrightarrow{\operatorname{embedding}}
\mathbf{x}_{t+1}^{(0)}
\in \mathbb{R}^{1 \times d}
\\
&\xrightarrow[
\mathbf{K}_{t}^{(1)},\,\mathbf{V}_{t}^{(1)}
]{
\substack{\text{Q/K/V projection}\\\text{append K/V to cache}}
}
\left(
\mathbf{q}_{t+1}^{(1)},
\mathbf{K}_{t+1}^{(1)},
\mathbf{V}_{t+1}^{(1)}
\right)
\xrightarrow{\text{attention}}
\mathbf{x}_{t+1}^{(1)}
\in \mathbb{R}^{1 \times d}
\\
&\xrightarrow[
\mathbf{K}_{t}^{(2)},\,\mathbf{V}_{t}^{(2)}
]{
\substack{\text{Q/K/V projection}\\\text{append K/V to cache}}
}
\left(
\mathbf{q}_{t+1}^{(2)},
\mathbf{K}_{t+1}^{(2)},
\mathbf{V}_{t+1}^{(2)}
\right)
\xrightarrow{\text{attention}}
\mathbf{x}_{t+1}^{(2)}
\in \mathbb{R}^{1 \times d}
\\
&\quad \cdots
\\
&\xrightarrow[
\mathbf{K}_{t}^{(L)},\,\mathbf{V}_{t}^{(L)}
]{
\substack{\text{Q/K/V projection}\\\text{append K/V to cache}}
}
\left(
\mathbf{q}_{t+1}^{(L)},
\mathbf{K}_{t+1}^{(L)},
\mathbf{V}_{t+1}^{(L)}
\right)
\xrightarrow{\text{attention}}
\mathbf{x}_{t+1}^{(L)}
\in \mathbb{R}^{1 \times d}
\\
&\xrightarrow{\text{output stage}}
y_{t+2}
\in \mathcal{V}.
\end{aligned}
$$

At each layer, the model projects the current hidden state into $\mathbf{q}_{t+1}^{(\ell)}$, $\mathbf{k}_{t+1}^{(\ell)}$, and $\mathbf{v}_{t+1}^{(\ell)}$, then updates the cache by concatenation:

$$
\mathbf{K}_{t+1}^{(\ell)} =
\begin{bmatrix}
\mathbf{K}_{t}^{(\ell)} \\
\mathbf{k}_{t+1}^{(\ell)}
\end{bmatrix},
\qquad
\mathbf{V}_{t+1}^{(\ell)} =
\begin{bmatrix}
\mathbf{V}_{t}^{(\ell)} \\
\mathbf{v}_{t+1}^{(\ell)}
\end{bmatrix}.
$$

The attention computation then uses the current query together with the updated key and value states. Only the resulting state for the current position is propagated to the next layer.

> [!note] Cache Initialization
> During prefill, every prompt token is processed in parallel, producing $\mathbf{K}_0^{(\ell)}, \mathbf{V}_0^{(\ell)} \in \mathbb{R}^{N_0 \times d}$ for each layer. Incremental decoding begins only after this one-time initialization.

### Storage and Computational Complexity

At step $t$, the cache contains $\mathbf{K}_{t}^{(\ell)}, \mathbf{V}_{t}^{(\ell)} \in \mathbb{R}^{N_t \times d}$ for every layer. Under the single-head assumption, this is exactly $2 L N_t d$ scalar elements. If each scalar occupies $b$ bytes, the cache requires $2bL N_t d$ bytes. Its storage therefore scales as $O(LN_t d)$ and grows linearly with context length. For standard multi-head attention, the same expression holds after substituting $d=h d_h$; architectures such as MQA and GQA reduce the total key/value width and therefore the cache size.

The following table compares one naïve full-sequence decoding step with one cached incremental step. It omits position-wise sublayers and the output projection because they do not change the asymptotic comparison:

| Computation | Naïve full recomputation | Cached incremental decoding |
| --- | --- | --- |
| Q/K/V projection | $O(L N_t d^2)$ | $O(L d^2)$ |
| Attention scores and weighted sum | $O(L N_t^2 d)$ | $O(L N_t d)$ |
| Total cost per decoding step | $O\left(L(N_t d^2 + N_t^2 d)\right)$ | $O\left(L(d^2 + N_t d)\right)$ |

> [!info] Interpretation
> Naïve decoding projects all $N_t$ positions and constructs a complete $N_t \times N_t$ attention computation at every step. Cached decoding projects only the appended token, but its query must still attend to $N_t$ keys and values. KV caching therefore reduces the projection term from linear to constant in $N_t$, and the attention term from quadratic to linear per generated token.
>
> These decoding costs exclude the one-time prefill computation. KV caching avoids repeated work after prefill; it does not reduce the cost of processing the prompt itself.

> [!note] Mathematical and Numerical Equivalence
> In exact arithmetic, cached and uncached decoding produce the same result under the stated assumptions. Real implementations may exhibit small floating-point differences because cached and full-sequence execution can select different kernels or accumulate operations in a different order.

## References

- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS, 2017. Defines the causal masking constraint used by the decoder.
- Shazeer. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150). 2019. Discusses key/value storage and memory bandwidth during incremental decoding.
- Hugging Face. [Caching](https://huggingface.co/docs/transformers/main/cache_explanation). Documents the `past_key_values` abstraction and per-layer cache update used in mainstream Transformer implementations.
- Athiwaratkun et al. [Bifurcated Attention for Single-Context Large-Batch Sampling](https://proceedings.mlr.press/v235/athiwaratkun24a.html). ICML, 2024. Separates prefill and decoding KV states for more efficient incremental attention.
