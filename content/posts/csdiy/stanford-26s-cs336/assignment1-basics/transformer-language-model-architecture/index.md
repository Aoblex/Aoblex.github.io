+++
title = 'Section 2: Transformer Language Model Architecture'
date = '2026-04-28T23:49:45+08:00'
summary = 'Build a Transformer model from scratch! 🥸'
weight = 20
draft = false
+++

# Introduction

In [the last section](../byte-pair-encoding/), we built a BPE tokenizer from scratch, so we can convert raw texts into token IDs that a language model can understand.

$$
\text{text: (batch,)} \xrightarrow{\text{BPE}} \text{input: (batch, seqlen)}
$$

In this section, we focus on building a **Transformer language model** to get output:

$$
\text{input: (batch, seqlen)} \xrightarrow{\text{model}} \text{output: (batch, seqlen, vocab)}
$$

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

```python
class Linear(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        weight = torch.empty(d_out, d_in, device=device, dtype=dtype)
        mean, std = 0.0, math.sqrt(2 / (d_in + d_out))
        nn.init.trunc_normal_(weight, mean=mean, std=std, a=-3 * std, b=3 * std)
        self.weight = torch.nn.Parameter(data=weight, requires_grad=True)

    def forward(
        self,
        x: Float[torch.Tensor, "... d_in"],
    ) -> Float[torch.Tensor, "... d_out"]:
        return x @ self.weight.T

```

Note that we do not include a bias term, following most modern LLMs.

---

## Embedding Module

Mathematical notation:

$$
y = \text{Embeddings[}x\text{]}
$$

It's actually an indexing operation.

```python
class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        weight = torch.empty(vocab_size, d_model, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

        self.weight = nn.Parameter(data=weight, requires_grad=True)

    def forward(
        self,
        token_ids: Int[torch.Tensor, "batch seqlen"],
    ) -> Float[torch.Tensor, "batch seqlen d_model"]:
        return einx.get_at("[vocab_size] d_model, batch seqlen -> batch seqlen d_model", self.weight, token_ids)

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

{{< figure
  src="figures/operator-proportions.png"
  alt="Proportions for operator classes in PyTorch"
  caption="Operator-class proportions reported in [Data Movement Is All You Need](https://arxiv.org/pdf/2007.00072)."
  width="60%"
  align="center"
>}}

One practical motivation for RMSNorm is that normalization is often much more expensive than its FLOP count suggests. In the table above, statistical normalization accounts for only $0.17\%$ of FLOPs, but takes $25.5\%$ of the PyTorch runtime. These operators are dominated by reductions and data movement, not arithmetic. RMSNorm removes the mean-centering step from LayerNorm, so it is simpler to compute and usually faster in practice. This is why modern LLM implementations commonly replace LayerNorm with RMSNorm: it can bring a noticeable speedup while leaving model quality almost unchanged.

```python
class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        weight = torch.ones(size=(d_model,), device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(data=weight, requires_grad=True)

    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]:
        in_type = x.dtype
        x = x.to(torch.float32)
        denom = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        numer = x * self.weight
        result = numer / denom
        return result.to(in_type)
```

---

## SwiGLU: Sigmoid Linear Unit + Gated Linear Unit

We will implement the position-wise feed-forward network in a Transformer block with a [**SwiGLU**](https://github.com/huggingface/transformers/blob/c6c7e189846ccaa3bb410569bfc6556d193c638d/src/transformers/models/qwen3/modeling_qwen3.py#L70) activation function, which combines [SiLU](https://arxiv.org/pdf/1702.03118) and [GLU](https://arxiv.org/abs/2002.05202):

$$
\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = \underbrace{W_2}_{\text{down proj}} (\underbrace{\text{SiLU}(W_1 x)}_{\text{gate branch}} \odot \underbrace{W_3 x}_{\text{value branch}})
$$

where $x \in \mathbb{R}^{d_{\text{model}}}$,
$W_1, W_3 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$,
$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$,
$\text{FFN}(x) \in \mathbb{R}^{d_{\text{model}}}$,
and canonically, $d_{\text{ff}} = \frac{8}{3} d_{\text{model}}$.

Some heuristic arguments from the writeup:

- GLUs are suggested to "reduce the vanishing gradient problem for deep architectures by providing **a linear path** for the gradients while retaining non-linear capabilities." To be specific:

    $$
    g_i = \sigma\left(\sum_j W_{g,ij}x_j\right), \quad
    v_i = \sum_j W_{v,ij}x_j, \quad
    y_i = g_i v_i
    $$

    For one input coordinate $x_k$, the scalar derivative is:

    $$
    \begin{aligned}
    \frac{\partial y_i}{\partial x_k}
    &=
    \underbrace{v_i \, \sigma'\left(\sum_j W_{g,ij}x_j\right) W_{g,ik}}_{\text{gate branch}}
    +
    \underbrace{g_i W_{v,ik}}_{\text{value branch}}
    \end{aligned}
    $$

    When $\sigma(x) = \frac{1}{1 + e^{-x}}$, we have $\sigma'(x) = \sigma(x)(1-\sigma(x)) \le 0.25$. Therefore, the gate branch is scaled by the derivative of the nonlinearity, while the value branch has a more direct linear path through $W_v$.

- It is fine to round the dimensions to a nearby multiple of 64 for hardware efficiency.

- Keep an empirical perspective: the exact dimension choice is ultimately an implementation and performance tradeoff.

- The model dimension ratio $\frac{8}{3} = 4 \times \frac{2}{3}$
    - 4: For $\text{FFN}(x)=\max(0, xW_1+b_1)W_2+b_2$, the best practice is $d_{\text{ff}} = 4 d_{\text{model}}$.
    - $\frac{2}{3}$: For GLU variants, the number of parameters scales down by $\frac{2}{3}$ to keep it the same as FFN.

```python
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.w1 = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_in=d_ff, d_out=d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]:
        values = self.w3(x)
        gates = self.silu(self.w1(x))
        return self.w2(values * gates)

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

in which $\theta_{i} = \theta^{-\frac{2i}{d}}$ and $\theta$ is a base constant for us to choose.
Then relative position information is encoded in the attention weights:

$$
(\mathcal{R}_m q)^\top (\mathcal{R}_n k)
= q^\top \mathcal{R}_m^\top \mathcal{R}_n k
= q^\top \mathcal{R}_{n-m} k
$$

```python
class RoPE(nn.Module):
    def __init__(
        self,
        d_k: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        assert d_k % 2 == 0, "d_k must be divisible by 2 in RoPE!"
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        pos = torch.arange(0, max_seq_len) # (max_seq_len,)
        dim = torch.arange(0, d_k // 2) # (d_k//2,)
        base_theta = theta ** (-(2.0 * dim) / d_k) # (d_k//2,)
        freqs = torch.outer(pos, base_theta).to(device=device, dtype=dtype) # (max_seq_len, d_k//2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seqlen d_k"],
        token_positions: Int[torch.Tensor, "... seqlen"],
    ) -> Float[torch.Tensor, "... seqlen d_k"]:
        seqlen, d_k = x.shape[-2], x.shape[-1]
        assert seqlen <= self.max_seq_len, f"Current seqlen={seqlen} > max_seq_len={self.max_seq_len}!"
        assert d_k == self.d_k, f"Input d_k={d_k} != RoPE d_k={self.d_k}"
        assert torch.max(token_positions) < self.max_seq_len, "Token position exceeded."

        # in case token_positions does not have the batch dimension
        assert token_positions.shape[-1] == seqlen
        token_positions = torch.broadcast_to(token_positions, x.shape[:-1])

        cos_freqs = einx.get_at("[position] half_dk, ... seqlen -> ... seqlen half_dk", self.cos, token_positions)
        sin_freqs = einx.get_at("[position] half_dk, ... seqlen -> ... seqlen half_dk", self.sin, token_positions)
        x_pairs = einx.id("... seqlen (half_dk two) -> ... seqlen half_dk two", x, two=2)
        rotation = einx.id(
            "... seqlen half_dk, ... seqlen half_dk, ... seqlen half_dk, ... seqlen half_dk -> ... seqlen half_dk (1 + 1) (1 + 1)",
            cos_freqs, -sin_freqs, sin_freqs, cos_freqs,
        )

        x_rotated = einx.dot(
            "... seqlen half_dk two_row [two_col], ... seqlen half_dk [two_col] -> ... seqlen (half_dk two_row)",
            rotation, x_pairs,
        )

        return x_rotated
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
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.softmax = Softmax()

    def forward(
        self,
        Q: Float[torch.Tensor, "... queries d_k"],
        K: Float[torch.Tensor, "... keys d_k"],
        V: Float[torch.Tensor, "... keys d_v"],
        mask: Bool[torch.Tensor, "... queries keys"] | None = None,
    ) -> Float[torch.Tensor, "... queries d_v"]:
        assert Q.shape[-1] == K.shape[-1], "Q, K last dim does not match!"
        d_k = Q.shape[-1]
        attention_scores = einx.dot("... queries d_k, ... keys d_k -> ... queries keys", Q, K)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, float("-inf"))
        return (self.softmax(attention_scores / math.sqrt(d_k))) @ V
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

```python
class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        assert d_model % num_heads == 0, "d_model not divisible by num_heads!"
        super().__init__()

        self.q_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.num_heads = num_heads
        self.sdpa = ScaledDotProductAttention()
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seqlen d_model"],
    ) -> Float[torch.Tensor, "... seqlen d_model"]:
        seqlen, d_model = x.shape[-2], x.shape[-1]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = einx.id("... seqlen (num_heads d_k) -> ... num_heads seqlen d_k", Q, num_heads=self.num_heads)
        K = einx.id("... seqlen (num_heads d_k) -> ... num_heads seqlen d_k", K, num_heads=self.num_heads)
        V = einx.id("... seqlen (num_heads d_v) -> ... num_heads seqlen d_v", V, num_heads=self.num_heads)

        causal_mask = self.mask[:seqlen, :seqlen]
        mha_result = einx.id("... num_heads seqlen d_v -> ... seqlen (num_heads d_v)", self.sdpa(Q, K, V, causal_mask))
        return self.output_proj(mha_result)
```

With RoPE, we apply the RoPE projections to **queries** and **keys**:

```python
class MultiheadSelfAttentionWithRoPE(MultiheadSelfAttention):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(d_model, num_heads, max_seq_len, device, dtype)
        self.rope = RoPE(
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... seqlen d_model"],
        token_positions: Int[torch.Tensor, "... seqlen"],
    ) -> Float[torch.Tensor, "... seqlen d_model"]:
        seqlen, d_model = x.shape[-2], x.shape[-1]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = einx.id("... seqlen (num_heads d_k) -> ... num_heads seqlen d_k", Q, num_heads=self.num_heads)
        K = einx.id("... seqlen (num_heads d_k) -> ... num_heads seqlen d_k", K, num_heads=self.num_heads)
        V = einx.id("... seqlen (num_heads d_v) -> ... num_heads seqlen d_v", V, num_heads=self.num_heads)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        causal_mask = self.mask[:seqlen, :seqlen]
        mha_result = einx.id("... num_heads seqlen d_v -> ... seqlen (num_heads d_v)", self.sdpa(Q, K, V, causal_mask))
        return self.output_proj(mha_result)
```

---

## Transformer Block

The structure of a Transformer block is shown [here](#transformer-lm).

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn = MultiheadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[torch.Tensor, "... seqlen d_model"],
        token_positions: Int[torch.Tensor, "... seqlen"],
    ) -> Float[torch.Tensor, "... seqlen d_model"]:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

```

---

## Transformer LM

Putting all these components together gives us the full Transformer model:

```python
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(
            vocab_size=vocab_size,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                theta=theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.lm_head = Linear(
            d_in=d_model,
            d_out=vocab_size,
            device=device,
            dtype=dtype,
        )

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

Here, we account for the compute and memory used by a Transformer forward pass.
Following the official writeup, we focus on **matrix multiplications**, since they dominate the FLOPs in a Transformer.

The core rule is:

$$
A \in \mathbb{R}^{m \times n},\quad B \in \mathbb{R}^{n \times p}
\quad\Longrightarrow\quad
AB \text{ costs } 2mnp \text{ FLOPs}.
$$

For now, we ignore the $\text{batch}$ dimension and assume that the input sequence has length $n$.

Notation:

- $n$: input sequence length;

- $d$: dimension of the model;

- $d_{\text{ff}}$: dimension of the FFN up projection;

- $h$: number of attention heads;

- $d^{\prime} = d / h$: dimension per attention head;

- $v$: vocabulary size;

- $l$: number of layers;

## Compute

Here we only count matrix multiplications and ignore other operations like embedding, RMSNorm, RoPE, softmax, masking, residual additions, _etc._.
These operations matter in real implementations, but their FLOP counts are usually smaller than the large matrix multiplications.

The matrix multiply FLOPs in one transformer forward is counted below (`batch_size=1`):

{{< collapse summary="Transformer: $8nd^2l + 4n^2dl + 6ndd_{\text{ff}}l + 2ndv$" >}}
{{< collapse summary="Transformer Block ($\times l$): $l \times (8nd^2 + 4n^2d + 6ndd_{\text{ff}})$" >}}
{{< collapse summary="Attention Block: $8nd^2 + 4n^2d$" >}}
{{< collapse summary="Q, K, V Projection: $6nd^2$" >}}
- Shape: $(n, d) \xrightarrow{(d, 3d)} (n, 3d) \rightarrow (3, n, d)$
- FLOPs: $6nd^2$
{{< /collapse >}}
{{< collapse summary="Attention Scores: $2n^2d$" >}}
- Shape: $(h, n, d^{\prime}) \xrightarrow{(h, d^{\prime}, n)} (h, n, n)$
- FLOPs: $h \times 2n^2d^{\prime} = 2n^2d$
{{< /collapse >}}
{{< collapse summary="Attention Weights: $2n^2d$" >}}
- Shape: $(h, n, n) \xrightarrow{(h, n, d^{\prime})} (h, n, d^{\prime}) \rightarrow (n, d)$
- FLOPs: $h \times 2n^2d^{\prime} = 2n^2d$
{{< /collapse >}}
{{< collapse summary="Output Projection: $2nd^2$" >}}
- Shape: $(n, d) \xrightarrow{(d, d)} (n, d)$
- FLOPs: $2nd^2$
{{< /collapse >}}
{{< /collapse >}}
{{< collapse summary="FFN Block: $6ndd_{\text{ff}}$" >}}
{{< collapse summary="Gate: $2ndd_{\text{ff}}$" >}}
- Shape: $(n, d) \xrightarrow{(d, d_{\text{ff}})} (n, d_{\text{ff}})$
- FLOPs: $2ndd_{\text{ff}}$
{{< /collapse >}}
{{< collapse summary="Value: $2ndd_{\text{ff}}$" >}}
- Shape: $(n, d) \xrightarrow{(d, d_{\text{ff}})} (n, d_{\text{ff}})$
- FLOPs: $2ndd_{\text{ff}}$
{{< /collapse >}}
{{< collapse summary="Output: $2ndd_{\text{ff}}$" >}}
- Shape: $(n, d_{\text{ff}}) \xrightarrow{(d_{\text{ff}}, d)} (n, d)$
- FLOPs: $2ndd_{\text{ff}}$
{{< /collapse >}}
{{< /collapse >}}
{{< /collapse >}}
{{< collapse summary="Output Embedding: $2ndv$" >}}
- Shape: $(n, d) \xrightarrow{(d, v)} (n, v)$
- FLOPs: $2ndv$
{{< /collapse >}}
{{< /collapse >}}

This formula makes the dominant terms easier to see.

## Memory

The number of trainable parameters of the given transformer architecture is counted below:

{{< collapse summary="Transformer: $4d^2l + 3dd_{\text{ff}}l + 2dl + 2dv + d$" >}}

{{< collapse summary="Input Embedding: $dv$" >}}
- Shape: $(d, v)$
- Parameters: $dv$
{{< /collapse >}}

{{< collapse summary="Transformer Block ($\times l$): $l \times (4d^2 + 3dd_{\text{ff}} + 2d)$" >}}
{{< collapse summary="Attention Block: $4d^2 + d$" >}}
{{< collapse summary="RMSNorm: $d$" >}}
- Shape: $(d,)$
- Parameters: $d$
{{< /collapse >}}
{{< collapse summary="Q: $d^2$" >}}
- Shape: $(h, d, d^{\prime})$
- Parameters: $d^2$
{{< /collapse >}}
{{< collapse summary="K: $d^2$" >}}
- Shape: $(h, d, d^{\prime})$
- Parameters: $d^2$
{{< /collapse >}}
{{< collapse summary="V: $d^2$" >}}
- Shape: $(h, d, d^{\prime})$
- Parameters: $d^2$
{{< /collapse >}}
{{< collapse summary="Output Projection: $d^2$" >}}
- Shape: $(d, d)$
- Parameters: $d^2$
{{< /collapse >}}
{{< /collapse >}}
{{< collapse summary="FFN Block: $3dd_{\text{ff}} + d$" >}}
{{< collapse summary="RMSNorm: $d$" >}}
- Shape: $(d,)$
- Parameters: $d$
{{< /collapse >}}
{{< collapse summary="Gate Projection: $dd_{\text{ff}}$" >}}
- Shape: $(d, d_{\text{ff}})$
- Parameters: $dd_{\text{ff}}$
{{< /collapse >}}
{{< collapse summary="Value Projection: $dd_{\text{ff}}$" >}}
- Shape: $(d, d_{\text{ff}})$
- Parameters: $dd_{\text{ff}}$
{{< /collapse >}}
{{< collapse summary="Output Projection: $dd_{\text{ff}}$" >}}
- Shape: $(d_{\text{ff}}, d)$
- FLOPs: $dd_{\text{ff}}$
{{< /collapse >}}
{{< /collapse >}}
{{< /collapse >}}

{{< collapse summary="Output Norm: $d$" >}}
- Shape: $(d,)$
- Parameters: $d$
{{< /collapse >}}

{{< collapse summary="Output Embedding: $dv$" >}}
- Shape: $(d, v)$
- FLOPs: $dv$
{{< /collapse >}}
{{< /collapse >}}

The memory usage can then be easily computed by multiplying the element size. $4$ for `float32` and 2 for `float16` or `bfloat16`.

Notice that this parameter count does not include activation memory, gradients, optimizer states, KV cache, temporary buffers, _etc._.
For example, a materialized causal mask uses $O(n^2)$ memory, and RoPE cosine/sine caches use $O(nd^{\prime})$ memory.
During training, activations and optimizer states usually dominate the additional memory beyond the parameters.

> [!code]- Resource accounting script
> ```python
> from rich.console import Console
> from rich.panel import Panel
> from rich.table import Table
>
>
> def humanize_number(x: int | float) -> str:
>     units = ["", "K", "M", "B", "T", "P", "E"]
>     x = float(x)
>     for unit in units:
>         if abs(x) < 1000:
>             return f"{x:,.2f}{unit}"
>         x /= 1000
>     return f"{x:,.2f}Z"
>
>
> def humanize_bytes(num_bytes: int) -> str:
>     units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
>     x = float(num_bytes)
>     for unit in units:
>         if abs(x) < 1024:
>             return f"{x:,.2f} {unit}"
>         x /= 1024
>     return f"{x:,.2f} EiB"
>
>
> def compute_resource(
>     n: int,
>     d: int,
>     d_ff: int,
>     h: int,
>     v: int,
>     l: int,
>     *,
>     bytes_per_param: int = 4,
>     tied_embeddings: bool = False,
>     save_svg_path: str | None = None,
> ) -> dict[str, int]:
>     if d % h != 0:
>         raise ValueError(f"d={d} must be divisible by h={h}.")
>
>     d_head = d // h
>     console = Console(record=save_svg_path is not None)
>
>     param_counts = {
>         "Token embedding": v * d,
>         "Attention projections": l * 4 * d**2,
>         "SwiGLU FFN": l * 3 * d * d_ff,
>         "RMSNorm": l * 2 * d + d,
>         "LM head": 0 if tied_embeddings else v * d,
>     }
>     total_params = sum(param_counts.values())
>     total_memory = total_params * bytes_per_param
>
>     flops = {
>         "Attention projections": l * 8 * n * d**2,
>         "Attention scores": l * 2 * n**2 * d,
>         "Attention value mix": l * 2 * n**2 * d,
>         "SwiGLU FFN": l * 6 * n * d * d_ff,
>         "LM head": 2 * n * d * v,
>     }
>     total_flops = sum(flops.values())
>
>     console.print(
>         Panel.fit(
>             "\n".join([
>                 f"[bold]n[/bold]={n:,}, [bold]d[/bold]={d:,}, "
>                 f"[bold]d_ff[/bold]={d_ff:,}, [bold]h[/bold]={h:,}, "
>                 f"[bold]d_head[/bold]={d_head:,}",
>                 f"[bold]v[/bold]={v:,}, [bold]l[/bold]={l:,}, "
>                 f"[bold]dtype bytes[/bold]={bytes_per_param}",
>             ]),
>             title="Transformer Resource Accounting",
>             border_style="cyan",
>         )
>     )
>
>     param_table = Table(title="Parameter Memory", show_lines=True)
>     param_table.add_column("Component", style="bold")
>     param_table.add_column("Parameters", justify="right")
>     param_table.add_column("Memory", justify="right")
>     param_table.add_column("Share", justify="right")
>     for name, count in param_counts.items():
>         param_table.add_row(
>             name,
>             f"{count:,} ({humanize_number(count)})",
>             humanize_bytes(count * bytes_per_param),
>             f"{count / total_params:.2%}",
>         )
>     param_table.add_section()
>     param_table.add_row(
>         "[bold]Total[/bold]",
>         f"[bold]{total_params:,} ({humanize_number(total_params)})[/bold]",
>         f"[bold]{humanize_bytes(total_memory)}[/bold]",
>         "[bold]100.00%[/bold]",
>     )
>     console.print(param_table)
>
>     flops_table = Table(title="Forward FLOPs", show_lines=True)
>     flops_table.add_column("Component", style="bold")
>     flops_table.add_column("FLOPs", justify="right")
>     flops_table.add_column("Share", justify="right")
>     for name, count in flops.items():
>         flops_table.add_row(
>             name,
>             f"{count:,} ({humanize_number(count)})",
>             f"{count / total_flops:.2%}",
>         )
>     flops_table.add_section()
>     flops_table.add_row(
>         "[bold]Total[/bold]",
>         f"[bold]{total_flops:,} ({humanize_number(total_flops)})[/bold]",
>         "[bold]100.00%[/bold]",
>     )
>     console.print(flops_table)
>
>     if save_svg_path is not None:
>         console.save_svg(save_svg_path, title="Transformer Resource Accounting")
>
>     return {
>         "total_params": total_params,
>         "total_memory_bytes": total_memory,
>         "total_flops": total_flops,
>         **{f"params/{k}": v for k, v in param_counts.items()},
>         **{f"flops/{k}": v for k, v in flops.items()},
>     }
>
>
> for n in [1024, 16384]:
>     compute_resource(
>         n=n,
>         d=1600,
>         d_ff=4288,
>         h=25,
>         v=50257,
>         l=48,
>         save_svg_path=f"figures/transformer_resource_accounting_n{n}.svg",
>     )
> ```

{{< gallery cols="2" gap="16px" align="start" >}}
{{< gallery-item src="figures/transformer_resource_accounting_n1024.svg" alt="Transformer resource accounting with context length 1024" caption="Context length: $n=1024$" >}}
{{< gallery-item src="figures/transformer_resource_accounting_n16384.svg" alt="Transformer resource accounting with context length 16384" caption="Context length: $n=16384$" >}}
{{< /gallery >}}

# Solutions

Here are my solutions to the problems given in the writeup.

> [!note]- Problem (transformer_accounting): Transformer LM resource accounting (5 points)
> > [!question]- Consider a GPT-2 XL-sized model using our assignment architecture, which has the following configuration:
> > ```txt
> > vocab_size: 50,257
> > context_length: 1,024
> > num_layers: 48
> > d_model: 1,600
> > num_heads: 25
> > d_ff: 4,288 (the nearest multiple of 64 to 8/3x1,600)
> > ```
> > Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?
> >
> > **Answer**: Under the assignment architecture, the parameter count is:
> >
> > | Component | Parameters |
> > | --- | ---: |
> > | Token embedding | $vd = 50{,}257 \times 1{,}600 = 80{,}411{,}200$ |
> > | Attention projections | $l \cdot 4d^2 = 48 \cdot 4 \cdot 1{,}600^2 = 491{,}520{,}000$ |
> > | SwiGLU FFN | $l \cdot 3dd_{\text{ff}} = 48 \cdot 3 \cdot 1{,}600 \cdot 4{,}288 = 987{,}955{,}200$ |
> > | RMSNorm | $l \cdot 2d + d = 155{,}200$ |
> > | LM head | $dv = 1{,}600 \times 50{,}257 = 80{,}411{,}200$ |
> > | **Total** | **$1{,}640{,}452{,}800$** |
> >
> > So the model has about **1.64B trainable parameters**.
> > With single-precision floating point, each parameter takes $4$ bytes, so loading just the parameters requires $1{,}640{,}452{,}800 \times 4 = 6{,}561{,}811{,}200$ bytes, or about **6.11 GiB**.
> >
>
> > [!question]- Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has `context_length` tokens.
> >
> > **Answer**: The matrix multiplies are:
> >
> > 1. Attention projections: $Q, K, V, O$, costing $8nd^2$ FLOPs per layer.
> > 2. Attention scores: $QK^\top$, costing $2n^2d$ FLOPs per layer.
> > 3. Attention weighted value sum: $AV$, costing $2n^2d$ FLOPs per layer.
> > 4. SwiGLU FFN: gate, up, and down projections, costing $6ndd_{\text{ff}}$ FLOPs per layer.
> > 5. LM head: final vocabulary projection, costing $2ndv$ FLOPs once.
> >
> > For GPT-2 XL-shaped configuration with $n=1024$, $d=1600$, $d_{\text{ff}}=4288$, $v=50257$, and $l=48$:
> >
> > | Component | FLOPs | Share |
> > | --- | ---: | ---: |
> > | Attention projections | $1.0066 \times 10^{12}$ | 28.62% |
> > | Attention scores | $1.6106 \times 10^{11}$ | 4.58% |
> > | Attention value mix | $1.6106 \times 10^{11}$ | 4.58% |
> > | SwiGLU FFN | $2.0233 \times 10^{12}$ | 57.53% |
> > | LM head | $1.6468 \times 10^{11}$ | 4.68% |
> > | **Total** | **$3.5168 \times 10^{12}$** | **100.00%** |
> >
> > So one forward pass costs about **3.52T FLOPs** for a single sequence of length $1024$.
>
> > [!question]- Based on your analysis above, which parts of the model require the most FLOPs?
> >
> > **Answer**: At context length $1024$, the dominant cost is the **SwiGLU FFN**, which accounts for about **57.53%** of the forward-pass FLOPs.
> > The next largest cost is the attention projection matrices, about **28.62%**.
> >
> > The explicitly quadratic attention terms, $QK^\top$ and $AV$, are only about **9.16%** together at $n=1024$.
> > This is because $n$ is still smaller than $d$ here, and the projection/FFN terms scale strongly with model width.
>
> > [!question]- Repeat your analysis with GPT-2 small (12 layers, 768 `d_model`, 12 heads), GPT-2 medium (24 layers, 1024 `d_model`, 16 heads), and GPT-2 large (36 layers, 1280 `d_model`, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?
> >
> > **Answer**: I use the same assignment convention $d_{\text{ff}} \approx \frac{8}{3}d$, rounded to a nearby multiple of $64$:
> > small uses $d_{\text{ff}}=2048$, medium uses $d_{\text{ff}}=2752$, and large uses $d_{\text{ff}}=3456$.
> >
> > | Model | Total FLOPs | Attention projections | Attention scores | Attention value mix | FFN | LM head |
> > | --- | ---: | ---: | ---: | ---: | ---: | ---: |
> > | Small | $2.9165 \times 10^{11}$ | 19.88% | 6.63% | 6.63% | 39.76% | 27.10% |
> > | Medium | $8.3017 \times 10^{11}$ | 24.83% | 6.21% | 6.21% | 50.05% | 12.70% |
> > | Large | $1.7867 \times 10^{12}$ | 27.04% | 5.41% | 5.41% | 54.76% | 7.37% |
> > | XL | $3.5168 \times 10^{12}$ | 28.62% | 4.58% | 4.58% | 57.53% | 4.68% |
> >
> > As the model gets wider and deeper while the context length stays fixed at $1024$, the **FFN and attention projection terms take up more of the total FLOPs**.
> > The **LM head becomes proportionally smaller**, because it scales like $O(ndv)$ while the per-layer projection and FFN costs scale like $O(lnd^2)$ and $O(lndd_{\text{ff}})$.
> > The quadratic sequence terms also become proportionally smaller because $n$ is fixed while $d$ and $l$ increase.
>
> > [!question]- Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?
> >
> > **Answer**: With the same GPT-2 XL-shaped model but $n=16{,}384$, the total forward-pass FLOPs become $1.3358 \times 10^{14}$ FLOPs, or about **133.58T FLOPs**.
> > This is about **38.0x** larger than the $n=1024$ case.
> >
> > | Component | FLOPs | Share |
> > | --- | ---: | ---: |
> > | Attention projections | $1.6106 \times 10^{13}$ | 12.06% |
> > | Attention scores | $4.1232 \times 10^{13}$ | 30.87% |
> > | Attention value mix | $4.1232 \times 10^{13}$ | 30.87% |
> > | SwiGLU FFN | $3.2373 \times 10^{13}$ | 24.24% |
> > | LM head | $2.6349 \times 10^{12}$ | 1.97% |
> > | **Total** | **$1.3358 \times 10^{14}$** | **100.00%** |
> >
> > The main change is that the quadratic attention terms dominate.
> > At $n=1024$, $QK^\top$ and $AV$ together were only about **9.16%** of the FLOPs.
> > At $n=16{,}384$, they become about **61.74%** of the FLOPs.
> > This is exactly the expected effect of the $4n^2d$ term in attention.
