+++
title = 'Section 5: Generating Text'
date = '2026-07-01T14:51:15+08:00'
summary = 'Generate text from our model! 🤬'
weight = 50
draft = false
+++

First of all, let's make the notations clear:

- $n$: sequence length;
- $v$: the vocabulary size;
- $\mathbb{V} = \\{0, 1, 2, \ldots, v-1 \\}$: the set of token IDs;
- $\Delta^v = \\{ (x_0, x_1, \ldots, x_{v-1}) \in \mathbb{R}^v | x_i \geq 0, \sum x_i = 1 \\}$: probability simplex;
- $X \in \mathbb{V}^{n}$: input token IDs;
- $O \in \mathbb{R}^{n \times v}$: output logits;
- $P \in (\Delta^v)^{n}$: output probabilities;

---

# Text Generation Loop

In order to sample discrete IDs from $O$, we usually use $\text{softmax}$ to convert the logits to normalized probabilities. So the generation loop could be represented as:

$$
X \in \mathbb{V}^{n} \xrightarrow{\text{model}} O \in \mathbb{R}^{n \times v} \xrightarrow{\text{softmax}} P \in (\Delta^v)^{n} \xrightarrow{\text{decoder}} x_{n+1} \in \mathbb{V}
$$

When decoding, we only take the last probability distribution for next token prediction, which is effectively:

$$
P[n] \in \Delta^v \xrightarrow{\text{decoder}} x_{n+1} \in \mathbb{V}
$$

Here I use $P[n]$ informally to mean the probability distribution at the current last position. It is not meant to emphasize Python-style indexing. The important idea is that the decoder only needs the next-token distribution from the final position of the current context.

When we get this token $x_{n+1}$, we append it to the input $X$ and then generate the next token ID. This loop continues until we meet an ending condition: eos token or maximum length reached _etc._.

---

# Decoder

The core of text generation is the decoder. There are many options for this component.

## Simple Sampling

We can generate new tokens by repeatedly sampling from this conditioned distribution:

$$
P[n, i] = P(x_{n+1} = i \mid x_{1 \ldots n}) = \frac{\exp(o_i)}{\sum_j \exp(o_j)}
$$

In code, I take the logits at the final position, convert them to probabilities, and sample one token ID from this categorical distribution:

```python
def sample_next_token(logits: torch.Tensor) -> torch.Tensor:
    """Sample token ids directly from the softmax distribution."""
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (batch, vocab), got {tuple(logits.shape)}")

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

## Sampling with Temperature

We modify our $\text{softmax}$ with a temperature parameter $\tau$:

$$
P[n, i] = P_{\tau}(x_{n+1} = i \mid x_{1 \ldots n}) = \frac{\exp(o_i/\tau)}{\sum_j \exp(o_j/\tau)}
$$

When $\tau < 1$, the distribution becomes sharper and the model becomes more conservative. When $\tau > 1$, the distribution becomes flatter and sampling becomes more diverse. The implementation is just simple sampling after rescaling logits:

```python
def sample_next_token_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample token ids after scaling logits by temperature."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    return sample_next_token(logits / temperature)
```

## Sampling with top-$p$ (nucleus)

Another trick is to truncate low-probability tokens:

$$
P[n, i] = P_p(x_{n+1} = i \mid x_{1 \ldots n}) = 
\begin{cases}
\frac{\exp(o_i)}{\sum_{j \in T(p)} \exp(o_j)} & \text{if } i \in T(p) \\
0 & \text{otherwise}
\end{cases}
$$

where $T(p)$ is the smallest prefix of tokens, sorted by probability from high to low, such that $\sum_{j \in T(p)} P[n, j] \ge p$.

The implementation first sorts tokens by probability, keeps the smallest high-probability prefix, renormalizes the remaining probabilities, and finally maps the sampled sorted position back to the original token ID:

```python
def sample_next_token_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Sample token ids from the smallest high-probability set with cumulative mass at least p."""
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (batch, vocab), got {tuple(logits.shape)}")
    if not 0 < p <= 1:
        raise ValueError(f"p must be in (0, 1], got {p}")

    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = cumulative_probs > p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    filtered_probs = sorted_probs.masked_fill(remove_mask, 0.0)
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    sampled_sorted_positions = torch.multinomial(filtered_probs, num_samples=1)
    return sorted_indices.gather(dim=-1, index=sampled_sorted_positions).squeeze(-1)
```
