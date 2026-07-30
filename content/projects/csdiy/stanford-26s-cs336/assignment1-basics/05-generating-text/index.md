+++
title = 'Section 5: Generating Text'
date = '2026-07-01T14:51:15+08:00'
summary = 'Generate text from our model.'
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

The core of text generation is the decoder, which could be separated into two settings: next-token prediction (NTP) and multi-token prediction (MTP).

---

## NTP

In NTP, the model gives me one next-token distribution at the current last position. We sample one token, append it to the context, and then repeat the same process.

### Simple Sampling

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

### Sampling with Temperature

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

### Sampling with top-$p$ (nucleus)

Top-$p$ sampling, also known as nucleus sampling, was introduced by [Holtzman et al. (2020)](https://arxiv.org/abs/1904.09751) in *The Curious Case of Neural Text Degeneration*. The main idea is to truncate low-probability tokens:

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

---

## MTP

MTP is a natural extension of the same idea. Instead of training the model to predict only $x_{n+1}$ at each position, the training objective asks it to predict several future tokens, such as:

$$
x_{n+1}, x_{n+2}, \ldots, x_{n+k}
$$

### Modern MTP

[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) is the most direct reference for this idea in modern decoder-only LLMs. Its setup is simple: keep a shared transformer trunk, and attach several output heads on top of it. Head 1 predicts the ordinary next token, head 2 predicts the token after that, and so on. During inference, the extra heads can either be discarded, or they can be used to propose draft tokens for self-speculative decoding.

{{< figure
  src="figures/gloeckle-mtp-overview.png"
  alt="Multi-token prediction overview from Gloeckle et al."
  caption="Multi-token prediction overview from Gloeckle et al."
  width="55%"
  align="center"
>}}

This is the modern LLM formulation of MTP. There were earlier ideas with a similar flavor, such as future n-gram prediction in [ProphetNet](https://arxiv.org/abs/2001.04063), but Gloeckle et al. is the clean reference for the decoder-only LLM setting discussed here.

### DeepSeek-V3 MTP

[DeepSeek-V3](https://arxiv.org/abs/2412.19437) also uses MTP, but its implementation is not just a copy of the parallel-head version above. In the technical report, the authors say their MTP design is inspired by Gloeckle et al., but they predict the additional tokens sequentially and keep a complete causal chain at each prediction depth.

{{< figure
  src="figures/deepseek-v3-mtp.png"
  alt="DeepSeek-V3 multi-token prediction module"
  caption="DeepSeek-V3 multi-token prediction module."
  width="80%"
  align="center"
>}}

The main model still handles ordinary next-token prediction. Then each MTP module takes the previous depth representation and the embedding of a future token, combines them with a projection, runs a Transformer block, and predicts the next future token through the shared output head. DeepSeek-V3 sets the MTP depth to 1, so in practice each position predicts the next token plus one additional token.
