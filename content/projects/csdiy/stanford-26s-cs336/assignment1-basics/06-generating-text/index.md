+++
title = 'Section 6: Generating Text'
date = '2026-07-01T14:51:15+08:00'
summary = 'Turn next-token logits into autoregressive text with temperature and nucleus sampling.'
weight = 60
draft = false
+++

# Introduction

Training evaluates all next-token predictions in parallel, but generation reveals tokens sequentially. At each step, the model produces logits for the current context, a decoding rule turns the final-position logits into a sampling distribution, and one sampled token is appended to the sequence.

The model determines which continuations are plausible; the decoder determines how probability mass is distributed among those possibilities. Temperature and top-$p$ sampling adjust this distribution without changing the model itself.

---

# Autoregressive Generation

Given a prompt $x_{1:t}$, the model returns one logit vector for every position. Only the final vector is needed to generate the next token:

$$
z_t=\operatorname{TransformerLM}(x_{1:t})_t\in\mathbb{R}^{v}.
$$

A sampling rule converts $z_t$ into a distribution $q_t$ and draws

$$
x_{t+1}\sim q_t.
$$

Generation then repeats the same recurrence:

1. Run the model on the available context.
2. Select the logits at the final position.
3. Transform the logits according to the decoding parameters.
4. Sample one token and append it to the sequence.
5. Stop after producing EOS or reaching the generation limit.

When the sequence exceeds the model's context length, only the most recent context window is passed to the model. Without a KV cache, this implementation recomputes the hidden states for that window at every step; the algorithm is simple, but repeated prefixes make decoding much more expensive than one parallel training pass.

---

# Sampling

## Temperature

Temperature rescales logits before softmax:

$$
q_i(\tau)=\frac{\exp(z_i/\tau-c)}{\sum_{j=1}^{v}\exp(z_j/\tau-c)},
\qquad
c=\max_j\frac{z_j}{\tau},
$$

where $\tau\gt0$. The transformation preserves the ranking of tokens while changing the concentration of the distribution:

- $\tau\lt1$ sharpens the distribution and favors high-probability tokens;
- $\tau=1$ recovers the original softmax distribution;
- $\tau\gt1$ flattens the distribution and increases diversity;
- as $\tau\to0$, sampling approaches greedy decoding.

Temperature changes relative probabilities across the entire vocabulary, including the low-probability tail. This can increase diversity, but it does not prevent implausible tail tokens from being sampled.

## Top-$p$ Sampling

[Top-$p$ sampling](https://arxiv.org/abs/1904.09751), or nucleus sampling, restricts sampling to the smallest high-probability set whose cumulative mass reaches a threshold $p$.

Sort the temperature-scaled probabilities so that

$$
q_{(1)}\ge q_{(2)}\ge\cdots\ge q_{(v)},
$$

and define

$$
k^*=\min\left\{k:\sum_{r=1}^{k}q_{(r)}\ge p\right\},
\qquad
S_p=\{(1),\ldots,(k^*)\}.
$$

The final sampling distribution is

$$
\widetilde q_i=
\begin{cases}
\dfrac{q_i}{\sum_{j\in S_p}q_j},&i\in S_p,\\[8pt]
0,&i\notin S_p.
\end{cases}
$$

Unlike top-$k$, the number of retained tokens adapts to the model's uncertainty. A confident prediction produces a small nucleus; a diffuse prediction retains more alternatives. The token that crosses the threshold must remain in the set, otherwise its cumulative mass can fall below $p$.

Temperature and top-$p$ address different aspects of sampling and can be applied in sequence: temperature reshapes the distribution, then top-$p$ removes its low-probability tail.

---

# Generation Interface

Generation uses the same experiment file as training. It reconstructs the model and tokenizer from checkpoint metadata, then reads the prompts, stopping budget, and sampling settings from the `generation` section.

> [!example]- Validate and run generation
> ```bash
> uv run python scripts/generate.py configs/tinystories-smoke.yaml --check
> make generate CONFIG=configs/tinystories-smoke.yaml
> ```

The generated text and the state needed to interpret it are stored together:

```text
outputs/<experiment>/generation/
├── config.yaml
├── metadata.json
└── samples.md
```

The metadata records the checkpoint step, prompt, random seed, device, token budget, temperature, and top-$p$. This makes stochastic samples attributable to a specific model and decoding configuration.

The smoke model exercises the complete generation path, but its output is not a meaningful quality result. The requested 256-token sample should come from the trained baseline model in [Section 7](../07-experiments/#tinystories).

---

# Multi-Token Prediction

The assignment uses ordinary next-token prediction: every position learns to predict one future token, and inference samples one new token per model call. Multi-token prediction (MTP) extends the training objective so that a representation predicts several future tokens.

## Parallel Prediction Heads

[Gloeckle et al.](https://arxiv.org/abs/2404.19737) attach several prediction heads to a shared Transformer trunk. Head $r$ predicts the token $r$ positions ahead, so a single representation is supervised by multiple future targets.

{{< figure
  src="figures/gloeckle-mtp-overview.png"
  alt="Multi-token prediction overview from Gloeckle et al."
  caption="Several prediction heads share one Transformer trunk and target different future offsets."
  width="55%"
  align="center"
>}}

The auxiliary heads can be discarded at inference, leaving the ordinary next-token model, or used to propose draft tokens for self-speculative decoding. Earlier work such as [ProphetNet](https://arxiv.org/abs/2001.04063) applies a related future n-gram objective, while Gloeckle et al. formulate the idea directly for decoder-only language models.

## DeepSeek-V3

[DeepSeek-V3](https://arxiv.org/abs/2412.19437) predicts additional tokens through sequential MTP modules rather than independent parallel heads. Each depth combines the previous representation with the embedding of a future token, applies another Transformer block, and predicts the next future target through the shared output head.

{{< figure
  src="figures/deepseek-v3-mtp.png"
  alt="DeepSeek-V3 multi-token prediction module"
  caption="DeepSeek-V3 preserves a causal chain across successive prediction depths."
  width="80%"
  align="center"
>}}

DeepSeek-V3 uses one additional prediction depth: the main model predicts the next token, while the MTP module predicts one token further ahead. This auxiliary objective improves representations during training and can support speculative generation, but it is separate from the decoding implementation required by this assignment.

---

# Solutions

> [!note]- Problem (`decoding`): Decoding (3 points)
> > [!question]- Deliverable: Implement autoregressive text generation
> > Generate a completion for a user-provided prompt until EOS or a configurable maximum number of new tokens. Support temperature scaling and top-$p$ sampling with user-provided values.
> >
> > **Answer**: Encode the prompt, repeatedly evaluate the final-position logits, apply temperature and nucleus filtering, sample the next token, and append it to the context. Stop when EOS is sampled or the token budget is exhausted, then decode the complete token sequence back to text.
