+++
title = 'Transformer Language Model Architecture'
date = '2026-04-28T23:49:45+08:00'
summary = 'Build a Transformer model from scratch! 🥸'
weight = 20
draft = false
+++

A language model takes as input a **batched sequence of integer token IDs** and returns a **(batched) normalized probability distribution** over a fixed vocabulary:

$$
\text{input (batch, seqlen)} \xrightarrow{\text{model}} \text{output (batch, seqlen, vocab)}
$$

Formally, the input and output spaces can be described as:

$$
\text{input} \in \mathbb{Z}^{\text{batch} \times \text{seqlen}}, \quad
\text{output} \in \mathbb{R}^{\text{batch} \times \text{seqlen} \times \text{vocab}}
$$

where:
- Each element in the input tensor is an integer token ID representing a token in the vocabulary.
- The output tensor contains, for each position in each sequence, a probability distribution over all vocabulary tokens.

For every fixed $(b, t)$ (batch index and time step), the model produces a vector:

$$
\text{output}[b, t, :] \in \mathbb{R}^{\text{vocab}}
$$

which satisfies:
- Non-negativity: $\text{output}[b, t, i] \ge 0$
- Normalization: $\sum_{i=1}^{\text{vocab}} \text{output}[b, t, i] = 1$

In practice, this distribution is typically obtained by applying a softmax function to the model’s logits at each sequence position.
