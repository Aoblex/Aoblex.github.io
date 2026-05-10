+++
title = 'Section 3: Training a Transformer LM'
date = '2026-05-09T21:27:10+08:00'
summary = 'Build the components we need to support training! 😌'
weight = 30
draft = false
+++

After building the tokenizer and the model, the remaining task is to build the code that supports training. This consists of three main components:

- **Loss**: define the objective function, usually cross-entropy.
- **Optimizer**: define the update rule used to minimize this loss, such as AdamW.
- **Training loop**: build the infrastructure that loads data, saves checkpoints, and manages training.

# Cross-entropy Loss

Recall that the Transformer model estimates **the distribution** of the next token given a context. Its final layer outputs a matrix $o \in \mathbb{R}^{n \times \text{vocab}}$, where the $i$-th row $o_i \in \mathbb{R}^{\text{vocab}}$ contains the next-token logits for the $i$-th input position.

Given a training dataset $\mathcal{D} = \\{ x \mid x \in \\{1, 2, \ldots, \text{vocab}\\}^{m+1} \\}$, where each $x$ is a token ID sequence, the cross-entropy loss is defined as:

$$
\begin{align*}
\ell(\theta; \mathcal{D}) &= \frac{1}{|\mathcal{D}| m} \sum_{x \in \mathcal{D}} \sum_{i=1}^{m} - \log p_{\theta}(x_{i+1} \mid x_{1:i}) \\
p_{\theta}(x_{i+1} \mid x_{1:i}) &= \operatorname{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{j=1}^{\text{vocab}} \exp(o_i[j])}
\end{align*}
$$

where $o_i \in \mathbb{R}^{\text{vocab}}$ is the $i$-th row of the Transformer output logits.

> [!note] Cross-entropy and KL divergence
> More formally, cross-entropy is defined for two probability distributions $p$ and $q$:
> $$
H(p, q) = \mathbb{E}_{x \sim p}\left[ -\log q(x) \right] = H(p) + D_{\text{KL}}(p \| q)
$$
> where
> $$
\begin{align*}
H(p) &= \mathbb{E}_{x \sim p} \left[- \log p(x)\right] \\
D_{\text{KL}}(p \| q) &= \mathbb{E}_{x \sim p} \left[\log \frac{p(x)}{q(x)}\right] \\
\end{align*}
$$
> The given definition $\ell(\theta; \mathcal{D})$ is an empirical version of $H(p, q)$, where $p$ is the true distribution of tokens and $q$ is the predicted distribution.
>
> Therefore, minimizing $\ell$ is equivalent to minimizing $D_{\text{KL}}$, since $H(p)$ is fixed with respect to the model parameters. In other words, the model is trying to learn a distribution that matches the data distribution as closely as possible.
>
> In information-theoretic terms, minimizing cross-entropy also corresponds to learning a probability model that gives a shorter expected code length for data from the true distribution.
