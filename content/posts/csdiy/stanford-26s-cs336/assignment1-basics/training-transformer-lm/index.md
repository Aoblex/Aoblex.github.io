+++
title = 'Section 3: Training a Transformer LM'
date = '2026-05-09T21:27:10+08:00'
summary = 'Build components that we need to support training! 😌'
weight = 30
draft = false
+++

After building the tokenizer and the model, what remains is to build all of the code to support training. This consists of the following:
- **Loss**: we need to define the loss function (cross-entropy).
- **Optimizer**: we need to define the optimizer to minimize this loss (AdamW).
- **Training loop**: we need all the supporting infrastructure that loads data, saves checkpoints, and manages training.

# Cross-entropy Loss

Recall that the Transformer model is modeling **the distribution** of the next token given a context. Its final layer outputs a $o \in \mathbb{R}^{n \times \text{vocab}}$ matrix where the $i$-th row $o_i \in \mathbb{R}^{\text{vocab}}$ represents the next token logits of the $i$-th input token.

Given a training dataset $\mathcal{D} = \\{ x | x \in \\{1, 2, \ldots, \text{vocab}\\}^{m+1} \\}$ ($x$ is a token ID sequence), the cross-entropy loss is defined as:

$$
\begin{align*}
\ell(\theta; \mathcal{D}) &= \frac{1}{|\mathcal{D}| m} \sum_{x \in \mathcal{D}} \sum_{i=1}^{m} - \log p_{\theta}(x_{i+1} \mid x_{1:i}) \\
p_{\theta}(x_{i+1} \mid x_{1:i}) &= \operatorname{softmax}(o_i)[x_{i+1}] = \frac{\exp{o_i[x_{i+1}]}}{\sum_{j=1}^{\text{vocab}} \exp o_i[j]}
\end{align*}
$$

where $o_i \in \mathbb{R}^{\text{vocab}}$ is the $i$-th row of the Transformer output logits.

> [!note] A more formal definition
> More formally, the cross-entropy is defined on two probability distributions $p$ and $q$:
> $$
H(p, q) = \mathbb{E}_{x \sim p}\left[ -\log q(x) \right] = H(p) + D_{\text{KL}}(p \| q)
$$
> where
> $$
\begin{align*}
H(p) &= \mathbb{E}_{x \sim p} \left[- \log p(x)\right] \\
D_{\text{KL}}(p \| q) &= \mathbb{E}_{x \sim p} \left[- \log \frac{p(x)}{q(x)}\right] \\
\end{align*}
$$
> The given definition $\ell(\theta; \mathcal{D})$ is an empirical version of $H(p, q)$, where $p$ is the true distribution of tokens and $q$ is the predicted distribution.
>
> Therefore, minimizing $\ell$ is equivalent to minimizing $D_{\text{KL}}$, since we consider $H(p)$ as a constant, _i.e._, the model is trying to learn from the data an optimal compression of the language.