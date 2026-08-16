+++
title = 'Section 4: Training a Transformer LM'
date = '2026-05-09T21:27:10+08:00'
summary = 'Define the objective and optimization mechanisms used to train a Transformer language model.'
weight = 40
draft = false
+++

# Introduction

Training turns next-token prediction into parameter updates. The model produces logits, cross-entropy measures how well they predict the following tokens, and AdamW uses the resulting gradients to update the model. A learning-rate schedule controls the update scale over time, while gradient clipping limits occasional unstable steps.

This section develops those components and accounts for the memory and computation required by one training step.

---

# Cross-Entropy Loss

For a sequence $x_1,\ldots,x_{m+1}$, the model produces logits $o_i\in\mathbb{R}^{v}$ at each position $i$. The probability assigned to the next token is

$$
p_\theta(x_{i+1}\mid x_{1:i})=\operatorname{softmax}(o_i)_{x_{i+1}}.
$$

Training minimizes the average negative log-likelihood over tokens and examples:

$$
\ell(\theta;\mathcal D)=-\frac{1}{|\mathcal D|m}
\sum_{x\in\mathcal D}\sum_{i=1}^{m}
\log p_\theta(x_{i+1}\mid x_{1:i}).
$$

For a single logit vector $o$ and target $y$, the same loss can be written without explicitly forming probabilities:

$$
\ell(o,y)=\log\sum_{j=1}^{v}\exp(o_j)-o_y.
$$

Subtracting $c=\max_j o_j$ from every logit gives the numerically stable form

$$
\ell(o,y)=\log\sum_{j=1}^{v}\exp(o_j-c)-(o_y-c).
$$

The shift changes neither the softmax distribution nor the loss, but prevents large logits from overflowing during exponentiation.

> [!note]- Cross-entropy and KL divergence
> Cross-entropy decomposes as $H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q)$. The data distribution $p$ is fixed during training, so minimizing cross-entropy with respect to the model distribution $q$ is equivalent to minimizing $D_{\mathrm{KL}}(p\|q)$.

> [!note]- Perplexity
> If the mean token-level cross-entropy is $\bar\ell$, then $\operatorname{PPL}=\exp(\bar\ell)$. Perplexity is the effective number of equally plausible next-token choices under the model. It equals $1$ for perfect prediction and $v$ for a uniform distribution over a vocabulary of size $v$.

---

# Optimization

An optimizer converts the gradient on a sampled batch $B_t$ into a parameter update. The basic form is

$$
\theta_{t+1}=\theta_t-\alpha_t u_t,
$$

where $u_t$ is the update direction and $\alpha_t$ controls its scale.

## Stochastic Gradient Descent

SGD uses the current batch gradient directly. The assignment's example additionally decays the step size as $1/\sqrt{t+1}$:

$$
g_t=\nabla_\theta\ell(\theta_t;B_t),
\qquad
\theta_{t+1}=\theta_t-\frac{\alpha}{\sqrt{t+1}}g_t.
$$

The learning rate determines how far the optimizer moves along the gradient direction. A small value makes reliable but slow progress; a larger value can converge faster until the update begins to overshoot and destabilize optimization.

Momentum replaces the instantaneous direction with an exponentially weighted history,

$$
v_t=\mu v_{t-1}+g_t,
\qquad
\theta_{t+1}=\theta_t-\alpha v_t,
$$

which suppresses oscillation while reinforcing directions that remain consistent across batches.

> [!example]- Representative applications
> SGD with momentum was central to the training of influential convolutional networks such as [VGG](https://arxiv.org/abs/1409.1556) and [ResNet](https://arxiv.org/abs/1512.03385). These models pair momentum with scheduled learning-rate reductions to optimize deep vision architectures.

## [Adagrad](https://jmlr.org/papers/v12/duchi11a.html)

Adagrad accumulates squared gradients separately for every parameter:

$$
r_t=r_{t-1}+g_t^2,
\qquad
\theta_{t+1}=\theta_t-\frac{\alpha}{\sqrt{r_t}+\varepsilon}g_t.
$$

Coordinates with frequent or large gradients receive smaller effective learning rates, making the method useful for sparse features. Because $r_t$ only grows, however, its learning rates can eventually become too small.

> [!example]- Representative application
> [GloVe](https://aclanthology.org/D14-1162/) uses Adagrad to learn word vectors from a sparse word co-occurrence objective, where different parameters are updated at very different frequencies.

## [RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

RMSProp replaces Adagrad's cumulative history with an exponential moving average:

$$
v_t=\beta v_{t-1}+(1-\beta)g_t^2,
\qquad
\theta_{t+1}=\theta_t-\frac{\alpha}{\sqrt{v_t}+\varepsilon}g_t.
$$

Forgetting distant gradients prevents the denominator from growing indefinitely, while retaining coordinate-wise scaling. RMSProp rescales the current gradient but does not maintain a first-moment estimate of its direction.

> [!example]- Representative applications
> RMSProp was used in the [DQN](https://www.nature.com/articles/nature14236) work on Atari and in early attention-based neural machine translation, including [Luong et al.](https://arxiv.org/abs/1508.04025).

## [Adam](https://arxiv.org/abs/1412.6980)

Adam combines momentum with RMSProp-style adaptive scaling:

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\end{aligned}
$$

The two moments provide a smoothed direction and a coordinate-wise scale. Bias correction is needed because both moving averages begin at zero.

> [!example]- Representative applications
> Adam underlies several foundational Transformer systems, including the original [Transformer](https://arxiv.org/abs/1706.03762), [BERT](https://arxiv.org/abs/1810.04805), [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [GPT-3](https://arxiv.org/abs/2005.14165), and [T5](https://arxiv.org/abs/1910.10683).

## [AdamW](https://arxiv.org/abs/1711.05101)

AdamW replaces the raw gradient with normalized first- and second-moment estimates:

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\\
\widehat m_t&=\frac{m_t}{1-\beta_1^t},
&
\widehat v_t&=\frac{v_t}{1-\beta_2^t}.
\end{aligned}
$$

The first moment smooths the update direction, while the second moment rescales coordinates according to their recent gradient magnitude. Bias correction compensates for initializing both estimates at zero.

AdamW applies weight decay directly to the parameters rather than mixing it into the gradient moments:

$$
\theta_t=(1-\alpha\lambda)\theta_{t-1}
-\alpha\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}.
$$

This separation preserves a uniform multiplicative shrinkage while leaving the adaptive gradient estimate unchanged. The trade-off is memory: every parameter retains both $m_t$ and $v_t$ throughout training.

{{< figure
  src="figures/AdamW-pseudo.png"
  alt="AdamW pseudocode from the original paper"
  caption="AdamW separates weight decay from the adaptive gradient update."
  width="60%"
  align="center"
>}}

> [!example]- Representative applications
> AdamW is used across modern language-model families, including [LLaMA](https://arxiv.org/abs/2302.13971), [LLaMA 2](https://arxiv.org/abs/2307.09288), [LLaMA 3](https://arxiv.org/abs/2407.21783), [Mistral](https://arxiv.org/abs/2310.06825), [Chinchilla](https://arxiv.org/abs/2203.15556), [PaLM](https://arxiv.org/abs/2204.02311), [Gemma](https://arxiv.org/abs/2403.08295), and [Falcon](https://arxiv.org/abs/2311.16867).

---

# Learning-Rate Scheduling

A fixed learning rate cannot serve every stage of training equally well. Linear warmup avoids large updates while the optimizer statistics are still poorly estimated; cosine decay then reduces the step size as optimization approaches a solution.

For peak rate $\alpha_{\max}$, final rate $\alpha_{\min}$, warmup endpoint $T_w$, and decay endpoint $T_c$,

$$
\alpha_t=
\begin{cases}
\dfrac{t}{T_w}\alpha_{\max},
&t\lt T_w,\\[8pt]
\alpha_{\min}
+\dfrac{1}{2}\left[1+\cos\left(\dfrac{t-T_w}{T_c-T_w}\pi\right)\right]
(\alpha_{\max}-\alpha_{\min}),
&T_w\le t\le T_c,\\[8pt]
\alpha_{\min},
&t\gt T_c.
\end{cases}
$$

{{< figure
  src="figures/learning_rate_schedule.png"
  alt="A cosine learning-rate schedule with linear warmup"
  caption="Linear warmup followed by cosine decay and a constant final learning rate."
  width="60%"
  align="center"
>}}

---

# Gradient Clipping

A single atypical batch can produce a gradient whose norm is large enough to destabilize training. Global norm clipping preserves the gradient direction but limits its magnitude:

$$
\widehat g=g\min\left(1,\frac{M}{\lVert g\rVert_2+\varepsilon}\right),
$$

where $g$ concatenates the gradients of all parameters and $M$ is the maximum allowed norm. Gradients below the threshold remain unchanged; larger gradients are rescaled together so their relative proportions are preserved.

---

# Training Resource Accounting

Training memory consists of model parameters, their gradients, optimizer state, and saved activations. We use the following notation:

- $b$: batch size
- $n$: context length
- $l$: number of Transformer layers
- $d$: model width
- $d_{\mathrm{ff}}$: feed-forward width
- $h$: number of attention heads
- $v$: vocabulary size

The parameter count is

$$
P=d+2dv+l\left(2d+4d^2+3dd_{\mathrm{ff}}\right).
$$

With $d_{\mathrm{ff}}=\frac{8}{3}d$,

$$
P=d+2dv+l(2d+12d^2).
$$

Counting the outputs of the operations specified in the assignment, the saved activations per sequence are

$$
A=l\left(8nd+4nd_{\mathrm{ff}}+2hn^2\right)+nd+2nv.
$$

AdamW training in `float32` therefore requires approximately

$$
\underbrace{P}_{\text{parameters}}
+\underbrace{P}_{\text{gradients}}
+\underbrace{2P}_{\text{AdamW states}}
+\underbrace{bA}_{\text{activations}}=4P+bA
$$

floating-point values. Since every value occupies four bytes, the corresponding memory is $4(P+P+2P+bA)=4(4P+bA)$ bytes. This estimate follows the simplified activation model requested by the assignment; framework workspaces and temporary buffers are not included.

For computation, let

$$
F=b\left(24nd^2l+4n^2dl+2ndv\right)
$$

be the forward-pass FLOPs when $d_{\mathrm{ff}}=\frac{8}{3}d$. Since the backward pass costs approximately twice the forward pass, the model computation for one training step is approximately $3F$. AdamW adds only $O(P)$ element-wise work, which is small compared with the Transformer matrix multiplications at practical batch sizes.

---

# Solutions

> [!note]- Problem (`cross_entropy`): Implement Cross-Entropy (1 point)
> > [!question]- Deliverable: Implement numerically stable cross-entropy
> > Write a function that accepts logits and target token IDs, subtracts the maximum logit for numerical stability, avoids explicitly computing unnecessary softmax probabilities, supports arbitrary leading batch dimensions, and returns the mean loss. The implementation is evaluated through `[adapters.run_cross_entropy]`.
> >
> > **Answer**: For each target $y$, compute $\operatorname{logsumexp}(o)-o_y$ after shifting the logits by their maximum, then average over every batch-like position.

> [!note]- Problem (`learning_rate_tuning`): Tuning the Learning Rate (1 point)
> > [!question]- Compare three learning rates in the SGD example
> > Run the SGD example for ten iterations with learning rates $10$, $100$, and $1000$. Determine whether the loss decreases slowly, converges quickly, or diverges.
> >
> > **Deliverable:** A one-to-two sentence description of the observed behavior.
> >
> > **Answer**:
> >
> > | Learning rate | Loss after 10 iterations | Behavior |
> > | ---: | ---: | --- |
> > | $10$ | $3.5307$ | Stable but slow decrease |
> > | $100$ | $2.31\times10^{-23}$ | Rapid convergence |
> > | $1000$ | $2.44\times10^{18}$ | Divergence |
> >
> > The intermediate rate converges fastest on this quadratic objective. Increasing the rate by another order of magnitude overshoots the stable region and makes the loss grow explosively.
> >
> > > [!example]- Reproduce the comparison
> > > ```bash
> > > uv run python cs336_basics/optimizer/sgd.py
> > > ```

> [!note]- Problem (`adamw`): Implement AdamW (2 points)
> > [!question]- Deliverable: Implement the AdamW optimizer
> > Implement AdamW as a subclass of `torch.optim.Optimizer`, accepting the learning rate, moment coefficients, numerical-stability constant, and weight-decay rate. Store the first moment, second moment, and step counter for each parameter. The implementation is evaluated through `[adapters.get_adamw_cls]`.
> >
> > **Answer**: Each step updates the biased moments, corrects their initialization bias, applies the normalized gradient update, and decays the parameter independently of those moments.

> [!note]- Problem (`adamw_accounting`): Resource Accounting for Training with AdamW (2 points)
> > Assume every tensor uses `float32` and $d_{\mathrm{ff}}=\frac{8}{3}d$.
>
> > [!question]- (a) Derive peak training memory
> > Decompose peak memory into parameters, activations, gradients, and AdamW optimizer state. Include the Transformer-block operations, final RMSNorm, LM head, and cross-entropy activations specified by the assignment.
> >
> > **Deliverable:** An algebraic expression for every component and their total.
> >
> > **Answer**: Define $P=d+2dv+l(2d+12d^2)$.
> >
> > | Component | Number of `float32` values |
> > | --- | ---: |
> > | Parameters | $P$ |
> > | Gradients | $P$ |
> > | AdamW first and second moments | $2P$ |
> > | Transformer-block activations | $bl\left(\frac{56}{3}nd+2hn^2\right)$ |
> > | Final RMSNorm | $bnd$ |
> > | LM-head logits | $bnv$ |
> > | Cross-entropy intermediates | $bnv$ |
> >
> > Thus $M_{\mathrm{peak}}=4\left[P+P+2P+b\left(l\left(\frac{56}{3}nd+2hn^2\right)+nd+2nv\right)\right]$ bytes.
>
> > [!question]- (b) Instantiate the estimate for GPT-2 XL
> > Use $v=50{,}257$, $n=1{,}024$, $l=48$, $d=1{,}600$, and $h=25$. Express memory as a function of batch size and find the largest integer batch size that fits within 80 GB.
> >
> > **Deliverable:** An expression $a\cdot b+c$ and the maximum batch size.
> >
> > **Answer**: $M_{\mathrm{peak}}(b)\approx16.36b+26.17$ GB. The largest batch size below 80 GB is therefore $b=3$. This is an accounting estimate rather than a guarantee, since it excludes allocator overhead and temporary workspaces.
> >
> > > [!example]- Reproduce the memory estimate
> > > ```bash
> > > uv run python scripts/resource_accounting.py training-memory
> > > ```
>
> > [!question]- (c) Count the FLOPs in one AdamW training step
> > Include the forward pass, backward pass, and element-wise optimizer update.
> >
> > **Deliverable:** An algebraic expression with a brief justification.
> >
> > **Answer**: Let $F=b\left(24nd^2l+4n^2dl+2ndv\right)$ denote the forward-pass cost. The forward and backward passes require approximately $3F$ FLOPs. Counting the scalar operations in the AdamW update gives another $14P$ FLOPs, so $F_{\mathrm{step}}\approx3F+14P$. The optimizer term is normally negligible beside the matrix multiplications.
>
> > [!question]- (d) Estimate GPT-2 XL training time on one H100
> > Assume 400,000 steps, batch size 1,024, 495 TFLOP/s theoretical throughput, 50% MFU, and a backward pass costing twice the forward pass.
> >
> > **Deliverable:** The number of training hours with a brief justification.
> >
> > **Answer**: The effective throughput is $0.5\times495=247.5$ TFLOP/s. Dividing the total work for 400,000 steps by this throughput gives approximately **4,836 hours**, or **202 days**, on a single H100.
> >
> > > [!example]- Reproduce the training-time estimate
> > > ```bash
> > > uv run python scripts/resource_accounting.py training-time
> > > ```

> [!note]- Problem (`learning_rate_schedule`): Implement Cosine Learning-Rate Scheduling (1 point)
> > [!question]- Deliverable: Implement linear warmup followed by cosine decay
> > Write a function of $t$, $\alpha_{\max}$, $\alpha_{\min}$, $T_w$, and $T_c$ that returns the piecewise schedule defined above. The implementation is evaluated through `[adapters.get_lr_cosine_schedule]`.
> >
> > **Answer**: The schedule increases linearly from zero to $\alpha_{\max}$ during warmup, follows half a cosine wave down to $\alpha_{\min}$, and remains at $\alpha_{\min}$ after $T_c$.

> [!note]- Problem (`gradient_clipping`): Implement Gradient Clipping (1 point)
> > [!question]- Deliverable: Clip the global gradient norm in place
> > Accept an iterable of parameters and a maximum $\ell_2$ norm. Compute the norm across all available parameter gradients and rescale them together when it exceeds the threshold, using $\varepsilon=10^{-6}$. The implementation is evaluated through `[adapters.run_gradient_clipping]`.
> >
> > **Answer**: Multiplying every gradient by $\min(1,M/(\lVert g\rVert_2+\varepsilon))$ preserves their joint direction while enforcing the global norm limit.
