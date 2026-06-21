+++
title = 'Section 3: Training a Transformer LM'
date = '2026-05-09T21:27:10+08:00'
summary = 'Build the components we need to support training! 😌'
weight = 30
draft = false
+++

In this section, we will train the transformer model. The training process mainly consists of the following three components:

- **Loss**: the objective function, usually cross-entropy.
- **Optimizer**: the update rule used to minimize this loss, such as [AdamW](https://arxiv.org/abs/1711.05101). We will need a direction and a step size for each update.
- **Training loop**: the infrastructure that loads data, saves checkpoints, and manages training.

---

# Loss: Cross-entropy Loss

Recall that the Transformer model estimates **the distribution** of the next token given a context. Its final layer outputs a matrix $o \in \mathbb{R}^{m \times \text{vocab}}$, where the $i$-th row $o_i \in \mathbb{R}^{\text{vocab}}$ contains the next-token logits for the $i$-th input position.

Given a training dataset $\mathcal{D} = \\{ x \mid x \in \\{1, 2, \ldots, \text{vocab}\\}^{m+1} \\}$, where each $x$ is a length-$m+1$ token sequence, the cross-entropy loss is defined as:

$$
\begin{align*}
\ell(\theta; \mathcal{D})
&= \frac{1}{|\mathcal{D}|} \sum_{x \in \mathcal{D}} \left[\frac{1}{m} \sum_{i=1}^{m} - \log p_{\theta}(x_{i+1} \mid x_{1:i}) \right] \\
p_{\theta}(x_{i+1} \mid x_{1:i}) &= \operatorname{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{j=1}^{\text{vocab}} \exp(o_i[j])}
\end{align*}
$$

where $o_i \in \mathbb{R}^{\text{vocab}}$ is the $i$-th row of the Transformer output logits.

```python
def cross_entropy(
    logits: Float[torch.Tensor, "... seqlen vocab"],
    targets: Int[torch.Tensor, "... seqlen"],
) -> torch.Tensor:
    logits_max = einx.max("... seqlen [vocab]", logits)
    logits = einx.subtract("... seqlen vocab, ... seqlen -> ... seqlen vocab", logits, logits_max)
    lse = einx.logsumexp("... seqlen [vocab]", logits)
    o = einx.get_at("... seqlen [vocab], ... seqlen -> ... seqlen", logits, targets)
    loss_items = einx.subtract("... seqlen , ... seqlen", lse, o)
    return einx.mean("[... seqlen]", loss_items)
```

> [!note]- Cross-entropy and KL divergence
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
> Therefore, minimizing $\ell$ is equivalent to minimizing $D_{\text{KL}}$, since $H(p)$ is fixed with respect to the model parameters.

> [!note]- Perplexity
> For a sequence of length $m$, the perplexity is defined as:
> $$
\text{perplexity} = \exp \left\{ \frac{1}{m} \sum_{i=1}^{m} \ell_i \right\}
= \exp \left\{ -\frac{1}{m} \sum_{i=1}^{m} \log p_{\theta}(x_{i+1} \mid x_{1:i}) \right\}
$$
>
> Intuitively, perplexity can be interpreted as the **effective branching factor** — the average number of tokens the model considers equally likely at each step. A perfect model achieves perplexity $1$, while uniform guessing over a vocabulary of size $V$ yields perplexity $V$.

---

# Optimizers

Here we will build our own optimizers, which could be generally described as:

$$
\theta_{t+1} \leftarrow \theta_t - \alpha_t g_t,
$$

where $\alpha_t$ and $g_t$ are step size and search direction respectively. They are usually a function of timestamp $t$ and gradient $\nabla \ell(\theta_t)$.

In practice, we inherit from [`torch.optim.Optimizer`](https://docs.pytorch.org/docs/2.12/optim.html#base-class) to define our own optimizer. There are at least two methods for us to implement:

- `__init__(self, params, ...)`: stores `params` that will be optimized. You can also take additional arguments depending on the optimizer (_e.g._, learning rate).
- `step(self)`: make one update of the parameters. It will be called after the backward pass, so you can use $\nabla \ell(\theta_t)$ and other state information to compute the search direction $g_t$ and step size $\alpha_t$, then updates parameters in-place: $\theta_t \leftarrow \theta_{t-1} - \alpha_t g_t$. Usually we decorate it with `@torch.no_grad()` to disable autograd in `step`.

> [!note]- A note on the learning rate
> For simplicity, all of the formulas and implementations below use a constant learning rate $\eta$. In practice, one typically uses a time-dependent schedule $\eta_t$ (e.g., linear warmup followed by cosine decay). Replacing $\eta$ with $\eta_t$ is straightforward and does not change the core logic of any optimizer described here.

---

## SGD: Stochastic Gradient Descent

The writeup implemented a slight variation of SGD to make the example richer:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{t+1}} \nabla L(\theta_t; B_t)
$$

```python
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # get state and grad
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad

                # update parameters
                p.add_(grad, alpha=-lr / math.sqrt(t + 1))

                # update state
                state["t"] = t + 1

        return loss

```

### SGD with Momentum

SGD with momentum smooths the update direction by accumulating an exponential moving average of past gradients, which dampens oscillations and accelerates progress along consistent descent directions.

$$
\begin{align*}
v_t &= \mu v_{t-1} + g_t \\[4pt]
\theta_{t+1} &= \theta_t - \eta \, v_t
\end{align*}
$$

where $g_t = \nabla \ell(\theta_t)$, $v_0 = 0$, and $\mu \in [0, 1)$ (typically $0.9$) controls how much of the previous velocity is retained.

> [!example]- Use cases
> While SGD is rarely used for modern LLM training, it was the workhorse behind many foundational deep learning models, especially in computer vision.
>
> - [ResNet](https://arxiv.org/abs/1512.03385) (He et al., 2016): the classic deep residual network trained with SGD + momentum, which popularized residual connections and batch normalization at scale.
> - [VGG](https://arxiv.org/abs/1409.1556) (Simonyan & Zisserman, 2015): demonstrated that depth alone could improve performance when paired with SGD and a simple step-wise learning rate schedule.

---

## [Adagrad](https://jmlr.org/papers/v12/duchi11a.html): Adaptive Gradient Algorithm

Adagrad adapts the per-parameter learning rate by accumulating the sum of squared gradients over time — parameters that receive frequent or large updates get a smaller effective learning rate, while infrequent parameters get a larger one, making it especially well suited for sparse features.

$$
\begin{align*}
r_t &= r_{t-1} + g_t^2 \\[4pt]
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{r_t} + \epsilon} \, g_t
\end{align*}
$$

where $g_t = \nabla \ell(\theta_t)$, $r_t$ is the cumulative sum of squared gradients, $\eta$ is the global learning rate, and $\epsilon$ is a small constant for numerical stability.

A key weakness of Adagrad is that $r_t$ grows monotonically, causing the effective learning rate $\eta / \sqrt{r_t}$ to shrink toward zero — the model eventually stops learning.

```python
class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = {
            "lr": lr,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # get state and grad
                grad = p.grad
                state = self.state[p]
                r = state.get("r", torch.zeros_like(p))
                r.add_(grad.pow(2))

                # p <- p - lr * grad / (sqrt(r) + eps)
                p.addcdiv_(grad, r.sqrt().add_(eps), value=-lr)

                # update state
                state["r"] = r
```

> [!example]- Use cases
> Adagrad's per-parameter adaptation makes it naturally effective for problems with sparse features (one-hot, ID, etc.), where some parameters receive updates far less frequently than others.
>
> - [GloVe](https://aclanthology.org/D14-1162/) (Pennington et al., 2014): used Adagrad to learn word embeddings from a global word co-occurrence matrix — a classic sparse-feature setting where Adagrad's adaptive per-parameter rates shine.

---

## [RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf): Root Mean Squared Propagation

RMSProp adapts the per-parameter learning rate by dividing the gradient by a running average of its recent magnitude — this keeps the update scale consistent across parameters that may have very different gradient scales, without the need for global learning rate tuning.

$$
\begin{align*}
v_t &= \beta v_{t-1} + (1 - \beta) \, g_t^2 \\[4pt]
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \, g_t
\end{align*}
$$

where $g_t = \nabla \ell(\theta_t)$, $\beta \in [0, 1)$ controls the decay of the squared-gradient moving average, $\eta$ is the learning rate, and $\epsilon$ is a small constant for numerical stability.

Unlike Adam, RMSProp lacks a momentum term — it only rescales the gradient but does not accumulate a consistent direction, which can lead to slower convergence along flat or ravine-like loss surfaces. It also has no bias correction for the initial estimate of $v_t$, so early updates can be overly large.

```python
class RMSProp(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= beta < 1:
            raise ValueError(f"Invalid beta: {beta}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = {
            "lr": lr,
            "beta": beta,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # get state and grad
                state = self.state[p]
                grad = p.grad

                # v <- beta * v + (1 - beta) * grad^2
                v = state.get("v", torch.zeros_like(p))
                v.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                # p <- p - lr * grad / (sqrt(v) + eps)
                denom = v.sqrt().add_(eps)
                p.addcdiv_(grad, denom, value=-lr)

                # update state
                state["v"] = v
```

> [!example]- Use cases
> RMSProp was widely adopted in reinforcement learning and early neural machine translation, serving as the direct predecessor to Adam's adaptive scaling.
>
> - [DQN](https://www.nature.com/articles/nature14236) (Mnih et al., 2015): the breakthrough deep Q-network that achieved human-level play on Atari games, trained with RMSProp for stable online RL.
> - [Effective NMT](https://arxiv.org/abs/1508.04025) (Luong et al., 2015): used RMSProp to train attention-based seq2seq models, establishing it as the preferred optimizer for early NMT systems.

---

## [Adam](https://arxiv.org/abs/1412.6980): A Method for Stochastic Optimization

Adam combines the momentum of SGD with the per-parameter adaptive learning rate of RMSProp, and adds bias correction to both moment estimates — this gives it the benefits of both approaches: a consistent search direction and a well-scaled step size, with stable early-training behavior.

$$
\begin{align*}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \, g_t \\[4pt]
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) \, g_t^2 \\[4pt]
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\[4pt]
\theta_{t+1} &= \theta_t - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align*}
$$

where $g_t = \nabla \ell(\theta_t)$, $\beta_1, \beta_2 \in [0, 1)$ control the decay rates of the first and second moment estimates, $t$ is the step counter, and $\epsilon$ is a small constant for numerical stability. The division by $1 - \beta^t$ corrects for the fact that $m_0 = 0$ and $v_0 = 0$ bias the early estimates toward zero.

```python
class Adam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # grad <- grad + weight_decay * p
                grad = p.grad.add(p, alpha=weight_decay)
                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                t += 1
                # m <- beta1 * m + (1 - beta1) * grad
                # v <- beta2 * v + (1 - beta2) * grad^2
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias correction
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                # p <- p - lr * m_hat / (sqrt(v_hat) + eps)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

                # update state
                state["t"] = t
                state["m"] = m
                state["v"] = v

```

> [!example]- Use cases
> Adam was the dominant optimizer throughout the late 2010s, powering the first wave of large-scale transformer language models.
>
> - [Transformer](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017): the original "Attention Is All You Need" paper used Adam with a custom warmup-then-decay schedule to train the first transformer model.
> - [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019): employed Adam for pre-training the bidirectional encoder that became the foundation of NLP fine-tuning pipelines.
> - [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019) and [GPT-3](https://arxiv.org/abs/2005.14165) (Brown et al., 2020): scaled up autoregressive pretraining with Adam, demonstrating the power of scaling data and model size together.
> - [T5](https://arxiv.org/abs/1910.10683) (Raffel et al., 2020): used Adam to train a unified text-to-text framework that reframed all NLP tasks as sequence generation.

---

## [AdamW](https://arxiv.org/abs/1711.05101): Adam with Decoupled Weight Decay

In standard Adam, weight decay is typically implemented as L2 regularization — adding $\lambda \theta$ to the gradient before computing the moment estimates. However, because the Adam update divides by the adaptive step size $\sqrt{\hat{v}_t} + \epsilon$, the weight decay term ends up scaled differently for each parameter: parameters with large gradient histories get less regularization, and vice versa. This coupling means weight decay no longer behaves as a pure, uniform regularizer.

AdamW fixes this by **decoupling** weight decay from the adaptive gradient — it applies weight decay directly to the parameters as a separate step, outside the moment computations:

$$
\begin{align*}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \, g_t \\[4pt]
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) \, g_t^2 \\[4pt]
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\[4pt]
\theta_t &= \theta_{t-1} - \eta \lambda \, \theta_{t-1} - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align*}
$$

where $g_t = \nabla \ell(\theta_t)$ and $\lambda$ is the weight decay coefficient. Compared with standard Adam, the only change is the explicit ($-\eta \lambda \theta_{t-1}$) term applied directly to the parameters rather than being folded into $g_t$. This simple shift restores the intended semantics of weight decay as a uniform shrinkage of all parameters.

The full procedure is summarized below. Decoupled weight decay has since become the standard in transformer training — it generalizes better than L2-regularized Adam with minimal implementation overhead.

{{< figure
  src="figures/AdamW-pseudo.png"
  alt="AdamW pseudocode from the original paper"
  caption="Pseudocode from the original paper, showing the key difference between Adam and AdamW."
  width="60%"
  align="center"
>}}

```python
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                t += 1
                # AdamW: p <- (1 - lr * weight_decay) * p
                p.mul_(1 - lr * weight_decay)

                # m <- beta1 * m + (1 - beta1) * grad
                # v <- beta2 * v + (1 - beta2) * grad^2
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias correction
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                # p <- p - lr * m_hat / (sqrt(v_hat) + eps)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

                # update state
                state["t"] = t
                state["m"] = m
                state["v"] = v
```

> [!example]- Use cases
> Since ~2021, AdamW has become the default optimizer for virtually every major transformer language model.
>
> - [LLaMA](https://arxiv.org/abs/2302.13971) / [LLaMA 2](https://arxiv.org/abs/2307.09288) / [LLaMA 3](https://arxiv.org/abs/2407.21783) (Touvron et al. & Meta): the most influential open-weight LLM family, all trained with AdamW and cosine learning rate schedules.
> - [Mistral](https://arxiv.org/abs/2310.06825) (Jiang et al., 2023): demonstrated strong 7B-scale performance with AdamW and grouped-query attention.
> - [Chinchilla](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022): used AdamW to establish the "compute-optimal" scaling laws that now guide most LLM training budgets.
> - [PaLM](https://arxiv.org/abs/2204.02311) (Chowdhery et al., 2022): scaled AdamW to a 540B-parameter model on 6144 TPUs, showing the optimizer's reliability at unprecedented scale.
> - [Gemma](https://arxiv.org/abs/2403.08295) (Gemma Team, 2024) and [Falcon](https://arxiv.org/abs/2311.16867) (Almazrouei et al., 2023): further confirmed AdamW as the standard choice for both research and deployment-oriented open LLMs.

---

## Learning Rate Scheduling

In practice, it's typical to use a learning rate _schedule_ instead of a constant learning rate. Here we will implement the cosine annealing schedule used in [LLaMA](https://arxiv.org/pdf/2302.13971). It consists of three stages:

- warm-up ($t < T_w$): $$\alpha_t = \frac{t}{T_w} \alpha_{\max},$$
- cosine annealing ($T_w \leq t \leq T_c$): $$\alpha_t = \alpha_{\min} + \frac{1}{2}\left(1 + \cos\left(\frac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{\max} - \alpha_{\min}),$$
- post-annealing ($t > T_c$): $$\alpha_t = \alpha_{\min},$$

Below is an example cosine learning rate schedule:

{{< figure
  src="figures/learning_rate_schedule.png"
  alt="Cosine learning rate schedule with warmup"
  caption="Cosine learning rate schedule with linear warmup, as used in LLaMA."
  width="60%"
  align="center"
>}}

>[!note]- Notation
> - $\alpha_t$: learning rate at step $t$
> - $\alpha_{\max}$: peak learning rate (reached at the end of warm-up)
> - $\alpha_{\min}$: minimum learning rate (held after cosine annealing ends)
> - $t$: current step/iteration number
> - $T_w$: number of warm-up steps
> - $T_c$: number of cosine annealing steps (end of the decay phase)

>[!experiment]- Learning Rate Plot Code
> ```python
> import numpy as np
> import matplotlib.pyplot as plt
>
> alpha_max = 3e-4
> alpha_min = 3e-5
> T_w = 500
> T_c = 6000
> T_total = 12000
>
>
> if __name__ == "__main__":
>     def learning_rate(t):
>         if t < T_w:
>             return (t / T_w) * alpha_max
>         elif t <= T_c:
>             cosine = 0.5 * (1 + np.cos((t - T_w) / (T_c - T_w) * np.pi))
>             return alpha_min + cosine * (alpha_max - alpha_min)
>         return alpha_min
>
>     steps = np.arange(T_total + 1)
>     lrs = np.array([learning_rate(t) for t in steps])
>
>     plt.plot(steps, lrs, label="Learning rate")
>     plt.axvline(T_w, linestyle="--", label=r"$T_w$ (warm-up end)")
>     plt.axvline(T_c, linestyle=":", label=r"$T_c$ (cosine decay end)")
>
>     plt.xlabel("Training step")
>     plt.ylabel("Learning rate")
>     plt.title("Cosine Annealing Learning Rate Schedule")
>     plt.legend()
>     plt.grid(True)
>
>     plt.savefig("learning_rate_schedule.png", dpi=300, bbox_inches="tight")
> ```

---

## Gradient Clipping

_Gradient clipping_ is a trick to stabilize training. We do it by scaling the gradient $g$ (for all parameters) by a factor to make sure $\\|g\\|_2 \leq M$, where $M$ is a given constant. It can be expressed as:

$$
\hat{g} \leftarrow
\begin{cases}
\dfrac{M}{\lVert g \rVert_2 + \varepsilon}\, g,
& \lVert g \rVert_2 > M, \\[8pt]
g,
& \lVert g \rVert_2 \le M.
\end{cases}
$$

where $\hat{g}$ is the clipped gradient and $\varepsilon$ is a small constant for numerical stability.

```python
@torch.no_grad()
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return

    l2_norm = sum(p.grad.float().pow(2).sum() for p in params).sqrt()
    scaling = (max_l2_norm / (l2_norm + eps)).clamp(max=1.0)

    for p in params:
        p.grad.mul_(scaling)
```

---

# Solutions

Here are my solutions to the problems given in the writeup.

> [!note]- Problem (learning_rate_tuning): Tuning the learning rate (1 point)
> > [!question]- As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s see that in practice in our toy example. Run the SGD example above with three other values for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of training)?
> > For `lr=1e1`, the loss converges slowly. But for `lr=1e2`, it converges quickly to 0. For `lr=1e3`, it diverges.
> > ```sh
> > sh> uv run python cs336_basics/trainer/optimizer/sgd.py
> > ========================================
> > Current lr=10.0
> > 26.271406173706055
> > 16.81369972229004
> > 12.394342422485352
> > 9.697249412536621
> > 7.854771614074707
> > 6.512506008148193
> > 5.492434501647949
> > 4.693441867828369
> > 4.053155899047852
> > 3.530749559402466
> > ========================================
> > Current lr=100.0
> > 26.271406173706055
> > 26.271404266357422
> > 4.507460117340088
> > 0.10787364840507507
> > 1.0858587174674276e-16
> > 1.2102574668033792e-18
> > 4.075361842979353e-20
> > 2.427720975674151e-21
> > 2.0826556462203992e-22
> > 2.3140628809483172e-23
> > ========================================
> > Current lr=1000.0
> > 26.271406173706055
> > 9483.9765625
> > 1638032.0
> > 182213568.0
> > 14759300096.0
> > 931480731648.0
> > 47819176017920.0
> > 2057385568894976.0
> > 7.583085165648282e+16
> > 2.4350128106110976e+18
> > ```

> [!note]- Problem (adamw_accounting): Resource accounting for training with AdamW (2 points)
> Let us compute how much memory and compute running AdamW requires. Assume we are using
float32 for every tensor.
> > [!question]- How much peak memory does running AdamW require?
> > Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the `batch_size` and the model hyperparameters (`vocab_size, context_length, num_layers, d_model, num_heads`). Assume $d_{\text{ff}} = \frac{8}{3} \times d_{\text{model}}$.
> > For simplicity, when calculating memory usage of activations, consider only the following components:
> > - Transformer block
> >     - RMSNorm(s)
> >     - Multi-head self-attention sublayer: $QKV$ projections, $QK^T$ matrix multiply, softmax, weighted sum of values, output projection
> >     - Position-wise feed-forward (SwiGLU): $W_1$, $W_2$, SiLU on the gate branch, element-wise product, $W_3$
> >     - final RMSNorm
> >     - output embedding
> >     - cross-entropy on logits
> >
> > **Answer**: Let's compute each component one by one.
> >
> > Notation: $b$: `batch_size`, $v$: `vocab_size`, $n$: `context_length`, $l$: `num_layers`, $d$: `d_model`, $h$: `num_heads`.
> >
> > _parameters_: $d + 2dv + l \times (2d + 12d^2)$ elements.
> > - Embedding: $dv$
> > - Transformer Block ($\times l$): $2d + 12d^2$
> >     - Attention Block: $d + 4d^2$
> >         - RMSNorm: $d$
> >         - Attention: $4d^2$
> >             - $W_Q$: $d^2$
> >             - $W_K$: $d^2$
> >             - $W_V$: $d^2$
> >             - $W_O$: $d^2$
> >     - FFN Block: $d + 3dd_{\text{ff}} = d + 8d^2$
> >         - RMSNorm: $d$
> >         - FFN: $3dd_{\text{ff}}$
> >             - $W_1$: $dd_{\text{ff}}$
> >             - $W_2$: $dd_{\text{ff}}$
> >             - $W_3$: $dd_{\text{ff}}$
> > - Output RMSNorm: $d$
> > - LM Head: $dv$
> >
> > _activations_ ($\times b$): $17ndl + 2hn^2l$ elements.
> > - Transformer Block ($\times l$): $17nd + 2hn^2$
> >     - Attention Block: $6nd + 2hn^2$
> >         - Input: $nd$
> >         - Normalized input: $nd$
> >         - Q, K, V: $3nd$
> >         - Attention score ($QK^T$): $hn^2$
> >         - Softmax output (attention weights): $hn^2$
> >         - Attention output: $nd$
> >     - FFN Block: $11nd$
> >         - Input: $nd$
> >         - Normalized input: $nd$
> >         - Gate ($W_1$): $\frac{8}{3}nd$
> >         - Value ($W_2$): $\frac{8}{3}nd$
> >         - Element-wise product (gate $\odot$ value): $\frac{8}{3}nd$
> >         - FFN output ($W_3$): $nd$
> >
> > _gradients_: each parameter has 1 corresponding gradient. So there are 1 _parameter_ elements.
> >
> > _optimizer state_: With AdamW, each parameter has an $m$ state ($g$) and a $v$ state ($g^2$). Ignore the $t$ state, then there are 2 _parameter_ elements.
> >
> > So in total:
> > $$
\begin{align*}
\text{memory}
&= (\text{parameters} + b \times \text{activations} \\
&\quad + \text{gradients} + \text{optimizer state}) \times 4 \text{bytes} \\
&= (4 \times \text{parameters} + b \times \text{activations}) \times 4 \text{bytes} \\
&= \left[ 4(d + 2dv + 2dl + 12d^2l) + b(17ndl + 2hn^2l) \right] \times 4 \text{bytes} \\
\end{align*}
$$
>
> > [!question]- Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the `batch_size`. What is the maximum batch size you can use and still fit within 80GB memory?
> > **Answer**: Official GPT-2 model is on [github](https://github.com/openai/gpt-2). According to this repository, we can get the [configuration](https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/hparams.json) for GPT-2 XL (1558M):
> > ```json
> > {
> >   "n_vocab": 50257,
> >   "n_ctx": 1024,
> >   "n_embd": 1600,
> >   "n_head": 25,
> >   "n_layer": 48
> > }
> > ```
> >
> > Instantiate our answer with this configuration, we get:
> > ```sh
> > memory usage = 26.17 GB + b x 15.41 GB
> > max batch size = 3
> > ```
> >
> > > [!experiment]- Compute on GPT-2 XL
> > > We can compute using this script:
> > > ```python
> > > import json
> > > import requests
> > >
> > > def main():
> > >     # get GPT-2 XL-shaped configuration
> > >     hparams_url = "https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/hparams.json"
> > >     response = requests.get(hparams_url, timeout=10)
> > >     response.raise_for_status()
> > >     hparams = response.json()
> > >
> > >     # read configuration
> > >     v = hparams["n_vocab"]
> > >     n = hparams["n_ctx"]
> > >     d = hparams["n_embd"]
> > >     h = hparams["n_head"]
> > >     l = hparams["n_layer"]
> > >
> > >     # compute memory usage
> > >     # C1: fixed memory (params + grads + optimizer) — does NOT scale with batch
> > >     # C2: per-sample activation memory — scales with batch
> > >     C1 = (4 * (d + 2*d*v + 2*d*l + 12 * (d ** 2) * l)) * 4
> > >     C2 = (17 * n * d * l + 2 * h * (n ** 2) * l) * 4
> > >     C1_GB = C1 / 1e9
> > >     C2_GB = C2 / 1e9
> > >     print(f"memory usage = {C1_GB:.2f} GB + b x {C2_GB:.2f} GB")
> > >
> > >     # compute max batch size
> > >     max_memory = 80 # GPU memory is 80 GB
> > >     max_b = int((max_memory - C1_GB) // C2_GB)
> > >     print(f"max batch size = {max_b}")
> > >
> > > if __name__ == "__main__":
> > >     main()
> > > ```
>
> >[!question]- How many FLOPs does running one step of AdamW take?
> > **Answer**: Total FLOPs consists of three parts: forward, backward, optimizer step.
> > - Forward: In [last section](../transformer-language-model-architecture/#compute), we computed the total FLOPs for 1 forward pass:
> > $$
\begin{align*}
\text{forward FLOPs} &= b\left[l(8nd^2 + 4n^2d + 6ndd_{\text{ff}}) + 2ndv \right] \\
&= b(24nd^2l + 4n^2dl + 2ndv)
\end{align*}
$$
> >
> > - Backward: The total FLOPs for 1 backward pass is approximately twice the amount of 1 forward pass. A toy matmul example explains this: suppose $C \in \mathbb{R}^{m \times n}, A \in \mathbb{R}^{m \times p}, B \in \mathbb{R}^{p \times n}, L \in \mathbb{R}$ is a scalar loss, $G = \frac{\partial L}{\partial C} \in \mathbb{R}^{m \times n}$ is the gradient of $C$, and $C = AB$, then the forward FLOPs is $2mnp$. For backward, we need to compute:
> >
> >     $$
\begin{align*}
\frac{\partial L}{\partial A} &= \frac{\partial L}{\partial C} B^T \Rightarrow 2 mnp \text{ FLOPs} \\
\frac{\partial L}{\partial B} &= A^T \frac{\partial L}{\partial C} \Rightarrow 2 mnp \text{ FLOPs} \\
\end{align*}
$$
> >
> >     so in total it's $4mnp$ FLOPs, which is $2\times$ the forward FLOPs.
> >
> > - Optimizer state: In AdamW, for each parameter at each step, we need to compute $m$, $v$ and update the parameter $\theta$. So the total FLOPs for this update is $C \times \\#\text{parameters}$, where $C$ is some constant corresponding to the FLOPs of 1 parameter at 1 step. I think this constant is around $20$.
> >
> > Add these up, we get:
> >
> > $$
\begin{align*}
\text{total FLOPs} &\approx 3 \text{ forward FLOPs} + \text{ update FLOPs} \\
&= 3b(24nd^2l + 4n^2dl + 2ndv)  \\
&\quad + C \times (d + 2dv + 2dl + 12d^2l)
\end{align*}
$$
>
> >[!question]- Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware’s theoretical peak FLOP throughput [A. Chowdhery et al., 2022]. An NVIDIA H100 GPU has a theoretical peak of 495 teraFLOP/s for "float32" (actually TensorFloat-32, which in reality is "bfloat19") operations. Assuming you are able to get 50\% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single H100? Following J. Kaplan et al. and J. Hoffmann et al. , assume that the backward pass has twice the FLOPs of the forward pass.
> > **Answer**: The result shows that it takes about $4836.2$ hours to train.
> >
> > >[!experiment]- Compute Training Hours
> > > ```python
> > > import requests
> > >
> > > def get_forward_FLOPs(
> > >     b, v, n, l, d,
> > >     ) -> int:
> > >     return b * (24 * n * (d**2) * l + 4 * (n**2) * d * l + 2 * n * d * v)
> > >
> > > def get_total_FLOPs(
> > >     steps,
> > >     b, v, n, l, d, h,
> > >     ) -> int:
> > >
> > >     one_step = 3 * get_forward_FLOPs(b, v, n, l, d) + 20 * (d + 2 * d * v + 2 * d * l + 12 * (d**2) * l)
> > >     return steps * one_step
> > >
> > > def main():
> > >
> > >     # get GPT-2 XL-shaped configuration
> > >     hparams_url = "https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/hparams.json"
> > >     response = requests.get(hparams_url, timeout=10)
> > >     response.raise_for_status()
> > >     hparams = response.json()
> > >
> > >     # read configuration
> > >     v = hparams["n_vocab"]
> > >     n = hparams["n_ctx"]
> > >     d = hparams["n_embd"]
> > >     h = hparams["n_head"]
> > >     l = hparams["n_layer"]
> > >     b = 1024
> > >     steps = 400 * 1000
> > >
> > >     total_FLOPs = get_total_FLOPs(steps, b, v, n, l, d, h)
> > >     mfu = 0.5
> > >     full_speed = 495 * 1e12
> > >     total_seconds = total_FLOPs / (full_speed * mfu)
> > >     total_hours = total_seconds / 3600
> > >
> > >     print(f"Takes {total_hours:.2f} hours to train")
> > >
> > >
> > > if __name__ == "__main__":
> > >     main()
> > > ```