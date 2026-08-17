+++
title = 'Section 7: Experiments'
date = '2026-07-01T22:46:46+08:00'
summary = 'Train and compare small language models on TinyStories and OpenWebText.'
weight = 70
draft = false
+++

The components developed in the preceding sections now form a complete language-model training system. The purpose of these experiments is not merely to obtain a low loss, but to connect optimization behavior, architectural choices, and data complexity through controlled comparisons.

---

# Experimental Setup

Unless an experiment states otherwise, the model uses:

- 4 Transformer blocks;
- model dimension $d_{\text{model}}=512$;
- 16 attention heads;
- SwiGLU feed-forward networks with $d_{\text{ff}}=1344$;
- pre-norm RMSNorm;
- RoPE with $\Theta=10{,}000$;
- a context length of 256 tokens.

The TinyStories and OpenWebText baselines each process approximately 327.68 million tokens:

$$
N_{\text{tokens}}
= B_{\text{micro}}G L S,
$$

where $B_{\text{micro}}$ is the microbatch size, $G$ is the number of gradient-accumulation steps, $L$ is the context length, and $S$ is the number of optimizer steps. Shorter sweeps preserve a fixed token budget within each comparison.

TinyStories isolates optimization and architecture effects in a small, regular domain. OpenWebText then tests the same model and compute budget against a broader and substantially less predictable distribution.

---

# Solutions

> [!note]- Problem (`experiment_log`): Experiment Logging (3 points)
> For your training and evaluation code, create experiment tracking infrastructure that allows you to track your experiments and loss curves with respect to gradient steps and wall-clock time.
>
> > [!question]- Deliverable: Logging infrastructure code for your experiments and an experiment log for the assignment problems below in this section
> > **Answer**: Each experiment is defined by one flat configuration file. A run records its resolved configuration, JSONL metrics, checkpoint metadata, and generated samples locally, while the same scalar series are mirrored to Weights & Biases. The records share three progress coordinates—optimizer step, tokens processed, and wall-clock time—so optimization quality and computational efficiency can be compared independently.
> >
> > Checkpoints preserve the optimizer, scheduler, random-number-generator states, and accumulated elapsed time. The OpenWebText baseline was stopped after step 4000 and resumed to step 5000; its continuous metrics and W&B history therefore also verify the recovery path.
> >
> > | Study | Variants tried | Main outcome |
> > | --- | --- | --- |
> > | TinyStories learning rate | $10^{-3}$, $3\times10^{-3}$, $10^{-2}$, $10^{-1}$ | $3\times10^{-3}$ converged fastest; $10^{-1}$ diverged |
> > | TinyStories baseline | 327.68M tokens | Validation loss $1.3430$ |
> > | Batch size | $1$, $16$, $64$, $128$, $256$ | $64$ gave the best efficiency–quality balance |
> > | Batch-1 retuning | $3\times10^{-3}$, $10^{-3}$ | The lower rate improved validation loss from $1.8221$ to $1.7203$ |
> > | Matched ablation baseline | 163.84M tokens | Validation loss $1.4364$ |
> > | RMSNorm | removed at $3\times10^{-3}$ and $3\times10^{-4}$ | Original rate diverged; lower rate was stable but worse |
> > | Normalization placement | pre-norm, post-norm | Pre-norm was better by $0.0085$ |
> > | Position information | RoPE, NoPE | RoPE was better by $0.0837$ |
> > | Feed-forward network | SwiGLU, parameter-matched SiLU | SwiGLU was better by $0.0125$ |
> > | OpenWebText baseline | 327.68M tokens | Validation loss $3.9337$ |
> > | 45-minute pilot | cosine decay over 5000 steps | The schedule decayed much too early on an A100 |
> > | 45-minute runs | clipping thresholds $0.25$ and $1.0$ | $1.0$ reached the best validation loss, $4.0922$ |
> >
> > The full metric history remains machine-readable rather than being reduced to final losses: training and validation loss, learning rate, pre-clipping gradient norm, clipping fraction, throughput, evaluation time, and checkpoint time are all retained.

> [!note]- Problem (`learning_rate`): Tune the Learning Rate (2 B200 hrs) (3 points)
> The learning rate is one of the most important hyperparameters to tune. Taking the base model you have trained, answer the following questions.
>
> > [!question]- (a) Perform a hyperparameter sweep over the learning rates and report the final losses or note divergence if the optimizer diverges
> > **Deliverables:** Learning curves associated with multiple learning rates, an explanation of the search strategy, and a TinyStories model with per-token validation loss at most $1.45$.
> >
> > **Answer**: A logarithmic sweep first located the useful scale and the unstable region. Every sweep run processed 65.536 million tokens with the same model, batch, and schedule; only the peak learning rate changed.
> >
> > | Peak learning rate | Best validation loss | Final validation loss | Outcome |
> > | ---: | ---: | ---: | --- |
> > | $10^{-3}$ | $1.7765$ | $1.7801$ | Stable |
> > | $3\times10^{-3}$ | **$1.6096$** | **$1.6122$** | Stable, fastest convergence |
> > | $10^{-2}$ | $2.1027$ | $2.1078$ | Stable but poorly optimized |
> > | $10^{-1}$ | $3.9343$ | $4.4081$ at step 190 | Divergent |
> >
> > ![Learning-rate sweep on TinyStories](figures/section7-learning-rate.svg)
> >
> > Training the selected rate, $3\times10^{-3}$, for the full 327.68-million-token budget reached a best validation loss of $1.3425$ and a final loss of $1.3430$, satisfying the target.
>
> > [!question]- (b) Investigate how the point at which learning rates diverge is related to the best learning rate
> > **Deliverable:** Learning curves of increasing learning rate including at least one divergent run, together with an analysis of convergence rates.
> >
> > **Answer**: The stable region ends between $10^{-2}$ and $10^{-1}$, whereas the best tested rate is $3\times10^{-3}$. Increasing the rate from $10^{-3}$ to $3\times10^{-3}$ accelerates convergence, but another increase to $10^{-2}$ already makes optimization worse before outright divergence appears. The useful interpretation of the “edge of stability” is therefore directional rather than literal: good rates exploit aggressive updates, but the numerically largest stable rate need not minimize validation loss for a finite training budget.

> [!note]- Problem (`batch_size_experiment`): Batch Size Variations (1 B200 hr) (1 point)
> Vary your batch size all the way from 1 to the GPU memory limit. Try at least a few batch sizes in between, including typical sizes like 64 and 128.
>
> > [!question]- Deliverables
> > Learning curves for different batch sizes, with learning rates re-optimized if necessary, and a discussion of their effects on training.
> >
> > **Answer**: Each run processes 32.768 million tokens, so changing the batch size changes the number of optimizer updates but not the amount of data observed.
> >
> > | Batch size | Learning rate | Optimizer steps | Final validation loss | Throughput |
> > | ---: | ---: | ---: | ---: | ---: |
> > | $1$ | $3\times10^{-3}$ | 128,000 | $1.8221$ | $20.1$k tokens/s |
> > | $1$ | $10^{-3}$ | 128,000 | **$1.7203$** | $21.5$k tokens/s |
> > | $16$ | $3\times10^{-3}$ | 8,000 | $1.7302$ | $64.4$k tokens/s |
> > | $64$ | $3\times10^{-3}$ | 2,000 | $1.7220$ | $69.9$k tokens/s |
> > | $128$ | $3\times10^{-3}$ | 1,000 | $1.7535$ | $70.3$k tokens/s |
> > | $256$ | $3\times10^{-3}$ | 500 | $1.9407$ | $69.7$k tokens/s |
> >
> > ![Batch-size comparison on TinyStories](figures/section7-batch-size.svg)
> >
> > Small batches provide many noisy parameter updates per token budget and can reach a strong loss after retuning, but batch size 1 is more than three times slower and clips gradients on nearly every late update. Throughput saturates near batch size 64; beyond that point, larger batches reduce the number of updates without increasing hardware utilization. Batch size 64 is therefore the most useful compromise here, while batch size 256 is the memory-limit case rather than the best training choice.

> [!note]- Problem (`generate`): Generate Text (1 point)
> Using your decoder and trained checkpoint, report the text generated by your model. You may need to manipulate decoder parameters such as temperature and top-$p$ to get fluent outputs.
>
> > [!question]- Deliverable
> > A text dump of at least 256 tokens, or until the first `<|endoftext|>` token, and a brief comment on fluency and at least two factors that affect output quality.
> >
> > **Answer**: Sampling uses temperature $1.0$ and top-$p=0.9$. The following generation emits 138 tokens before `<|endoftext|>`:
> >
> > > [!example]- TinyStories generation
> > > **Prompt:** `Once upon a time`
> > >
> > > Once upon a time, there was a big castle. In the castle, there lived a clever girl named Lucy. Lucy liked to play in the castle all day. She had many toys and games to spend in the fun.
> > > One day, Lucy saw a little cat outside her window. The cat was hungry and could not find any food. Lucy wanted to help the cat, so she had an idea. She thought she could make the cat's food.
> > > Lucy took the cat to her mom and dad. They made the cat's food and gave it some milk. The cat was happy and not hungry anymore. Lucy and the cat became best friends and had lots of fun together.
> >
> > The sample maintains a recognizable story arc, stable characters, and an appropriate ending, although phrases such as “games to spend in the fun” expose imperfect local semantics. Its quality depends principally on the coverage and regularity of the training corpus, the model and training budget, and the sampling distribution controlled by temperature and top-$p$.

> [!note]- Problem (`layer_norm_ablation`): Remove RMSNorm and Train (0.5 B200 hrs) (1 point)
> Remove all RMSNorms from the Transformer and train. What happens at the previous optimal learning rate? Can stability be recovered with a lower learning rate?
>
> > [!question]- Deliverables
> > A learning curve without RMSNorm, a curve for the best lower learning rate, and commentary on the impact of RMSNorm.
> >
> > **Answer**: At the baseline learning rate $3\times10^{-3}$, removing RMSNorm causes catastrophic instability by step 120: the gradient norm reaches approximately $6.35\times10^6$ and the training loss exceeds $1.6\times10^5$. Reducing the rate tenfold restores finite training, but its final validation loss is $1.8244$, compared with $1.4364$ for the normalized model under the same 163.84-million-token budget.
> >
> > ![RMSNorm ablation on TinyStories](figures/section7-rmsnorm.svg)
> >
> > RMSNorm therefore does more than improve the final fit: it makes a much larger and more effective learning rate usable. Lowering the rate can suppress divergence without recovering the optimization speed lost by removing normalization.

> [!note]- Problem (`pre_norm_ablation`): Implement Post-Norm and Train (0.5 B200 hrs) (1 point)
> Modify the pre-norm Transformer into a post-norm one and train it.
>
> > [!question]- Deliverable
> > A learning curve for a post-norm Transformer compared with the pre-norm model.
> >
> > **Answer**: Under matched architecture, token budget, and learning-rate schedule, post-norm reaches a final validation loss of $1.4449$, while pre-norm reaches $1.4364$.
> >
> > ![Pre-norm and post-norm comparison](figures/section7-norm-placement.svg)
> >
> > Both variants remain stable, but pre-norm is consistently ahead near the end of training. The difference is small at this scale—$0.0085$ in final validation loss—yet it favors placing normalization inside the residual branch rather than after the residual addition.

> [!note]- Problem (`no_pos_emb`): Implement NoPE (0.5 B200 hrs) (1 point)
> Remove position information entirely from the Transformer and compare the result with RoPE.
>
> > [!question]- Deliverable
> > A learning curve comparing RoPE and NoPE.
> >
> > **Answer**: NoPE reaches a final validation loss of $1.5201$, compared with $1.4364$ for RoPE.
> >
> > ![RoPE and NoPE comparison](figures/section7-position-encoding.svg)
> >
> > The causal mask provides an implicit ordering signal, so removing explicit position information does not prevent learning. It does, however, weaken the model's ability to represent token distance and order, which leaves a measurable validation penalty. NoPE is computationally cheaper in this implementation—about $101.8$k rather than roughly $69$k tokens/s—but the speedup does not compensate for the lost modeling quality at a fixed token budget.

> [!note]- Problem (`swiglu_ablation`): SwiGLU vs. SiLU (0.5 B200 hrs) (1 point)
> Compare the default SwiGLU feed-forward network with an ungated SiLU network, $\operatorname{FFN}_{\text{SiLU}}(x)=W_2\operatorname{SiLU}(W_1x)$, using $d_{\text{ff}}=4d_{\text{model}}$ for SiLU to approximately match parameter counts.
>
> > [!question]- Deliverables
> > A learning curve comparing SwiGLU and SiLU with approximately matched parameter counts, together with a discussion of the result.
> >
> > **Answer**: The SwiGLU model has 22,696,448 parameters, while the SiLU model has 22,827,520, a difference of only $0.58\%$. SiLU reaches a final validation loss of $1.4489$, compared with $1.4364$ for SwiGLU.
> >
> > ![SwiGLU and SiLU comparison](figures/section7-ffn.svg)
> >
> > Approximately matching parameter count isolates the effect of gating from model size. SwiGLU improves final validation loss by $0.0125$, a modest but consistent advantage for the multiplicative gate under this budget.

> [!note]- Problem (`main_experiment`): Experiment on OpenWebText (2 B200 hrs) (2 points)
> Train the language model on OpenWebText with the same model architecture and total training iterations as TinyStories. How well does it perform?
>
> > [!question]- Deliverables
> > An OpenWebText learning curve; an interpretation of its loss relative to TinyStories; generated text in the same format as the TinyStories output; and an explanation of the difference in fluency.
> >
> > **Answer**: With the same 5000 optimizer steps and 327.68-million-token budget, OpenWebText reaches a final validation loss of $3.9337$, whereas TinyStories reaches $1.3430$.
> >
> > ![TinyStories and OpenWebText validation curves](figures/section7-datasets.svg)
> >
> > These losses are conditional cross-entropies under different datasets and different 10k/32k tokenizers. They are not direct scores of corpus quality: OpenWebText contains a much wider range of topics, styles, entities, and long-range dependencies, so its next-token distribution has higher irreducible uncertainty. Perplexity, $\operatorname{PPL}=\exp(\mathcal{L})$, is approximately $3.83$ on TinyStories and $51.09$ on OpenWebText, but the tokenizer difference still prevents a clean cross-dataset ranking.
> >
> > > [!example]- OpenWebText generation
> > > **Prompt:** `The`
> > >
> > > The Clinton campaign and the Democratic candidate in her husband’s bedroom were forced to walk away, they are not being violent or rebellious, they are not even dealing with these late-night pictures of a drug addict.”
> > >
> > > Sharia has been released from prison without any major charges. For the same reason she was in prison, she was handed the terms of her second life sentence. She could have been charged by state police.
> > >
> > > In the end, the FBI isn’t seeking prosecution, instead, for the suspects that should serve the country in the country or in other cases. In truth, cases like this now aren’t without merit, or coercion, or criminal activity.
> > >
> > > The Trump campaign has kept the promise from the FBI and is calling on it to dissolve local prosecutors. The DOJ says that without law, the FBI does not use such a policy to give some details about the criminal justice system in a racially diverse style.
> > >
> > > The first three judges are Comey. And then he’s pushing them.
> > >
> > > They basically just say they’re important for law enforcement.
> > >
> > > There are five Republicans in charge of the Republican Party, with Democrats out on the front line. In 2016, they were major targets for special elections and they finally
> >
> > The 256-token sample is locally grammatical and resembles political news, but entities and causal relations drift across paragraphs and it ends without completing the final sentence. TinyStories repeatedly exposes a small model to short, regular narrative patterns; the same compute budget covers only a tiny fraction of OpenWebText's diversity, so surface fluency emerges before reliable discourse coherence.

> [!note]- Problem (`leaderboard`): Leaderboard (10 B200 hrs) (6 points)
> Train a model under the leaderboard rules with the goal of minimizing validation loss within 0.75 B200-hours. Training may use only the provided OpenWebText data.
>
> > [!question]- Deliverable
> > The final validation loss, a learning curve with a wall-clock axis shorter than 45 minutes, and a description of the approach. A submission is expected to beat the naive baseline loss of $5.0$.
> >
> > **Answer**: These runs use an A100 rather than the official B200 evaluation environment, so they are a local 45-minute proxy rather than a leaderboard submission. The model uses $d_{\text{model}}=448$, 4 layers, 7 heads, $d_{\text{ff}}=1344$, batch size 32, a peak learning rate of $2\times10^{-3}$, and a cosine schedule designed around the measured A100 step rate.
> >
> > A short pilot exposed the decisive scheduling issue: a 5000-step cosine schedule exhausted the learning rate in roughly 15 minutes. Extending it to 17,000 steps aligns decay with the available wall-clock budget.
> >
> > | Gradient-clipping threshold | Final step | Tokens processed | Final validation loss | Wall-clock time |
> > | ---: | ---: | ---: | ---: | ---: |
> > | $0.25$ | 16,431 | 134.60M | $4.0963$ | 44m 52s |
> > | $1.0$ | 16,737 | 137.11M | **$4.0922$** | 44m 52s |
> >
> > ![OpenWebText 45-minute training runs](figures/section7-leaderboard.svg)
> >
> > The more restrictive threshold clips nearly every update but does not improve validation loss. Standard clipping permits slightly more progress and gives the best result, beating the $5.0$ baseline by a wide margin. The small difference between the two runs also indicates that schedule alignment and tokens processed matter more here than fine tuning the clipping threshold.
