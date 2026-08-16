+++
title = 'Section 5: Training Loop'
date = '2026-07-01T13:49:13+08:00'
summary = 'Assemble data sampling, optimization, evaluation, and checkpointing into a complete training system.'
weight = 50
draft = false
+++

# Introduction

A training loop repeatedly turns tokenized text into parameter updates. Its core operation is simple—sample a batch, compute next-token loss, backpropagate, and update the model—but a usable training system must also evaluate progress, preserve state, and expose enough information to compare experiments.

This section connects the model and optimizer from the previous sections with three supporting mechanisms: random-access data loading, resumable checkpointing, and experiment-level observability.

---

# Data Loading

After tokenization, the corpus is stored as one sequence of token IDs,

$$
x=(x_0,x_1,\ldots,x_{N-1}).
$$

A training example is a contiguous window paired with the same window shifted by one position. For each batch element $j$, sample a start index $s_j$ satisfying $0\le s_j\le N-n-1$, then define

$$
X_{j,k}=x_{s_j+k},
\qquad
Y_{j,k}=x_{s_j+k+1},
\qquad
0\le k\lt n.
$$

Both $X$ and $Y$ have shape $(b,n)$. Every target is therefore the token immediately following its corresponding input, which lets a single causal forward pass train all $n$ next-token predictions in parallel.

Sampling fixed-length windows has three useful consequences:

- every valid start position defines a training example;
- batches require no padding, so every token contributes useful work;
- document boundaries need no special loader logic because the tokenizer has already inserted delimiter tokens into the stream.

## Memory-Mapped Datasets

The token array can be larger than physical memory. Memory mapping exposes the file as an array while loading only the pages that are actually accessed. Random windows can therefore be sampled without reading the complete corpus into RAM.

The file's dtype must match the dtype used during tokenization. The sampled arrays are converted to integer tensors only after indexing and then moved to the selected device:

- `cpu` for CPU training;
- `mps` for Apple Silicon;
- `cuda` for NVIDIA GPUs.

---

# Checkpointing

A checkpoint represents the state of optimization at a particular step. The minimum state required by the assignment is

$$
\mathcal C_t=(\theta_t,\omega_t,t),
$$

where $\theta_t$ contains the model parameters and $\omega_t$ contains the optimizer state, including AdamW's moment estimates. Restoring only the model weights would recover its predictions but not the update trajectory.

The training system stores several additional fields because resuming and later analysis require slightly different information:

| State | Purpose |
| --- | --- |
| Model parameters | Restore the learned function |
| Optimizer state | Continue the same adaptive updates |
| Completed step | Resume the schedule and training budget |
| Random-number-generator state | Continue the same sampling sequence |
| Metadata | Record elapsed time, tokens seen, model settings, and tokenizer settings |

Checkpoint writes are atomic: the new state is written to a temporary file and then moved into place. An interrupted write therefore cannot replace the last valid checkpoint with a partial file.

---

# Training Loop

One optimizer step coordinates the mechanisms developed across the assignment:

1. Sample input and target windows from the training token stream.
2. Compute logits and the mean cross-entropy loss.
3. Backpropagate the loss, accumulating gradients across microbatches when requested.
4. Set the scheduled learning rate, clip the global gradient norm, and apply AdamW.
5. Record training metrics and periodically run evaluation or save a checkpoint.

With microbatch size $b$, gradient-accumulation factor $a$, and context length $n$, each optimizer step processes

$$
T_{\mathrm{step}}=ban
$$

tokens. Dividing each microbatch loss by $a$ before backpropagation makes the accumulated gradient equal to the mean over the effective batch of size $ba$.

## Evaluation

Evaluation samples batches from a separate validation stream and measures loss without constructing a backward graph. The model is temporarily placed in evaluation mode, then returned to its previous mode after validation.

Training loss measures progress on the batches used for optimization; validation loss measures whether that progress transfers to unseen data. Perplexity, $\exp(\ell_{\mathrm{val}})$, expresses the same validation result on a more interpretable multiplicative scale.

---

# Experiments

Each experiment is described by one complete YAML file in `configs/`. The file name identifies the run, and all outputs for that experiment are placed under the corresponding directory. There is no configuration inheritance or directory hierarchy to resolve.

The smoke test uses the same schema and training path as a full run while reducing only the scale:

> [!example]- Smoke-test configuration
> ```yaml
> device: auto
> data:
>   train_path: data/TinyStoriesV2-GPT4-train.npy
>   valid_path: data/TinyStoriesV2-GPT4-valid.npy
> model:
>   vocab_size: 10000
>   d_model: 16
>   num_layers: 1
>   d_ff: 32
>   num_heads: 2
>   max_seq_len: 8
> train:
>   batch_size: 1
>   gradient_accumulation_steps: 1
>   total_steps: 10
>   compile: false
> ```

Device selection is automatic in the order CUDA, MPS, then CPU. A configuration can be checked without loading data or starting training, while the smoke command exercises the complete training path:

> [!example]- Validate and run the smoke experiment
> ```bash
> uv run python scripts/train.py configs/tinystories-smoke.yaml --check
> make train CONFIG=configs/tinystories-smoke.yaml
> ```

## Output Structure

Training and generation are two views of the same named experiment. The repository keeps their artifacts together:

```text
outputs
└── <experiment>
    ├── train
    │   ├── checkpoint.pt
    │   ├── config.yaml
    │   ├── metadata.json
    │   ├── metrics.jsonl
    │   └── wandb.json
    └── generation
        ├── config.yaml
        ├── metadata.json
        └── samples.md
```

If a checkpoint exists, the run restores its model, optimizer, completed step, and sampling state. A completed run is skipped; a meaningfully different run receives a different experiment file and output directory.

---

# Observability

[Weights & Biases](https://wandb.ai/site) provides a shared view of multiple experiments, while the local backend writes the same metric stream to `metrics.jsonl`. An experiment may use local logging, W&B, both, or neither; the smoke test remains fully local and requires no network connection.

Metrics serve four distinct questions:

| Question | Metrics |
| --- | --- |
| Is optimization progressing? | `train/loss`, `train/lr` |
| Does the model generalize? | `eval/loss`, `eval/perplexity` |
| Are updates stable? | `train/grad_norm`, `train/grad_clip_fraction` |
| Is the system efficient? | `time/step_sec`, `time/tokens_per_sec`, `eval/time_sec` |

The same run is indexed by three coordinates:

- `step` compares optimizer progress;
- `tokens_seen` normalizes experiments with different batch sizes, accumulation factors, or context lengths;
- `time/elapsed_sec` compares wall-clock efficiency.

Together, these views distinguish an optimization problem from a systems problem: loss and gradient statistics describe learning behavior, while throughput and elapsed time describe the cost of obtaining it.

---

# Solutions

> [!note]- Problem (`data_loading`): Implement Data Loading (2 points)
> > [!question]- Deliverable: Sample next-token training batches
> > Write a function that accepts a one-dimensional NumPy token array, batch size, context length, and device. Return input and target tensors of shape $(b,n)$ on the requested device, with every target shifted one token ahead of its input. The implementation is evaluated through `[adapters.run_get_batch]`.
> >
> > **Answer**: Sample $b$ valid start indices, add the offsets $0,\ldots,n-1$ to construct the input windows, and add one more position for the targets. Vectorized indexing constructs the whole batch without a Python loop.

> [!note]- Problem (`checkpointing`): Implement Model Checkpointing (1 point)
> > [!question]- Deliverable: Save and restore training state
> > Implement `save_checkpoint(model, optimizer, iteration, out)` and `load_checkpoint(src, model, optimizer)`. Save both state dictionaries and the iteration number; loading must restore the supplied objects and return the saved iteration. The implementation is evaluated through `[adapters.run_save_checkpoint]` and `[adapters.run_load_checkpoint]`.
> >
> > **Answer**: The serialized checkpoint contains the model state, optimizer state, and completed iteration. Loading applies both state dictionaries to newly constructed objects and returns the iteration from which training should continue.

> [!note]- Problem (`training_together`): Put It Together (4 points)
> > [!question]- Deliverable: Implement the complete training script
> > Write a script that trains the model on user-provided data. It should expose model and optimizer hyperparameters, load large training and validation arrays through memory mapping, serialize checkpoints to a chosen path, and periodically report training and validation performance.
> >
> > **Answer**: The training entry point validates one experiment configuration, constructs the model and AdamW optimizer, memory-maps the datasets, and either starts a run or resumes its checkpoint. The loop applies the learning-rate schedule, gradient accumulation, clipping, evaluation, logging, and checkpointing around the shared next-token training step.
