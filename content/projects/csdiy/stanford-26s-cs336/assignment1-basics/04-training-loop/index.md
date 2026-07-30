+++
title = 'Section 4: Training Loop'
date = '2026-07-01T13:49:13+08:00'
summary = 'Build the full training loop! 🤩'
weight = 40
draft = false
+++

A full training system should include at least the following components:

- **Data loading**: samples fixed-length token windows from the tokenized corpus.
- **Checkpointing**: saves enough state for later inspection or resuming.
- **Configuration**: stores model, data, optimizer, and training hyperparameters.
- **Logging**: records training progress so I can tell whether the run is healthy.

---

# Data Loading

After tokenization, our input data is a single sequence of token IDs $x = (x_1, \ldots, x_n)$. During training, we repeatedly sample mini-batches of contiguous token sequences from this stream. This makes optimization computationally manageable, introduces useful stochasticity into gradient estimates, and allows efficient training even when the full dataset is too large to fit in memory.

```python
def get_batch(
    x: Int[np.ndarray, "..."],
    batch_size: int,
    context_length: int,
    device: torch.device | str,
) -> tuple[
    Int[torch.Tensor, "batch_size context_length"],
    Int[torch.Tensor, "batch_size context_length"],
]:
    if x.ndim != 1:
        raise ValueError("x must be a 1D array of token IDs.")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive.")
    if len(x) <= context_length:
        raise ValueError("x must contain more tokens than context_length.")

    # Each start position must leave room for context_length inputs
    # plus one next-token target.
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)
    offsets = np.arange(context_length)

    inputs = x[starts[:, None] + offsets[None, :]]
    targets = x[starts[:, None] + offsets[None, :] + 1]

    return (
        torch.as_tensor(inputs, dtype=torch.long, device=device),
        torch.as_tensor(targets, dtype=torch.long, device=device),
    )
```

---

# Checkpointing

_Checkpointing_ allows us to resume a training run that stopped midway through. To resume, the essential state is still simple: model parameters, optimizer state, and the current iteration number. Extra metadata is not required for resuming, but it is useful when I later inspect a checkpoint and want to know when it was saved and how much training had already happened.

```python
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    metadata: dict | None = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    if metadata is not None:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]
```

For my training loop, the metadata contains fields such as `started_at`, `saved_at`, `elapsed_sec`, and `tokens_seen`. These fields do not change the training semantics, but they make checkpoints easier to reason about outside the training script.

---

# Configuration

There are too many choices in a training run to keep them as command-line flags: model size, dataset, tokenizer files, optimizer parameters, checkpoint paths, device selection, logging metadata, and so on. So I use [Hydra](https://hydra.cc/) to keep these choices explicit and composable.

## Structure

My hydra configuration structure is roughly:

```sh
configs
├── data
├── device
├── inference
│   ├── config.yaml
│   ├── experiment
│   └── run
├── model
├── optimizer
├── paths
├── tokenizer
└── train
    ├── config.yaml
    ├── experiment
    └── run
```

The top-level `train/config.yaml` and `inference/config.yaml` files are only entry points. They choose defaults and define the Hydra output directory. The actual settings live in smaller config groups:

- `data`: tokenized dataset paths and vocabulary size;
- `tokenizer`: vocabulary file, merges file, special tokens, and EOS token;
- `model`: Transformer size and the `_target_` used by Hydra to instantiate the model;
- `optimizer`: optimizer constructor plus learning-rate schedule and gradient clipping settings;
- `device`: the device policy, such as `auto`, `gpu`, `cpu`, etc.;
- `train/run`: batch size, number of steps, validation frequency, checkpoint frequency, etc.;
- `train/experiment`: complete training experiment presets;
- `inference/run`: decoding parameters, prompt, and checkpoint path;
- `inference/experiment`: inference presets.

I also want experiment files to stay small. The default runtime settings are loaded by the entry points: `train/config.yaml` loads `/train/run@run: default`, and `inference/config.yaml` loads `/inference/run@run: default`. An experiment preset should mostly choose the model, dataset, tokenizer, and optimizer, then override only the few settings that make that run special.

## Instantiation

One useful feature of hydra is that the model and optimizer could be instantiated from configuration. For example, a base model config contains:

```yaml
_target_: cs336_basics.transformer.Transformer
vocab_size: ${data.vocab_size}
d_model: 512
num_layers: 4
d_ff: 1344
num_heads: 16
max_seq_len: 256
theta: 10000.0
```

The training script can then simply call:

```python
model = hydra.utils.instantiate(cfg.model, device=device)
```

The optimizer is slightly different because `model.parameters()` is only available at runtime. I keep the optimizer constructor under `optimizer.init` and make it partial:

```yaml
init:
  _target_: cs336_basics.optimizer.AdamW
  _partial_: true
  _convert_: all
  lr: 0.001
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0.1

warmup_steps: 100
cosine_steps: 10000
min_lr: 1e-5
max_grad_norm: 1.0
```

Then the script supplies parameters explicitly:

```python
optimizer = hydra.utils.instantiate(cfg.optimizer.init)(params=model.parameters())
```

This way, the config decides what object to build, and the script only wires runtime objects together.

## Output

I also organize outputs by experiment name first:

```sh
outputs
└── tinystories-smoke
    ├── train
    │   ├── checkpoints
    │   │   ├── latest.pt
    │   │   └── step_10.pt
    │   ├── console.log
    │   └── .hydra
    └── inference
        └── .hydra
```

This makes `train` and `inference` parallel views of the same experiment. The tradeoff is that rerunning the same experiment name overwrites the current run. For this project, I prefer that behavior because `latest.pt` always points to the checkpoint I want to inspect or decode from.

---

# Logging

I use [Weights & Biases](https://wandb.ai/site) to track training runs. The main reason is simple: once I start running multiple experiments, I need a place to compare their curves and recover the exact configuration that produced each result.

I keep the basic W&B metadata in config:

```yaml
project: cs336-assignment1
name: tinystories-smoke
notes: "TinyStories smoke test"
tags: []
```

These fields are passed to `wandb.init`, together with the resolved Hydra config:

```python
wandb.init(
    project=cfg.project,
    name=cfg.name,
    notes=cfg.get("notes"),
    tags=cfg.get("tags", []),
    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
)
```

During training, I keep three coordinates around:

- `step`;
- `tokens_seen`;
- `time/elapsed_sec`.

I still use `step` as the default W&B step, because it is the most natural axis for debugging the optimizer. But `tokens_seen` and elapsed wall-clock time answer different questions. `tokens_seen` is better when comparing experiments with different batch sizes, gradient accumulation, or context lengths. `time/elapsed_sec` is better when comparing how quickly two runs reach the same loss or perplexity in real time.

The main metrics I record are:

- `train/loss`;
- `train/lr`;
- `eval/loss`;
- `eval/perplexity`;
- `eval/time_sec`;
- `time/elapsed_sec`;
- `time/step_sec`;
- `time/tokens_per_sec`;
- `checkpoint/time_sec`;
- `tokens_seen`.

This gives me both an optimization view and an efficiency view. Loss and perplexity show whether the model is improving; step time, throughput, and checkpoint/eval time show whether the training system itself is behaving as expected.
