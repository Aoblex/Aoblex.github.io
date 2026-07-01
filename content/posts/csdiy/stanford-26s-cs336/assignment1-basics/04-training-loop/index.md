+++
title = 'Section 4: Training Loop'
date = '2026-07-01T13:49:13+08:00'
summary = 'Put things together and build a full training loop! 🤩'
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

_Checkpointing_ allows us to resume a training run tht stopped midway through. To resume, we need to save model state, optimizer state, current iteration number and other necessary information.

```python
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
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

---

# Configuration

There are too many hyperparameters to set in a training experiment, so I use [hydra](https://hydra.cc/) to manage these configurations.
Here is a [basic tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/). Using this tool allows me to put different configuration sets in different yaml files and switch easily between them.

My configuration structure is like this:

```sh
.
├── config.yaml
├── data
├── experiment
├── model
├── optimizer
├── paths
└── training

6 directories, 1 file
```

- config.yaml: the entrance configuration that orchestrates all components;
- data: data path, vocabulary size, etc.;
- experiment: configuration for different experiments, each file gives a complete set of training settings;
- model: specifies how the model is instantiated;
- optimizer: offers different optimizer choices;
- paths: specifies input/output paths;
- training: details for training, including batch size, gradient accumulation steps, log steps, etc.;

---

# Logging

I use [wandb](https://wandb.ai/site) for logging.
