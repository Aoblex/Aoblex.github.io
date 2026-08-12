+++
title = 'Section 7: Experiments'
date = '2026-07-01T22:46:46+08:00'
summary = 'Put everything together and train our own model.'
weight = 70
draft = false
+++

Now it is time to put everything together and train (small) language models on a pretraining dataset.

---

# TinyStories



---

# Solutions

Here are my solutions to the problems given in the writeup.

> [!note]- Problem (experiment_log): Experiment logging (3 points)
> For your training and evaluation code, create experiment tracking infrastructure that allows you
> to track your experiments and loss curves with respect to gradient steps and wall-clock time.
> > [!question]- Deliverable: Logging infrastructure code for your experiments and an experiment log (a document of all the things you tried) for the assignment problems below in this section.
> > **Answer**:
> >
> > **Logging infrastructure code**
> >
> > I use Hydra for reproducible configuration and a local JSONL file for experiment tracking. Each
> > experiment name maps to a fixed output root containing the resolved Hydra configuration,
> > checkpoints, `metrics.jsonl`, and derived plots. This keeps experiment records local and makes
> > checkpoint recovery independent of the logging system.
> > Single runs and sweep jobs share the same per-run layout; Hydra's sweep-level `multirun.yaml`
> > is kept separately under `outputs/.hydra/sweeps/<sweep-name>/`.
> >
> > ```python
> > # cs336_basics/train.py
> > def _log_metrics(metrics, metrics_log_path):
> >     with open(metrics_log_path, "a", encoding="utf-8") as f:
> >         f.write(json.dumps(dict(metrics), sort_keys=True) + "\n")
> >
> > loss, grad_norm, grad_clipped = train_step(...)
> > _log_metrics(
> >     {
> >         "event": "train",
> >         "step": step,
> >         "tokens_seen": tokens_seen,
> >         "time/elapsed_sec": elapsed_sec,
> >         "time/step_sec": recent_train_sec / recent_steps,
> >         "time/tokens_per_sec": recent_tokens / recent_train_sec,
> >         "train/loss": loss,
> >         "train/lr": current_lr,
> >         "train/grad_norm": grad_norm,
> >         "train/grad_clip_fraction": recent_clipped_steps / recent_steps,
> >     },
> >     metrics_log_path,
> > )
> > ```
> >
> > The most important design choice is to keep several progress axes:
> >
> > - `step`: the optimizer step;
> > - `tokens_seen`: the total number of tokens processed so far;
> > - `time/elapsed_sec`: wall-clock time since the beginning of training.
> >
> > This lets me inspect loss curves in different ways. `loss vs step` is useful for checking optimizer behavior, `loss vs tokens_seen` is better for comparing runs with different batch sizes or context lengths, and `loss vs time/elapsed_sec` shows real training efficiency.
> >
> > The training loop records:
> >
> > - `train/loss`;
> > - `train/lr`;
> > - `train/grad_norm`;
> > - `train/grad_clip_fraction`;
> > - `eval/loss`;
> > - `eval/perplexity`;
> > - `eval/time_sec`;
> > - `time/elapsed_sec`;
> > - `time/step_sec`;
> > - `time/tokens_per_sec`;
> > - `checkpoint/time_sec`;
> > - `tokens_seen`.
> >
> > The gradient norm is measured before clipping. The clipping fraction reports the fraction of optimizer
> > steps clipped since the previous training log, which is more informative than a single Boolean value.
> >
> > A local plotting script compares one or more runs and produces loss curves against optimizer steps and
> > wall-clock time, together with gradient norm and clipping diagnostics. Each completed training run
> > automatically writes its own plot to `outputs/<name>/plots/metrics.svg`; comparisons only need the run
> > names:
> >
> > ```bash
> > uv run python scripts/plot_metrics.py run-a run-b
> > ```
> >
> > The quantitative experiment table is also regenerated after each completed run. It can be rebuilt
> > at any time by scanning the local Hydra configurations and JSONL metrics:
> >
> > ```bash
> > make experiment-log
> > ```
> >
> > The generated Markdown is written to `outputs/experiment-log.md`. I keep qualitative decisions and
> > interpretations in this document rather than attempting to infer them from metrics.
> >
> > I also save checkpoint metadata including timing information and the training and validation RNG states.
> > This keeps resumed training, validation sampling, and elapsed time consistent with an uninterrupted run.
> >
> > **Experiment log**
> >
> > TODO

> [!note]- Problem (learning_rate): Tune the learning rate (2 B200 hrs) (3 points)
> The learning rate is one of the most important hyperparameters to tune. Taking the base model you’ve trained, answer the following questions:
> > [!question]- (a) Perform a hyperparameter sweep over the learning rates and report the final losses (or note divergence if the optimizer diverges).
> > Deliverable: Learning curves associated with multiple learning rates. Explain your hyperparameter search strategy.
> >
> > Deliverable: A model with validation loss (per-token) on TinyStories of at most 1.45
> >
> > **Answer**:
> >
> > TODO
>
> > [!question]- (b) Folk wisdom is that the best learning rate is “at the edge of stability.” Investigate how the point at which learning rates diverge is related to your best learning rate.
> > Deliverable: Learning curves of increasing learning rate which include at least one divergent run and an analysis of how this relates to convergence rates.
> >
> > **Answer**:
> >
> > TODO

> [!note]- Problem (batch_size_experiment): Batch size variations (1 B200 hr) (1 point)
> Vary your batch size all the way from 1 to the GPU memory limit. Try at least a few batch sizes in between, including typical sizes like 64 and 128.
> > [!question]- Deliverables
> > Deliverable: Learning curves for runs with different batch sizes. The learning rates should be optimized again if necessary.
> >
> > Deliverable: A few sentences discussing your findings on batch sizes and their impacts on training.
> >
> > **Answer**:
> >
> > TODO
