+++
title = 'Section 6: Experiments'
date = '2026-07-01T22:46:46+08:00'
summary = 'Put everything together and train our own model! 🥳'
weight = 60
draft = false
+++

Now it is time to put everything together and train (small) language models on a pretraining dataset.

---

# Tinystories



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
> > I use Hydra for reproducible configuration and W&B for experiment tracking. The core structure is:
> >
> > ```python
> > # scripts/train.py
> > wandb.init(
> >     project=cfg.project,
> >     name=cfg.name,
> >     notes=cfg.get("notes"),
> >     tags=cfg.get("tags", []),
> >     config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
> > )
> >
> > # cs336_basics/train.py
> > run_started_at = now_iso()
> > timer = Timer(device)
> > tokens_per_step = batch_size * gradient_accumulation_steps * max_seq_len
> >
> > for step in range(1, total_steps + 1):
> >     step_start = timer.measure_start()
> >     loss = train_one_optimizer_step()
> >     step_sec = timer.measure_end(step_start)
> >
> >     tokens_seen = step * tokens_per_step
> >     elapsed_sec = timer.elapsed_sec()
> >
> >     if should_eval(step):
> >         eval_start = timer.measure_start()
> >         val_loss = compute_validation_loss(...)
> >         eval_sec = timer.measure_end(eval_start)
> >
> >         wandb.log(
> >             {
> >                 "step": step,
> >                 "tokens_seen": tokens_seen,
> >                 "time/elapsed_sec": timer.elapsed_sec(),
> >                 "eval/loss": val_loss,
> >                 "eval/perplexity": math.exp(val_loss),
> >                 "eval/time_sec": eval_sec,
> >             },
> >             step=step,
> >         )
> >
> >     if should_log(step):
> >         wandb.log(
> >             {
> >                 "step": step,
> >                 "tokens_seen": tokens_seen,
> >                 "time/elapsed_sec": elapsed_sec,
> >                 "time/step_sec": recent_train_sec / recent_steps,
> >                 "time/tokens_per_sec": recent_tokens / recent_train_sec,
> >                 "train/loss": loss,
> >                 "train/lr": current_lr,
> >             },
> >             step=step,
> >         )
> >
> >     if should_checkpoint(step):
> >         save_checkpoint(
> >             model,
> >             optimizer,
> >             step,
> >             checkpoint_path,
> >             metadata={
> >                 "started_at": run_started_at,
> >                 "saved_at": now_iso(),
> >                 "elapsed_sec": timer.elapsed_sec(),
> >                 "tokens_seen": tokens_seen,
> >             },
> >         )
> > ```
> >
> > The most important design choice is to keep several progress axes:
> >
> > - `step`: the optimizer step, used as the default W&B step;
> > - `tokens_seen`: the total number of tokens processed so far;
> > - `time/elapsed_sec`: wall-clock time since the beginning of training.
> >
> > This lets me inspect loss curves in different ways. `loss vs step` is useful for checking optimizer behavior, `loss vs tokens_seen` is better for comparing runs with different batch sizes or context lengths, and `loss vs time/elapsed_sec` shows real training efficiency.
> >
> > The training loop records:
> >
> > - `train/loss`;
> > - `train/lr`;
> > - `eval/loss`;
> > - `eval/perplexity`;
> > - `eval/time_sec`;
> > - `time/elapsed_sec`;
> > - `time/step_sec`;
> > - `time/tokens_per_sec`;
> > - `checkpoint/time_sec`;
> > - `tokens_seen`.
> >
> > I also save timing metadata in checkpoints, including `started_at`, `saved_at`, `elapsed_sec`, and `tokens_seen`. This does not affect resuming training, but it makes the checkpoint easier to inspect later.
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
