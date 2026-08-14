+++
title = 'Section 1: Assignment Overview'
date = '2026-08-12T18:09:26+08:00'
summary = 'Build a Transformer language model from its fundamental components.'
weight = 10
draft = false
+++

CS336 Assignment 1 asks us to build the complete path from raw text to a trained Transformer language model. Instead of assembling existing high-level PyTorch layers, we implement the important components ourselves, connect them into a reproducible training system, and then use experiments to understand how the choices interact.

---

# What We Will Implement

The implementation follows the data flow of a language model:

1. [Byte-Pair Encoding](../02-byte-pair-encoding/) converts raw UTF-8 text into a compact sequence of token IDs.
2. [Transformer Language Model Architecture](../03-transformer-language-model-architecture/) maps those token IDs to next-token logits using embeddings, attention, RoPE, RMSNorm, and SwiGLU blocks.
3. [Training a Transformer LM](../04-training-transformer-lm/) provides cross-entropy, AdamW, learning-rate scheduling, and gradient clipping.
4. [Training Loop](../05-training-loop/) joins the model, optimizer, data loader, checkpointing, configuration, and experiment logging into one executable system.

The result is not just a model definition. It is a minimal language-model training stack whose behavior can be inspected from tokenizer training through text generation.

---

# What We Will Run

The assignment uses two datasets with different purposes:

- **TinyStories** is small and structurally simple enough for rapid iteration. We train a tokenizer, encode the corpus, tune optimization settings, compare architectural choices, and generate short stories.
- **OpenWebText** is broader and noisier. It provides a more realistic pretraining task and is used for the final time-constrained training run.