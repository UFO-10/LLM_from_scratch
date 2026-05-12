# Transformer From Scratch in PyTorch

This project implements a GPT-style Transformer language model from scratch using PyTorch.  
The focus is on **clarity, intuition, and understanding how modern LLMs work internally**, rather than using high-level libraries.

The project walks through the full pipeline of building a language model — from raw text to text generation.

---

# Learning Path (Recommended Order)

Follow the notebooks in the exact order below to understand the full system step-by-step.

---

## 1️⃣ Transformer Architecture
📁 `notebooks/transformer_architectures.ipynb`

Implements the core building blocks of a Transformer:

- Linear layers
- Token + positional embeddings
- RMS normalization
- SiLU activation
- Self-attention mechanism
- Multi-head attention
- KV-cache (for fast autoregressive inference)
- Feed-forward network (MLP)
- Transformer block
- Full LLM assembly

👉 **Goal:** Understand how a GPT-style model is built internally.

---

## 2️⃣ Tokenization / Pretokenization
📁 `notebooks/pretokenize_data.ipynb`

Converts raw text into token IDs:

- Byte Pair Encoding (BPE) (educational implementation)
- GPT-2 tokenizer (`tiktoken`)
- Token vocabulary creation
- Conversion of text → token sequences
- Saving tokenized dataset as `.bin` files

👉 **Goal:** Transform raw text into model-ready numerical data.

---

## 3️⃣ Data Loading
📁 `notebooks/dataloader.ipynb`

Prepares data for training:

- Loads pretokenized binary files
- Creates input–target pairs `(X, Y)`
- Batching sequences for training
- Efficient memory handling

👉 **Goal:** Feed structured training data into the model.

---

## 4️⃣ Optimization
📁 `notebooks/optimizer.ipynb`

Implements training mechanics:

- Loss function (cross-entropy)
- Gradient descent intuition
- SGD vs Adam optimizer

👉 **Goal:** Understand how neural networks actually learn.

---

## 6️⃣ Text Generation
📁 `notebooks/generation.ipynb`

Uses the trained model to generate text:

- Prompt → generated output

👉 **Goal:** Generate coherent text using the trained model.

---

## 5️⃣ Full Training Pipeline
📁 `notebooks/Train_LLM.ipynb`

Full training pipeline:

- Model initialization
- Data loading
- Forward + backward pass
- Loss tracking
- Weight updates
- Generate texts from a prompt

👉 **Goal:** Train a working Transformer language model.

---
