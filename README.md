## Transformer From Scratch in PyTorch

This project implements a GPT-style Transformer language model from scratch using PyTorch, with a focus on clarity, educational value, and understanding how modern LLMs work internally.

The project walks through the complete pipeline of building a language model:

- Transformer architecture
- Pretokenization
- Data loading
- Optimization
- Training
- Text generation
- Project Structure
.
├── transformer_architectures.ipynb
├── pretokenize_data.ipynb
├── dataloader.ipynb
├── optimizer.ipynb
├── generation.ipynb
└── README.md

## Features

- Transformer Components
- Linear layer implementation
- Token embeddings
- Positional embeddings
- RMSNorm
- SiLU activation
- Self-attention
- Multi-head attention
- KV-cache
- Feed-forward MLP
- Transformer blocks
- Causal masking
- Training Components
- Cross entropy loss
- Adam optimizer
- Gradient-based optimization
- Data loading pipeline
- Tokenization
- Educational BPE tokenizer implementation
- GPT-2 tokenizer via tiktoken
- Pretokenized binary datasets
- Inference
- Autoregressive generation
- Temperature sampling
- KV-cache decoding

## Requirements

Install dependencies:

pip install torch numpy matplotlib tiktoken huggingface_hub

Step-by-Step Workflow

1. Build the Transformer Architecture

Open:

transformer_architectures.ipynb

This notebook implements:

attention
MLP
transformer blocks
RMSNorm
KV-cache
full LLM architecture

Goal:
Understand how Transformer models are built internally.

2. Pretokenize the Dataset

Open:

pretokenize_data.ipynb

This notebook:

downloads TinyStories
loads the GPT-2 tokenizer
converts text into token IDs
writes tokens into binary .bin files

Example:

tokenizer = tiktoken.get_encoding("gpt2")

Goal:
Prepare training data efficiently before training.

3. Load Training Data

Open:

dataloader.ipynb

This notebook:

loads tokenized .bin files
creates batches
constructs (X, Y) token prediction pairs

Where:

X = input tokens
Y = next-token targets

Goal:
Feed training sequences into the Transformer.

4. Optimization and Training

Open:

optimizer.ipynb

This notebook:

implements SGD and Adam
explains gradient descent
trains the Transformer model
computes cross entropy loss

Goal:
Learn how neural networks optimize parameters during training.

5. Generate Text

Open:

generation.ipynb

This notebook:

loads the trained model
generates text autoregressively
uses temperature sampling
demonstrates KV-cache inference

Goal:
Use the trained Transformer as a language model.

## Training Pipeline

Raw Text
   ↓
GPT-2 Tokenizer (tiktoken)
   ↓
Pretokenized Binary Files (.bin)
   ↓
DataLoader
   ↓
Transformer LLM
   ↓
Training
   ↓
Text Generation

## Goal
This project is designed to help understand:

how tokenization works
how attention works
how transformers process sequences
how language models train
how autoregressive generation works
why Adam optimization is commonly used

## Requirements

- Python 3.8+
- PyTorch