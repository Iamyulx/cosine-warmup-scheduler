# Cosine Learning Rate Scheduler with Warmup

Implementation of a **Cosine Learning Rate Scheduler with Linear Warmup** in PyTorch.

This learning rate schedule is widely used in training **Transformer models** and modern deep learning systems.

## Motivation

Large neural networks often benefit from:

1. **Warmup phase**
   - Gradually increases the learning rate
   - Stabilizes early training

2. **Cosine decay**
   - Smoothly reduces learning rate
   - Improves convergence

## Learning Rate Schedule

Warmup phase:

LR increases linearly

Cosine decay phase:

LR follows a cosine function until reaching the minimum learning rate.

## Visualization

![Learning Rate Curve](lr_curve.png)

## Project Structure

scheduler.py → scheduler implementation  
train_demo.py → training example  
plot_schedule.py → LR visualization  

## Usage

```python
scheduler = CosineWarmupScheduler(
    optimizer,
    warmup_steps=1000,
    total_steps=10000,
    lr_max=1e-3,
    lr_min=1e-5
)




## Applications

This schedule is used in training models such as:

Transformer models

BERT

GPT-style architectures

Vision Transformers

## Requirements
torch
matplotlib
