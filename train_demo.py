import torch
import torch.nn as nn
import torch.optim as optim

from scheduler import CosineWarmupScheduler


model = nn.Linear(10, 1)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

scheduler = CosineWarmupScheduler(
    optimizer,
    warmup_steps=1000,
    total_steps=10000,
    lr_max=1e-3,
    lr_min=1e-5
)


for step in range(10000):

    optimizer.zero_grad()

    x = torch.randn(32, 10)

    y = model(x).sum()

    y.backward()

    optimizer.step()

    scheduler.step()
