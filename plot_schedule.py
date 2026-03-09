import matplotlib.pyplot as plt
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


lrs = []

for _ in range(10000):

    scheduler.step()

    lrs.append(optimizer.param_groups[0]["lr"])


plt.plot(lrs)

plt.xlabel("Step")

plt.ylabel("Learning Rate")

plt.title("Cosine LR Schedule with Warmup")

plt.savefig("lr_curve.png")

plt.show()
