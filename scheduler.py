import math


class CosineWarmupScheduler:
    """
    Cosine learning rate scheduler with linear warmup.

    LR increases linearly during warmup and then decays
    following a cosine curve.

    Commonly used in Transformer training.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, lr_max, lr_min=0):

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min

        self.step_num = 0

    def get_lr(self):

        if self.step_num < self.warmup_steps:

            # linear warmup
            lr = self.lr_max * self.step_num / self.warmup_steps

        else:

            progress = (self.step_num - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )

            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + math.cos(math.pi * progress)
            )

        return lr

    def step(self):

        self.step_num += 1

        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:

            param_group["lr"] = lr
