# Source: https://github.com/OpenLMLab/LOMO
# Source: https://github.com/OpenLMLab/collie/tree/dev/collie

import copy
from dataclasses import dataclass

import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.utils import PaddingStrategy
from transformers.trainer import *
import wandb


class LearningRateScheduler:
    r"""
    Learning rate scheduler with warmup.

        :param warmup: if ``warmup`` is an integer, ``warmup`` stands for warmup steps, if ``warmup`` is a float,
            such as 0.1, then it stands for warmup_ratio.
        :param schedule: the learning rate will be adjusted according to ``schedule`` strategy,
            which can be: linear or constant.
    """

    def __init__(
        self, warmup: float, schedule: str, learning_rate: float, n_steps: int = 0
    ):

        self.warmup = max(warmup, 0.0)
        self.schedule = schedule
        self.initial_lr = learning_rate

        if self.warmup > 1:
            self.warmup = self.warmup / n_steps
        self.t_steps = max(2, n_steps)

        if self.schedule == "constant":
            self.get_lr = self._get_constant_lr
        elif self.schedule == "linear":
            self.get_lr = self._get_linear_lr
        else:
            raise NotImplementedError("Only support 'linear', 'constant'.")

    def _get_constant_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1

    def _get_linear_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.0) / (self.warmup - 1.0), 0.0)

    def step(self, global_step):
        progress = global_step / self.t_steps
        return self.initial_lr * self.get_lr(progress)


class DynamicLossScaler:
    def __init__(
        self,
        init_scale=2**32,
        scale_factor=2.0,
        scale_window=1000,
        min_scale=1,
        delayed_shift=1,
        consecutive_hysteresis=False,
        raise_error_at_min_scale=True,
        dtype=torch.half,
    ):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis
        self.raise_error_at_min_scale = raise_error_at_min_scale
        self.dtype = dtype
        self.has_overflow_serial = False

    @property
    def loss_scale(self):
        return self.cur_scale

    # `x` is a torch.Tensor
    def _has_inf_or_nan(self, x):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum in [float("inf"), -float("inf")] or cpu_sum != cpu_sum:
                return True
            return False

    # `overflow` is boolean indicating whether the gradient overflowed
    def update_scale(self, overflow):
        if overflow:
            # self.cur_scale /= self.scale_factor
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                if (self.cur_scale == self.min_scale) and self.raise_error_at_min_scale:
                    raise Exception(
                        "Current loss scale already at minimum - cannot decrease scale anymore. Exiting run."
                    )
                else:
                    next_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
                    if torch.distributed.get_rank() == 0:
                        overflow_msg = f"[deepspeed] OVERFLOW! Rank {torch.distributed.get_rank()} Skipping step."
                        if self.dtype == torch.half:
                            overflow_msg += f" Attempted loss scale: {int(self.cur_scale)}, reducing to {int(next_scale)}"
                        print(overflow_msg)
                    self.cur_scale = next_scale
            else:
                if torch.distributed.get_rank() == 0:
                    overflow_msg = f"[deepspeed] OVERFLOW! Rank {torch.distributed.get_rank()} Skipping step."
                    if self.dtype == torch.half:
                        overflow_msg += f" Attempted loss scale: {int(self.cur_scale)}, but hysteresis is {self.cur_hysteresis}. Reducing hysteresis to {self.cur_hysteresis - 1}"
                    print(overflow_msg)
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                if torch.distributed.get_rank() == 0:
                    hysteresis_msg = f"Consecutive hysteresis is enabled. Restoring hysteresis to {self.delayed_shift}"
                    print(hysteresis_msg)
                self.cur_hysteresis = self.delayed_shift
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1


def get_loss(logits, labels, clip_loss_value=None):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    # Flatten the tokens
    if clip_loss_value is not None:
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
            shift_labels.view(-1).cuda(),
        )
        loss.data.clamp_(min=-clip_loss_value, max=clip_loss_value)
        loss = loss.mean()
    else:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
            shift_labels.view(-1).cuda(),
        )
    return loss
