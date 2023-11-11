# Source: https://github.com/OpenLMLab/LOMO
# Source: https://github.com/OpenLMLab/collie/tree/dev/collie

import torch
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import AutoConfig
import sys
import os
from collections import OrderedDict
import tqdm
import deepspeed
from lomo.lomo_orig import LOMO
from lomo.adalomo_orig import AdaLomo
from lomo.lomo_utils import LearningRateScheduler, DynamicLossScaler

try:
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.accelerator import get_accelerator
except:
    pass


def setup_lomo(model_name_or_path):
    torch.set_default_dtype(torch.float16)
    ds_config = __file__.replace("lomo_base.py", "ds_config.json")
    dschf = HfDeepSpeedConfig(ds_config)
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.gradient_checkpointing = True
    return config


def create_lomo_lr_scheduler(
    learning_rate=0.03,
    n_steps=1000,
    num_train_epochs=10,
    warmup=0.1,
    lr_scheduler_type="linear",
):
    return LearningRateScheduler(
        learning_rate=learning_rate,
        warmup=warmup,
        schedule=lr_scheduler_type,
        n_steps=n_steps,
    )


name_to_lomo = {
    "lomo": LOMO,
    "adalomo": AdaLomo,
}


class Functor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self):
        return self.forward(**self.kwargs)

    def forward(self, **kwargs):
        pass


def setup_env_vars():
    import os

    default_vals = {
        "LOCAL_RANK": 0,
        "RANK": 0,
        "WORLD_SIZE": 1,
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": 6001,
    }
    for k, v in default_vals.items():
        if k not in os.environ:
            os.environ[k] = str(v)


class LOMOBaseLite:
    def __init__(
        self,
        optimizer_name,
        model,
        clip_grad_norm=1.0,
        clip_grad_value=None,
        lr_scheduler=None,
    ):
        self.allow_print = True

        if "deepspeed" not in sys.modules:
            raise ModuleNotFoundError(
                "Detected DeepSpeed is not installed. See https://github.com/microsoft/DeepSpeed"
            )

        # Initialize deepspeed engine
        setup_env_vars()
        ds_config = __file__.replace("lomo_base.py", "ds_config.json")
        deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)
        self.model, _, _, _ = deepspeed.initialize(
            config=ds_config,
            model=model,
        )

        # setup learning rate
        self.lr_scheduler = None if isinstance(lr_scheduler, float) else lr_scheduler
        self.lr = lr_scheduler if isinstance(lr_scheduler, float) else 0.0

        self.optimizer = name_to_lomo[optimizer_name](
            model,
            lr=self.lr,
            clip_grad_norm=clip_grad_norm,
            clip_grad_value=clip_grad_value,
        )

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        self.global_step = 0
        self.optimizer.loss_scaler = None
        get_accelerator().empty_cache()

    def step(self, functor):
        loss = functor()
        self.model.train()
        # update the learning rate
        if self.lr_scheduler is not None:
            self.lr = self.lr_scheduler.step(self.global_step)
        self.global_step += 1
        if self.clip_grad_norm is not None and self.clip_grad_norm > 0:
            self.optimizer.grad_norm(loss)

            if (
                self.optimizer.loss_scaler
                and self.optimizer.loss_scaler.has_overflow_serial
            ):
                print(f"Gradient overflow, skipping step {self.global_step}")
                self.model.optimizer.get_param_coordinator(training=True).reset_step()
                return
            else:
                self.model.optimizer.get_param_coordinator(training=True).reset_step()
            # 第二次forward
            loss = functor()

        self.optimizer.fused_backward(loss, self.lr)
        self.model.optimizer.get_param_coordinator(training=True).reset_step()
        return loss

    def save_pretrained(self, output_dir):
        torch.distributed.barrier()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        state_dict = OrderedDict()
        shared_params = {}

        # Prepare for checkpoint save by ensuring all parameters are partitioned
        self.model.optimizer.partition_all_parameters()

        for name, param in self.model.module.named_parameters():
            with deepspeed.zero.GatheredParameters(param):
                if torch.distributed.get_rank() == 0:
                    # can't rely on param.data_ptr() as it will be reused as weights gets
                    # gathered and reduced, but param.ds_id is unique across all zero weights
                    # (and shared params will have the same param.ds_id)
                    if param.ds_id in shared_params:
                        # shared weights
                        state_dict[name] = state_dict[shared_params[param.ds_id]]
                    else:
                        state_dict[name] = param.detach().cpu()
                        shared_params[param.ds_id] = name

        if len(self.model.optimizer.persistent_parameters) > 0:
            self.model.optimizer.persistent_parameters[0].all_gather(
                self.model.optimizer.persistent_parameters
            )

        self.model.module.config.save_pretrained(output_dir)
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        print(f"Save model to {output_dir}")

        torch.distributed.barrier()
