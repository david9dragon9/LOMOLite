# 💡 LOMOLite: A Lightweight Adapter For Low-Memory Optimization

Train full-parameter 7B language models with only 22 GB of GPU RAM!

LOMO and AdaLOMO are low-memory optimization methods that use in-place gradient updates to significantly decrease the amount of GPU memory necessary.

This repository, based off of the LOMO and collie repositories from OpenLMLab, allows for easy use of the LOMO and AdaLOMO optimizers. 

While the LOMO and collie repositories are bulky and hard to use (you have to integrate your code into their code), this repository allows you to simply take the LOMO and AdaLOMO optimizers and use them with few modifications.

# How to use
1. Add the following imports to your code:
```python
import sys; sys.path.append("/path/to/LOMOLite/")
from lomo.lomo_base import setup_lomo, create_lomo_lr_scheduler, Functor, LOMOBaseLite
```

2. (optional) Call `setup_lomo` for `transformers` models in order to receive a config with additional LOMO settings.
```python
config = setup_lomo(pretrained_model_name_or_path)
```

3. (optional) Run `create_lomo_lr_scheduler` to get a learning rate scheduler. Alternatively, you can pass in a single number as the learning rate.
```python
lr_scheduler = create_lomo_lr_scheduler(
            learning_rate=lr,
            n_steps=1000,
            num_train_epochs=10,
            warmup=0.1,
            lr_scheduler_type="linear",            
        )
```

4. Create your optimizer. `optimizer_name` is one of: `lomo` or `adalomo`, depending on which one you would like to use. 
```python
optimizer = LOMOBaseLite(
            optimizer_name, model, clip_grad_norm=1.0, clip_grad_value=None, lr_scheduler=lr_scheduler
        )
```

5. Subclass the `Functor` class and override the `forward` function to create a class that has a collection of attributes, and that, when run, runs a forward pass through the model and returns the loss. Then, instantiate your custom class by passing in keyword arguments that will be then be passed into your `forward` function.
```python
class MyFunctor(Functor):
    def forward(self, loss_fn, model, batch, train_config):
        loss = loss_fn(model, batch, train_config)
        return loss
```

6. During training, instead of directly calculating your loss, instead instantiate your custom class and pass in the functor to the `optimizer.step` method. This will return the loss value. **Make sure to pass in the optimizer as the model.**
```python
functor = MyFunctor(loss_fn=loss_fn, model=optimizer, batch=batch, train_config=train_config)
loss = optimizer.step(functor)
```

7. Instead of calling `torch.save(model.state_dict())`, call `optimizer.save_pretrained` with a `save_folder` path. The state dict will then be saved to a `pytorch_model.bin` file in the specified `save_folder`.
```python
optimizer.save_pretrained(save_path)
```

8. When running your python file, you can override the default environment variables: `LOCAL_RANK`, `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`. In most cases, you will not need to override any of these, though sometimes `MASTER_PORT` will be overridden if two processes are running on the same machine.
```python
MASTER_PORT=6001 python3 main.py ...
```

# Limitations
Currently LOMOLite only supports training on a single GPU, and by default is set to use `bfloat16` precision with no loss scaling.

# Roadmap
- Support multi-GPU training
- Integrate into PyTorch

# References
https://github.com/OpenLMLab/collie/tree/dev/collie

https://github.com/OpenLMLab/LOMO

[Full Parameter Fine-Tuning for Large Language Models with Limited Resources](https://arxiv.org/pdf/2306.09782.pdf)

[AdaLomo: Low-memory Optimization with Adaptive Learning Rate](https://arxiv.org/pdf/2310.10195.pdf)
