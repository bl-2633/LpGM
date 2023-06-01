from functools import partial
from typing import Union, Iterable, Callable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
from torch import Tensor, nn
from enum import Enum


class EvalTy(Enum):
    """Evaluation type (used to determine settings during model training)"""
    TRAINING, TESTING, VALIDATION = "training", 'testing', 'validation'


def get_dataloader(
        dataset: Union[Dataset, None],
        batch_size: int,
        shuffle: bool = True,
        n_workers: int = 3
) -> Union[None, DataLoader]:
    """Gets a PyTorch DataLoader for the given dataset"""
    if dataset is None:
        return None
    return DataLoader(  # noqa
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(n_workers, 1),
        drop_last=False,
        prefetch_factor=2 * batch_size // n_workers,
    )


def linear_update_func(batch_size, warmup_steps, batch_idx) -> float:
    """linear update function"""
    return min(1, (1 + batch_idx * batch_size) / warmup_steps)


def update_func(batch_size, warmup_steps, ty='LINEAR') -> Callable:
    """Update function"""
    if warmup_steps <= 0:
        return lambda *args: 1
    if ty == 'LINEAR':
        return partial(linear_update_func, batch_size, warmup_steps)
    raise Exception(f'got unknown update func type : {ty}')


def safe_round(x, decimals=3) -> Union[int, float, np.ndarray]:
    """Rounds x"""
    if isinstance(x, torch.Tensor):
        x = x.item()
    if isinstance(x, int):
        return x
    return np.round(x, decimals)


def _valid_grad_params(model: nn.Module) -> Iterable:
    """gets all parameters of model with gradient attribute"""
    return filter(lambda p: p is not None and p.grad is not None, model.parameters())


def get_grad_norm(model: nn.Module) -> Tensor:
    """Get the (global) norm of the model's gradients"""
    return torch.sqrt(sum([torch.sum(torch.square(p.grad)) for p in _valid_grad_params(model)]))


def check_nan_in_grad(model: nn.Module) -> bool:
    """Determines if any model gradients are nan"""
    return any([torch.any(torch.isnan(p.grad)) for p in _valid_grad_params(model)])