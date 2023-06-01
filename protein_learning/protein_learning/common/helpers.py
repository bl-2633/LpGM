import torch
from typing import Optional
from torch import Tensor
from contextlib import contextmanager

get_eps = lambda x: torch.finfo(x.dtype).eps  # noqa
get_max_val = lambda x: torch.finfo(x.dtype).max  # noqa
get_min_val = lambda x: torch.finfo(x.dtype).min  # noqa


def maybe_add_batch(x: Tensor, unbatched_dim: int) -> Tensor:
    """Unsqueezes first dimension iff x.ndim!=unbatched_dim"""
    assert x.ndim >= unbatched_dim
    return x.unsqueeze(0) if x.ndim == unbatched_dim else x


def safe_norm(x: Tensor, dim: int, keepdim: bool = False, eps: float = 1e-12):
    """Safe norm of a vector"""
    return torch.sqrt(torch.sum(torch.square(x), dim=dim, keepdim=keepdim) + eps)


def safe_normalize(x: Tensor, eps: float = 1e-12):
    """Safe normalization of a vector"""
    return x / safe_norm(x, dim=-1, keepdim=True, eps=eps)


def parse_bool(x: str) -> bool:
    x = x.strip().lower()
    return x == "true" or x == "1"


def exists(x):
    """Returns true iff x is not None.
    """
    return x is not None


def default(x, y):
    """Returns x if x exists, otherwise y.
    """
    return x if x is not None else y


def to_tensor(x, dtype=None) -> Optional[torch.Tensor]:
    """Converts x to a tensor.
    x can be a list, numpy.ndarray, or tensor.

    :param x: object to convert to tensor
    :param dtype: dtype of return tensor
    :return: x as a tensor
    """
    if x is None:
        return None
    dtype = torch.float32 if dtype is None else dtype
    if torch.is_tensor(x):
        return x.type(dtype)
    else:
        return torch.tensor(x, dtype=dtype)


def safe_to_device(x, device):
    """maps x to given device iff x is a tensor"""
    return x.to(device) if torch.is_tensor(x) else x


def masked_mean(tensor, mask, dim=-1, keepdim=False):
    """Performs a masked mean over a given dimension
    :param tensor: tensor to apply mean to
    :param mask: mask to use for calculating mean
    :param dim: dimension to extract mean over
    :param keepdim: keep the dimension where mean is taken
    :return: masked mean of tensor according to mask along dimension dim.
    """
    if not exists(mask):
        return torch.mean(tensor, dim=dim, keepdim=keepdim)
    assert tensor.ndim == mask.ndim
    tensor = torch.masked_fill(tensor, ~mask, 0)
    total_el = mask.sum(dim=dim, keepdim=keepdim)
    mean = tensor.sum(dim=dim, keepdim=keepdim) / total_el.clamp(min=1.)
    mean = mean.masked_fill(total_el == 0, 0.)
    return mean


def batched_index_select(values: Tensor, indices: Tensor, dim: int = 1) -> Tensor:
    """Selects (from values) the data at given indices
    :param values: values to select from
    :param indices: the indices to select for each (sub) value
    :param dim: the dimension to select from
    :return: tensor with selected (or sub-selected) values according to indices.
    """
    assert dim >= 0, f"ERROR: batched index selection requires dim argument to be " \
                     f"non-negative!, got {dim}"
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))  # noqa
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


def calc_tm_torch(deviations: Tensor, norm_len: Optional[int] = None) -> Tensor:
    """Calculate TM-score based on residue-wise distance deviations
    :param deviations: distance deviations
    :param norm_len: length to normalize by
    :return: tm score (0-1)
    """
    norm_len = norm_len if norm_len else len(deviations.numel())
    d0 = 0.5 if norm_len <= 15 else 1.24 * ((norm_len - 15.0) ** (1. / 3.)) - 1.8
    return torch.mean(1 / (1 + (torch.square(deviations) / (d0 ** 2))))


@contextmanager
def disable_tf32():
    """temporarily disable 32-bit float ops"""
    orig_value = torch.backends.cuda.matmul.allow_tf32  # noqa
    torch.backends.cuda.matmul.allow_tf32 = False  # noqa
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_value  # noqa
