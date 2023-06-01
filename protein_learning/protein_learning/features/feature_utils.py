"""Utility classes and functions for feature representations
"""
from typing import Optional, Union, List, Tuple, Dict, Any
from torch import Tensor, nn
import torch


def string_encode(
        mapping: Dict[str, int],
        *x,
        device: Any = "cpu"
) -> Tensor:
    """Encodes a string (or list of strings) according to the given mapping
    :param x: string(s) to encode
    :param mapping: map from string to integer defining encoding
    :param device: device to place output tensor on
    :return: encoded strings accordint to given mapping
    """
    assert all([len(el) == len(x[0]) for el in x]), "ERROR: all strings must have same length"
    out = torch.tensor([[mapping[pos] for pos in el] for el in x], device=device)
    return out[0] if len(x) == 1 else out


def fourier_encode(x: Tensor, num_encodings=4, include_self=True) -> Tensor:
    """Applies fourier encoding (sin + cos scaled by freq.) to input x

    :param x: tensor to apply encoding to
    :param num_encodings: number of frequencies to encode for (1,...1/2**(num_encodings-1))
    :param include_self: whether to append x[...-1] to encodings
    :return: fourier encoding of x
    """
    trailing_one = x.shape[-1] == 1
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x.squeeze(-2) if trailing_one else x


def bin_encode(data: Tensor, bins: Tensor):
    """Assigns each value in data to
    :param data: the data to apply bin encoding to
    :param bins: description of bin positions to encode into
        [(bins[i],bins[i+1])] is used to define each position.
    :return: bin index of each value in input data
    """
    assert torch.min(data) >= bins[0] and torch.max(data) < bins[-1], \
        f"incorrect bins, got min/max of data: ({torch.min(data)},{torch.max(data)})\n" \
        f"but bin min/max = ({bins[0]},{bins[-1]}])"
    binned_data = -torch.ones_like(data)
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        mask = torch.logical_and(data >= low, data < high)
        binned_data[mask] = i
    return binned_data.long()
