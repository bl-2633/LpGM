"""Feature representation
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import torch
from torch import Tensor

from protein_learning.common.helpers import safe_to_device, exists


class FeatureTy(Enum):
    """Feature Type flag
    """
    SCALAR, COORD, PAIR = 1, 2, 3


class Feature:
    """Represents a single feature
    """

    def __init__(
            self,
            raw_data: Any,
            encoded_data: Any,
            name: str,
            ty: FeatureTy,
            dtype: torch.dtype = torch.float32,
            n_classes: Optional[int] = None,
    ):
        self.raw_data, self.encoded_data = raw_data, encoded_data
        self.name = name
        self.ty, self.dtye = ty, dtype
        self.n_classes = n_classes
        self.masked = False

    def to(self, device: Any) -> Feature:
        """maps underlying data to given device
        :param device: the device to map to
        """
        self.raw_data = safe_to_device(self.raw_data, device)
        self.encoded_data = safe_to_device(self.encoded_data, device)
        return self

    def get_raw_data(self):
        """Returns features raw data"""
        if self.masked:
            print(f"[WARNING] : feature {self.name} has been masked, raw data is not safe to use!")
        return self.raw_data

    def get_encoded_data(self):
        """Returns features encoded data"""
        return self.encoded_data.long()

    @property
    def encoded_shape(self):
        """Returns shape of underlying encoded data object"""
        return self.encoded_data.shape if torch.is_tensor(self.encoded_data) else None

    @property
    def raw_shape(self):
        """Returns shape of underlying raw data object"""
        return self.raw_data.shape if torch.is_tensor(self.raw_data) else None

    def __len__(self):
        if exists(self.encoded_data):
            idx = 1 if self._has_batch_dim(self.encoded_data) else 0
            return self.encoded_data.shape[idx]
        raise Exception(f"no encoded data found for feature {self.name}")

    def add_batch(self, other: Feature):
        """Adds the other features to this feature"""
        raise Exception("not implemented")

    def _maybe_add_batch(self, feat: Any) -> Any:
        if not exists(feat) or not torch.is_tensor(feat):
            return feat
        return feat if self._has_batch_dim(feat) else feat.unsqueeze(0)

    def maybe_add_batch(self) -> Feature:
        """Adds batch dimension if not present"""
        self.raw_data = self._maybe_add_batch(self.raw_data)
        self.encoded_data = self._maybe_add_batch(self.encoded_data)
        return self

    def _has_batch_dim(self, feat):
        if not exists(feat) or not torch.is_tensor(feat):
            return False
        if self.ty == FeatureTy.PAIR:
            if feat.ndim == 3:
                return False
            assert feat.ndim == 4
            return True
        if self.ty == FeatureTy.SCALAR:
            if feat.ndim == 2:
                return False
            assert feat.ndim == 3, f"[{self.name}] expected feature dimension 3, got shape : {feat.shape}"
            return True
        raise Exception("this line should be unreachable!")

    def _crop(self, feat: Any, start: int, end: int) -> Any:
        if not exists(feat) or not torch.is_tensor(feat):
            return feat
        if self.ty == FeatureTy.PAIR:
            return feat[..., start:end, start:end, :]
        elif self.ty == FeatureTy.SCALAR:
            return feat[..., start:end, :]
        else:
            raise Exception("not implemented")

    def crop(self, start, end) -> Feature:
        """crop the feature from start..end"""
        self.raw_data = self._crop(self.raw_data, start, end)
        self.encoded_data = self._crop(self.encoded_data, start, end)
        return self

    def apply_mask(self, mask: Tensor):
        """Apply mask to feature information"""
        assert mask.shape[0] == len(self)
        if self._has_batch_dim(self.encoded_data):
            mask = mask.unsqueeze(0)
        assert self.encoded_shape[:mask.ndim] == mask.shape, \
            f"{self.name}: {self.encoded_shape}, {mask.shape}"
        self.encoded_data[mask] = self.n_classes
        self.masked = True
