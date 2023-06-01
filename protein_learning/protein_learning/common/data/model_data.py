"""Input for protein-based learning model
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from typing import Optional, Tuple, List, Union, Any
from protein_learning.features.input_features import InputFeatures
from protein_learning.common.helpers import default, exists
from protein_learning.common.protein_constants import BB_ATOMS, SC_ATOMS
from protein_learning.common.data.protein import Protein
from protein_learning.protein_utils.align.kabsch_align import kabsch_align
from einops import rearrange
import numpy as np
import random

try:
    from functools import cached_property  # noqa
except:  # noqa
    from cached_property import cached_property

SC_ATOM_SET, BB_ATOM_SET = set(BB_ATOMS), set(SC_ATOMS)


class ExtraInput(ABC):
    """Extra information to augment ModelInput with

    Must define "crop" and "to"
    """

    def __init__(self):
        pass

    @abstractmethod
    def crop(self, start, end) -> ExtraInput:
        """Crop data from start..end"""
        pass

    @abstractmethod
    def to(self, device: Any) -> ExtraInput:
        """Place data on given device"""
        pass


class ModelInput:
    """Input for protein-based learning model"""

    def __init__(
            self,
            decoy: Protein,
            native: Protein,
            input_features: InputFeatures,
            extra: Optional[ExtraInput] = None,
    ):
        self.decoy, self.native = decoy, native
        self.input_features = input_features
        self.extra = extra
        self.crop_posns = None

    def crop(self, max_len) -> ModelInput:
        """Randomly crop model input to max_len"""
        start, end = 0, len(self.decoy.seq)
        start = random.randint(0, (end - max_len)) if end > max_len else start
        end = min(end, start + max_len)
        self.crop_posns = (start, end)
        self.input_features = self.input_features.crop(start=start, end=end)
        self.decoy = self.decoy.crop(start, end)
        self.native = self.native.crop(start, end)
        self.extra = self.extra.crop(start, end) if exists(self.extra) else None
        return self

    def to(self, device: Any) -> ModelInput:
        """Places all data on given device"""
        self.input_features = self.input_features.to(device)
        self.decoy = self.decoy.to(device)
        self.native = self.native.to(device) if exists(self.native) else None
        self.extra = self.extra.to(device) if exists(self.extra) else None
        return self

    def _get_protein(self, native: bool = False, decoy: bool = False) -> Protein:
        """Gets native or decoy protein"""
        assert native ^ decoy
        return self.native if native else self.decoy

    def bb_atom_tys(self, native: bool = False, decoy: bool = False):
        """Backbone atom types for native or decoy"""
        return self._get_protein(native=native, decoy=decoy).bb_atom_tys

    def sc_atom_tys(self, native: bool = False, decoy: bool = False):
        """Side-chain atom types for native or decoy"""
        return self._get_protein(native=native, decoy=decoy).sc_atom_tys

    def get_atom_coords(
            self,
            atom_tys: Union[str, List[str]],
            native: bool = False,
            decoy: bool = False,
            coords: Optional[Tensor] = None
    ) -> Tensor:
        """Gets the atom coordinates for the given atom types

        Returns: Tensor of shape (...,n,3) if atom_tys is a string, otherwise a tensor of shape
        (...,n,a,3) where a is the number of atom_tys given.
        """
        protein = self._get_protein(native=native, decoy=decoy)
        coords = default(coords, protein.atom_coords)
        return protein.get_atom_coords(atom_tys=atom_tys, coords=coords)

    def get_atom_masks(
            self,
            atom_tys: Union[str, List[str]],
            native: bool = False,
            decoy: bool = False,
    ) -> Tensor:
        """Gets the atom masks for the given atom types

        Returns: Tensor of shape (...,n) if atom_tys is a string, otherwise a tensor of shape
        (...,n,a) where a is the number of atom_tys given.
        """
        return self._get_protein(native=native, decoy=decoy).get_atom_masks(atom_tys)

    def get_atom_coords_n_masks(
            self,
            atom_tys: Union[str, List[str]],
            native: bool = False,
            decoy: bool = False,
            coords: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Get coords and masks for given atom types"""
        coords = self.get_atom_coords(native=native, decoy=decoy, coords=coords, atom_tys=atom_tys)
        masks = self.get_atom_masks(native=native, decoy=decoy, atom_tys=atom_tys)
        return coords, masks

    def __getattr__(self, attr):
        """Called only if this class does not have the given attribute"""
        try:
            return getattr(self.extra, attr)
        except:
            raise AttributeError(f"No attribute {attr} found for this class")


class ModelOutput:
    """Model Output"""

    def __init__(
            self,
            predicted_coords: Tensor,
            scalar_output: Tensor,
            pair_output: Tensor,
            model_input: ModelInput,
            predicted_atom_tys: Optional[List[str]] = None,
            extra: Optional = None,
    ):
        self.predicted_coords = predicted_coords
        self.predicted_atom_tys = predicted_atom_tys
        self.scalar_output = scalar_output
        self.pair_output = pair_output
        self.model_input = model_input
        self.extra = extra

    @property
    def native_protein(self):
        """Input native protein"""
        return self.model_input.native

    @property
    def decoy_protein(self):
        """Input decoy protein"""
        return self.model_input.decoy

    @property
    def valid_residue_mask(self):
        return self.native_protein.valid_residue_mask & self.decoy_protein.valid_residue_mask

    def get_atom_coords(
            self,
            native: bool = False,
            decoy: bool = False,
            atom_tys: Optional[Union[str, List[str]]] = None,
            coords: Optional[Tensor] = None,
    ) -> Tensor:
        """Get native or decoy atom coordinates for given atom types"""
        assert native ^ decoy
        protein = self.native_protein if native else self.decoy_protein
        coords = default(coords, protein.atom_coords)
        return protein.get_atom_coords(atom_tys=atom_tys, coords=coords)

    def get_atom_mask(
            self,
            native: bool = False,
            decoy: bool = False,
            atom_tys: Optional[Union[str, List[str]]] = None,

    ) -> Tensor:
        """Get native or decoy atom masks for given atom types"""
        assert native ^ decoy
        protein = self.native_protein if native else self.decoy_protein
        return protein.get_atom_masks(atom_tys=atom_tys)

    def get_native_n_decoy_coords_n_masks(
            self,
            atom_tys: Optional[List[str]] = None,
            align_by_kabsch: bool = False,

    ) -> Tuple[Tensor, ...]:
        """Gets coordinates and masks from model output for given atom types"""
        pred_coords = self.predicted_coords
        native_coords = self.get_atom_coords(native=True, atom_tys=atom_tys).unsqueeze(0)
        pred_coords = self.get_atom_coords(decoy=True, atom_tys=atom_tys, coords=pred_coords)
        native_mask = self.get_atom_mask(native=True, atom_tys=atom_tys).unsqueeze(0)
        pred_mask = self.get_atom_mask(decoy=True, atom_tys=atom_tys).unsqueeze(0)
        joint_mask = torch.logical_and(native_mask, pred_mask)
        assert native_coords.shape == pred_coords.shape
        assert native_mask.shape == pred_mask.shape
        assert native_mask.shape == pred_coords.shape[:3]
        if align_by_kabsch:
            tmp, native_coords, mask = map(lambda x: rearrange(x, "b n a ... -> b (n a) ..."),
                                           (pred_coords, native_coords, joint_mask))
            _, native_coords = kabsch_align(
                align_to=tmp, align_from=native_coords, mask=mask
            )
            native_coords = rearrange(
                native_coords, "b (n a) c -> b n a c", n=pred_coords.shape[1]
            )

        return pred_coords, native_coords, joint_mask

    @property
    def seq_len(self):
        """Sequence length"""
        return len(self.native_protein.seq)

    @property
    def input_masks(self) -> Optional[Tuple[Optional[Tensor], Optional[Tensor]]]:
        """Get masks for input sequence and features"""
        return self.model_input.input_features.masks

    def __getattr__(self, attr):
        """Called only if this class does not have the given attribute"""
        try:
            if isinstance(self.extra, dict):
                return self.extra[attr]
            return getattr(self.extra, attr)
        except:
            raise AttributeError(f"No attribute {attr} found for this class")


class ModelLoss:
    """Tracks model loss"""

    def __init__(self, seq_len: Optional[int] = None, pdb: Optional[str] = None):
        self.loss_dict = {}
        self.seq_len = seq_len
        self.pdb = pdb

    def add_loss(
            self,
            loss: Tensor,
            loss_weight: float = 1,
            baseline: Union[Tensor, float] = 0,
            loss_name: Optional[str] = None,
    ):
        """Add loss term"""
        loss_name = default(loss_name, f"loss_{len(self.loss_dict)}")
        assert loss.numel() == 1, f"[{loss_name}] expected scalar loss, got shape {loss.shape}"
        assert loss_name not in self.loss_dict
        self.loss_dict[loss_name] = dict(
            raw_loss=loss,
            loss_val=loss * loss_weight,
            loss_weight=loss_weight,
            baseline=baseline
        )

    def display_loss(self):
        """Print loss values for each term"""
        item = lambda x: np.round(x.detach().cpu().item() if torch.is_tensor(x) else x, 4)
        if exists(self.seq_len):
            print(f"model pdb : {self.pdb}, sequence length : {self.seq_len}")
        for name, vals in self.loss_dict.items():
            print(f"[{name}] : baseline : {item(vals['baseline'])}, "
                  f"actual : {item(vals['raw_loss'])} "
                  f"loss_val : {item(vals['loss_val'])}"
                  )

    def get_loss(self) -> Tensor:
        """Gets (weighted) loss value"""
        return sum(v["loss_val"] for v in self.loss_dict.values() if v["loss_weight"] != 0)
