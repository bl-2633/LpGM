"""Protein dataset"""
from typing import Any
from protein_learning.common.data.protein import Protein
from protein_learning.common.data.model_data import ExtraInput
from protein_learning.networks.common.rigid import Rigids


class ExtraRefine(ExtraInput):
    """Store True rigids"""

    def __init__(self,
                 true_rigids: Rigids,
                 ):
        super(ExtraRefine, self).__init__()
        self.true_rigids = true_rigids

    def crop(self, start, end) -> ExtraInput:
        """Crop rigids"""
        self.true_rigids.translations = self.true_rigids.translations[:, start:end, :]
        self.true_rigids.quaternions = self.true_rigids.quaternions[:, start:end, :]
        return self

    def to(self, device: Any) -> ExtraInput:
        """Send rigids to device"""
        self.true_rigids = self.true_rigids.to(device)
        return self


def augment(decoy_protein: Protein, native_protein: Protein) -> ExtraMDS:  # noqa
    return ExtraRefine(true_rigids=Rigids.RigidFromBackbone(native_protein.atom_coords.unsqueeze(0)))
