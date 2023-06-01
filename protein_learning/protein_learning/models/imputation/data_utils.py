from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from protein_learning.common.data.model_data import ExtraInput
from protein_learning.common.data.protein import Protein
from protein_learning.common.protein_constants import AA_TO_INDEX
from protein_learning.networks.common.rigid import Rigids


class ExtraImpute(ExtraInput):
    """Store Encoded Native Sequence"""

    def __init__(self,
                 native_seq_enc: Tensor,
                 true_rigids: Rigids
                 ):
        super(ExtraImpute, self).__init__()
        self.native_seq_enc = native_seq_enc if native_seq_enc.ndim == 2 \
            else native_seq_enc.unsqueeze(0)
        self.true_rigids = true_rigids

    def crop(self, start, end) -> ExtraImpute:
        """Crop native seq. encoding"""
        self.native_seq_enc = self.native_seq_enc[:, start:end]
        self.true_rigids = self.true_rigids.crop(start,end)
        return self

    def to(self, device: Any) -> ExtraImpute:
        """Send native sequence encoding to device"""
        self.native_seq_enc = self.native_seq_enc.to(device)
        self.true_rigids = self.true_rigids.to(device)
        return self


def augment(decoy_protein: Protein, native_protein: Protein) -> ExtraDesign:  # noqa
    """Augment function for storing native seq. encoding in ModelInput object"""
    seq = native_protein.seq
    native_seq_enc = [AA_TO_INDEX[r] for r in seq]
    native_coords = native_protein.get_atom_coords(["N", "CA", "C"])
    true_rigids = Rigids.RigidFromBackbone(native_coords.unsqueeze(0))
    return ExtraImpute(torch.tensor(native_seq_enc).long(), true_rigids=true_rigids)
