"""Generator for design features"""
from __future__ import annotations

from typing import List, Callable, Any

import numpy as np
import torch
from torch import Tensor

from protein_learning.common.data.model_data import ExtraInput
from protein_learning.common.data.protein import Protein
from protein_learning.common.protein_constants import AA_TO_INDEX
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.generator import FeatureGenerator
from protein_learning.features.input_features import InputFeatures, get_input_features
from protein_learning.features.maked_feature_generator import apply_mask_to_seq


class DesignFeatureGenerator(FeatureGenerator):
    """Generator for Design Features"""

    def __init__(
            self,
            config: InputFeatureConfig,
            mask_strategies: List[Callable[..., str]],
            strategy_weights: List[float],

    ):
        super(DesignFeatureGenerator, self).__init__(config=config)
        self.mask_strategies = mask_strategies
        self.strategy_weights = np.array(strategy_weights) / sum(strategy_weights)
        assert len(strategy_weights) == len(mask_strategies)

    def mask_sequence(self, seq: str, coords: Tensor) -> str:
        """Masks sequence"""
        strategy_idx = np.random.choice(len(self.strategy_weights), p=self.strategy_weights)
        return apply_mask_to_seq(seq, self.mask_strategies[strategy_idx](len(seq), coords))

    def generate_features(
            self,
            seq: str,
            coords: Tensor,
            res_ids: Tensor,
            coord_mask: Tensor,
            atom_tys: List[str],
    ) -> InputFeatures:
        """Generate input features for ProteinModel"""
        feats = get_input_features(
            seq=self.mask_sequence(seq, coords),
            coords=coords,
            res_ids=res_ids,
            atom_ty_to_coord_idx={a: i for i, a in enumerate(atom_tys)},
            config=self.config
        )
        return InputFeatures(features=feats, batch_size=1, length=len(seq)).maybe_add_batch()


class ExtraDesign(ExtraInput):
    """Store Encoded Native Sequence"""

    def __init__(self,
                 native_seq_enc: Tensor,
                 ):
        super(ExtraDesign, self).__init__()
        self.native_seq_enc = native_seq_enc if native_seq_enc.ndim == 2 \
            else native_seq_enc.unsqueeze(0)

    def crop(self, start, end) -> ExtraDesign:
        """Crop native seq. encoding"""
        self.native_seq_enc = self.native_seq_enc[:, start:end]
        return self

    def to(self, device: Any) -> ExtraDesign:
        """Send native sequence encoding to device"""
        self.native_seq_enc = self.native_seq_enc.to(device)
        return self


def augment(decoy_protein: Protein, native_protein: Protein) -> ExtraDesign:  # noqa
    """Augment function for storing native seq. encoding in ModelInput object"""
    seq = native_protein.seq
    native_seq_enc = [AA_TO_INDEX[r] for r in seq]
    return ExtraDesign(torch.tensor(native_seq_enc).long())
