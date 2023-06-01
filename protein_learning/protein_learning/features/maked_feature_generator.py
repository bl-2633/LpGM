"""Masked Feature Generation"""
from __future__ import annotations

import random
from functools import partial
from typing import List, Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from protein_learning.common.helpers import exists
from protein_learning.common.helpers import safe_norm
from protein_learning.features.feature import FeatureTy
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.generator import FeatureGenerator
from protein_learning.features.input_features import InputFeatures, get_input_features, FeatureNames
from einops import repeat  # noqa

max_value = lambda x: torch.finfo(x.dtype).max  # noqa


def bool_tensor(n, fill=True, posns: Optional[Tensor] = None):
    """Bool tensor of length n, initialized to 'fill' in all given positions
    and ~fill in other positions.
    """
    # create bool tensor initialized to ~fill
    mask = torch.zeros(n).bool() if fill else torch.ones(n).bool()
    mask[posns if exists(posns) else torch.arange(n)] = fill
    return mask


def apply_mask_to_seq(seq: str, mask: Tensor):
    """Modifies the input sequence so that s[i]="-" iff mask[i]"""
    assert mask.ndim == 1
    return "".join([s if mask[i] else "-" for i, s in enumerate(seq)])


def contiguous_mask(num_residue: int, coords: Tensor, min_len: int, max_len: int) -> Tensor:  # noqa
    """Masks a contiguous segment of a sequence"""
    mask_len = min(num_residue, np.random.randint(min_len, max_len))
    mask_start = np.random.randint(0, num_residue - mask_len)
    mask_posns = torch.arange(start=mask_start, end=mask_start + mask_len)
    return bool_tensor(num_residue, posns=mask_posns, fill=True)


def spatial_mask(
        num_residue: int,
        coords: Tensor,
        top_k=30,
        max_radius=12,
        mask_self: bool = False,
        atom_pos: int = 1
) -> Tensor:
    """Masks positions in a sequence based on spatial proximity to a random query residue"""
    coords = coords.squeeze(0) if coords.ndim == 4 else coords
    top_k = min(top_k, num_residue - 1)
    residue_idx = np.random.choice(num_residue)
    dists = safe_norm(coords[:, atom_pos] - coords[residue_idx, atom_pos].unsqueeze(0), dim=-1)
    assert dists.ndim == 1
    if not mask_self:
        dists[residue_idx] = max_value(dists)
    nbr_dists, nbr_indices = dists.topk(k=top_k, dim=-1, largest=False)
    mask_posns = nbr_indices[nbr_dists < max_radius]
    return bool_tensor(num_residue, posns=mask_posns, fill=True)


def random_mask(num_residue: int, coords: Tensor, min_p: float, max_p: float) -> Tensor:  # noqa
    """Randomly masks each sequence position w.p. in range (min_p, max_p)"""
    mask_prob = np.random.uniform(min_p, max_p)
    mask_posns = torch.arange(num_residue)[torch.rand(num_residue) < mask_prob]
    return bool_tensor(num_residue, posns=mask_posns, fill=True)


def no_mask(num_residue: int, *args, **kwargs) -> Tensor:  # noqa
    """Does not mask any sequence positions"""
    return bool_tensor(num_residue, fill=False)


def full_mask(num_residue: int, *args, **kwargs) -> Tensor:  # noqa
    """Does not mask any sequence positions"""
    return bool_tensor(num_residue, fill=True)


def get_mask_strategies_n_weights(
        random_mask_min_p: float = 0,
        random_mask_max_p: float = 0,
        spatial_mask_top_k: int = 30,
        spatial_mask_max_radius: float = 12,
        spatial_mask_mask_self: bool = False,
        spatial_mask_atom_pos: int = 1,
        contiguous_mask_max_len: int = 60,
        contiguous_mask_min_len: int = 5,
        no_mask_weight: float = 0,
        random_mask_weight: float = 0,
        contiguous_mask_weight: float = 0,
        spatial_mask_weight: float = 0,
        full_mask_weight: float = 0,

):
    """Get mask strategy functions and strategy weights"""
    # mask options
    mask_strategies = [
        no_mask,
        full_mask,
        partial(
            random_mask,
            min_p=random_mask_min_p,
            max_p=random_mask_max_p
        ),
        partial(
            spatial_mask,
            top_k=spatial_mask_top_k,
            max_radius=spatial_mask_max_radius,
            mask_self=spatial_mask_mask_self,
            atom_pos=spatial_mask_atom_pos
        ),
        partial(
            contiguous_mask,
            min_len=contiguous_mask_min_len,
            max_len=contiguous_mask_max_len
        )
    ]

    strategy_weights = [
        no_mask_weight,
        full_mask_weight,
        random_mask_weight,
        spatial_mask_weight,
        contiguous_mask_weight,
    ]
    return mask_strategies, strategy_weights


class MaskedFeatureGenerator(FeatureGenerator):
    """Feature Generator with masking functionality"""

    def __init__(
            self,
            config: InputFeatureConfig,
            mask_strategies: List[Callable[..., str]],
            strategy_weights: List[float],
            mask_feats: bool = False,
            mask_seq: bool = False,
            mask_feat_n_seq_indep_prob: float = 0
    ):
        super(MaskedFeatureGenerator, self).__init__(config=config)
        self.mask_strategies = mask_strategies
        self.strategy_weights = np.array(strategy_weights) / sum(strategy_weights)
        self.mask_feats = mask_feats
        self.mask_seq = mask_seq
        self.mask_feat_n_seq_indep_prob = mask_feat_n_seq_indep_prob
        assert mask_feats or mask_seq
        assert len(strategy_weights) == len(mask_strategies)
        assert config.pad_embeddings

    def get_seq_n_feat_mask(self, seq: str, coords: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Gets masks to apply to sequence and coordinate features"""
        feat_mask, seq_mask = None, None
        if self.mask_seq:
            seq_mask = self.sample_mask(len(seq), coords)
        if self.mask_feats:
            feat_mask = self.sample_mask(len(seq), coords)
        if self.mask_seq and self.mask_feats:
            if self.mask_feat_n_seq_indep_prob < random.random():
                feat_mask = seq_mask
        return seq_mask, feat_mask

    def sample_mask(self, num_residue: int, coords: Tensor) -> Tensor:
        """Sample a mask from mask strategy distribution"""
        strategy_idx = np.random.choice(len(self.strategy_weights), p=self.strategy_weights)
        return self.mask_strategies[strategy_idx](num_residue, coords)

    @staticmethod
    def bb_dihedral_mask(feat_mask: Tensor) -> Tensor:
        """Get backbone dihedral feature mask"""
        bb_dihedral_mask = feat_mask.clone()
        bb_dihedral_mask[:-1] = torch.logical_or(feat_mask[1:], bb_dihedral_mask[:-1])
        bb_dihedral_mask[1:] = torch.logical_or(feat_mask[:-1], bb_dihedral_mask[1:])
        return repeat(bb_dihedral_mask, "i -> i a", a=3)

    @staticmethod
    def pair_feat_mask(feat_mask):
        """Generate mask for pairwise features"""
        return torch.einsum("i,j->ij", feat_mask, feat_mask)

    def generate_features(
            self,
            seq: str,
            coords: Tensor,
            res_ids: Tensor,
            coord_mask: Tensor,
            atom_tys: List[str],
    ) -> InputFeatures:
        """Generate ProteinModel input features"""
        seq_mask, feat_mask = self.get_seq_n_feat_mask(seq, coords)
        feats = get_input_features(
            seq=apply_mask_to_seq(seq, seq_mask) if exists(seq_mask) else seq,
            coords=coords,
            res_ids=res_ids,
            atom_ty_to_coord_idx={a: i for i, a in enumerate(atom_tys)},
            config=self.config
        )
        # mask the features
        if feat_mask is not None:
            pair_mask = self.pair_feat_mask(feat_mask)
            dihedral_mask = self.bb_dihedral_mask(feat_mask)
            for feature_name, feature in feats.items():
                if feature.ty == FeatureTy.SCALAR:
                    if feature_name == FeatureNames.BB_DIHEDRAL:
                        feature.apply_mask(dihedral_mask)
                    else:
                        feature.apply_mask(feat_mask)
                elif feature.ty == FeatureTy.PAIR:
                    feature.apply_mask(pair_mask)
                else:
                    raise Exception(f"Error masking {feature.name}, feature type "
                                    f": {feature.ty.value} Not yet supported")

        return InputFeatures(features=feats, batch_size=1, length=len(seq),
                             masks=(seq_mask, feat_mask)).maybe_add_batch()
