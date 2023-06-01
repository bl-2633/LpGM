"""Helper methods for generating model input features
"""
from __future__ import annotations

from enum import Enum
from math import pi
from typing import Tuple, List, Optional, Dict

import torch
from einops import rearrange
from torch import Tensor

from protein_learning.common.helpers import exists, default
from protein_learning.common.protein_constants import AA_INDEX_MAP
from protein_learning.features.constants import SMALL_SEP_BINS
from protein_learning.features.feature import Feature, FeatureTy
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.feature_utils import (
    string_encode,
    bin_encode,
)
from protein_learning.protein_utils.dihedral.orientation_utils import (
    get_bb_dihedral,
    get_tr_rosetta_orientation_mats,
)

PI = pi + 1e-10


class InputFeatures:
    """Dictionary-like wrapper for ProteinModel input features"""

    def __init__(
            self,
            features: Dict[str, Feature],
            batch_size: int,
            length: int,
            masks: Optional[Tuple[Optional[Tensor], Optional[Tensor]]] = None,
    ):
        """
        :param features: Dict of input features
        :param batch_size: batch size (should be same for all features)
        :param length: the length of the input features  (e.g. number of residues -
        should be same for all features)
        :param masks: (Optional) sequence and feature masks applied to input features
        and input sequence encodings.
        """
        self.features = features
        self._length = length
        self._batch_size = batch_size
        self.crop_posns = None
        self.masks = masks

    @property
    def length(self) -> int:
        """Get the input length"""
        return self._length

    @property
    def batch_size(self) -> int:
        """Get the batch size"""
        return self._batch_size

    def crop(self, start, end) -> InputFeatures:
        """crop underlying features"""
        self.crop_posns = (start, end)
        self._length = end - start
        self.features = {k: v.crop(start, end) for k, v in self.features.items()}
        self.masks = tuple(x[start:end] if exists(x) else x for x in self.masks) \
            if exists(self.masks) else None
        return self

    def maybe_add_batch(self) -> InputFeatures:
        """Add batch dimension if not present"""
        self.features = {k: v.maybe_add_batch() for k, v in self.features.items()}
        self.masks = tuple(x.reshape(1, -1) if exists(x) else x for x in self.masks) \
            if exists(self.masks) else None
        return self

    def to(self, device: str) -> InputFeatures:
        """Send input features to specified device"""
        self.features = {k: v.to(device) for k, v in self.features.items()}
        self.masks = tuple(x.to(device) if exists(x) else x for x in self.masks) \
            if exists(self.masks) else None
        return self

    def add_batch(self, features: Dict[str, Feature]):
        """Add batch of features to the input"""
        raise Exception("not yet implemented!")

    def items(self):
        """feature dict items"""
        return self.features.items()

    def keys(self):
        """feature dict keys"""
        return self.features.keys()

    def values(self):
        """feature dict values"""
        return self.features.values()

    def __getitem__(self, item):
        return self.features[item]


class FeatureNames(Enum):
    """Identifiers for each feature type
    """
    REL_POS = "rel_pos"
    REL_SEP = "rel_sep"
    REL_DIST = "rel_dist"
    BB_DIHEDRAL = "bb_dihedral"
    CENTRALITY = "centrality"
    RES_TY = "res_ty"
    TR_ORI = "tr_ori"


def res_ty_encoding(
        seq: str,
) -> Feature:
    """Encodes sequence either as a Tensor of ints.
    :param seq: sequence to encode
    :return: encoded sequence.
    """
    seq_emb = string_encode(AA_INDEX_MAP, seq).unsqueeze(-1)
    return Feature(
        raw_data=seq,
        encoded_data=seq_emb,
        name=FeatureNames.RES_TY.value,
        dtype=torch.long,
        ty=FeatureTy.SCALAR,
        n_classes=len(AA_INDEX_MAP)
    )


def rel_pos_encoding(res_ids: Tensor, n_classes: int = 10) -> Feature:
    """Encodes each residue position based on the relative position in the sequence.
    """
    assert torch.all(res_ids >= 0), f"{res_ids}"
    max_posn, _ = torch.max(res_ids, dim=-1, keepdim=True)
    rel_pos_enc = torch.floor((res_ids.float() * n_classes) / (max_posn + 1)).long()
    assert torch.all(rel_pos_enc >= 0)
    return Feature(
        raw_data=res_ids.unsqueeze(-1),
        encoded_data=rel_pos_enc.unsqueeze(-1),
        name=FeatureNames.REL_POS.value,
        dtype=torch.long,
        ty=FeatureTy.SCALAR,
        n_classes=n_classes
    )


def bb_dihedral_encoding(
        bb_coords: Optional[List[Tensor]] = None,
        n_classes: int = 36,
        encode: bool = True,
        bb_dihedrals: Optional[Tuple[Tensor, ...]] = None
) -> Feature:
    """BB DIhedral Features (encoded or raw)
    """
    assert exists(bb_dihedrals) or exists(bb_coords)
    phi, psi, omega = bb_dihedrals if exists(bb_dihedrals) else get_bb_dihedral(*bb_coords)
    bb_dihedrals = torch.cat([x.unsqueeze(-1) for x in (phi, psi, omega)], dim=-1)
    encoded_bb_dihedrals = None
    if encode:
        encoded_bb_dihedrals = torch.clamp(((bb_dihedrals / PI) + 1) / 2, 0, 1) * (n_classes - 1)
    return Feature(
        raw_data=bb_dihedrals,
        encoded_data=encoded_bb_dihedrals,
        name=FeatureNames.BB_DIHEDRAL.value,
        dtype=torch.long,
        ty=FeatureTy.SCALAR,
        n_classes=n_classes
    )


def degree_centrality_encoding(
        coords: Tensor,
        n_classes: int = 6,
        max_radius=12,
        bounds=(6, 30),
) -> Feature:
    """Residue degree centrality features
    """
    dists = torch.cdist(coords, coords)
    cmin, cmax = bounds
    res_centrality = torch.sum(dists <= max_radius, dim=-1) - 1
    norm_res_centrality = (torch.clamp(res_centrality, cmin, cmax) - cmin) / (cmax - cmin)
    binned_res_centrality = norm_res_centrality * (n_classes - 1)
    assert torch.all(binned_res_centrality >= 0)
    return Feature(
        raw_data=res_centrality.unsqueeze(-1),
        encoded_data=binned_res_centrality.unsqueeze(-1).long(),
        name=FeatureNames.CENTRALITY.value,
        dtype=torch.long,
        ty=FeatureTy.SCALAR,
        n_classes=n_classes
    )


def rel_sep_encoding(res_ids: Tensor, sep_bins: Optional[List] = None) -> Feature:
    """Relative Separation Encoding
    """
    sep_bins = default(sep_bins, SMALL_SEP_BINS)
    res_posns = res_ids
    sep_mat = rearrange(res_posns, "n -> () n ()") - rearrange(res_posns, "n -> n () ()")
    enc_sep_mat = bin_encode(sep_mat, bins=sep_bins)
    assert torch.all(enc_sep_mat >= 0)
    return Feature(
        encoded_data=enc_sep_mat,
        raw_data=sep_mat,
        name=FeatureNames.REL_SEP.value,
        dtype=torch.long,
        ty=FeatureTy.PAIR,
        n_classes=len(sep_bins)
    )


def rel_dist_encoding(
        rel_dists: Tensor,
        dist_bounds=(2.5, 16.5),
        n_classes=32,
) -> Feature:
    """Relative Distance Encoding
    """
    min_dist, max_dist = dist_bounds
    normed_dists = (rel_dists - min_dist) / (max_dist - min_dist)
    dist_bins = torch.clamp(normed_dists, 0, 1) * (n_classes - 1)
    return Feature(
        raw_data=rel_dists,
        encoded_data=dist_bins,
        name=FeatureNames.REL_DIST.value,
        dtype=torch.long,
        ty=FeatureTy.PAIR,
        n_classes=n_classes,
    )


def tr_rosetta_ori_encoding(
        bb_coords: List[Tensor] = None,
        n_classes: int = 36,
        encode: bool = True,
        tr_angles: Optional[Tuple[Tensor, ...]] = None,
) -> Feature:
    """trRosetta dihedral features
    """
    phi, psi, omega = get_tr_rosetta_orientation_mats(*bb_coords) \
        if not exists(tr_angles) else tr_angles
    ori_feats = torch.cat([x.unsqueeze(-1) for x in (phi, psi, omega)], dim=-1)
    encoded_ori_feats = None
    if encode:
        encoded_ori_feats = torch.clamp(((ori_feats / PI) + 1) / 2, 0, 1) * (n_classes - 1)
    return Feature(
        raw_data=ori_feats,
        encoded_data=encoded_ori_feats,
        name=FeatureNames.TR_ORI.value,
        dtype=torch.long if encode else torch.float32,
        ty=FeatureTy.PAIR,
        n_classes=n_classes
    )


def get_input_features(
        seq: str,
        coords: Tensor,
        res_ids: Tensor,
        atom_ty_to_coord_idx: Dict[str, int],
        config: InputFeatureConfig,
) -> Dict[str, Feature]:
    """
    :param seq: String sequence for protein
    :param coords: Tensor of shape (b,n,a,3) or (n,a,3) where n is the sequence length,
    a is the number of atoms.
    :param atom_ty_to_coord_idx: mapping from atom type to atom index in coord tensor
    e.g. coords[...,atom_ty_to_coord_idx["CA"],:] should yield CA coordinates
    :param config: feature config
    :param res_ids: Residue id's (list of integers describing residue sequence positions
    in underlying protein).
    :return: Dict mapping feature name to corresponding feature
    """
    feats = {}
    atom_coords = lambda atom_ty: coords[..., atom_ty_to_coord_idx[atom_ty], :]

    def add_feat(feat):
        """Adds feature to feats dict
        """
        feats[feat.name] = feat

    if config.include_res_ty:
        add_feat(res_ty_encoding(seq=seq))

    if config.include_rel_pos:
        add_feat(rel_pos_encoding(res_ids=res_ids, n_classes=config.res_rel_pos_bins))

    if config.include_bb_dihedral:
        N, CA, C = [atom_coords(ty) for ty in ["N", "CA", "C"]]
        dihedrals = get_bb_dihedral(N=N, CA=CA, C=C)
        add_feat(bb_dihedral_encoding(
            bb_dihedrals=dihedrals,
            encode=True,
            n_classes=config.bb_dihedral_bins,

        ))

    if config.include_centrality:
        add_feat(degree_centrality_encoding(
            coords=atom_coords("CB" if "CB" in atom_ty_to_coord_idx else "CA"),
            n_classes=config.centrality_embed_bins,
        ))

    if config.include_rel_sep:
        add_feat(rel_sep_encoding(res_ids=res_ids, sep_bins=SMALL_SEP_BINS))

    if config.include_tr_ori:
        N, CA, CB = [atom_coords(ty) for ty in ["N", "CA", "CB"]]
        phi, psi, omega = get_tr_rosetta_orientation_mats(N=N, CA=CA, CB=CB)
        add_feat(tr_rosetta_ori_encoding(
            tr_angles=(phi, psi, omega),
            encode=True,
            n_classes=config.tr_rosetta_ori_embed_bins,
        ))

    if config.include_rel_dist:
        rel_dists = []
        for (a1, a2) in config.rel_dist_atom_pairs:
            c1, c2 = atom_coords(a1), atom_coords(a2)
            rel_dists.append(torch.cdist(c1, c2).unsqueeze(-1))
        add_feat(rel_dist_encoding(
            rel_dists=torch.cat(rel_dists, dim=-1),
            n_classes=config.rel_dist_embed_bins
        ))

    return feats
