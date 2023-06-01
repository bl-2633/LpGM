"""Configuration for Input Feature Generation
"""
from typing import NamedTuple, List, Tuple

from protein_learning.features.constants import SMALL_SEP_BINS


class InputFeatureConfig(NamedTuple):
    # Global settings
    # pad embeddings with an extra bin (n_classes+1) - can be used
    # in conjunction with sequence or coordinate masking, e.g.
    pad_embeddings: bool = False

    # Residue Type (SCALAR)
    embed_res_ty: bool = False
    res_ty_embed_dim: int = 32
    one_hot_res_ty: bool = True

    # Residue Relative Position (SCALAR)
    embed_res_rel_pos: bool = False
    res_rel_pos_embed_dim: int = 6
    res_rel_pos_bins: int = 10
    one_hot_res_rel_pos: bool = True

    # BB Dihedral (SCALAR)
    embed_bb_dihedral: bool = False
    bb_dihedral_embed_dim: int = 6
    fourier_encode_bb_dihedral: bool = True
    n_bb_dihedral_fourier_feats: int = 2
    one_hot_bb_dihedral: bool = False
    bb_dihedral_bins: int = 36

    # Centrality (SCALAR)
    embed_centrality: bool = False
    centrality_embed_bins: int = 6
    centrality_embed_dim: int = 6
    one_hot_centrality: bool = True

    # Relative Separation (PAIR)
    embed_rel_sep: bool = False
    rel_sep_embed_dim: int = 32
    one_hot_rel_sep: bool = False
    rel_sep_embed_bins: int = len(SMALL_SEP_BINS)

    # Relative Distance (PAIR)
    embed_rel_dist: bool = False
    rel_dist_embed_dim: int = 16
    one_hot_rel_dist: bool = True
    rel_dist_atom_tys: List[str] = ["CA", "CA", "N", "CA"]
    rel_dist_embed_bins: int = 32

    # trRosetta Orientation (PAIR)
    embed_tr_rosetta_ori: bool = False
    tr_rosetta_ori_embed_dim: int = 6
    tr_rosetta_ori_embed_bins: int = 36
    fourier_encode_tr_rosetta_ori: bool = True
    tr_rosetta_fourier_feats: int = 2
    one_hot_tr_rosetta_ori: bool = False

    # Joint Embedding for Pair and Sep (PAIR)
    joint_embed_res_pair_rel_sep: bool = True
    joint_embed_res_pair_rel_sep_embed_dim: int = 48

    @property
    def include_res_ty(self):
        return self.joint_embed_res_pair_rel_sep or self.embed_res_ty \
               or self.one_hot_res_ty

    @property
    def include_rel_pos(self):
        return self.embed_res_rel_pos or self.one_hot_res_rel_pos

    @property
    def include_bb_dihedral(self):
        return self.embed_bb_dihedral or self.one_hot_bb_dihedral or \
               self.fourier_encode_bb_dihedral

    @property
    def include_centrality(self):
        return self.embed_centrality or self.one_hot_centrality

    @property
    def include_rel_sep(self):
        return self.one_hot_rel_sep or self.embed_rel_sep \
               or self.joint_embed_res_pair_rel_sep

    @property
    def include_rel_dist(self):
        return self.one_hot_rel_dist or self.embed_rel_dist

    @property
    def rel_dist_atom_pairs(self) -> List[Tuple[str]]:
        tys = self.rel_dist_atom_tys
        return list(zip(tys[::2], tys[1::2]))

    @property
    def include_tr_ori(self):
        return self.embed_tr_rosetta_ori or self.fourier_encode_tr_rosetta_ori \
               or self.one_hot_tr_rosetta_ori

