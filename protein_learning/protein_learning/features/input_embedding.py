"""Input Feature Embedding
"""
from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn.functional import one_hot as to_one_hot

from protein_learning.common.helpers import exists
from protein_learning.common.protein_constants import AA_ALPHABET
from protein_learning.features.feature import Feature, FeatureTy
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.feature_utils import fourier_encode
from protein_learning.features.input_features import FeatureNames, InputFeatures


class FeatEmbedding(nn.Module):
    """One Hot encoding (wrapped, so it can be used in module dict)
    """

    def __init__(self, num_classes, embed_dim, mult=1):
        super(FeatEmbedding, self).__init__()
        self.mult, self.offsets = mult, None
        self.embed_dim, self.num_classes = embed_dim, num_classes
        self.embedding = nn.Embedding(mult * num_classes, embed_dim)

    @property
    def embedding_dim(self) -> int:
        """flattened shape of embedding"""
        return self.mult * self.embed_dim

    def get_offsets(self, feat: Tensor):
        """values to shift bins by for case of multiple embeddings
        """
        if not exists(self.offsets):
            offsets = [i * self.num_classes for i in range(self.mult)]
            self.offsets = torch.tensor(offsets, device=feat.device, dtype=torch.long)
        return self.offsets

    def forward(self, feat: Feature):
        """Embed the feature"""
        to_emb = feat.get_encoded_data()
        assert to_emb.shape[-1] == self.mult
        return self.embedding(to_emb + self.get_offsets(to_emb))


class FeatOneHotEncoding(nn.Module):
    """One Hot encoding (wrapped, so it can be used in module dict)
    """

    def __init__(self, num_classes, mult: int = 1):  # noqa
        super(FeatOneHotEncoding, self).__init__()
        self.mult, self.num_classes = mult, num_classes

    @property
    def embedding_dim(self) -> int:
        """flattened shape of embedding"""
        return self.mult * self.num_classes

    def forward(self, feat: Feature):
        """One hot encode the features"""
        to_hot = feat.get_encoded_data()
        assert to_hot.shape[-1] == self.mult
        return to_one_hot(to_hot, self.num_classes)


class FeatFourierEncoding(nn.Module):
    """Fourier (sin and cos) encoding (wrapped so it can be used in module dict)
    """

    def __init__(self, n_feats, include_self=False, mult: int = 1):  # noqa
        super(FeatFourierEncoding, self).__init__()
        self.n_feats, self.include_self, self.mult = n_feats, include_self, mult
        self.num_embeddings = (2 * self.n_feats + (1 if include_self else 0))

    @property
    def embedding_dim(self) -> int:
        """flattened shape of embedding"""
        return self.mult * self.num_embeddings

    def forward(self, feat: Feature):
        """fourier encode the features"""
        with torch.no_grad():
            to_encode = feat.get_raw_data()
            assert to_encode.shape[-1] == self.mult
            return fourier_encode(
                feat.get_raw_data(),
                num_encodings=self.n_feats,
                include_self=self.include_self
            )


def count_embedding_dim(embeddings) -> int:
    """sums output dimension of list of embeddings"""
    return sum([e.embedding_dim for e in embeddings])


def get_embeddings(
        embed: bool = False,
        one_hot: bool = False,
        fourier: bool = False,
        fourier_feats: int = None,
        n_classes: int = None,
        embed_dim: int = None,
        mult: int = 1,
) -> nn.ModuleDict:
    """Gets embedding dict for input"""
    embeddings = nn.ModuleDict()
    if embed:
        embeddings["emb"] = FeatEmbedding(
            n_classes, embed_dim, mult=mult)
    if one_hot:
        embeddings["one_hot"] = FeatOneHotEncoding(
            num_classes=n_classes, mult=mult)
    if fourier:
        embeddings["fourier"] = FeatFourierEncoding(
            n_feats=fourier_feats, mult=mult)
    return embeddings


class InputEmbedding(nn.Module):
    """Input Embedding"""

    def __init__(self, feature_config: InputFeatureConfig):
        super(InputEmbedding, self).__init__()
        self.config = feature_config
        dims, embeddings = self._feat_dims_n_embeddings(feature_config)
        self.scalar_dim, self.pair_dim = dims
        self.scalar_embeddings, self.pair_embeddings = embeddings

    def forward(self, feats: InputFeatures) -> Tuple[Tensor, Tensor]:
        """Get pair and scalar input features"""
        leading_shape = (feats.batch_size, feats.length)
        scalar_feats = self.get_scalar_input(feats, leading_shape)
        pair_feats = self.get_pair_input(feats, (*leading_shape, leading_shape[-1]))  # noqa
        return scalar_feats.float(), pair_feats.float()

    @property
    def dims(self) -> Tuple[int, int]:
        """Scalar and Pair Feature dimension"""
        return self.scalar_dim, self.pair_dim

    @staticmethod
    def _feat_dims_n_embeddings(
            config: InputFeatureConfig
    ) -> Tuple[Tuple[int, int], Tuple[nn.ModuleDict, nn.ModuleDict]]:
        """gets scalar and pair input dimensions as well as embedding/encoding
        functions for each input feature.

        Feature types can be found in features/input_features.py
        """
        pad = 1 if config.pad_embeddings else 0
        scalar_dim, pair_dim = 0, 0
        scalar_embeddings, pair_embeddings = nn.ModuleDict(), nn.ModuleDict()

        if config.include_res_ty:
            scalar_embeddings[FeatureNames.RES_TY.value] = get_embeddings(
                embed=config.embed_res_ty,
                one_hot=config.one_hot_res_ty,
                embed_dim=config.res_ty_embed_dim,
                n_classes=len(AA_ALPHABET) + pad,
            )

        if config.include_rel_pos:
            scalar_embeddings[FeatureNames.REL_POS.value] = get_embeddings(
                embed=config.embed_res_rel_pos,
                one_hot=config.one_hot_res_rel_pos,
                embed_dim=config.res_rel_pos_embed_dim,
                n_classes=config.res_rel_pos_bins + pad,
            )

        if config.include_bb_dihedral:
            scalar_embeddings[FeatureNames.BB_DIHEDRAL.value] = get_embeddings(
                embed=config.embed_bb_dihedral,
                one_hot=config.one_hot_bb_dihedral,
                embed_dim=config.bb_dihedral_embed_dim,
                n_classes=config.bb_dihedral_bins + pad,
                fourier=config.fourier_encode_bb_dihedral,
                fourier_feats=config.n_bb_dihedral_fourier_feats,
                mult=3
            )

        if config.include_centrality:
            scalar_embeddings[FeatureNames.CENTRALITY.value] = get_embeddings(
                embed=config.embed_centrality,
                one_hot=config.one_hot_centrality,
                embed_dim=config.centrality_embed_dim,
                n_classes=config.centrality_embed_bins + pad,
            )

        if config.include_rel_sep:
            pair_embeddings[FeatureNames.REL_SEP.value] = get_embeddings(
                embed=config.embed_rel_sep,
                one_hot=config.one_hot_rel_sep,
                embed_dim=config.rel_sep_embed_dim,
                n_classes=config.rel_sep_embed_bins + pad,
            )

        if config.joint_embed_res_pair_rel_sep:
            pair_dim -= 2 * config.joint_embed_res_pair_rel_sep_embed_dim
            pair_embeddings["joint_pair_n_sep"] = nn.ModuleDict()
            pair_embeddings["joint_pair_n_sep"][FeatureNames.REL_SEP.value] = FeatEmbedding(
                config.rel_sep_embed_bins + pad, config.joint_embed_res_pair_rel_sep_embed_dim
            )
            pair_embeddings["joint_pair_n_sep"][FeatureNames.RES_TY.value + "_a"] = FeatEmbedding(
                len(AA_ALPHABET) + pad, config.joint_embed_res_pair_rel_sep_embed_dim
            )
            pair_embeddings["joint_pair_n_sep"][FeatureNames.RES_TY.value + "_b"] = FeatEmbedding(
                len(AA_ALPHABET) + pad, config.joint_embed_res_pair_rel_sep_embed_dim
            )

        if config.include_tr_ori:
            pair_embeddings[FeatureNames.TR_ORI.value] = get_embeddings(
                embed=config.embed_tr_rosetta_ori,
                one_hot=config.one_hot_tr_rosetta_ori,
                embed_dim=config.tr_rosetta_ori_embed_dim,
                n_classes=config.tr_rosetta_ori_embed_bins + pad,
                fourier=config.fourier_encode_tr_rosetta_ori,
                fourier_feats=config.tr_rosetta_fourier_feats,
                mult=3
            )

        if config.include_rel_dist:
            n_classes = config.rel_dist_embed_bins * (len(config.rel_dist_atom_pairs) if config.embed_rel_dist else 1)
            pair_embeddings[FeatureNames.REL_DIST.value] = get_embeddings(
                embed=config.embed_rel_dist,
                one_hot=config.one_hot_rel_dist,
                embed_dim=config.rel_dist_embed_dim,
                n_classes=n_classes + pad,
                mult=len(config.rel_dist_atom_pairs)
            )

        scalar_dim += sum([count_embedding_dim(e.values()) for e in scalar_embeddings.values()])
        pair_dim += sum([count_embedding_dim(e.values()) for e in pair_embeddings.values()])
        return (scalar_dim, pair_dim), (scalar_embeddings, pair_embeddings)

    def get_scalar_input(self, features: InputFeatures, leading_shape: Tuple[int, int]) -> Tensor:
        """Get scalar input"""
        scalar_feats = []
        for feat_name, feat in features.items():
            if feat.ty == FeatureTy.SCALAR:
                if feat.name not in self.scalar_embeddings:
                    continue
                feat_embeddings = self.scalar_embeddings[feat.name]
                for emb_name, emb in feat_embeddings.items():
                    emb_feat = emb(feat)
                    scalar_feats.append(emb_feat.reshape(*leading_shape, -1))
        return torch.cat(scalar_feats, dim=-1) if len(scalar_feats) > 0 else None

    def get_pair_input(self, features: InputFeatures, leading_shape: Tuple[int, int, int]) -> Tensor:
        """Get pair input"""
        pair_feats = []
        for feat_name, feat in features.items():
            if feat.ty == FeatureTy.PAIR:
                if feat.name not in self.pair_embeddings:
                    continue
                feat_embeddings = self.pair_embeddings[feat.name]
                for emb_name, emb in feat_embeddings.items():
                    emb_feat = emb(feat)
                    pair_feats.append(emb_feat.reshape(*leading_shape, -1))

        # optional joint embedding
        if "joint_pair_n_sep" in self.pair_embeddings:
            joint_embs = self.pair_embeddings["joint_pair_n_sep"]
            res_ty = features[FeatureNames.RES_TY.value]
            sep = features[FeatureNames.REL_SEP.value]
            emb_sep = joint_embs[FeatureNames.REL_SEP.value](sep).reshape(*leading_shape, -1)
            emb_a = joint_embs[FeatureNames.RES_TY.value + "_a"](res_ty).reshape(*leading_shape[:-1], -1)
            emb_b = joint_embs[FeatureNames.RES_TY.value + "_b"](res_ty).reshape(*leading_shape[:-1], -1)
            joint_emb = rearrange(emb_a, '... n d-> ... n () d') + \
                        rearrange(emb_b, '... n d-> ... () n d') + emb_sep  # noqa
            pair_feats.append(joint_emb)
        return torch.cat(pair_feats, dim=-1) if len(pair_feats) > 0 else None
