"""MDS model"""
from collections import namedtuple
from typing import List, Any

import torch
from einops.layers.torch import Rearrange
from torch import nn

from protein_learning.common.data.model_data import ModelInput, ModelOutput, ModelLoss
from protein_learning.common.global_constants import get_logger
from protein_learning.common.model_config import ModelConfig
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.models.utils.model_abc import ProteinModel
from protein_learning.networks.evoformer.evoformer import Evoformer
from protein_learning.networks.evoformer.evoformer_config import EvoformerConfig
from protein_learning.networks.ipa.invariant_point_attention import IPATransformer
from protein_learning.networks.ipa.ipa_config import IPAConfig

logger = get_logger(__name__)

Extra = namedtuple("Extra", "pred_rigids")


class InvariantCoordProjection(nn.Module):
    """Predict coordinates form (invariant) scalar and pair features"""

    def __init__(self, scalar_dim_in, pair_dim_in, coord_dim_out):
        super(InvariantCoordProjection, self).__init__()
        self.edge_pool_proj = nn.Sequential(
            nn.Linear(pair_dim_in, scalar_dim_in),
            nn.LayerNorm(scalar_dim_in)
        )
        self.edge_pool_norm = nn.LayerNorm(scalar_dim_in)
        self.scalar_norm = nn.LayerNorm(scalar_dim_in)
        self.point_proj = nn.Sequential(
            nn.Linear(2 * scalar_dim_in, 2 * scalar_dim_in),
            nn.GELU(),
            nn.Linear(2 * scalar_dim_in, coord_dim_out * 3),
            Rearrange("b n (a c) -> b n a c", a=coord_dim_out)
        )

    def forward(self, scalar_feats, pairwise_repr):
        pairwise_repr = self.edge_pool_norm(torch.mean(self.edge_pool_proj(pairwise_repr), dim=-2))
        scalar_feats = self.scalar_norm(scalar_feats)
        feats = torch.cat((scalar_feats, pairwise_repr), dim=-1)
        return self.point_proj(feats)


class MDS(ProteinModel):
    """Model for performing Multi Dimensional Scaling (MDS)"""

    def __init__(
            self,
            model_config: ModelConfig,
            input_embedding: InputEmbedding,
            scalar_dim_hidden: int,
            pair_dim_hidden: int,
            evoformer_scalar_heads_n_dim: List[int],
            evoformer_pair_heads_n_dim: List[int],
            ipa_heads: int,
            ipa_head_dims: List[int],
            loss_fn: Any,
            coord_dim_out: int = 4,
            evoformer_depth: int = 4,
            ipa_depth: int = 3,
            use_ipa: bool = True,
            predict_rigids: bool = True,
            detach_frames: bool = False,

    ):
        super(MDS, self).__init__()
        self.predict_rigids = predict_rigids
        self.detach_frames = detach_frames

        self.model_config = model_config
        self.evoformer_config = EvoformerConfig(
            node_dim_in=scalar_dim_hidden,
            edge_dim_in=pair_dim_hidden,
            depth=evoformer_depth,
            edge_attn_heads=evoformer_pair_heads_n_dim[0],
            edge_dim_head=evoformer_pair_heads_n_dim[1],
            node_attn_heads=evoformer_scalar_heads_n_dim[0],
            node_dim_head=evoformer_scalar_heads_n_dim[1],
            edge_ff_mult=2,
            node_ff_mult=2,
            use_rezero=True,
        )
        self.ipa_config = IPAConfig(
            scalar_dim_in=scalar_dim_hidden,
            pair_dim=pair_dim_hidden,
            scalar_key_dim=ipa_head_dims[0],
            scalar_value_dim=ipa_head_dims[0],
            point_value_dim=ipa_head_dims[1],
            point_key_dim=ipa_head_dims[1],
            heads=ipa_heads,
            depth=ipa_depth,
            coord_dim_out=coord_dim_out,
        )

        self.evoformer = Evoformer(self.evoformer_config)
        self.to_coords = IPATransformer(self.ipa_config) if \
            use_ipa else InvariantCoordProjection(
            scalar_dim_in=scalar_dim_hidden,
            pair_dim_in=pair_dim_hidden,
            coord_dim_out=coord_dim_out)

        self.input_embedding = input_embedding
        s_in, p_in = self.input_embedding.dims

        # input projections
        self.scalar_project_in = nn.Linear(s_in, scalar_dim_hidden)
        self.pair_project_in = nn.Linear(p_in, pair_dim_hidden)
        self.to_scalar_structure = nn.Sequential(
            nn.LayerNorm(scalar_dim_hidden),
            nn.Linear(scalar_dim_hidden, scalar_dim_hidden),
        ) if use_ipa else nn.Identity()
        self.to_pair_structure = nn.Sequential(
            nn.LayerNorm(scalar_dim_hidden),
            nn.Linear(pair_dim_hidden, pair_dim_hidden),
        ) if use_ipa else nn.Identity()

        self.loss_fn = loss_fn
        self.use_ipa = use_ipa

    def forward(self, sample: ModelInput, **kwargs) -> ModelOutput:
        """Run the model"""
        # get input features
        scalar_feats, pair_feats = self.input_embedding(sample.input_features)
        scalar_feats = self.scalar_project_in(scalar_feats)
        pair_feats = self.pair_project_in(pair_feats)
        scalar_feats, pair_feats = self.evoformer(scalar_feats, pair_feats)
        scalar_feats = self.to_scalar_structure(scalar_feats)
        pair_feats = self.to_pair_structure(pair_feats)
        rigids, aux_loss = None, None
        if self.use_ipa:
            scalar_feats, rigids, coords = self.to_coords(scalar_feats, pairwise_repr=pair_feats)
            rigids = rigids.detach_all() if self.detach_frames else rigids
            rigids = rigids.scale(10) if self.predict_rigids else None
        else:
            coords = self.to_coords(scalar_feats, pair_feats)

        return ModelOutput(
            predicted_coords=coords * 10,
            scalar_output=scalar_feats,
            pair_output=pair_feats,
            predicted_atom_tys=None,
            model_input=sample,
            extra=Extra(pred_rigids=rigids)
        )

    def compute_loss(self, output: ModelOutput, compute_zero_wt_loss: bool = False) -> ModelLoss:
        """Compute model loss"""
        return self.loss_fn(output, compute_zero_wt_loss=compute_zero_wt_loss)
