from torch import nn, Tensor
from protein_learning.networks.evoformer.node_updates import NodeUpdateLayer
from protein_learning.networks.evoformer.triangle_updates import PairUpdateLayer
from protein_learning.networks.ipa.invariant_point_attention import IPATransformer
from protein_learning.networks.common.rigid import Rigids
from protein_learning.networks.evoformer.evoformer_config import EvoformerConfig
from protein_learning.networks.ipa.ipa_config import IPAConfig
from typing import Optional

from protein_learning.networks.common.invariant.units import (
    Residual,
    LearnedOuterProd
)
from protein_learning.networks.vae.net_utils import RBF, Coords2Pair


class Decoder(nn.Module):
    """Decoder Module"""
    def __init__(
            self,
            scalar_dim: int,
            pair_dim: int,
            depth: int,
            evoformer_config: EvoformerConfig,
            ipa_config: IPAConfig,
            coord_scale: float = 1e-1,
    ):
        super(Decoder, self).__init__()
        self.coord_scale = coord_scale
        self.evo_config, self.ipa_config = evoformer_config, ipa_config
        self.coord_to_pair = Coords2Pair(ipa_config.coord_dim_out, RBF(2, 18, 18), pair_dim)
        self.layers = nn.ModuleList([
            self.get_layer(scalar_dim, pair_dim)
            for _ in range(depth)])

    def forward(
            self,
            scalar_feats: Tensor,
            pair_feats: Tensor,
            coords: Tensor,
            rigids: Optional[Rigids] = None,
    ):
        coords = coords * self.coord_scale
        for layer in self.layers:
            scalar_out = layer["scalar_update"](scalar_feats, pair_feats)
            scalar_feats = layer["scalar_residual"](scalar_out, res=scalar_feats)
            scalar_to_pair = layer["scalar_to_pair"](scalar_feats)
            pair_feats = layer["scalar_to_pair_residual"](scalar_to_pair, res=pair_feats)
            coord_to_pair = self.coord_to_pair(coords * (1 / self.coord_scale))
            pair_out = layer["pair_update"](
                edges=pair_feats + coord_to_pair,
                mask=None
            )
            pair_feats = layer["pair_residual"](pair_out, res=pair_feats)
            ipa_out = layer["coord_update"](
                single_repr=scalar_feats,
                pairwise_repr=pair_feats,
                rigids=rigids
            )
            scalar_out, rigids, coords = ipa_out
            scalar_feats = layer["scalar_ipa_residual"](scalar_out, res=scalar_feats)
            # TODO: stop gradient ??
        coords, rigids = coords * (1 / self.coord_scale), rigids.scale(factor=(1 / self.coord_scale))
        return scalar_feats, pair_feats, coords, rigids

    def get_layer(self, scalar_dim, pair_dim) -> nn.ModuleDict:
        c = self.evo_config
        return nn.ModuleDict(
            dict(
                scalar_update=NodeUpdateLayer(
                    dim=scalar_dim,
                    dim_head=c.node_dim_head,
                    heads=c.node_attn_heads,
                    edge_dim=pair_dim,
                    ff_mult=c.node_ff_mult,
                    dropout=c.node_dropout,
                ),
                # residual
                scalar_residual=Residual(use_rezero=c.use_rezero),
                # scalar to pair
                scalar_to_pair=LearnedOuterProd(
                    dim_in=scalar_dim, dim_out=pair_dim, dim_hidden=16, gate=True
                ),
                # outer product residual
                scalar_to_pair_residual=Residual(use_rezero=c.use_rezero),
                pair_update=PairUpdateLayer(
                    dim=pair_dim,
                    heads=c.edge_attn_heads,
                    dim_head=c.edge_dim_head,
                    dropout=c.edge_dropout,
                    tri_mul_dim=c.triangle_mul_dim,
                    do_checkpoint=c.checkpoint,
                    ff_mult=c.edge_ff_mult,
                ),
                # pair residual
                pair_residual=Residual(use_rezero=c.use_rezero),
                # coord update
                coord_update=IPATransformer(
                    self.ipa_config
                ),
                # scalar_coord_residual
                scalar_ipa_residual=Residual(use_rezero=True),
            )
        )
