"""Design Model"""

from typing import List, Optional, Any, Tuple, Dict

import torch
from torch import nn, Tensor

from protein_learning.common.data.model_data import ModelOutput, ModelLoss, ModelInput
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import default, maybe_add_batch
from protein_learning.common.helpers import safe_norm
from protein_learning.common.model_config import ModelConfig
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.networks.common.helpers.neighbor_utils import get_neighbor_info
from protein_learning.networks.se3_transformer.se3_transformer_config import SE3TransformerConfig
from protein_learning.networks.structure_net.structure_net import StructureNet
from protein_learning.networks.se3_transformer.tfn_transformer import TFNTransformer

logger = get_logger(__name__)


class Designer(StructureNet):
    """Protein Design Model"""

    def __init__(
            self,
            model_config: ModelConfig,
            input_embedding: InputEmbedding,
            loss_fn: nn.Module,
            scalar_dim_hidden: int,
            pair_dim_hidden: int,
            tfn_heads: int,
            tfn_head_dims: List[int],
            evoformer_scalar_heads_n_dim: List[int],
            evoformer_pair_heads_n_dim: List[int],
            evoformer_depth: int = 6,
            coord_dim_hidden: int = 16,
            coord_dim_out: int = 32,
            tfn_depth: int = 6,
            use_dist_sim: bool = False,
            append_rel_dist: bool = False,
            use_coord_layernorm: bool = False,
            append_sc_coord_norms_nsr: bool = False,
            max_nbr_radius: float = 16,
            max_nbrs: int = 16,
            pre_norm_pair: bool = False,
    ):
        super(Designer, self).__init__(
            model_config=model_config,
            scalar_dim_hidden=scalar_dim_hidden,
            pair_dim_hidden=pair_dim_hidden,
            input_embedding=input_embedding,
            pre_norm_pair=pre_norm_pair,
            evoformer_scalar_heads_n_dim=evoformer_scalar_heads_n_dim,
            evoformer_pair_heads_n_dim=evoformer_pair_heads_n_dim,
            evoformer_depth=evoformer_depth,
            tfn_transformer=TFNTransformer(
                SE3TransformerConfig(
                    fiber_in={0: scalar_dim_hidden, 1: 3},  # bb relative coordinates used as input features
                    fiber_out={0: scalar_dim_hidden, 1: default(coord_dim_out, coord_dim_hidden)},
                    fiber_hidden={0: scalar_dim_hidden, 1: coord_dim_hidden},
                    heads=(tfn_heads, tfn_heads),
                    dim_heads=(tfn_head_dims[0], tfn_head_dims[1]),
                    edge_dim=pair_dim_hidden,
                    depth=tfn_depth,
                    use_dist_sim=use_dist_sim,
                    append_rel_dist=append_rel_dist,
                    use_coord_layernorm=use_coord_layernorm,
                )
            )
        )

        self.loss_fn = loss_fn
        self.append_sc_norms_nsr = append_sc_coord_norms_nsr
        self.nsr_feat_norm = nn.LayerNorm(scalar_dim_hidden)
        self.sc_feat_norm = nn.LayerNorm(32) if append_sc_coord_norms_nsr else None
        self.top_k, self.max_radius = max_nbrs, max_nbr_radius

    def get_structure_input_kwargs(self, scalar_feats, pair_feats, sample: ModelInput) -> Dict:
        """Get input for structure module"""
        CA = maybe_add_batch(sample.get_atom_coords(["CA"], decoy=True), 3)
        N_C_O = maybe_add_batch(sample.get_atom_coords(["N", "C", "O"], decoy=True), 3)
        nbr_info = get_neighbor_info(CA.squeeze(-2), max_radius=self.max_radius, top_k=self.top_k)
        feats = {"0": scalar_feats, "1": (N_C_O - CA)}
        return dict(feats=feats, edges=pair_feats, neighbor_info=nbr_info)

    def get_structure_output(
            self, structure_out: Any, structure_input: Dict
    ) -> Tuple[Tensor, Tensor, Optional[Any]]:
        """Get output of structure module (e.g. scalar,coord features)"""
        CA = structure_input["neighbor_info"].coords.unsqueeze(-2)
        coord_out, scalar_out = structure_out["1"] + CA, structure_out["0"].squeeze(-1)
        return scalar_out, coord_out, dict(neighbor_info=structure_input["neighbor_info"])

    def augment_output(self, model_input, coords, scalar_out, evo_scalar, pair, extra_out):
        """Augment the output"""
        nsr_scalar = scalar_out
        neighbor_info = extra_out["neighbor_info"]
        if self.append_sc_norms_nsr:
            sc_coord_norms = safe_norm(coords - neighbor_info.coords.unsqueeze(-2), dim=-1)
            nsr_scalar = torch.cat((scalar_out, self.sc_feat_norm(sc_coord_norms)), dim=-1)
        return {"nsr_scalar": nsr_scalar}

    def compute_loss(self, output: ModelOutput, **kwargs) -> ModelLoss:
        """Compute model loss"""
        output.scalar_output = output.extra["nsr_scalar"]
        return self.loss_fn(output)
