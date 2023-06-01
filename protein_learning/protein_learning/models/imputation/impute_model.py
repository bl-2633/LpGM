"""Imputation Model"""

from torch import nn, Tensor
from typing import List, Dict, Optional, Tuple, Any
from protein_learning.common.data.model_data import ModelOutput, ModelLoss, ModelInput
from protein_learning.common.global_constants import get_logger
from protein_learning.common.model_config import ModelConfig
from protein_learning.common.helpers import exists
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.networks.structure_net.structure_net import StructureNet
from protein_learning.networks.ipa.invariant_point_attention import IPATransformer
from protein_learning.networks.ipa.ipa_config import IPAConfig
from protein_learning.networks.common.rigid import Rigids

logger = get_logger(__name__)


class Imputer(StructureNet):
    """Protein Imputation Model"""

    def __init__(
            self,
            model_config: ModelConfig,
            input_embedding: InputEmbedding,
            loss_fn: nn.Module,
            scalar_dim_hidden: int,
            pair_dim_hidden: int,
            evoformer_scalar_heads_n_dim: List[int],
            evoformer_pair_heads_n_dim: List[int],
            evoformer_depth: int,
            coord_dim_out: int,
            ipa_heads: int,
            ipa_head_dims: List[int],
            ipa_depth: int,
            evoformer_ff_mults: List[int] = None,
            recompute_rigids: bool = True,
            share_weights: bool = False,
    ):
        super(Imputer, self).__init__(
            model_config=model_config,
            input_embedding=input_embedding,
            scalar_dim_hidden=scalar_dim_hidden,
            pair_dim_hidden=pair_dim_hidden,
            pre_norm_pair=False,
            evoformer_ff_mults=evoformer_ff_mults,
            evoformer_scalar_heads_n_dim=evoformer_scalar_heads_n_dim,
            evoformer_pair_heads_n_dim=evoformer_pair_heads_n_dim,
            evoformer_depth=evoformer_depth,
            ipa_transformer=IPATransformer(
                IPAConfig(
                    scalar_dim_in=scalar_dim_hidden,
                    pair_dim=pair_dim_hidden,
                    coord_dim_out=coord_dim_out,
                    heads=ipa_heads,
                    scalar_key_dim=ipa_head_dims[0],
                    scalar_value_dim=ipa_head_dims[0],
                    point_key_dim=ipa_head_dims[1],
                    point_value_dim=ipa_head_dims[1],
                    depth=ipa_depth,
                    share_weights=share_weights
                )
            )
        )
        self.share_weights = share_weights
        self.loss_fn = loss_fn
        self.nsr_feat_norm = nn.LayerNorm(scalar_dim_hidden)
        self.pair_pre_norm = nn.LayerNorm(pair_dim_hidden)
        self.recompute_rigids = recompute_rigids

    @property
    def structure_net(self) -> nn.Module:
        """return the underlying structure network"""
        return self.ipa_transformer

    def get_structure_input_kwargs(self, scalar_feats, pair_feats, sample: ModelInput) -> Dict:
        """Get input for structure module"""
        return dict(
            single_repr=scalar_feats,
            pairwise_repr=self.pair_pre_norm(pair_feats),
            true_rigids=getattr(sample, "true_rigids", None),
            mask=sample.native.valid_residue_mask.unsqueeze(0)
        )

    def get_structure_output(
            self, structure_out: Any, structure_input: Dict
    ) -> Tuple[Tensor, Tensor, Optional[Any]]:
        """Get output of structure module (e.g. scalar,coord features)"""
        aux_loss = None
        if not self.share_weights:
            scalar_out, rigids, coords = structure_out
        else:
            scalar_out, rigids, aux_loss, coords = structure_out
        return scalar_out, coords, (rigids, aux_loss)

    def augment_output(self, model_input, coords, scalar_out, evo_scalar, pair, extra_out):
        """Augment the output"""
        scalar_feats = self.nsr_feat_norm(scalar_out)
        rigids, aux_loss = extra_out
        rigids = Rigids.RigidFromBackbone(coords) if self.recompute_rigids else rigids
        return dict(nsr_scalar=scalar_feats, pred_rigids=rigids, aux_loss=aux_loss)

    def compute_loss(self, output: ModelOutput, **kwargs) -> ModelLoss:
        """Compute model loss"""
        output.scalar_output = output.extra["nsr_scalar"]
        aux_loss = output.extra["aux_loss"]
        loss = self.loss_fn(output, **kwargs)
        if exists(aux_loss):
            loss.add_loss(loss_name="fape-aux", loss_weight=1, loss=aux_loss)
        return loss
