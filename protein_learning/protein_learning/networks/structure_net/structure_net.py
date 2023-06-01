"""Side-chain Packing Model"""
from abc import abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from torch import nn, Tensor

from protein_learning.common.data.model_data import ModelInput, ModelOutput, ModelLoss
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import default
from protein_learning.common.model_config import ModelConfig
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.models.utils.model_abc import ProteinModel
from protein_learning.networks.evoformer.evoformer import Evoformer
from protein_learning.networks.evoformer.evoformer_config import EvoformerConfig

logger = get_logger(__name__)


class StructureNet(ProteinModel):
    """Evoformer + Coord prediction network"""

    def __init__(
            self,
            model_config: ModelConfig,
            input_embedding: InputEmbedding,
            scalar_dim_hidden: int,
            pair_dim_hidden: int,
            evoformer_scalar_heads_n_dim: List[int],
            evoformer_pair_heads_n_dim: List[int],
            evoformer_depth: int = 6,
            evoformer_ff_mults: List[int] = None,
            pre_norm_pair: bool = False,
            tfn_transformer: nn.Module = None,
            ipa_transformer: nn.Module = None,
    ):
        super(StructureNet, self).__init__()
        self.model_config = model_config
        evoformer_ff_mults = default(evoformer_ff_mults, [2, 2])
        self.evoformer_config = EvoformerConfig(
            node_dim_in=scalar_dim_hidden,
            edge_dim_in=pair_dim_hidden,
            depth=evoformer_depth,
            edge_attn_heads=evoformer_pair_heads_n_dim[0],
            edge_dim_head=evoformer_pair_heads_n_dim[1],
            node_attn_heads=evoformer_scalar_heads_n_dim[0],
            node_dim_head=evoformer_scalar_heads_n_dim[1],
            edge_ff_mult=evoformer_ff_mults[0],
            node_ff_mult=evoformer_ff_mults[1],
            use_rezero=True,
        )

        self.evoformer = Evoformer(self.evoformer_config)

        # Needed for backward compatibility :/
        self.tfn_transformer = tfn_transformer
        self.ipa_transformer = ipa_transformer

        self.input_embedding = input_embedding
        s_in, p_in = self.input_embedding.dims

        # input projections
        self.scalar_project_in = nn.Linear(s_in, scalar_dim_hidden)
        self.pair_project_in = nn.Linear(p_in, pair_dim_hidden)
        self.to_scalar_structure = nn.Sequential(
            nn.LayerNorm(scalar_dim_hidden),
            nn.Linear(scalar_dim_hidden, scalar_dim_hidden),
        )
        self.to_pair_structure = nn.Sequential(
            nn.LayerNorm(pair_dim_hidden),
            nn.Linear(pair_dim_hidden, pair_dim_hidden),
            nn.LayerNorm(pair_dim_hidden) if pre_norm_pair else nn.Identity()
        )

    def forward(self, sample: ModelInput, **kwargs) -> ModelOutput:
        """Run the model"""
        # get input features
        scalar_feats, pair_feats = self.input_embedding(sample.input_features)
        scalar_feats = self.scalar_project_in(scalar_feats)
        pair_feats = self.pair_project_in(pair_feats)
        evo_scalar_feats, pair_feats = self.evoformer(scalar_feats, pair_feats)
        structure_scalar_feats = self.to_scalar_structure(evo_scalar_feats)
        structure_pair_feats = self.to_pair_structure(pair_feats)

        structure_input = self.get_structure_input_kwargs(
            scalar_feats=structure_scalar_feats,
            pair_feats=structure_pair_feats,
            sample=sample
        )
        # Needed for backward compatibility :/
        structure_net = default(self.tfn_transformer, self.ipa_transformer)
        out = structure_net(**structure_input)

        scalar_out, coord_out, extra = self.get_structure_output(
            structure_out=out, structure_input=structure_input
        )

        return ModelOutput(
            predicted_coords=coord_out,
            scalar_output=scalar_out,
            pair_output=pair_feats,
            predicted_atom_tys=None,
            model_input=sample,
            extra=self.augment_output(
                model_input=sample,
                coords=coord_out,
                scalar_out=scalar_out,
                evo_scalar=evo_scalar_feats,
                pair=pair_feats,
                extra_out=extra,
            )
        )

    @abstractmethod
    def get_structure_output(
            self, structure_out: Any, structure_input: Dict
    ) -> Tuple[Tensor, Tensor, Optional[Any]]:
        """Get output of structure module (e.g. scalar,coord features)"""

    @abstractmethod
    def get_structure_input_kwargs(self, scalar_feats, pair_feats, sample: ModelInput) -> Dict:
        """Get input for structure module"""
        pass

    @abstractmethod
    def augment_output(self, model_input, coords, scalar_out, evo_scalar, pair, extra_out):
        """Augment the output"""
        pass

    @abstractmethod
    def compute_loss(self, output: ModelOutput, **kwargs) -> ModelLoss:
        """Compute the loss"""
        pass
