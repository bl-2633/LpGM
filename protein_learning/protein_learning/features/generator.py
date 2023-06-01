from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.input_features import InputFeatures, get_input_features
from torch import Tensor
from abc import abstractmethod
from typing import List, Optional


class FeatureGenerator:

    def __init__(
            self,
            config: InputFeatureConfig,
    ):
        self.config = config

    @abstractmethod
    def generate_features(
            self,
            seq: str,
            coords: Tensor,
            coord_mask: Tensor,
            atom_tys: List[str],
            res_ids: Optional[Tensor],
    ) -> InputFeatures:
        """Generate input features"""
        pass


class DefaultFeatureGenerator(FeatureGenerator):
    def __init__(
            self,
            config: InputFeatureConfig,
    ):
        super(DefaultFeatureGenerator, self).__init__(config)

    def generate_features(
            self,
            seq: str,
            coords: Tensor,
            res_ids: Tensor,
            coord_mask: Tensor,
            atom_tys: List[str],
    ) -> InputFeatures:
        feats = get_input_features(
            seq=seq,
            coords=coords,
            res_ids=res_ids,
            atom_ty_to_coord_idx={a: i for i, a in enumerate(atom_tys)},
            config=self.config
        )
        return InputFeatures(features=feats, batch_size=1, length=len(seq)).maybe_add_batch()
