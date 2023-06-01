"""VAE model"""
from protein_learning.common.data.model_data import ModelInput, ModelOutput, ModelLoss
from protein_learning.models.utils.model_abc import ProteinModel
from protein_learning.common.model_config import ModelConfig
from protein_learning.networks.evoformer.evoformer_config import EvoformerConfig
from protein_learning.networks.ipa.ipa_config import IPAConfig
from protein_learning.networks.loss.coord_loss import FAPELoss
from protein_learning.networks.evoformer.evoformer import Evoformer
from protein_learning.networks.vae.decoder import Decoder
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.networks.vae.net_utils import (
    kl_diag_gaussian,
    sample_diag_gaussian,
)
from typing import List, Tuple
from torch import Tensor, nn
import torch
from protein_learning.networks.loss.coord_loss import TMLoss
from protein_learning.common.global_constants import get_logger

logger = get_logger(__name__)


class DesignVAE(ProteinModel):
    """VAE for protein imputation"""

    def __init__(
            self,
            model_config: ModelConfig,
            input_embedding: InputEmbedding,
            latent_dim: int,
            scalar_dim_hidden: int,
            pair_dim_hidden: int,
            evoformer_scalar_heads_n_dim: List[int],
            evoformer_pair_heads_n_dim: List[int],
            ipa_heads: int,
            ipa_head_dims: List[int],
            encoder_decoder_depth: List[int],
            use_diag_cov: bool = True,
            coord_dim_out: int = 4
    ):
        super(DesignVAE, self).__init__()
        self.use_diag_cov = use_diag_cov
        self.model_config = model_config
        self.latent_dim = latent_dim
        self.evoformer_config = EvoformerConfig(
            node_dim_in=scalar_dim_hidden,
            edge_dim_in=pair_dim_hidden,
            depth=encoder_decoder_depth[0],
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
            depth=1,
            coord_dim_out=coord_dim_out
        )

        self.encoder = Evoformer(config=self.evoformer_config)
        self.decoder = Decoder(
            scalar_dim=scalar_dim_hidden,
            pair_dim=pair_dim_hidden,
            depth=encoder_decoder_depth[1],
            evoformer_config=self.evoformer_config,
            ipa_config=self.ipa_config,
        )
        self.input_embedding = input_embedding
        s_in, p_in = self.input_embedding.dims

        # encoder projections
        self.encoder_scalar_project_in = nn.Linear(s_in, scalar_dim_hidden)
        self.encoder_pair_project_in = nn.Linear(p_in, pair_dim_hidden)
        self.latent_scalar_pre_norm = nn.LayerNorm(scalar_dim_hidden)
        self.latent_pair_pre_norm = nn.LayerNorm(pair_dim_hidden)
        dim = scalar_dim_hidden + pair_dim_hidden
        self.to_latent_scalar = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2 * latent_dim)
        )
        dim = pair_dim_hidden
        self.to_latent_pair = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2 * latent_dim)
        )

        # decoder projections
        self.to_decoder_pair = nn.Linear(latent_dim, pair_dim_hidden)
        self.to_decoder_scalar = nn.Linear(latent_dim, scalar_dim_hidden)

        # fape loss
        self.fape = FAPELoss()
        self.tm_loss = TMLoss()

    def forward(self, sample: ModelInput, **kwargs) -> ModelOutput:
        """Run the model"""
        # get input features
        scalar_feats, pair_feats = self.input_embedding(sample.input_features)
        scalar_feats = self.encoder_scalar_project_in(scalar_feats)
        pair_feats = self.encoder_pair_project_in(pair_feats)
        mu_s, sigma_s, mu_e, sigma_e = self.encode(scalar_feats, pair_feats)
        # latent_sample = self.sample(mu, sigma)
        latent_scalar = mu_s
        latent_pair = mu_e
        scalar_feats, pair_feats, coords, rigids = self.decode(latent_scalar, latent_pair)
        return ModelOutput(
            predicted_coords=coords,
            scalar_output=scalar_feats,
            pair_output=pair_feats,
            predicted_atom_tys=None,
            model_input=sample,
            extra=dict(mu=mu_s, sigma=sigma_s, rigids=rigids)
        )

    def compute_loss(self, output: ModelOutput, **kwargs) -> ModelLoss:
        """Compute model loss"""
        logger.info("computing sample loss")
        loss = ModelLoss()
        # compute kl divergence
        tmp, sample = output.extra, output.model_input
        mu, sigma, rigids = tmp["mu"], tmp["sigma"], tmp["rigids"]
        native_coords, native_mask = sample.native.full_coords_n_mask
        res_mask = torch.all(native_mask, dim=-1)

        loss.add_loss(
            loss=torch.mean(self.get_kld(mu, sigma)),
            loss_name="kl-divergence",
            loss_weight=0.,
        )
        loss.add_loss(
            loss=self.fape.forward(
                pred_rigids=rigids,
                true_rigids=None,
                true_coords=native_coords.unsqueeze(0),
                pred_coords=output.predicted_coords,
                residue_mask=res_mask.unsqueeze(0),
                coord_mask=native_mask.unsqueeze(0)
            ),
            loss_name="FAPE",
            loss_weight=1.,
        )
        CA_pred = output.predicted_coords[:, res_mask, 1]
        CA_True = native_coords[res_mask, 1].unsqueeze(0)
        loss.add_loss(
            loss=self.tm_loss(predicted_coords=CA_pred, actual_coords=CA_True),
            loss_name="tm-score",
            loss_weight=0.,
        )

        return loss

    def decode(self, latent_scalar: Tensor, latent_pair: Tensor):
        """Run decoder"""
        logger.info(f"running decoder - latent_sample_shape {latent_scalar.shape}")
        scalar_feats = self.to_decoder_scalar(latent_scalar)
        pair_feats = self.to_decoder_pair(latent_pair)
        b, n, *_ = scalar_feats.shape
        coords = torch.zeros(b, n, self.ipa_config.coord_dim_out, 3, device=pair_feats.device)
        return self.decoder(
            scalar_feats=scalar_feats,
            pair_feats=pair_feats,
            coords=coords,
        )

    def encode(self, scalar_feats, pair_feats) -> Tuple[Tensor, ...]:
        """Run Encoder"""
        logger.info(f"running encoder - scalar {scalar_feats.shape}, pair: {pair_feats.shape}")
        # encode input features
        scalar_feats, pair_feats = self.encoder(scalar_feats, pair_feats)
        # pool pair features and get latent distribution parameters
        pooled_pair = self.latent_pair_pre_norm(pair_feats.mean(dim=-2))
        scalar_feats = self.latent_scalar_pre_norm(scalar_feats)
        feats = torch.cat((scalar_feats, pooled_pair), dim=-1)
        mu_s, sigma_s = self.to_latent_scalar(feats).chunk(2, dim=-1)
        mu_e, sigma_e = self.to_latent_pair(pair_feats).chunk(2, dim=-1)
        return mu_s, sigma_s ** 2, mu_e, sigma_e ** 2

    def sample(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """Get a sample from encoder latent distribution"""
        if self.use_diag_cov:
            return sample_diag_gaussian(mu=mu, sigma=sigma)
        raise Exception("Not yet implemented!")

    def get_kld(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """Get a sample from encoder latent distribution"""
        if self.use_diag_cov:
            return kl_diag_gaussian(mu, sigma)
        raise Exception("Not yet implemented!")
