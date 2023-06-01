from torch import nn, Tensor
from protein_learning.networks.vae.net_utils import (
    kl_diag_gaussian,
    sample_diag_gaussian,
)
from protein_learning.networks.vae.vae_config import VAEConfig
from protein_learning.networks.vae.vae_abc import EncoderABC, DecoderABC
from typing import Dict, Any, Optional
from protein_learning.common.helpers import default


class VAE(nn.Module):
    """VAE for proteins"""

    def __init__(
            self,
            config: VAEConfig,
            encoder: EncoderABC,
            decoder: DecoderABC,
    ):
        super(VAE, self).__init__()
        self.config = config
        self.encoder, self.decoder = encoder, decoder
        # pair and scalar input projection

    def forward(
            self,
            scalar_feats: Tensor,
            pair_feats: Tensor,
            coords: Tensor,
            mask: Tensor,
            encoder_kwargs: Optional[Dict[str, Any]],
            decoder_kwargs: Optional[Dict[str, Any]],
    ):
        """Run the model"""
        encoder_kwargs = default(encoder_kwargs, {})
        decoder_kwargs = default(decoder_kwargs, {})

        mu, sigma = self.encoder(
            scalar_feats=scalar_feats,
            pair_feats=pair_feats,
            coords=coords,
            mask=mask,
            **encoder_kwargs
        )
        latent_scalars = self.sample(mu, sigma)

        decoder_out = self.decoder(
            **self.decoder.get_input_from_encoder_input(
                latent_scalars=latent_scalars,
                pair_feats=pair_feats,
                coords=coords,
                mask=mask,
            ),
            **decoder_kwargs
        )
        return decoder_out, self.get_kld(mu, sigma)

    def sample(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """Get a sample from encoder latent distribution"""
        if self.config.use_diag_cov:
            return sample_diag_gaussian(mu=mu, sigma=sigma)
        raise Exception("Not yet implemented!")

    def get_kld(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """Get a sample from encoder latent distribution"""
        if self.config.use_diag_cov:
            return kl_diag_gaussian(mu, sigma)
        raise Exception("Not yet implemented!")
