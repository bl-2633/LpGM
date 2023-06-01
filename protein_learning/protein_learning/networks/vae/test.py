import torch
from protein_learning.networks.evoformer.evoformer_config import EvoformerConfig
from protein_learning.networks.ipa.ipa_config import IPAConfig
from protein_learning.networks.vae.decoder import Decoder

if __name__ == "__main__":
    batch_size = 2
    scalar_dim = 32
    pair_dim = 64
    coord_dim = 4
    num_res = 50

    ipa_config = IPAConfig(
        scalar_dim_in=scalar_dim,
        pair_dim=pair_dim,
        coord_dim_out=coord_dim,
    )
    evoformer_config = EvoformerConfig(
        node_dim_in=scalar_dim,
        edge_dim_in=pair_dim,
        edge_dim_head=16,
        node_dim_head=16,
        node_attn_heads=4,
        edge_attn_heads=2,
    )

    decoder = Decoder(
        scalar_dim=scalar_dim,
        pair_dim=pair_dim,
        depth=3,
        evoformer_config=evoformer_config,
        ipa_config=ipa_config,
        coord_scale=1e-1,
    )

    scalar_in = torch.randn(batch_size, num_res, scalar_dim)
    pair_in = torch.randn(batch_size, num_res, num_res, pair_dim)
    coord_in = torch.randn(batch_size, num_res, coord_dim, 3)
    out = decoder.forward(scalar_in, pair_in, coord_in)
    print("scalar : ", out[0].shape)
    print("pair : ", out[1].shape)
    print("coord : ", out[2].shape)
