from typing import Optional
from protein_learning.networks.config.net_config import NetConfig
from typing import Union, Tuple
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from protein_learning.common.helpers import parse_bool


class IPAConfig(NetConfig):
    """Evoformer configuration"""

    def __init__(
            self,
            scalar_dim_in: int,
            pair_dim: Optional[int],
            coord_dim_out: Optional[int] = 4,
            heads: int = 8,
            scalar_key_dim: int = 16,
            scalar_value_dim: int = 16,
            point_key_dim: int = 4,
            point_value_dim: int = 4,
            depth: int = 1,
            dropout: float = 0,
            ff_mult: float = 1,
            num_ff_layers: int = 3,
            share_weights: bool = False,
    ):
        super(IPAConfig, self).__init__()
        self.scalar_dim_in = scalar_dim_in
        self.pair_dim = pair_dim
        self.depth = depth
        self.dropout = dropout
        self.coord_dim_out = coord_dim_out
        self.heads = heads
        self.scalar_key_dim = scalar_key_dim
        self.scalar_value_dim = scalar_value_dim
        self.point_key_dim = point_key_dim
        self.point_value_dim = point_value_dim
        self.ff_mult = ff_mult
        self.num_ff_layers = num_ff_layers
        self.share_weights = share_weights

    @property
    def require_pair_repr(self):
        return self.pair_dim is not None

    @property
    def compute_coords(self):
        return self.coord_dim_out is not None

    @property
    def scalar_dims(self) -> Tuple[int, int, int]:
        """pair dimensions (input, hidden, output)"""
        return self.scalar_dim_in, -1, self.scalar_dim_in

    @property
    def pair_dims(self) -> Tuple[int, int, int]:
        """pair dimensions (input, hidden, output)"""
        return [self.pair_dim] * 3

    @property
    def coord_dims(self) -> Tuple[int, int, int]:
        return -1, -1, self.coord_dim_out

    @property
    def attn_kwargs(self):
        return dict(
            dim=self.scalar_dim_in,
            heads=self.heads,
            scalar_key_dim=self.scalar_key_dim,
            scalar_value_dim=self.scalar_value_dim,
            point_key_dim=self.point_key_dim,
            point_value_dim=self.point_value_dim,
            pairwise_repr_dim=self.pair_dim,
            require_pairwise_repr=self.require_pair_repr,
        )


def get_config(arg_list):
    parser = ArgumentParser(description="IPA Configuration Settings",  # noqa
                            epilog='IPA configuration settings',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--scalar_dim_in',
        help='',
        type=int,
        default=-1
    )

    parser.add_argument(
        '--pair_dim',
        help='',
        type=Optional[int],
        default=-1
    )

    parser.add_argument(
        '--coord_dim_out',
        help='',
        type=Optional[int],
        default=4
    )

    parser.add_argument(
        '--heads',
        help='',
        type=int,
        default=8
    )

    parser.add_argument(
        '--scalar_key_dim',
        help='',
        type=int,
        default=16
    )

    parser.add_argument(
        '--scalar_value_dim',
        help='',
        type=int,
        default=16
    )

    parser.add_argument(
        '--point_key_dim',
        help='',
        type=int,
        default=4
    )

    parser.add_argument(
        '--point_value_dim',
        help='',
        type=int,
        default=4
    )

    parser.add_argument(
        '--depth',
        help='',
        type=int,
        default=1
    )

    parser.add_argument(
        '--dropout',
        help='',
        type=float,
        default=0
    )

    parser.add_argument(
        '--ff_mult',
        help='',
        type=float,
        default=1
    )

    parser.add_argument(
        '--num_ff_layers',
        help='',
        type=int,
        default=3
    )

    return parser.parse_args(arg_list)
