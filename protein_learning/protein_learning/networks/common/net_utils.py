import torch
from einops import rearrange
from torch import nn
from typing import Optional, Tuple
from protein_learning.common.helpers import default, exists
from torch import Tensor, tensor, tensor_split  # noqa

REZERO_INIT = 0.025


class SplitLinear(nn.Module):
    """Linear projection from one input to several outputs of varying sizes
    """

    def __init__(self, dim_in, dim_out, bias=True, chunks=1, sizes=None):
        super(SplitLinear, self).__init__()
        self.dim_in, self.dim_out = dim_in, dim_out
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        if not exists(sizes) and chunks == 1:
            self.to_out = lambda x: x
        else:
            sizes = tensor(default(sizes, [dim_out // chunks] * chunks))
            assert sum(sizes) == dim_out
            self.sizes = torch.cumsum(sizes, dim=0).long()[:-1]
            self.to_out = lambda x: tensor_split(x, self.sizes, dim=-1)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Compute linear projections"""
        return self.to_out(self.linear(x))


class _Identity(nn.Module):

    def __init__(self, *args, **kwargs):  # noqa
        super(_Identity, self).__init__()

    def forward(self, *args):  # noqa
        return args


class LearnedOuterProd(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=32, pre_norm=True):
        super(LearnedOuterProd, self).__init__()
        self.pre_norm = nn.LayerNorm(dim_in) if pre_norm else nn.Identity()
        self.project_in_a = nn.Linear(dim_in, dim_hidden)
        self.project_in_b = nn.Linear(dim_in, dim_hidden)
        self.project_out = nn.Linear(dim_hidden, dim_out)

    def forward(self, feats):
        feats = self.pre_norm(feats)
        a, b = self.project_in_a(feats), self.project_in_b(feats)
        outer = rearrange(a, 'b n d -> b n () d') * rearrange(b, 'b n d -> b () n d')
        return self.project_out(outer)


class Transition(nn.Module):
    """FeedForward Transition"""

    def __init__(self, dim_in: int, dim_out: Optional[int], mult: int = 2, pre_norm=True, nonlin=nn.GELU):
        super().__init__()
        dim_out = default(dim_out, dim_in)
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in) if pre_norm else nn.Identity(),
            nn.Linear(dim_in, mult * dim_in),
            nonlin(),
            nn.Linear(2 * dim_in, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class IdentityResidual(nn.Module):
    """Identity Residual"""

    def __init__(self):
        super().__init__()

    def forward(self, out, res):  # noqa
        return res


class ReZero(nn.Module):
    """ReZero Residual"""

    def __init__(self):
        super(ReZero, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1).float() + REZERO_INIT)

    def forward(self, out, res):
        if out.shape != res.shape:
            print("WARNING: rezero sizes don't match!")
            return out
        return self.alpha * out + res


class Residual(nn.Module):
    """Residual"""

    def __init__(self, use_rezero: bool = True, use_identity: bool = False):
        super(Residual, self).__init__()
        self.residual = ReZero() if use_rezero else None
        self.residual = IdentityResidual() if use_identity else self.residual
        self.residual = self.residual if exists(self.residual) else lambda out, res: out + res

    def forward(self, out, res):
        return self.residual(out, res)
