import torch
from torch import nn, Tensor
from einops import rearrange
from torch import Tensor, einsum
from typing import Optional
import torch
from torch.distributions import LowRankMultivariateNormal

batch_mat_vec = lambda A, x: einsum("... i j, ... j->i", A, x)

EPS = 1e-6


class RBF(nn.Module):
    def __init__(self, start: float, end: float, steps: int):
        super(RBF, self).__init__()
        self.start, self.end, self.steps = start, end, steps
        self._radii = None

    def radii(self, x: Tensor) -> Tensor:
        if self._radii is None:
            radii = torch.linspace(self.start, self.end, self.steps)
            self._radii = radii.unsqueeze(0).to(x.device)
        return self._radii

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "... -> ... ()") if x.shape[-1] > 1 else x
        in_shape, radii = x.shape, self.radii(x)
        x = x.reshape(-1, 1)
        diffs = torch.square(x - radii)
        return torch.exp(-diffs).reshape(*in_shape[:-1], -1)


class Coords2Pair(nn.Module):
    """Takes coordinate features to pair features"""

    def __init__(
            self,
            coord_dim: int,
            rbf: RBF,
            dim_out: int = 48,
    ):
        super().__init__()
        self.rbf, self.coord_dim, self.dim_out = rbf, coord_dim, dim_out
        self.dim_in = self.rbf.steps * coord_dim
        self.project_out = nn.Sequential(nn.LayerNorm(self.dim_in), nn.Linear(self.dim_in, self.dim_out))

    def forward(self, coords) -> Tensor:
        """Generate pair features from coordinates
        :param coords: coordinates of shape (b,n,self.coord_dim,3)
        :return: tensor of pair features with shape (b,n,n,self.dim_out)
        """
        b, n = coords.shape[:2]
        rel_coords = rearrange(coords, "b n a c -> b () n a c") - rearrange(coords, "b n a c -> b n () a c")
        dists = torch.clamp_min(torch.norm(rel_coords, dim=-1), EPS)
        radial_encoding = self.rbf(dists).reshape(b, n, n, self.dim_in)
        return self.project_out(radial_encoding.detach())  # stop gradient before projecting out


def sample_kl_divergence(z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)
    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)
    # kl
    kl = (log_qzx - log_pz)
    # sum over last dim to go from single dim distribution to multi-dim
    kl = kl.sum(-1)
    return kl


def sample_diag_gaussian(mu, sigma):
    """Sample from a multivariate gaussian with diagonal covariance matrix"""
    return torch.randn_like(mu) * sigma + mu


def kl_diag_gaussian(mu, sigma):
    return -(1 / 2) * torch.sum(-sigma ** 2 - mu ** 2 + torch.log(sigma ** 2) + 1, dim=-1)


def sample_multivar_gaussian(mu, sigma):
    """Sample from a multivariate gaussian
    :param mu: mean vector of shape (...,n)
    :param sigma: sqrt(covaraince) of shape (...,n,n)
    :return: sample from normal distribution with params (mu, sigma*sigma^T)
    """
    batch_mat_vec(sigma, torch.randn_like(mu)) + mu
