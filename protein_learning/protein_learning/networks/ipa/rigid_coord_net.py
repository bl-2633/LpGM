import torch
from torch import nn, Tensor
from protein_learning.networks.common.rigid import Rigids
from einops import rearrange


class RigidCoordNet(nn.Module):
    def __init__(self,
                 dim_in,
                 compute_backbone: bool = True,
                 compute_sidechain: bool = False,
                 compute_cb: bool = True,
                 n_blocks: int = 2,
                 ):
        super(RigidCoordNet, self).__init__()
        self.compute_backbone = compute_backbone
        self.compute_sidechain = compute_sidechain
        self.compute_cb = compute_cb
        self.n_blocks = n_blocks

        num_angles = 0
        num_angles += 3 if compute_backbone else 0
        num_angles += 4 if compute_sidechain else 0

        self.project_in = nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, dim_in))
        self.angle_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_in)
        )
        self.to_angles = nn.Linear(dim_in, 2 * num_angles)

    def forward(self, scalar_feats: Tensor, rigids: Rigids) -> Tensor:
        angle_feats = self.project_in(scalar_feats)
        for _ in range(self.n_blocks):
            angle_feats = angle_feats + self.angle_net(angle_feats)
        angles = self.to_angles(angle_feats)
        angles = rearrange(angles, "b n (a c) -> b n a c", c=2)

    def compute_bb_rigids(self, rigids: Rigids, angles: Tensor):
        pass

    def make_rot_x(self, angles: Tensor):
        b, n, a, _ = angles.shape
        rots = torch.zeros(b, n, a, 3, 3, device=angles.device)
        rots[:, :, :, 0, 0] = 1
        for idx in range(a):
            rots[:, :, idx, 1, 1] = angles[:, :, idx, 0]
            rots[:, :, idx, 1, 2] = -angles[:, :, idx, 1]
            rots[:, :, idx, 2, 1] = angles[:, :, idx, 1]
            rots[:, :, idx, 2, 2] = angles[:, :, idx, 0]
        return rots
