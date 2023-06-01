from __future__ import annotations
from typing import Tuple
from torch import Tensor
import torch
from protein_learning.common.helpers import safe_normalize
from pytorch3d.transforms import (  # noqa
    quaternion_multiply,  # noqa
    quaternion_to_matrix,  # noqa
    matrix_to_quaternion,  # noqa
    quaternion_invert,  # noqa
    quaternion_apply  # noqa
)
from einops import rearrange

rot_mul_vec = lambda x, y: torch.einsum("... i j, ... j -> ... i", x, y)


class Rigids:
    """Rigid Transformations (rotation + translation)"""

    def __init__(
            self,
            quaternions: Tensor,
            translations: Tensor,
    ):
        self.quaternions, self.translations = quaternions, translations

    @property
    def rotations(self) -> Tensor:
        """Gets underlying rotation matrices"""
        return quaternion_to_matrix(self.quaternions)

    def difference(self, other: Rigids):
        quat_diff = torch.norm(self.quaternions - other.quaternions)
        trans_diff = torch.norm(self.translations - other.translations)
        return quat_diff, trans_diff

    def compose(self, other: Rigids) -> Rigids:
        """Compose two rigid transformations"""
        rots = self.rotations
        quaternions = matrix_to_quaternion(rots @ other.rotations)
        translations = self.translations + rot_mul_vec(rots, other.translations)
        return Rigids(quaternions=quaternions, translations=translations)

    def detach_rot(self) -> Rigids:
        """detach rotation from computational graph"""
        self.quaternions = self.quaternions.detach()
        return self

    def detach_all(self) -> Rigids:
        """Detaches rotation and translation from gradient computations"""
        self.quaternions = self.quaternions.detach()
        self.translations = self.translations.detach()
        return self

    def to(self, device: str = 'cpu') -> Rigids:
        """Send the Rigids object to a specified device"""
        self.quaternions = self.quaternions.to(device)
        self.translations = self.translations.to(device)
        return self

    def rotate(self, points):
        """Rotate points according to the underlying `Rigids` rotation"""
        quats = self.quaternions if points.ndim == 3 else self.quaternions.unsqueeze(-2)
        return quaternion_apply(quats, points)

    def apply(self, points: Tensor) -> Tensor:
        """Applies rigid transformation to points
        :param points: tensor of shape (b,n,*,3) where * is (optional) atom axis
        to apply i=1..nth quaternion t0.
        :return:
        """
        quats, trans = self.quaternions, self.translations
        if points.ndim == 4:
            quats, trans = quats.unsqueeze(-2), trans.unsqueeze(-2)
        return quaternion_apply(quats, points) + trans

    def apply_by_rot(self, points):
        quats, trans = self.quaternions, self.translations
        if points.ndim == 4:
            quats, trans = quats.unsqueeze(-2), trans.unsqueeze(-2)
        return rot_mul_vec(quaternion_to_matrix(quats), points) + trans

    def apply_inverse(self, points: Tensor) -> Tensor:
        """Inverse of apply
        """
        quats, trans = quaternion_invert(self.quaternions), self.translations
        if points.ndim == 4:
            quats, trans = quats.unsqueeze(-2), trans.unsqueeze(-2)
        return quaternion_apply(quats, points - trans)

    def get_inverse(self) -> Rigids:
        """Get the inverse of this transformation"""
        quats = quaternion_invert(self.quaternions)
        trans = quaternion_apply(quats, self.translations)
        return Rigids(quats, -trans)

    def scale(self, factor: float) -> Rigids:
        """Scale transformation by factor"""
        return Rigids(self.quaternions, self.translations * factor)

    @classmethod
    def IdentityRigid(cls, leading_shape: Tuple[int, ...], device: str = 'cpu') -> Rigids:
        """Rigid transformation representing identity (no transform)
        :param leading_shape: (b,n) or (n,)
        :param device: device to place translation/rotation tensors on
        :return: Identity rigid transformation
        """
        translations = torch.zeros((*leading_shape, 3), device=device)
        quaternions = torch.zeros((*leading_shape, 4), device=device)
        quaternions[..., 0] = 1
        return cls(quaternions=quaternions, translations=translations)

    @classmethod
    def RigidFromBackbone(cls, coords: Tensor) -> Rigids:
        """Rigid transformation mapping backbone coordinates to local framse
        :param coords: backbone coordinates of shape (b,n,3,3) order = (N, CA, C)
        :return: Rigid Transformation (T_1..T_n) where (R_i,t_i)
        represents rotation
        """
        assert coords.ndim == 4
        b, n, *_ = coords.shape
        N, CA, C = [coords[:, :, i] for i in range(3)]
        rotations = rotation_from_3_points(N, CA, C).reshape(b, n, 3, 3)
        return cls.RididFromRotNTrans(rotations, translations=CA.reshape(b, n, 3))

    @classmethod
    def RididFromRotNTrans(cls, rotations: Tensor, translations: Tensor) -> Rigids:
        """Get a rigid transformation for rotations and translations"""
        return cls(quaternions=matrix_to_quaternion(rotations), translations=translations)

    def crop(self, start: int, end: int) -> Rigids:
        """crop rigids from start : end"""
        self.translations = self.translations[:, start:end]
        self.quaternions = self.quaternions[:, start:end]
        return self


def rotation_from_3_points(p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """Get rotation matrix from three lists of points
    (p1 -> x-axis, p2-> origin, p3-> xy-plane)
    """
    v1, v2 = p1 - p2, p3 - p2
    e1 = safe_normalize(v1)
    e2 = safe_normalize(v2 - (torch.sum(e1 * v2, dim=-1, keepdim=True) * e1))
    e3 = safe_normalize(torch.cross(e1, e2))
    rot = torch.cat((e1, e2, e3), dim=-1).reshape(-1, 3, 3)
    return rearrange(rot, "... i j->... j i")


def rotmul(rot: Tensor, xs: Tensor) -> Tensor:
    """multiply points xs through rotation matrices"""
    return torch.einsum('... c r, ... r -> ... c', rot, xs)
