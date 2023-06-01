import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Union, List

cos_max, cos_min = (1 - 1e-9), -(1 - 1e-9)


def get_bb_dihedral(N: Tensor, CA: Tensor, C: Tensor) -> Tuple[Tensor, ...]:
    """
    Gets backbone dihedrals for 
    :param N: (n,3) or (b,n,3) tensor of backbone Nitrogen coordinates
    :param CA: (n,3) or (b,n,3) tensor of backbone C-alpha coordinates
    :param C: (n,3) or (b,n,3) tensor of backbone Carbon coordinates
    :return: phi, psi, and omega dihedrals angles (each of shape (n,) or (b,n))
    """
    assert all([len(N.shape) == len(x.shape) for x in (CA, C)])
    squeeze = len(N.shape) == 2
    N, CA, C = map(lambda x: x.unsqueeze(0), (N, CA, C)) if squeeze else (N, CA, C)
    b, n = N.shape[:2]
    phi, psi, omega = [torch.zeros(b, n, device=N.device) for _ in range(3)]
    phi[:, 1:] = signed_dihedral_4_torch([C[:, :-1], N[:, 1:], CA[:, 1:], C[:, 1:]])
    psi[:, :-1] = signed_dihedral_4_torch([N[:, :-1], CA[:, :-1], C[:, :-1], N[:, 1:]])
    omega[:, :-1] = signed_dihedral_4_torch([CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:]])
    return map(lambda x: x.squeeze(0), (phi, psi, omega)) if squeeze else (phi, psi, omega)


def signed_dihedral_4_torch(ps: Union[Tensor, List[Tensor]],return_mask=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    computes (signed) dihedral angle of four points
    ps is assumed to have 4 columns, each row corresponds
    to four points whose dihedral angle is to be taken
    """
    p0, p1, p2, p3 = ps
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    mask = torch.norm(b1, dim=-1) > 1e-7
    b1 = torch.clamp_min(b1, 1e-6)
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v) * w, dim=-1)
    res = torch.atan2(y, x)
    return res if not return_mask else (res, mask)


if __name__ == '__main__':
    N_tensor = torch.randn(size = (100, 3))
    CA_tensor = torch.randn(size= (100, 3))
    C_tensor = torch.randn(size = (100, 3))

    x = get_bb_dihedral(N_tensor, CA_tensor, C_tensor)
    print(list(x)[0])
