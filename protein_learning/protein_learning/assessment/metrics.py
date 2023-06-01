"""Functions for computing scoring metrics on proteins
"""
import math
from itertools import combinations
from math import pi as PI  # noqa
from typing import Optional, List, Tuple

import torch
from torch import Tensor

from protein_learning.common.helpers import (
    get_eps,
    calc_tm_torch,
    default,
    masked_mean,
    exists
)
from protein_learning.common.protein_constants import (
    AA_INDEX_MAP,
    ALL_ATOM_POSNS,
)


def tensor_to_list(x: Tensor) -> List:
    """Convert torch tensor to python list"""
    return x.detach().cpu().numpy().tolist()


def batch_coords(predicted_coords: Tensor, actual_coords: Tensor, batched_len: int):
    """(potentially) adds batch dimension to coordinates and returns whether
    coordinates already had a batch dimension
    """
    batched = predicted_coords.ndim == batched_len
    actual = actual_coords if actual_coords.ndim == batched_len else actual_coords.unsqueeze(0)
    pred = predicted_coords if predicted_coords.ndim == batched_len else predicted_coords.unsqueeze(0)
    assert actual.ndim == pred.ndim == batched_len
    return batched, pred, actual


def compute_coord_lddt(
        predicted_coords: Tensor,
        actual_coords: Tensor,
        cutoff: float = 15.,
        per_residue: bool = True
) -> Tensor:
    """Computes LDDT of predicted and actual coords.

    :param predicted_coords: tensor of shape (b, n, 3) or (n,3)
    :param actual_coords: tensor of shape (b, n, 3) or (n,3)
    :param cutoff: LDDT cutoff value
    :param per_residue: whether to compute LDDT per-residue or for all coords.
    :return: LDDT or pLDDT tensor
    """
    # reshape so that each set of coords has batch dimension
    batched, actual_coords, predicted_coords = batch_coords(
        predicted_coords=predicted_coords,
        actual_coords=actual_coords,
        batched_len=3
    )
    n = predicted_coords.shape[1]
    pred_dists = torch.cdist(predicted_coords, predicted_coords)
    actual_dists = torch.cdist(actual_coords, actual_coords)
    not_self = (1 - torch.eye(n, device=pred_dists.device)).bool()
    mask = torch.logical_and(pred_dists < cutoff, not_self).float()
    l1_dists = torch.abs(pred_dists - actual_dists).detach()

    scores = 0.25 * ((l1_dists < 0.5).float() +
                     (l1_dists < 1.0).float() +
                     (l1_dists < 2.0).float() +
                     (l1_dists < 4.0).float())

    dims = (1, 2) if not per_residue else (2,)
    eps = get_eps(l1_dists)
    scale = 1 / (eps + torch.sum(mask, dim=dims))
    scores = eps + torch.sum(scores * mask, dim=dims)
    return scale * scores if batched else (scale * scores)[0]


def compute_coord_tm(predicted_coords: Tensor, actual_coords: Tensor, norm_len: Optional[int] = None) -> Tensor:
    """Compute TM-Score of predicted and actual coordinates"""
    # reshape so that each set of coords has batch dimension
    batched, actual_coords, predicted_coords = batch_coords(
        predicted_coords=predicted_coords,
        actual_coords=actual_coords,
        batched_len=3
    )
    deviations = torch.norm(predicted_coords - actual_coords, dim=-1)
    norm_len = default(norm_len, predicted_coords.shape[1])
    tm = calc_tm_torch(deviations, norm_len=norm_len)
    return tm if batched else tm[0]


def mean_aligned_error(
        predicted_coords: Tensor,
        actual_coords: Tensor,
        mask: Optional[Tensor],
        per_residue: bool,
        fn=lambda x: torch.square(x)
):
    """mean per-residue error w.r.t given function"""
    # reshape so that each set of coords has batch dimension
    batched, actual_coords, predicted_coords = batch_coords(
        predicted_coords=predicted_coords,
        actual_coords=actual_coords,
        batched_len=4 if per_residue else 3
    )
    if exists(mask):
        mask = mask if batched else mask.unsqueeze(0)
        assert mask.ndim == actual_coords.ndim - 1


    tmp = torch.sum(fn(predicted_coords - actual_coords), dim=-1)
    mean_error = masked_mean(tmp, mask, dim=-1)
    return tmp if per_residue else mean_error


def compute_coord_rmsd(
        predicted_coords: Tensor,
        actual_coords: Tensor,
        atom_mask: Optional[Tensor],
        per_res: bool = True
) -> Tensor:
    """Computes RMSD between predicted and actual coordinates

    :param predicted_coords: tensor of shape (...,n,a,3) if per_res
    is specified, otherwise (...,n,3) - where a is number of atom types
    :param actual_coords: tensor of shape (...,n,a,3)  if per_res
    is specified, otherwise (...,n,3) - where a is number of atom types
    :param atom_mask: mask tensor of shape (...,n,a) if per_res is specified
    otherwise (...,n)
    :param per_res: whether to return deviation for each residue,
    or for the entire structure.
    :return: RMSD
    """
    # reshape so that each set of coords has batch dimension
    mse = mean_aligned_error(
        predicted_coords=predicted_coords,
        actual_coords=actual_coords,
        mask=atom_mask,
        fn=torch.square,
        per_residue=per_res
    )
    return torch.sqrt(mse)


def compute_coord_mae(
        predicted_coords: Tensor,
        actual_coords: Tensor,
        atom_mask: Optional[Tensor],
        per_res: bool = True
) -> Tensor:
    """Computes mean l1 deviatoin between predicted and actual coordinates

    :param predicted_coords: tensor of shape (...,n,a,3) if per_res
    is specified, otherwise (...,n,3) - where a is number of atom types
    :param actual_coords: tensor of shape (...,n,a,3)  if per_res
    is specified, otherwise (...,n,3) - where a is number of atom types
    :param atom_mask: mask tensor of shape (...,n,a) if per_res is specified
    otherwise (...,n)
    :param per_res: whether to return deviation for each residue,
    or for the entire structure.
    :return: Mean l1 coordinate deviation
    """
    # reshape so that each set of coords has batch dimension
    return mean_aligned_error(
        predicted_coords=predicted_coords,
        actual_coords=actual_coords,
        mask=atom_mask,
        fn=lambda x: torch.sqrt(torch.square(x) + get_eps(x)),
        per_residue=per_res
    )


def per_residue_neighbor_counts(atom_coords: Tensor, mask: Optional[Tensor], dist_cutoff=10):
    """Computes the number of coordinates within a dist_cutoff radius of each input coordinate
    :param atom_coords: Tensor of shape (...,n,3).
    :param mask: ignore coordinate i if mask[...,i] is False (optional)
    :param dist_cutoff: cutoff distance for two coordinates to be considered neighbors
    :return: number of neighbors per atom
    """
    batched = atom_coords.ndim == 3
    rel_dists = torch.cdist(atom_coords, atom_coords)
    dist_mask = torch.einsum("... n, ... m-> ... nm", mask, mask) if exists(mask) else \
        torch.ones(1, device=atom_coords.device).bool()
    exclude_self_mask = torch.eye(atom_coords.shape[-2], device=atom_coords.device)
    mask = torch.logical_and(dist_mask, exclude_self_mask.unsqueeze(0) if batched else exclude_self_mask)
    rel_dists[mask] = dist_cutoff + 1
    return torch.sum((rel_dists < dist_cutoff), dim=-1)


def compute_angle_mae(source: Tensor, target: Tensor) -> Tensor:
    """computes absolute error between two lists of angles"""
    a = source - target
    a[a > PI] -= 2 * PI
    a[a < -PI] += 2 * PI
    return torch.abs(a)


def calculate_sequence_identity(pred_seq: Tensor, target_seq: Tensor) -> Tensor:
    """Calculate average sequence identity between pred_seq and target_seq"""
    return torch.mean((pred_seq == target_seq).float())


def detect_disulfide_bond_pairs(target_seq: Tensor, target_coords: Tensor) -> List[Tuple[int, int]]:
    """Returns Cysteine pairs forming disulfide bonds"""
    # calculate cystine positions
    cys_posns = torch.arange(target_seq.numel())[target_seq == AA_INDEX_MAP["CYS"]]
    SG = ALL_ATOM_POSNS["SG"]
    is_bond = lambda p1, p2: torch.norm(target_coords[p1, SG] - target_coords[p2, SG]) < 2.5
    return list(filter(lambda x: is_bond(*x), combinations(tensor_to_list(cys_posns), 2)))


def calculate_average_entropy(pred_aa_logits: Tensor):
    """Calculate Average Entropy"""
    log_probs = torch.log_softmax(pred_aa_logits, dim=-1)
    probs = torch.exp(log_probs)
    return torch.mean(torch.sum(-probs * (log_probs * math.log2(math.e)), dim=-1))  # entropy


def calculate_perplexity(pred_aa_logits: Tensor, true_labels: Tensor):
    """Calculate Perplexity"""
    ce = torch.nn.CrossEntropyLoss()
    return torch.exp(ce(pred_aa_logits, true_labels))


def calculate_unnormalized_confusion(pred_labels: Tensor, true_labels: Tensor):
    """Calculate (un-normalized) confusion"""
    pred_one_hot, target_one_hot = map(lambda x: torch.nn.functional.one_hot(x, 21), (pred_labels, true_labels))
    return torch.einsum("n i, n j -> i j", pred_one_hot.float(), target_one_hot.float())
