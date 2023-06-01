"""Utility functions for working with protein side-chains"""
import torch
from protein_learning.common.protein_constants import (
    SYMM_SC_RES_ATOMS,
    SYMM_SC_RES_TYPES,
    SYMM_SC_RES_TYPE_SET,
    ALL_ATOM_POSNS,
    ONE_TO_THREE,
    chi1_atom_posns,
    chi2_atom_posns,
    chi3_atom_posns,
    chi4_atom_posns,
    chi_pi_periodic,
)
from protein_learning.common.helpers import masked_mean, batched_index_select
from protein_learning.protein_utils.dihedral.angle_utils import signed_dihedral_4
from torch import Tensor
from typing import List, Dict, Tuple
from einops import rearrange, repeat  # noqa

RES_KEY_TO_ATOM_POSNS = torch.tensor([
    [
        [ALL_ATOM_POSNS[c], ALL_ATOM_POSNS[d]],
        [ALL_ATOM_POSNS[a], ALL_ATOM_POSNS[b]]
    ]
    for [[a, b], [c, d]] in SYMM_SC_RES_ATOMS
]).long()
RES_TO_KEY = {a: i for i, a in enumerate(SYMM_SC_RES_TYPES)}


def chi_mask_n_indices(
        seq: str,
        sc_atom_mask: Tensor,
        chi: List[str],
        chi_posns: Dict[str, int]
) -> Tuple[Tensor, Tensor]:
    """Gets chi-dihedral mask and atom indices"""
    mask, posns = torch.zeros(len(seq)), []
    for idx, res in enumerate(seq):
        res = ONE_TO_THREE[res]
        if res in chi:
            if torch.all(sc_atom_mask[..., idx, chi_posns[res]]):
                mask[idx] = 1
                posns.append(chi_posns[res])
    return mask.bool(), torch.tensor(posns).long()


def get_chi_pi_periodic_mask(seq: str) -> Tensor:
    """Get chi1-4 pi-periodic mask"""
    masks = [chi_pi_periodic[r] for r in seq]
    return rearrange(torch.tensor(masks).bool(), "n c -> c n")


# data entries -> (pdb, res_ty, res_idx, rmsd, num_neighbors, chis)
def _per_residue_rmsd(predicted, native, mask, fn=lambda x: torch.square(x), reduce=False):
    tmp = torch.sum(fn(predicted - native), dim=-1)
    if not reduce:
        return masked_mean(tmp, mask, dim=-1)
    else:
        return torch.sum(masked_mean(tmp, mask, dim=-1)) / max(1, torch.sum(mask.any(dim=-1)))


def per_residue_chi_indices_n_mask(coord_mask, seq):
    assert coord_mask.ndim == 2, f"{coord_mask.shape}"
    posns, chi_mask = [[] for _ in range(4)], [[] for _ in range(4)]
    default_posns = [0, 1, 2, 3]
    chi_maps = [chi1_atom_posns, chi2_atom_posns, chi3_atom_posns, chi4_atom_posns]
    for chi_idx, chi_posns in enumerate(chi_maps):
        for res_idx, res in enumerate(seq):
            if res in chi_posns:
                atom_posns = chi_posns[res]
                posns[chi_idx].append(atom_posns)
                if not torch.all(coord_mask[res_idx, atom_posns]):
                    chi_mask[chi_idx].append(0)
                else:
                    chi_mask[chi_idx].append(1)
            else:
                chi_mask[chi_idx].append(0)
                posns[chi_idx].append(default_posns)

    return torch.tensor(posns), torch.tensor(chi_mask).bool()


def get_symmetric_residue_keys_n_indices(seq) -> Tuple[Tensor, Tensor]:
    keys, indices = [], []
    for idx, res_ty in enumerate(seq):
        if res_ty in SYMM_SC_RES_TYPE_SET:
            keys.append(RES_TO_KEY[res_ty])
            indices.append(idx)
    return torch.tensor(keys).long(), torch.tensor(indices).long()


def swap_symmetric_atoms(atom_coords: Tensor, symm_keys_n_indices: Tuple[Tensor, Tensor]) -> Tensor:
    """swap symmetric side chain atom coordinates"""
    keys, indices = symm_keys_n_indices
    keys, indices = keys.to(atom_coords.device), indices.to(atom_coords.device)
    # make a copy of the original coordinates
    swapped_coords = atom_coords.detach().clone()
    # get positions of equivalent atoms
    equiv_atom_positions = RES_KEY_TO_ATOM_POSNS[keys]
    for i in range(2):
        # swap the equivalent atoms
        posns_to, posns_from = equiv_atom_positions[:, i, 0], equiv_atom_positions[:, i, 1]
        swapped_coords[..., indices, posns_to, :] = atom_coords[..., indices, posns_from, :]
        swapped_coords[..., indices, posns_from, :] = atom_coords[..., indices, posns_to, :]
    return swapped_coords


def get_sc_dihedral(coords: Tensor, chi_mask: Tensor, chi_indices: Tensor) -> Tensor:
    """Get side-chain dihedral angles"""
    m = chi_mask.shape[0]
    assert m <= 4, f"{chi_mask.shape}"
    coords = coords if coords.ndim == 4 else coords.unsqueeze(0)
    rep_coords = repeat(coords, "b n a c -> (m b) n a c", m=m)
    assert rep_coords.shape[:2] == chi_indices.shape[:2]
    assert chi_indices.shape[2] == 4
    select_coords = batched_index_select(rep_coords, chi_indices, dim=2)
    assert select_coords.shape[:3] == chi_indices.shape
    return signed_dihedral_4(ps=rearrange(select_coords, "b n a c -> a b n c"))


def align_symmetric_sidechains(native_sc, predicted_sc, sc_mask, symm_keys_n_indxs, align_fn=_per_residue_rmsd):
    """Align side chain atom coordinates"""
    with torch.no_grad():
        initial_rmsds = _per_residue_rmsd(predicted_sc, native_sc, sc_mask)
        # swap atoms in symmetric sidechains and get swapped rmsds
        swapped_native = swap_symmetric_atoms(native_sc, symm_keys_n_indxs)
        swapped_rmsds = align_fn(predicted_sc, swapped_native, sc_mask)
        swap_mask = initial_rmsds >= swapped_rmsds
        native_sc[swap_mask] = swapped_native[swap_mask]
        return native_sc, predicted_sc
