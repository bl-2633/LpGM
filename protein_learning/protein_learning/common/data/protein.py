"""Input for protein-based learning model
"""
from __future__ import annotations

import functools
import inspect
import os
#from functools import property  # noqa
from typing import Optional, Dict, List, Union, Any, Tuple

import torch
from torch import Tensor

from protein_learning.common.helpers import default
from protein_learning.common.helpers import exists
from protein_learning.common.io.pdb_utils import (
    extract_atom_coords_n_mask_tensors,
    extract_pdb_seq_from_pdb_file,
)
from protein_learning.common.io.sequence_utils import load_fasta_file
from protein_learning.common.pdb_writer import write_pdb
from protein_learning.common.protein_constants import BB_ATOMS, SC_ATOMS
from protein_learning.protein_utils.align.per_residue import impute_beta_carbon
from protein_learning.protein_utils.sidechain_utils import (
    get_symmetric_residue_keys_n_indices,
    per_residue_chi_indices_n_mask,
)

BB_ATOM_SET, SC_ATOM_SET = set(BB_ATOMS), set(SC_ATOMS)


# TODO: not batched

class Protein:
    """Represents a protein (or batch of proteins)"""

    def __init__(
            self,
            atom_coords: Tensor,
            atom_masks: Tensor,
            atom_tys: List[str],
            seq: str,
            name: str,
            res_ids: Optional[Tensor]
    ):
        self.atom_coords = atom_coords
        self.atom_masks = atom_masks
        self.atom_tys = atom_tys
        self.seq = seq
        self._name = name
        self.res_ids = res_ids

    @classmethod
    def FromPDBAndSeq(
            cls,
            pdb_path: str,
            seq: str,
            atom_tys: Optional[List[str]],
            remove_invalid_residues: bool = True,
    ) -> Protein:
        """Creates Protein object from pdb and sequence"""
        coords, mask, res_ids, seq = extract_atom_coords_n_mask_tensors(
            seq=seq,
            pdb_path=pdb_path,
            atom_tys=atom_tys,
            remove_invalid_residues=remove_invalid_residues
        )
        return cls(
            atom_coords=coords,
            atom_masks=mask,
            seq=seq,
            name=pdb_path,
            atom_tys=atom_tys,
            res_ids=res_ids
        )

    @property
    def name(self):
        return os.path.basename(self._name)

    @property
    def full_coords_n_mask(self) -> Tuple[Tensor, Tensor]:
        """Get full atom coordinates and corresponding atom masks"""
        return self.atom_coords, self.atom_masks

    @property
    def valid_residue_mask(self) -> Tensor:
        return torch.all(self.bb_atom_mask, dim=-1)

    @property
    def atom_ty_set(self):
        return set(self.atom_tys)

    @property
    def atom_positions(self) -> Dict[str, int]:
        """Mapping from atom type to atom position in native coord tensor"""
        return {a: i for i, a in enumerate(self.atom_tys)}

    @property
    def bb_atom_tys(self) -> List[str]:
        """Mapping from atom type to atom position in decoy coord tensor"""
        return list(filter(lambda x: x in BB_ATOM_SET, self.atom_tys))

    @property
    def sc_atom_tys(self) -> List[str]:
        """Mapping from atom type to atom position in decoy coord tensor"""
        return list(filter(lambda x: x in SC_ATOM_SET, self.atom_tys))

    @property
    def bb_atom_coords(self):
        """Coordinates of backbone atoms"""
        return self.get_atom_coords(atom_tys=self.bb_atom_tys)

    @property
    def bb_atom_mask(self):
        """Coordiantes of sidechain atoms"""
        return self.get_atom_masks(atom_tys=self.bb_atom_tys)

    @property
    def sc_atom_coords(self):
        """Coordiantes of sidechain atoms"""
        return self.get_atom_coords(atom_tys=self.sc_atom_tys)

    @property
    def sc_atom_mask(self):
        """Coordiantes of sidechain atoms"""
        return self.get_atom_masks(atom_tys=self.sc_atom_tys)

    @property
    def symmetric_sc_data(self):
        """gets residue positions and atom indices for residues with symmetric sidechains"""
        return get_symmetric_residue_keys_n_indices(self.seq)

    @property
    def sc_dihedral_data(self):
        """Gets residue and sidechain positions for chi1-4 dihedrals"""
        mask = self.get_atom_masks(self.sc_atom_tys)
        return per_residue_chi_indices_n_mask(coord_mask=mask, seq=self.seq)

    def get_atom_coords(
            self,
            atom_tys: Optional[Union[str, List[str]]] = None,
            coords: Optional[Tensor] = None
    ) -> Tensor:
        """Gets the atom coordinates for the given atom types

        Returns:
            - Tensor of shape (...,n,3) if atom_tys is a string
            - Tensor of shape(...,n,a,3) if atom_tys is a list, where a is the
             number of atom_tys given.
            - All atom coordinates if atom_tys is None
        """
        coords = default(coords, self.atom_coords)
        if atom_tys is None:
            return coords
        atom_posns = self.atom_positions
        return coords[..., atom_posns[atom_tys], :] if isinstance(atom_tys, str) else \
            torch.cat([coords[..., atom_posns[ty], :].unsqueeze(-2) for ty in atom_tys], dim=-2)

    def get_atom_masks(self, atom_tys: Optional[Union[str, List[str]]] = None) -> Tensor:
        """Gets the atom masks for the given atom types

        Returns:
            - Tensor of shape (...,n) if atom_tys is a string
            - Tensor of shape(...,n,a) if atom_tys is a list, where a is the
             number of atom_tys given.
            - All atom coordinate mask if atom_tys is None
        """

        if atom_tys is None:
            return self.atom_masks
        atom_posns = self.atom_positions
        return self.atom_masks[..., atom_posns[atom_tys]] if isinstance(atom_tys, str) else \
            torch.cat([self.atom_masks[..., atom_posns[ty]].unsqueeze(-1) for ty in atom_tys], dim=-1)

    def get_atom_coords_n_masks(
            self,
            atom_tys: Optional[Union[str, List[str]]] = None,
            coords: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Gets the atom coordinates and masks for the given atom types

        Returns:
            - Tuple of tensors with shapes (...,n,3), (...,n) if atom_tys is a string
            - Tuple of tensors with shapes (...,n,3), (...,n) if atom_tys is a list,
             where a is the number of atom types given.
            - All atom coordinates and all atom coord. masks if atom_tys is None
        """
        coords = self.get_atom_coords(atom_tys=atom_tys, coords=coords)
        masks = self.get_atom_masks(atom_tys=atom_tys)
        return coords, masks

    def kabsch_align_coords(
            self,
            target: Protein,
            coords: Optional[Tensor] = None,
            atom_tys: Optional[List[str]] = None,
            overwrite: bool = True,
    ) -> Tensor:
        """Aligns this proteins coordinates to other_coords
        :param target: protein to align to
        :param overwrite: overwrite this proteins coordinates
        :param atom_tys: atom types to align on (if none given, all atom types are assumed).
        :return: the aligned coordinates.
        """
        raise Exception("not yet implemented")

    def align_symmetric_sc_atoms(
            self,
            target: Protein,
            atom_tys: Optional[List[str]] = None,
            overwrite: bool = True,
    ):
        """Aligns the sidechain atoms of this protein to those of the target protein.

        For residues with with side-chain symmetries, the coordinates of atoms constituting those
        symmetries will be swapped to minimize side-chain RMSD between this protein and the
        target structure.

        :param target:
        :param atom_tys:
        :param overwrite:
        :return:
        """
        raise Exception("not yet implemented")

    def per_residue_align(
            self,
            atom_tys: List[str],
            target: Protein,
            overwrite: bool = True,
    ):
        """Applies separate rotation/translation to each residue.
        :param atom_tys:
        :param target:
        :param overwrite:
        :return:
        """
        raise Exception("not yet implemented")

    def to_pdb(self, pdb_dir, coords=None) -> None:
        """saves this protein to a .pdb file"""
        assert not pdb_dir.endswith(".pdb")
        os.makedirs(pdb_dir, exist_ok=True)
        pdb_name = self.name if self.name.endswith(".pdb") else self.name + ".pdb"
        pdb_path = os.path.join(pdb_dir, pdb_name)
        coord_dicts = []
        coords = default(coords, self.atom_coords)
        assert coords.ndim == 3
        coords = coords.detach().cpu().numpy()
        for i in range(len(self.seq)):
            atoms = {}
            for atom_idx, atom in enumerate(self.atom_tys):
                if self.atom_masks[i, atom_idx]:
                    atoms[atom] = coords[i, atom_idx].tolist()
            coord_dicts.append(atoms)
        write_pdb(coord_dicts, self.seq, pdb_path)

    def impute_cb(self, override: bool = False, exists_ok: bool = False) -> Tuple[Tensor, Tensor]:
        """Impute CB atom position"""
        coords = self.atom_coords
        n, a, c = coords.shape
        bb_coords, bb_mask = self.get_atom_coords_n_masks(atom_tys=["N", "CA", "C"], coords=coords)
        cb_mask, cb_coords = torch.all(bb_mask, dim=-1), torch.zeros(n, 1, 3, device=coords.device)
        cb_coords[cb_mask] = impute_beta_carbon(bb_coords[cb_mask]).unsqueeze(-2)
        if override:
            self.add_atoms(cb_coords, cb_mask.unsqueeze(-1), ["CB"], exists_ok=exists_ok)
        return cb_coords, cb_mask.unsqueeze(-1)

    def add_atoms(self, atom_coords: Tensor, atom_masks: Tensor, atom_tys: List[str],
                  exists_ok: bool = False) -> Protein:
        atom_exists = any([atom_ty in self.atom_tys for atom_ty in atom_tys])
        if not exists_ok:
            assert not atom_exists
        assert atom_coords.ndim == self.atom_coords.ndim, f"{atom_coords.shape},{self.atom_coords.shape}"
        assert atom_masks.ndim == self.atom_masks.ndim, f"{atom_masks.shape},{self.atom_masks.shape}"
        assert atom_masks.shape[0] == self.atom_masks.shape[0], f"{atom_masks.shape[0]},{self.atom_masks.shape}"
        assert atom_coords.shape[0] == self.atom_coords.shape[0], f"{atom_coords.shape}, {self.atom_coords.shape}"

        new_atom_tys, curr_atom_tys = [x for x in self.atom_tys], self.atom_ty_set
        new_coords, new_masks = torch.clone(self.atom_coords), torch.clone(self.atom_masks)
        for i, atom in enumerate(atom_tys):
            if atom in curr_atom_tys:
                atom_pos = self.atom_positions[atom]
                new_coords[..., atom_pos, :] = atom_coords[..., i, :]
                new_masks[..., atom_pos] = atom_masks[..., i]
            else:
                new_atom_tys.append(atom)
                new_coords = torch.cat((new_coords, atom_coords[..., i, :].unsqueeze(-2)), dim=-2)
                new_masks = torch.cat((new_masks, atom_masks[..., i].unsqueeze(-1)), dim=-1)

        self.atom_coords = new_coords
        self.atom_masks = new_masks
        self.atom_tys = new_atom_tys
        self.__clear_cache__()
        return self

    def to(self, device: Any) -> Protein:
        """Places coords and mask on given device"""
        self.atom_coords = self.atom_coords.to(device)
        self.atom_masks = self.atom_masks.to(device)
        self.res_ids = self.res_ids.to(device) if exists(self.res_ids) else None
        return self

    def crop(self, start: int, end: int) -> Protein:
        assert end <= len(self.seq)
        self.seq = self.seq[start:end]
        self.atom_coords = self.atom_coords[..., start:end, :, :]
        self.atom_masks = self.atom_masks[..., start:end, :]
        self.res_ids = self.res_ids[..., start:end] if exists(self.res_ids) else None
        return self

    def __clear_cache__(self):
        cache_keys = [
            name for name, value in inspect.getmembers(Protein)
            if isinstance(value, property)
        ]
        for key in cache_keys:
            if key in self.__dict__:
                del self.__dict__[key]


def safe_load_sequence(seq_path: Optional[str], pdb_path: str) -> str:
    """Loads sequence, either from fasta or given pdb file"""
    if exists(seq_path):
        return load_fasta_file(seq_path)
    pdbseqs, residueLists, chains = extract_pdb_seq_from_pdb_file(pdb_path)
    return pdbseqs[0]
