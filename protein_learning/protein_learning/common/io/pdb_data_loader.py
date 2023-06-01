"""Loads a Protein object from a pdb and sequence file
"""
from protein_learning.common.io.pdb_utils import extract_atom_coords_n_mask_tensors, extract_pdb_seq_from_pdb_file
from protein_learning.common.io.sequence_utils import load_fasta_file
from protein_learning.common.data.protein import Protein
from protein_learning.common.protein_constants import ALL_ATOMS
from typing import List, Tuple, Optional, Union
from torch import Tensor
from protein_learning.common.helpers import default, exists
import os


def extract_coords_n_masks(pdb_path: str, seq: str, atom_tys: List[str]
                           ) -> Union[Tuple[Tensor, Tensor, Tensor, str], Tuple[Tensor, Tensor, str]]:
    """Extracts coords and atom mask for given atom types from input pdb and sequence
    :return: atom coordinates and mask of shape (n,a,3) and (n,a) where n is the sequence length
    and a is the number of input atom types given.
    """

    return extract_atom_coords_n_mask_tensors(
        seq=seq,
        pdb_path=pdb_path,
        atom_tys=atom_tys,
        warn=True
    )


def load_protein(
        pdb_path: str,
        seq_path: Optional[str] = None,
        atom_tys: Optional[List[str]] = None,
) -> Protein:
    """Loads a protein object from the given pdb file mapped to the given sequence
    :param pdb_path: pdb path containing structure data
    :param seq_path: path to sequence (pdb sequence used if not provided)
    :param atom_tys: atom types to extract coordinates for
    :return: protein object
    """
    # get pdb path
    if not os.path.exists(pdb_path) and not pdb_path.endswith(".pdb"):
        pdb_path += ".pdb"
    assert os.path.exists(pdb_path), f"pdb_path: {pdb_path} does not exist!"
    if exists(seq_path):
        if not os.path.exists(seq_path) and not seq_path.endswith(".fasta"):
            seq_path += ".fasta"
        assert os.path.exists(seq_path), f"seq_path: {seq_path} does not exist!"
        seq = load_fasta_file(seq_path)
    else:
        seqs, _, _ = extract_pdb_seq_from_pdb_file(pdb_path)
        seq = seqs[0]
    atom_tys = default(atom_tys, ALL_ATOMS)

    atom_coords, atom_mask, res_ids, seq = extract_coords_n_masks(
        seq=seq,
        pdb_path=pdb_path,
        atom_tys=atom_tys,
    )

    return Protein(
        atom_coords=atom_coords,
        atom_masks=atom_mask,
        atom_tys=atom_tys,
        seq=seq,
        res_ids=res_ids,
        name=os.path.basename(pdb_path)[:-4]
    )
