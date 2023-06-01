"""Protein dataset"""
from typing import Optional, List, Any, Callable

from protein_learning.common.data.datasets.dataset import ProteinDatasetABC
from protein_learning.common.data.model_data import ModelInput, ExtraInput
from protein_learning.common.data.protein import Protein
from protein_learning.common.helpers import default
from protein_learning.features.generator import FeatureGenerator


class ProteinDataset(ProteinDatasetABC):
    """Generic protein dataset"""

    def __init__(
            self,
            model_list: str,
            decoy_folder: str,
            native_folder: str,
            seq_folder: str,
            max_samples: int,
            raise_exceptions: bool,
            feat_gen: FeatureGenerator,
            atom_tys: Optional[List[str]] = None,
            augment_fn: Callable[[Protein, Protein], ExtraInput] = None,
            impute_decoy_cb: bool = False,
            name: Optional[str] = None
    ):
        super(ProteinDataset, self).__init__(
            model_list=model_list,
            decoy_folder=decoy_folder,
            native_folder=native_folder,
            seq_folder=seq_folder,
            max_samples=max_samples,
            raise_exceptions=raise_exceptions
        )
        self.feat_gen = feat_gen
        self.atom_tys = default(atom_tys, self.default_atom_tys())
        self.augment_fn = default(augment_fn, lambda *args, **kwargs: None)
        self.impute_decoy_cb = impute_decoy_cb
        self.name = name

    def get_item_from_pdbs_n_seq(
            self,
            seq_path: Optional[str],
            decoy_pdb_path: Optional[str],
            native_pdb_path: str,
    ) -> ModelInput:
        """Load data given native and decoy pdb paths and sequence path"""
        seq = self.safe_load_sequence(seq_path, decoy_pdb_path)
        decoy_pdb_path = default(decoy_pdb_path, native_pdb_path)
        native_protein, decoy_protein = map(lambda x: Protein.FromPDBAndSeq(
            pdb_path=x,
            seq=seq,
            atom_tys=self.atom_tys,
        ), (native_pdb_path, decoy_pdb_path))

        atom_tys = self.atom_tys
        if self.impute_decoy_cb:
            decoy_protein.impute_cb(override=True, exists_ok=True)
            atom_tys = atom_tys + ["CB"] if "CB" not in atom_tys else atom_tys

        return ModelInput(
            decoy=decoy_protein,
            native=native_protein,
            input_features=self.feat_gen.generate_features(
                seq=decoy_protein.seq,
                coords=decoy_protein.atom_coords,
                res_ids=decoy_protein.res_ids,
                coord_mask=decoy_protein.atom_masks,
                atom_tys=atom_tys,
            ),
            extra=self.augment(decoy_protein, native_protein)
        )

    def augment(self, decoy_protein: Protein, native_protein: Protein) -> Any:  # noqa
        """Override in subclass to augment model input with extra informaiton"""
        return self.augment_fn(decoy_protein, native_protein)
