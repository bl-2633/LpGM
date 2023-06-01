"""Compare PDB files
"""
import os

# make sure cuda devices are listed according to PCI_BUS_ID beofre any torch modules are loaded
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from protein_learning.common.io.pdb_data_loader import load_protein
import time
from protein_learning.common.helpers import exists
import torch

from protein_learning.common.data.protein import Protein
from typing import Dict, Any
import numpy as np


def to_npy(x: Any) -> np.ndarray:
    """Convert input to numpy.ndarray"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, list):
        return np.array(x)
    assert isinstance(x, np.ndarray)
    return x


def assess_sidechains(native: Protein, decoy: Protein) -> Dict[str, Any]:
    """Gather quality metrics from protein sidechains
    :param native: native protein to compare to
    :param decoy: decoy protein to compare with
    :return: dictionary of side chain quality metrics
    """
    # If the decoy backbone is not the same as the native, the
    # backbone atom coordinates (and sc atom coordinates) of the
    # native must be aligned to the decoy.

    # residues with side-chain symmetries should have their atoms
    # reconfigured to minimize RMSD with decoy structure

    # We can now compute coordinate RMSD and MAE, as well as dihedral MAE
    # for each residue.
    rmsd = None  # TODO
    chi_mae = None  # TODO

    # In addition, neighbor counts for (based on CB atom), and valid residue
    # mask should also be returned
    neighbor_counts = None  # TODO
    return dict(
        seq=decoy.seq,
        valid_residue_mask=to_npy(decoy.valid_residue_mask),
        neighbor_counts=to_npy(neighbor_counts),

    )


if __name__ == "__main__":
    parser = ArgumentParser(description=" Compare predictive and native sidechain conformations",  # noqa
                            epilog='runs model on validation, train, or test dataset',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('model_list', help='path to list of pdbs to run')
    parser.add_argument('native_folder', help='path to folder storing native pdb files')
    parser.add_argument('decoy_folder', help="path to folder storing decoy pdb files")
    parser.add_argument("out_path", type=str, help="file to store output in (.npy)")
    parser.add_argument('--seq_folder', default=None, help="path to folder storing sequence files")
    parser.add_argument("--gpu_idx", default=None, type=int, help="specify if you wish to run on gpu")
    parser.add_argument("--compare_bb", action="store_true", help="compare decoy and native backbones")
    parser.add_argument("--compare_sc", action="store_true", help="compare native and decoy sidechains")
    args = parser.parse_args(sys.argv[1:])

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    device = f"cuda:{args.gpu_idx}" if args.gpu_idx is not None else "cpu"
    assert os.path.exists(args.native_folder)
    assert os.path.exists(args.decoy_folder)
    assert os.path.exists(args.model_list)

    with open(args.model_list, "r+") as model_list:
        model_names = [x.strip() for x in model_list]  # noqa

    all_stats, n_processed, start_time = dict(sc={}, bb={}), 0, time.time()
    for idx, target in enumerate(model_list):
        start = time.time()
        try:
            decoy = load_protein(
                pdb_path=os.path.join(args.decoy_folder, target),
                seq_path=os.path.join(args.seq_folder, target) if exists(args.seq_folder) else None,
            ).to_device(device)
            native = load_protein(
                pdb_path=os.path.join(args.native_folder, target),
                seq_path=os.path.join(args.seq_folder, target) if exists(args.seq_folder) else None,
            ).to_device(device)
        except Exception as e:
            print(f"Error loading target {target}")
            raise e

        if args.compare_sc:
            all_stats["sc"][decoy.name] = assess_sidechains(native=native, decoy=decoy)

        if args.compare_bb:
            raise Exception("compare_bb not yet implemented!")

        n_processed += 1
        if n_processed % max(1, len(model_list) // 10) == 0:
            avg_time = np.round((time.time() - start_time) / n_processed, 2)
            print(f"gathered stats for {n_processed}/{len(model_list)} targets."
                  f" Avg. Time per target: {avg_time} (s)")

    np.save(args.out_path, all_stats)
    print("Finished!")
