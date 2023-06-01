import os

# make sure cuda devices are listed according to PCI_BUS_ID beofre any torch modules are loaded
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import gc
import torch
from protein_learning.common.global_constants import MAX_SEQ_LEN
from protein_learning.common.model_config import (
    load_npy,
    ModelConfig,
    parse_arg_file
)
from argparse import Namespace
from protein_learning.common.protein_constants import ALL_ATOMS
from protein_learning.models.design.design_model import Designer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from protein_learning.common.helpers import exists
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.models.design.data_utils import get_mask_strategies_n_weights
from protein_learning.models.design.data_utils import (
    DesignFeatureGenerator,
    augment as design_augment,
)
from protein_learning.models.utils.model_io import load_args
from protein_learning.models.design.eval.design_stats import DesignStats
from protein_learning.features.input_embedding import InputEmbedding
import sys
from protein_learning.common.global_constants import get_logger
from protein_learning.models.utils.default_loss import DefaultLoss
from protein_learning.models.design.main import get_args as _get_args_main
from protein_learning.common.data.datasets.protein_dataset import ProteinDataset

logger = get_logger(__name__)


def get_args():
    assert len(sys.argv) == 2
    arg_file = sys.argv[1]
    print(f"getting args from {arg_file}")
    arg_list = parse_arg_file(arg_file) if exists(arg_file) else sys.argv[1:]
    parser = ArgumentParser(description="Evaluate Classic Design Model",  # noqa
                            epilog='',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--gpu_idx', help="gpu index to run model on", default="0")
    parser.add_argument("--stats_root", type=str, help="root of directory to store stats in")
    parser.add_argument("--n_cycles", type=int, default=1)

    # For Loading Dataset
    parser.add_argument("--data_root", type=str, help="root directory for data loading")
    parser.add_argument("--data_list", type=str, help="path to model list")
    parser.add_argument("--seq_folder", type=str, help="sequence folder")
    parser.add_argument("--max_samples", type=int, help="maximum number of samples to evaluate on", default=None)
    parser.add_argument("--native_folder", type=str, help="path to folder storing native pdbs")

    # For Loading Model Info
    parser.add_argument("--config_root", type=str, help="root directory to load configs from")
    parser.add_argument('--model_config_paths', help="path to load global config args from", nargs="+", type=str)
    parser.add_argument('--model_names', help="name to use for each model when saving statistics", nargs="+", type=str)

    # Mask Options
    mask_options = parser.add_argument_group("mask_args")
    # spatial mask
    mask_options.add_argument("--spatial_mask_weight", type=float, default=0)
    mask_options.add_argument("--spatial_mask_top_k", type=int, default=30)
    mask_options.add_argument("--spatial_mask_max_radius", type=float, default=12.)
    # contiguous mask
    mask_options.add_argument("--contiguous_mask_weight", type=float, default=0)
    mask_options.add_argument("--contiguous_mask_min_len", type=int, default=5)
    mask_options.add_argument("--contiguous_mask_max_len", type=int, default=60)
    # random mask
    mask_options.add_argument("--random_mask_weight", type=float, default=0)
    mask_options.add_argument("--random_mask_min_p", type=int, default=5)
    mask_options.add_argument("--random_mask_max_p", type=int, default=60)
    # No Mask
    mask_options.add_argument("--no_mask_weight", type=float, default=0)

    args = parser.parse_args(arg_list)
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = Namespace(**group_dict)

    return args, arg_groups


if __name__ == "__main__":
    print("Evaluating Designer Model")
    gc.collect()
    torch.cuda.empty_cache()

    # Set up model args
    eval_args, eval_arg_groups = get_args()
    mask_args = eval_arg_groups["mask_args"]
    main_args, main_arg_groups = _get_args_main(["none"])
    os.makedirs(eval_args.stats_root, exist_ok=True)

    for model_path, model_name in zip(eval_args.model_config_paths, eval_args.model_names):
        model_config_path = os.path.join(eval_args.config_root, model_path + ".npy")
        model_arg_path = os.path.join(eval_args.config_root, model_path + "_designer.npy")
        if not os.path.exists(model_config_path):
            print(f"[WARNING] : no config found at {model_config_path}\n SKIPPING...")
            continue
        config_kwargs = load_npy(model_config_path)
        # override load state, max seq. length, and gpu indices
        config_kwargs["gpu_indices"] = [eval_args.gpu_idx]
        config_kwargs["load_state"] = True
        loaded_config = ModelConfig(**load_npy(model_config_path))
        loaded_config, loaded_args, loaded_arg_groups = load_args(
            loaded_config, main_args, main_arg_groups,
            override={"gpu_indices": [eval_args.gpu_idx], "max_len": MAX_SEQ_LEN},
            suffix="designer"
        )
        loaded_model_args, loaded_loss_args, loaded_feature_args = map(
            lambda x: loaded_arg_groups[x],
            "model_args loss_args feature_args".split(" ")
        )

        # mask options
        mask_strategies, strategy_weights = get_mask_strategies_n_weights(**vars(mask_args))

        feature_config = InputFeatureConfig(
            fourier_encode_tr_rosetta_ori=loaded_feature_args.use_tr_ori,
            one_hot_rel_dist=loaded_feature_args.use_dist,
            fourier_encode_bb_dihedral=loaded_feature_args.use_bb_dihedral,
        )

        feat_gen = DesignFeatureGenerator(
            config=feature_config,
            mask_strategies=mask_strategies,
            strategy_weights=strategy_weights,
        )

        model = Designer(
            input_embedding=InputEmbedding(feature_config).to(loaded_config.device),
            model_config=loaded_config,
            loss_fn=DefaultLoss(
                pair_dim=loaded_args.pair_dim_hidden,
                scalar_dim=loaded_args.scalar_dim_hidden + (32 if loaded_args.append_sc_coord_norms_nsr else 0),
                **vars(loaded_loss_args)
            ),
            **vars(loaded_model_args),
        )

        # set up dataset
        dataset = ProteinDataset(
            model_list=eval_args.data_list,
            decoy_folder=eval_args.native_folder,
            native_folder=eval_args.native_folder,
            seq_folder=eval_args.seq_folder,
            max_samples=eval_args.max_samples,
            raise_exceptions=False,
            feat_gen=feat_gen,
            atom_tys=ALL_ATOMS,
            augment_fn=design_augment,
            impute_decoy_cb=True,
        )
        out_path = os.path.join(eval_args.stats_root, model_name + ".npy")
        print(f"[INFO] : Beginning evaluation, out_path : {out_path}")

        stats = DesignStats(
            config=loaded_config,
            model=model.to(loaded_config.device),
        )

        stats.evaluate_dataset(dataset, n_cycles=eval_args.n_cycles)
        stats.save(out_path)