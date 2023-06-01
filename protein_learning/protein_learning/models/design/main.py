import os

# make sure cuda devices are listed according to PCI_BUS_ID beofre any torch modules are loaded
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['HYDRA_FULL_ERROR'] = '1'  # TODO:REMOVE

import gc
import torch
from protein_learning.common.model_config import (
    make_config,
    parse_arg_file,
)
from argparse import Namespace
from protein_learning.common.protein_constants import ALL_ATOMS
from protein_learning.models.design.design_model import Designer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from protein_learning.common.helpers import exists
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.models.design.data_utils import (
    DesignFeatureGenerator,
    augment as design_augment,
)
from protein_learning.features.maked_feature_generator import (
    get_mask_strategies_n_weights
)

from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.training.trainer import Trainer
import sys
from protein_learning.common.global_constants import get_logger
from protein_learning.models.utils.model_io import load_args, save_args, print_args, get_datasets
from protein_learning.models.utils.default_loss import DefaultLoss

logger = get_logger(__name__)


def get_args(arg_list=None):
    print("getting args")
    arg_file = sys.argv[1] if len(sys.argv) == 2 else None
    logger.info(f"Parsing arguments from {arg_file}")
    if not exists(arg_list):
        arg_list = parse_arg_file(arg_file) if exists(arg_file) else sys.argv[1:]
    parser = ArgumentParser(description="SC-Packing",  # noqa
                            epilog='',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('model_config',
                        help="path to global config args")

    model_options = parser.add_argument_group("model_args")

    model_options.add_argument('--scalar_dim_hidden',
                               help="scalar hidden dimension",
                               default=128, type=int
                               )

    model_options.add_argument('--pair_dim_hidden',
                               help="pair hidden dimension",
                               default=128, type=int
                               )

    model_options.add_argument('--coord_dim_hidden',
                               help="coordinate hidden dimension",
                               default=16, type=int
                               )

    model_options.add_argument('--evoformer_scalar_heads_n_dim',
                               help="number of heads and head dimension for evoformer"
                                    " scalar features",
                               nargs="+", default=[10, 16], type=int
                               )

    model_options.add_argument('--evoformer_pair_heads_n_dim',
                               help="number of heads for evoformer scalar features",
                               nargs="+", default=[4, 32], type=int
                               )

    model_options.add_argument('--tfn_heads',
                               help="number of heads for TFN-Transformer",
                               default=10, type=int
                               )

    model_options.add_argument('--tfn_head_dims',
                               help="number of heads for IPA (scalar, coord)",
                               default=(16, 4), type=int, nargs="+"
                               )

    model_options.add_argument('--tfn_depth',
                               help="number of heads for IPA (scalar, coord)",
                               default=3, type=int
                               )

    model_options.add_argument('--evoformer_depth',
                               help="number of heads for IPA (scalar, coord)",
                               default=3, type=int
                               )
    model_options.set_defaults(pre_norm_pair=False)
    model_options.add_argument("--pre_norm_pair", action="store_true",
                               help="use layernorm for pair feats. before passing to TFN Transformer")
    model_options.set_defaults(use_dist_sim=False)
    model_options.add_argument("--use_dist_sim", action="store_true", help="use dist. similarity TFN Transformer")
    model_options.set_defaults(use_coord_layernorm=False)
    model_options.add_argument("--use_coord_layernorm", action="store_true", help="use layernorm for TFN Transformer")
    model_options.set_defaults(append_rel_dist=False)
    model_options.add_argument("--append_rel_dist", action="store_true", help="append rel. dist in TFN Transformer")
    model_options.set_defaults(append_sc_coord_norms_nsr=False)
    model_options.add_argument("--append_sc_coord_norms_nsr", action="store_true", help="")
    model_options.add_argument("--max_nbr_radius", default=16., type=float)
    model_options.add_argument("--max_nbrs", default=16, type=int)

    # Loss options
    loss_options = parser.add_argument_group("loss_args")
    loss_options.add_argument("--pair_wt", default=0, type=float)
    loss_options.add_argument("--pair_loss_atom_tys", default=None, type=str, nargs="+")
    loss_options.add_argument("--sc_rmsd_wt", default=0, type=float)
    loss_options.add_argument("--nsr_wt", default=1, type=float)

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
    # Full Mask
    mask_options.add_argument("--full_mask_weight", type=float, default=0)

    feature_options = parser.add_argument_group("feature_args")
    feature_options.set_defaults(use_dist=False)
    feature_options.set_defaults(use_tr_ori=False)
    feature_options.set_defaults(use_bb_dihedral=False)
    feature_options.add_argument("--use_dist", action="store_true")
    feature_options.add_argument("--use_tr_ori", action="store_true")
    feature_options.add_argument("--use_bb_dihedral", action="store_true")

    args = parser.parse_args(arg_list)
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = Namespace(**group_dict)

    return args, arg_groups


if __name__ == "__main__":
    print("RUNNING DESIGNER")
    # sometimes needed -- unclear why ?
    gc.collect()
    torch.cuda.empty_cache()

    # Set up model args
    args, arg_groups = get_args()
    config = make_config(args.model_config)

    if config.load_state:
        override = dict(gpu_indices=config.gpu_indices)
        config, args, arg_groups = load_args(
            config, args, arg_groups, override=override, suffix="designer"
        )

    arg_dict = vars(args)
    model_args, loss_args, feature_args, mask_args = map(
        lambda x: arg_groups[x],
        "model_args loss_args feature_args mask_args".split(" ")
    )

    save_args(config, args, suffix="designer")
    print_args(config, args, arg_groups)

    # set up feature generator
    feature_config = InputFeatureConfig(
        fourier_encode_tr_rosetta_ori=feature_args.use_tr_ori,
        one_hot_rel_dist=feature_args.use_dist,
        fourier_encode_bb_dihedral=feature_args.use_bb_dihedral,
    )

    # mask options

    mask_strategies, strategy_weights = get_mask_strategies_n_weights(
        **vars(mask_args)
    )
    val_strategies, val_weights = get_mask_strategies_n_weights(full_mask_weight=1)

    val_feat_gen = DesignFeatureGenerator(
        config=feature_config,
        mask_strategies=val_strategies,
        strategy_weights=val_weights,
    )

    feat_gen = DesignFeatureGenerator(
        config=feature_config,
        mask_strategies=mask_strategies,
        strategy_weights=strategy_weights,
    )

    model = Designer(
        input_embedding=InputEmbedding(feature_config).to(config.device),
        model_config=config,
        loss_fn=DefaultLoss(
            pair_dim=args.pair_dim_hidden,
            scalar_dim=args.scalar_dim_hidden + (32 if args.append_sc_coord_norms_nsr else 0),
            **vars(loss_args)
        ),
        **vars(model_args),
    )

    # set up datasets
    train_data, val_data, test_data = get_datasets(
        global_config=config,
        augment_fn=design_augment,
        atom_tys=ALL_ATOMS,
        feature_gen=[feat_gen, val_feat_gen, val_feat_gen],
        impute_decoy_cb=True,
    )

    trainer = Trainer(
        config=config,
        model=model,
        train_data=train_data,
        valid_data=val_data,
        test_data=test_data,
    )
    try:
        detect_anomoly = False
        with torch.autograd.set_detect_anomaly(detect_anomoly):
            trainer.train()
    except Exception as e:
        print("caught exception in training")
        raise e
