"""Main file for MDS"""
import os

# make sure cuda devices are listed according to PCI_BUS_ID beofre any torch modules are loaded
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import gc
import torch
from protein_learning.common.model_config import (
    make_config,
    parse_arg_file,
)

from protein_learning.models.mds.mds_model import MDS
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from protein_learning.common.helpers import exists
from protein_learning.features.generator import DefaultFeatureGenerator
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.training.trainer import Trainer
from protein_learning.models.mds.data_utils import augment
from protein_learning.models.utils.default_loss import DefaultLoss
import sys
from protein_learning.common.global_constants import get_logger
from protein_learning.models.utils.model_io import load_args, save_args, print_args, get_datasets

logger = get_logger(__name__)


def get_args():
    print("getting args")
    arg_file = sys.argv[1] if len(sys.argv) == 2 else None
    logger.info(f"Parsing arguments from {arg_file}")
    arg_list = parse_arg_file(arg_file) if exists(arg_file) else sys.argv[1:]
    parser = ArgumentParser(description="MDS",  # noqa
                            epilog='',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('model_config',
                        help="path to global config args")

    parser.add_argument('--atom_tys',
                        help="atom types to predict",
                        nargs="+",
                        default="N CA C CB".split(" "))

    model_args = parser.add_argument_group("model_args")

    model_args.add_argument('--scalar_dim_hidden',
                            help="scalar hidden dimension",
                            default=128, type=int
                            )

    model_args.add_argument('--pair_dim_hidden',
                            help="pair hidden dimension",
                            default=128, type=int
                            )

    model_args.add_argument('--evoformer_scalar_heads_n_dim',
                            help="number of heads and head dimension for evoformer"
                                 " scalar features",
                            nargs="+", default=[10, 16], type=int
                            )

    model_args.add_argument('--evoformer_pair_heads_n_dim',
                            help="number of heads for evoformer scalar features",
                            nargs="+", default=[4, 32], type=int
                            )

    model_args.add_argument('--ipa_heads',
                            help="number of heads for IPA",
                            default=10, type=int
                            )

    model_args.add_argument('--ipa_head_dims',
                            help="number of heads for IPA (scalar, coord)",
                            default=(16, 4), type=int, nargs="+"
                            )

    model_args.add_argument('--evoformer_depth',
                            help="number of heads for IPA (scalar, coord)",
                            default=3, type=int
                            )

    model_args.add_argument('--ipa_depth',
                            help="number of heads for IPA (scalar, coord)",
                            default=3, type=int
                            )
    model_args.set_defaults(use_ipa=False)
    model_args.add_argument('--use_ipa', action="store_true")
    model_args.set_defaults(predict_rigids=False)
    model_args.add_argument('--predict_rigids', action="store_true")
    model_args.set_defaults(detach_frames=False)
    model_args.add_argument('--detach_frames', action="store_true")

    loss_args = parser.add_argument_group("loss_args")
    keys = "fape pair tm rmsd dist lddt nsr"
    for key in keys.split(" "):
        loss_args.add_argument(f"--{key}_wt", type=float, default=None)

    feature_args = parser.add_argument_group("feature_args")
    feature_args.set_defaults(use_dist=False)
    feature_args.set_defaults(use_tr_ori=False)
    feature_args.set_defaults(use_bb_dihedral=False)
    feature_args.add_argument("--use_dist", action="store_true")
    feature_args.add_argument("--use_tr_ori", action="store_true")
    feature_args.add_argument("--use_bb_dihedral", action="store_true")

    args = parser.parse_args(arg_list)
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    return args, arg_groups


if __name__ == "__main__":
    print("RUNNING MDS")
    # sometimes needed -- unclear why ?
    gc.collect()
    torch.cuda.empty_cache()

    # Set up model args
    args, arg_groups = get_args()
    config = make_config(args.model_config)

    if config.load_state:
        override = dict()
        config, args, arg_groups = load_args(config, args, arg_groups, override=override, suffix="mds")

    arg_dict = vars(args)
    arg_dict.update({"model_config": config})
    model_args, loss_args, feature_args = map(lambda x: arg_groups[x],
                                              "model_args loss_args feature_args".split(" "))

    save_args(config, args, suffix="mds")
    print_args(config, args, arg_groups)

    # set up feature generator
    feature_config = InputFeatureConfig(
        fourier_encode_tr_rosetta_ori=feature_args.use_tr_ori,
        one_hot_rel_dist=feature_args.use_dist,
        fourier_encode_bb_dihedral=feature_args.use_bb_dihedral,
    )

    feat_gen = DefaultFeatureGenerator(
        config=feature_config,
    )

    # Loss Function
    loss_fn = DefaultLoss(
        scalar_dim=args.scalar_dim_hidden,
        pair_dim=args.pair_dim_hidden,
        **vars(loss_args)
    )

    model = MDS(
        model_config=config,
        input_embedding=InputEmbedding(feature_config).to(config.device),
        coord_dim_out=len(args.atom_tys),
        loss_fn=loss_fn,
        **vars(model_args)
    )

    train, val, test = get_datasets(
        global_config=config,
        augment_fn=augment,
        feature_gen=feat_gen,
        atom_tys=args.atom_tys
    )

    trainer = Trainer(
        config=config,
        model=model,
        train_data=train,
        valid_data=val,
        test_data=test,
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"Exception caught during training {e}")
        raise e
