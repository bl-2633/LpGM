import os
from argparse import Namespace
from typing import Dict, Union, Optional, Tuple, Any, Callable, List

import numpy as np

from protein_learning.common.data.datasets.protein_dataset import ProteinDataset
from protein_learning.common.helpers import exists, default
from protein_learning.common.model_config import (
    ModelConfig,
    load_config,
    load_npy,
    save_config,
    print_config,
)
from protein_learning.features.generator import FeatureGenerator


def load_args(
        curr_config: ModelConfig,
        curr_model_args: Namespace,
        arg_groups: Optional[Dict[str, Namespace]] = None,
        suffix: str = "model_args",
        override: Optional[Dict[str, Any]] = None
) -> Union[Tuple[ModelConfig, Namespace], Tuple[ModelConfig, Namespace, Dict[str, Namespace]]]:
    config = load_config(curr_config, **default(override, {}))
    path = os.path.join(config.param_directory, config.name + f"_{suffix}.npy")
    loaded_args = load_npy(path)
    curr_arg_dict = vars(curr_model_args)
    curr_arg_dict.update(loaded_args)

    if not exists(arg_groups):
        return config, Namespace(**curr_arg_dict)

    loaded_arg_groups = {}
    for group_name in arg_groups:
        group_dict = vars(arg_groups[group_name])
        for key in group_dict:
            if key not in curr_arg_dict:
                print(f"[WARNING] : no value for key {key}, arg_group : {group_name}")
                continue
            group_dict[key] = curr_arg_dict[key]
        loaded_arg_groups[group_name] = Namespace(**group_dict)
    return config, Namespace(**curr_arg_dict), loaded_arg_groups


def save_args(config: ModelConfig, args: Namespace, suffix: str = "model_args"):
    path = os.path.join(config.param_directory, config.name + f"_{suffix}.npy")
    np.save(path, vars(args))
    save_config(config)


def print_args(config: ModelConfig, args: Namespace, arg_groups: Optional[Dict[str, Namespace]] = None):
    print("---------- GLOBAL CONFIG ----------")
    print_config(config)

    print("---------- MODEL CONFIG ----------")
    if not exists(arg_groups):
        for k, v in vars(args).items():
            print(f"    {k} : {v}")
    else:
        for group in arg_groups:
            print(f"    ---- {group} ----")
            for k, v in vars(arg_groups[group]).items():
                print(f"        {k} : {v}")


def get_datasets(
        global_config: ModelConfig,
        augment_fn: Optional[Callable],
        feature_gen: Union[FeatureGenerator, List[FeatureGenerator]],
        **kwargs
) -> Tuple[ProteinDataset, Optional[ProteinDataset], Optional[ProteinDataset]]:
    c = global_config
    feature_gens = feature_gen if isinstance(feature_gen, list) else [feature_gen] * 3
    dataset = lambda lst, nat, decoy, seq, gen, samples=-1: ProteinDataset(
        model_list=lst,
        native_folder=nat,
        decoy_folder=decoy,
        seq_folder=seq,
        max_samples=samples,
        raise_exceptions=c.raise_exceptions,
        feat_gen=gen,
        augment_fn=augment_fn,
        **kwargs,
    ) if exists(lst) else None

    train_data = dataset(c.train_list, c.train_native_folder,
                         c.train_native_folder, c.train_seq_folder, feature_gens[0])
    valid_data = dataset(c.val_list, c.val_native_folder, c.val_native_folder,
                         c.val_seq_folder, feature_gens[1], c.max_val_samples)
    test_data = dataset(c.test_list, c.test_native_folder, c.test_native_folder,
                        c.test_seq_folder, feature_gens[2], c.max_test_samples)

    return train_data, valid_data, test_data
