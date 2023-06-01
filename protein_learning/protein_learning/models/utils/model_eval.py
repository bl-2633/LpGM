from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch

from protein_learning.common.data.datasets.protein_dataset import ProteinDataset
from protein_learning.common.data.model_data import ModelOutput
from protein_learning.common.model_config import ModelConfig
from protein_learning.models.utils.model_abc import ProteinModel
from protein_learning.training.train_utils import EvalTy
from protein_learning.training.trainer import Trainer


def to_npy(x):
    """Converts x to numpy.ndarray"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


class ModelStats(ABC):
    """Caclulate and store statistics for design model"""

    def __init__(
            self,
            config: ModelConfig,
            model: ProteinModel,
    ):
        self.model = model
        self.config = config
        self.data = self._init_data()

    def evaluate_dataset(self, dataset: ProteinDataset, n_cycles: int = 1):
        """Evaluate on all samples in dataset"""
        trainer = Trainer(
            config=self.config,
            model=self.model,
            train_data=dataset,
        )
        trainer.model = trainer.model.eval()
        for _ in range(n_cycles):
            data = dataset.get_dataloader(batch_size=min(len(dataset), 32), num_workers=4, shuffle=False)
            n_processed = 0
            with torch.no_grad():
                for idx, batch_data in enumerate(data):
                    for sample in batch_data:
                        model_out = trainer.safe_eval(sample, eval_ty=EvalTy.VALIDATION, raise_exceptions=True)
                        model_out = model_out if not isinstance(model_out, List) else model_out[-1]
                        if n_processed % max(1, (len(dataset) // 10)) == 0:
                            print(f" progress {1000 * (n_processed / len(dataset)) // 10}%, "
                                  f"-- {n_processed}/{len(dataset)}")
                        self.log_stats(model_out=model_out, model=trainer.model)
                        n_processed += 1

    @abstractmethod
    def _init_data(self):
        pass

    @abstractmethod
    def log_stats(self, model_out: ModelOutput, model: ProteinModel):
        """Log statistics for single input sample"""
        pass

    def save(self, path):
        """Save statistics"""
        np.save(path, self.data)
