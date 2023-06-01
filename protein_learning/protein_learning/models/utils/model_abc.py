import torch.cuda
from torch import nn
from abc import abstractmethod
from protein_learning.common.data.model_data import ModelInput, ModelOutput, ModelLoss
from protein_learning.common.model_config import ModelConfig


class ProteinModel(nn.Module):
    """Abstract base class for all protein learning models"""

    def __init__(self):
        super(ProteinModel, self).__init__()

    @abstractmethod
    def forward(self, sample: ModelInput) -> ModelOutput:
        """Perform forward pass"""
        pass

    @abstractmethod
    def compute_loss(self, output: ModelOutput, **kwargs) -> ModelLoss:
        """Compute loss on model output"""
        pass
