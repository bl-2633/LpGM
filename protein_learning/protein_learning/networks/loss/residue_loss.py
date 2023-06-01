"""Networks for Computing Residue Feature Loss"""
from typing import Optional

import torch
import torch.nn.functional as F  # noqa
from einops import rearrange
from torch import Tensor, nn

from protein_learning.assessment.metrics import compute_coord_lddt
from protein_learning.common.data.model_data import ModelOutput
from protein_learning.common.helpers import exists, default
from protein_learning.networks.loss.utils import FeedForward
from protein_learning.networks.loss.utils import (
    softmax_cross_entropy,
)

get_eps = lambda x: torch.finfo(x.dtype).eps  # noqa
max_float = lambda x: torch.finfo(x.dtype).max  # noqa


class SequenceRecoveryLossNet(nn.Module):
    """Loss for predicted residue type

    Output residue features are converted to residue type predictions via a
    feed-formard network. Cross Entropy loss between predicted labels and
    true labels is averaged to obtain the output.
    """

    def __init__(self, dim_in: int, n_labels=21, hidden_layers: int = 2, pre_norm: bool = False):
        """Native Sequence Recovery Loss

        :param dim_in: residue feature dimension
        :param n_labels:number of underlying residue labels
        :param hidden_layers: number of hidden layers to use in feed forward network
        """
        super(SequenceRecoveryLossNet, self).__init__()
        self.net = FeedForward(
            dim_in=dim_in, dim_out=n_labels,
            n_hidden_layers=hidden_layers, pre_norm=pre_norm
        )

    def forward(
            self,
            residue_feats: Tensor,
            true_labels: Tensor,
            mask: Optional[Tensor] = None,
            reduce: bool = True
    ) -> Tensor:
        """Compute Native Sequence Recovery Loss

        :param residue_feats: Residue features of shape (b,n,d) where d is the feature dimension
        :param true_labels: LongTensor of shape (b,n) storing residue class labels
        :param mask: residue mask of shape (b,n) indicating which residues to compute loss on.
        :param reduce : take mean of loss (iff reduce)
        :return: cross entropy loss of predicted and true labels.
        """
        assert residue_feats.shape[:2] == true_labels.shape, f"{residue_feats.shape},{true_labels.shape}"
        if exists(mask):
            assert mask.shape == true_labels.shape, f"{mask.shape},{true_labels.shape}"
        labels = torch.nn.functional.one_hot(true_labels, 21)
        logits = self.get_predicted_logits(residue_feats)
        ce = softmax_cross_entropy(logits, labels)
        if reduce:
            return torch.mean(ce[mask]) if exists(mask) else torch.mean(ce)
        return ce.masked_fill(~mask, 0) if exists(mask) else ce

    def get_predicted_logits(self, residue_feats: Tensor) -> Tensor:
        """Get predicted logits from residue features"""
        return self.net(residue_feats)

    def predict_classes(self, residue_feats: Tensor) -> Tensor:
        """Get predicted class labels from residue features"""
        return torch.argmax(self.get_predicted_logits(residue_feats), dim=-1)

    def get_acc(self, residue_feats: Tensor, true_labels: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Get label prediction accuracy"""
        pred_labels = self.predict_classes(residue_feats)
        assert pred_labels.shape == true_labels.shape
        correct_preds = pred_labels == true_labels
        correct_preds = correct_preds[mask] if exists(mask) else correct_preds
        return torch.mean(correct_preds.float(), dim=(-1))

    def forward_from_output(self, output: ModelOutput, reduce: bool = True) -> Tensor:
        """Run forward from ModelOutput Object"""
        if not (hasattr(output, "native_seq_enc") or hasattr(output.model_input, "native_seq_enc")):
            raise Exception("output or input must have native_seq_enc as attribute!")
        native_seq_enc = default(getattr(output, "native_seq_enc", None),
                                 getattr(output.model_input, "native_seq_enc", None)
                                 )
        return self.forward(
            residue_feats=output.scalar_output,
            true_labels=native_seq_enc,
            mask=output.valid_residue_mask.unsqueeze(0),
            reduce=reduce,
        )


class PredLDDTLossNet(nn.Module):
    """Loss for LDDT Prediction
    """

    def __init__(self, dim_in, n_bins: int = 48, n_hidden_layers: int = 1, atom_ty: Optional[str] = "CA"):
        """
        :param dim_in: residue feature dimension
        :param n_bins: number of lddt bins to project residue features into
        """
        super().__init__()
        self.net = FeedForward(
            dim_in=dim_in, dim_out=n_bins,
            pre_norm=True, n_hidden_layers=n_hidden_layers
        )
        bins = torch.linspace(start=0, end=1, steps=n_bins + 1)
        self._bins = rearrange((bins[1:] + bins[:-1]) / 2, "i -> () () i")
        self.n_bins = n_bins
        self.atom_ty = atom_ty

    def get_bins(self, device) -> torch.Tensor:
        """Gets the underlying LDDT bins"""
        if self._bins.device != device:
            self._bins = self._bins.to(device)
        return self._bins

    def get_plddt_logits(self, residue_feats):
        """Get predicted pLDDT logits"""
        return self.net(residue_feats)

    def get_predicted_plddt(self, residue_feats) -> torch.Tensor:
        """Computes per-residue LDDT score in range [0,1].
        :return: tensor of pLDDT scores with shape (b, n)
        """
        logits = self.get_plddt_logits(residue_feats)
        return torch.sum(F.softmax(logits, dim=-1) * self.get_bins(logits.device), dim=-1)

    @staticmethod
    def get_true_plddt(predicted_coords: Tensor, actual_coords: Tensor):
        """pLDDT score between two lists of coordinates"""
        with torch.no_grad():
            return compute_coord_lddt(
                predicted_coords=predicted_coords,
                actual_coords=actual_coords,
                cutoff=15.,
                per_residue=True,
            )

    def plddt_to_one_hot_labels(self, plddt: Tensor) -> Tensor:
        """Convert lddt values to one-hot labels"""
        nearest_bins = torch.abs(plddt.unsqueeze(-1) - self.get_bins(plddt.device))
        return F.one_hot(torch.argmin(nearest_bins, dim=-1), self.n_bins)

    def forward(
            self,
            residue_feats: Tensor,
            predicted_coords: Tensor,
            actual_coords: Tensor,
            mask: Optional[Tensor] = None,
            reduce: bool = True,
    ) -> Tensor:
        """Compute Predicted LDDT Loss
        :param residue_feats: residue features of shape (b,n,d) where d = self.dim_in
        :param predicted_coords: predicted coordinates of shape (b,n,3) used in ground-truth LDDT calculation
        :param actual_coords: actual coordinates of shape (b,n,3) used in ground-truth LDDT calculation
        :param mask: residue mask of shape (b,n) indicating which residues to compute LDDT scores for
        :param reduce : take mean of loss (iff reduce)
        :return: cross entropy loss between predicted logits and true LDDT labels.
        """
        assert residue_feats.ndim == predicted_coords.ndim == actual_coords.ndim == 3
        # compute true LDDT and class labels
        true_plddt = self.get_true_plddt(predicted_coords=predicted_coords, actual_coords=actual_coords)
        true_labels = self.plddt_to_one_hot_labels(true_plddt)
        # get predicted logits
        pred_plddt_logits = self.get_plddt_logits(residue_feats)
        ce = softmax_cross_entropy(pred_plddt_logits, true_labels)
        if reduce:
            return torch.mean(ce[mask]) if exists(mask) else torch.mean(ce)
        return ce.masked_fill(~mask, 0) if exists(mask) else ce

    def forward_from_output(self, output: ModelOutput, reduce: bool = True) -> Tensor:
        """Run forward from ModelOutput Object"""
        loss_input = output.get_native_n_decoy_coords_n_masks([self.atom_ty], align_by_kabsch=False)
        pred_coords, native_coords, mask = loss_input
        return self.forward(
            residue_feats=output.scalar_output,
            predicted_coords=rearrange(pred_coords, "b n a c -> b (n a) c"),
            actual_coords=rearrange(native_coords, "b n a c -> b (n a) c"),
            mask=output.valid_residue_mask.unsqueeze(0),
            reduce=reduce
        )
