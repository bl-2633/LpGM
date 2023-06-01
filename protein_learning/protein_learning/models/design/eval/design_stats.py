"""Compute Design Model statistics
"""
import torch

from protein_learning.assessment.metrics import (
    calculate_sequence_identity,
    calculate_perplexity,
    calculate_unnormalized_confusion,
    calculate_average_entropy,
)
from protein_learning.common.data.model_data import ModelOutput
from protein_learning.common.model_config import ModelConfig
from protein_learning.models.utils.model_abc import ProteinModel
from protein_learning.models.utils.model_eval import (
    ModelStats,
    to_npy,
)


class DesignStats(ModelStats):
    """Caclulate and store statistics for design model"""

    def __init__(
            self,
            config: ModelConfig,
            model: ProteinModel,
    ):
        super(DesignStats, self).__init__(model=model, config=config)

    def _init_data(self):
        return dict(seqs=[], perplexity=[], entropy=[], confusion=[], recovery=[], names=[],
                    logits=[], true_labels=[])

    def log_stats(self, model_out: ModelOutput, model: ProteinModel):
        """Log statistics for single input sample"""
        nsr_feats = model_out.extra["nsr_scalar"]
        nsr_loss_fn = model.loss_fn.loss_fns["nsr"]

        mask = model_out.valid_residue_mask
        mask = mask.unsqueeze(0) if mask.ndim == 1 else mask
        true_labels = model_out.model_input.native_seq_enc[mask]
        labels = to_npy(true_labels).squeeze()
        logits = nsr_loss_fn.get_predicted_logits(nsr_feats)[mask]
        pred_labels = torch.argmax(logits, dim=-1)
        assert true_labels.ndim == pred_labels.ndim

        perp = to_npy(calculate_perplexity(pred_aa_logits=logits, true_labels=true_labels))
        ent = to_npy(calculate_average_entropy(pred_aa_logits=logits))
        rec = to_npy(calculate_sequence_identity(pred_labels, true_labels))
        conf = to_npy(calculate_unnormalized_confusion(pred_labels=pred_labels, true_labels=true_labels))
        self.data["names"].append(model_out.native_protein.name)
        self.data["seqs"].append(labels)
        self.data["perplexity"].append(perp)
        self.data["entropy"].append(ent)
        self.data["recovery"].append(rec)
        self.data["confusion"].append(conf)
        self.data["logits"].append(to_npy(logits))
        self.data["true_labels"].append(to_npy(true_labels))
