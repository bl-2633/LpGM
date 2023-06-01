"""Reconstruction Loss Functi0n"""
from protein_learning.networks.loss.coord_loss import FAPELoss, DistanceInvLoss, TMLoss
from protein_learning.networks.loss.pair_loss import PairDistLossNet
from protein_learning.networks.loss.residue_loss import SequenceRecoveryLossNet, PredLDDTLossNet
from protein_learning.common.helpers import exists, default
from protein_learning.common.data.model_data import ModelOutput, ModelLoss
from einops import repeat, rearrange  # noqa
import torch
from torch import nn
from typing import Optional, List


class ReconstructionLoss(nn.Module):
    """Reconstruction Loss function"""

    def __init__(self,
                 scalar_dim: int,
                 pair_dim: int,
                 unmasked_rel_weight: float = 1,
                 fape_wt: Optional[float] = None,
                 pred_lddt_wt: Optional[float] = None,
                 nsr_wt: Optional[float] = None,
                 tm_wt: Optional[float] = None,
                 dist_inv_wt: Optional[float] = None,
                 pair_dist_wt: Optional[float] = None,
                 pair_loss_atom_tys: Optional[List[str]] = None,
                 fape_atom_tys: Optional[List[str]] = None,
                 recompute_rigids: bool = True,
                 ):
        super(ReconstructionLoss, self).__init__()
        assert 0 <= unmasked_rel_weight <= 1, \
            f"unmasked relative weight must be in [0,1], got {unmasked_rel_weight}"
        self.unmasked_rel_weight = unmasked_rel_weight
        self.recompute_rigids = recompute_rigids
        self.loss_fns, self.loss_wts = nn.ModuleDict(), {}
        pair_loss_atom_tys = default(pair_loss_atom_tys, "CA CA CA CB CB CB".split())
        self.fape_atom_tys = fape_atom_tys
        self.pair_loss_atom_tys = pair_loss_atom_tys

        self._register_loss(FAPELoss(atom_tys=fape_atom_tys), "fape", fape_wt)
        self._register_loss(SequenceRecoveryLossNet(scalar_dim), "nsr", nsr_wt)
        self._register_loss(PairDistLossNet(
            dim_in=pair_dim, atom_tys=pair_loss_atom_tys),
            "pair-dist", pair_dist_wt
        )
        self._register_loss(PredLDDTLossNet(scalar_dim), "plddt", pred_lddt_wt)
        self._register_loss(DistanceInvLoss(), "dist-inv", dist_inv_wt)
        self._register_loss(TMLoss(), "tm", tm_wt)

    def _register_loss(self, loss, name, wt):
        if exists(wt):
            self.loss_fns[name] = loss
            self.loss_wts[name] = wt

    def forward(self, model_out: ModelOutput, compute_zero_wt_loss: bool = False) -> ModelLoss:
        """Compute the loss"""
        seq_mask, feat_mask = model_out.input_masks

        model_loss = ModelLoss(seq_len=model_out.seq_len, pdb=model_out.native_protein.name)
        rel_wt, pair_mask = self.unmasked_rel_weight, None
        if exists(feat_mask):
            pair_mask = torch.einsum("bi,bj->bij", feat_mask, feat_mask)

        # collect all loss information
        for loss_name, loss_fn in self.loss_fns.items():
            loss_wt = self.loss_wts[loss_name]
            if loss_wt == 0 and not compute_zero_wt_loss:
                continue
            reduce = self.unmasked_rel_weight == 1
            loss, loss_mask, fn = None, None, None
            # FAPE
            if loss_name == "fape":
                loss = loss_fn.forward_from_output(
                    model_out, reduce=reduce, recompute_rigids=self.recompute_rigids
                )
                loss_mask, fn, reduce = pair_mask, torch.sum, reduce or not exists(feat_mask)

            # Predicted Distance
            if loss_name == "pair-dist":
                reduce = reduce or not exists(feat_mask)
                out = loss_fn.forward_from_output(model_out, reduce=reduce)
                loss, loss_masks = (out, None) if reduce else out
                if not reduce:
                    loss_mask = torch.logical_and(loss_masks, pair_mask)[pair_mask]  # noqa
                    loss = loss[pair_mask]

            # Native Seq. Recovery
            if loss_name == "nsr":
                assert exists(seq_mask), "must supply sequence mask for computing nsr loss!"
                loss = loss_fn.forward_from_output(model_out, reduce=reduce)
                loss_mask = seq_mask

                # TM score
            if loss_name == "tm":  # always reduce
                loss, reduce = loss_fn.forward_from_output(model_out), True

            # Predicted pLDDT
            if loss_name == "plddt":
                loss = loss_fn.forward_from_output(model_out, reduce=reduce)
                loss_mask = feat_mask

            # Inverse Distance Loss
            if loss_name == "dist-inv":
                loss = loss_fn.forward_from_output(model_out, reduce=reduce)
                loss_mask = pair_mask

            if not reduce:
                loss[~loss_mask] = loss[~loss_mask] * rel_wt
                loss = fn(loss) if exists(fn) else torch.mean(loss)


            model_loss.add_loss(loss=loss, loss_weight=loss_wt, loss_name=loss_name)

        return model_loss
