from typing import List, Optional

from torch import nn

from protein_learning.common.data.model_data import ModelOutput, ModelLoss
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists, default
from protein_learning.networks.loss.coord_loss import (
    FAPELoss,
    TMLoss,
    CoordDeviationLoss,
    DistanceInvLoss,
)
from protein_learning.networks.loss.pair_loss import (
    PairDistLossNet
)
from protein_learning.networks.loss.residue_loss import (
    SequenceRecoveryLossNet,
    PredLDDTLossNet,
)
from protein_learning.networks.loss.side_chain_loss import (
    SideChainDeviationLoss,
)

logger = get_logger(__name__)


class DefaultLoss(nn.Module):
    """Default Loss Function"""

    def __init__(
            self,
            pair_dim: int,
            scalar_dim: int,
            fape_wt: Optional[float] = None,
            pair_wt: Optional[float] = None,
            tm_wt: Optional[float] = None,
            rmsd_wt: Optional[float] = None,
            dist_wt: Optional[float] = None,
            lddt_wt: Optional[float] = None,
            nsr_wt: Optional[float] = None,
            sc_rmsd_wt: Optional[float] = None,
            pair_loss_atom_tys: Optional[List[str]] = None,
    ):
        super(DefaultLoss, self).__init__()
        loss_fns = nn.ModuleDict()
        loss_wts = {}
        pair_loss_atom_tys = default(pair_loss_atom_tys, "CA CA CA CB CB CB".split())

        def register_loss(loss, name, wt):
            if exists(wt):
                loss_fns[name] = loss
                loss_wts[name] = wt

        register_loss(loss=FAPELoss(), name="fape", wt=fape_wt)
        register_loss(loss=DistanceInvLoss(), name="dist-inv", wt=dist_wt)
        register_loss(loss=CoordDeviationLoss(), name="coord-l1", wt=rmsd_wt)
        register_loss(loss=TMLoss(), name="coord-tm", wt=tm_wt)
        register_loss(
            loss=PairDistLossNet(dim_in=pair_dim, atom_tys=pair_loss_atom_tys, use_hidden=False),
            name="pair-dist-loss", wt=pair_wt
        )
        register_loss(loss=SequenceRecoveryLossNet(dim_in=scalar_dim), name="nsr", wt=nsr_wt)
        register_loss(loss=PredLDDTLossNet(dim_in=scalar_dim, n_hidden_layers=1),
                      name="pred-lddt", wt=lddt_wt)
        register_loss(loss=SideChainDeviationLoss(p=1), name="sc-rmsd", wt=sc_rmsd_wt)

        self.loss_wts = loss_wts
        self.loss_fns = loss_fns

    def forward(self, output: ModelOutput, compute_zero_wt_loss: bool = False):
        """Compute model loss"""
        loss = ModelLoss(seq_len=output.seq_len, pdb=output.native_protein.name)
        for loss_name, loss_fn in self.loss_fns.items():
            if self.loss_wts[loss_name] != 0 or compute_zero_wt_loss:
                loss.add_loss(
                    loss=loss_fn.forward_from_output(output),
                    loss_weight=self.loss_wts[loss_name],
                    loss_name=loss_name,
                )
        return loss
