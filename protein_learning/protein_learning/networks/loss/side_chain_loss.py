"""Side Chain Loss Functions"""
import math

import torch
from torch import nn, Tensor

from protein_learning.common.data.model_data import ModelOutput
from protein_learning.common.helpers import masked_mean


class SideChainDihedralLoss(nn.Module):
    """
    Loss on predicted sidechain dihedral angles
    """

    def __init__(self, *args, **kwargs):  # noqa
        super(SideChainDihedralLoss, self).__init__()

    def forward(  # noqa
            self,
            pred_chis: Tensor,
            actual_chis: Tensor,
            chi_mask: Tensor,
            chi_pi_periodic: Tensor
    ) -> Tensor:
        """Chi-Dihedral angle Loss
        :param pred_chis:
        :param actual_chis:
        :param chi_mask:
        :param chi_pi_periodic:
        :return:
        """
        assert actual_chis.shape == pred_chis.shape, f"{actual_chis.shape},{pred_chis.shape}"
        assert actual_chis.shape == chi_pi_periodic.shape, f"{actual_chis.shape},{chi_pi_periodic.shape}"
        assert chi_mask.shape == actual_chis.shape, f"{actual_chis.shape},{chi_mask.shape}"
        actual_chis, pred_chis = actual_chis.detach()[chi_mask], pred_chis[chi_mask]
        actual_chi_pi = actual_chis + chi_pi_periodic.float()[chi_mask] * math.pi
        deviations = 1 - torch.cos(pred_chis - actual_chis)
        deviations_plus_pi = 1 - torch.cos(pred_chis - actual_chi_pi)
        pi_replace_mask = (deviations_plus_pi < deviations).float()
        return torch.mean(pi_replace_mask * deviations_plus_pi + (1 - pi_replace_mask) * deviations)


class SideChainDeviationLoss(nn.Module):
    """
    Loss on predicted sidechain residue RMSD's
    """

    def __init__(self, p=2, *args, **kwargs):  # noqa
        super(SideChainDeviationLoss, self).__init__()
        self.p = p

    def forward(self, predicted_coords, actual_coords, sc_atom_mask: Tensor) -> Tensor:
        """Per-Residue Side-Chain RMSD Loss
        :param predicted_coords: predicted side-chain coordinates (b,n,32,3)
        :param actual_coords: true side-chain coordinates (b,n,32,3)
        :param sc_atom_mask: side-chain atom mask. Side-chain atoms are assumed to
        be in the order given SC_ATOMS list (see common/protein_constants.py).
        :return: Average residue side-chain RMSD
        """
        assert predicted_coords.shape == actual_coords.shape, f"{predicted_coords.shape},{actual_coords.shape}"
        assert predicted_coords.shape[:3] == sc_atom_mask.shape, f"{predicted_coords.shape},{sc_atom_mask.shape}"
        actual_coords = actual_coords.detach()
        deviations = torch.square(predicted_coords - actual_coords)
        if self.p == 2:
            deviations = torch.sqrt(torch.sum(deviations, dim=-1) + 1e-8)
        elif self.p == 1:
            deviations = torch.sum(torch.sqrt(deviations + 1e-8), dim=-1)
        else:
            raise Exception("Not Implemented", self.p)
        assert deviations.shape == sc_atom_mask.shape, f"{deviations.shape},{sc_atom_mask.shape}"
        res_means = masked_mean(deviations, sc_atom_mask, dim=-1)
        residue_mask = torch.any(sc_atom_mask, dim=-1)
        return torch.mean(res_means[residue_mask])

    def forward_from_output(self, output: ModelOutput) -> Tensor:
        """Run forward from ModelOutput Object"""
        return self.forward(
            predicted_coords=output.predicted_coords,
            actual_coords=output.native_protein.sc_atom_coords.unsqueeze(0),
            sc_atom_mask=output.native_protein.sc_atom_mask.unsqueeze(0)
        )
