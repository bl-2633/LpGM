"""Pair Loss Functions and Networks"""
import torch
from torch import nn, Tensor
from protein_learning.networks.loss.utils import softmax_cross_entropy, partition
from protein_learning.common.data.model_data import ModelOutput
import torch.nn.functional as F  # noqa
from protein_learning.networks.common.utils import exists
from typing import Dict, Optional, Tuple, Union
from einops.layers.torch import Rearrange


class PairDistLossNet(nn.Module):
    """
    This module is used to predict pairwise distances (for given atom types) from
    output pair features. The predictions are compared to the true distances and
    cross entropy loss is used to derive the final output.
    A shallow FeedForward network is used to obtain predicted distance logits.
    """

    def __init__(self, dim_in, atom_tys, step=0.4, d_min=2.5, d_max=20, use_hidden: bool = False):
        """

        :param dim_in: pair feature dimension
        :param atom_tys: atom types to compute loss for - should be given as
        a list [a_1,a_2,a_3,...,a_2k]. distances between atoms a_2i and a_{2i+1} will
        be predicted... i.e. the atom pairs are (a1,a2),...(a_{2k-1}, a_2k)
        :param step: step size between predicted distances
        :param d_min: minimum distance to predict
        :param d_max: maximum distance to predict
        :param use_hidden : whether to add a hidden layer in distance logit prediction network
        **An extra bin for distances greater than d_max is also appended***
        """
        super().__init__()
        assert len(atom_tys) % 2 == 0, f"must have even number of atom types, got: {atom_tys}"
        self.step, self.d_min, self.d_max = step, d_min, d_max
        self._bins = torch.arange(self.d_min, self.d_max + 2 * step, step=step)
        self.atom_ty_set = set(atom_tys)
        self.num_pair_preds = len(atom_tys) // 2
        self.atom_tys = [(x, y) for x, y in partition(atom_tys, chunk=2)]
        dim_hidden = self.num_pair_preds * self._bins.numel()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_hidden),
            nn.GELU() if use_hidden else nn.Identity(),
            nn.Linear(dim_hidden, dim_hidden) if use_hidden else nn.Identity(),
            Rearrange("b n m (p d) -> b n m p d", p=len(self.atom_tys))
        )
        self.loss_fn = softmax_cross_entropy

    def bins(self, device) -> Tensor:
        """Gets the bins used to define the predicted distances"""
        # makes sure devices match
        if self._bins.device == device:
            return self._bins
        self._bins = self._bins.to(device)
        return self._bins

    def _to_labels(self, dists: Tensor) -> Tensor:
        """Convert native distances to one-hot labels"""
        dists = torch.clamp(dists, self.d_min, self.d_max + self.step) - self.d_min
        labels = torch.round(dists / self.step).long()
        return F.one_hot(labels, num_classes=self._bins.numel())

    def _get_true_dists_n_masks(self, atom_ty_map: Dict[str, int],
                                atom_coords: Tensor,
                                atom_masks: Optional[Tensor] = None,
                                pair_mask: Optional[Tensor] = None,
                                ) -> Tuple[Tensor, Optional[Tensor]]:
        a1_a2_dists, a1_a2_masks = [], []
        with torch.no_grad():
            for (a1, a2) in self.atom_tys:
                a1_pos, a2_pos = atom_ty_map[a1], atom_ty_map[a2]
                a1_coords, a2_coords = atom_coords[:, :, a1_pos], atom_coords[:, :, a2_pos]

                # add mask
                if exists(atom_masks):
                    a1_mask, a2_mask = atom_masks[:, :, a1_pos], atom_masks[:, :, a2_pos]
                    a1_a2_mask = torch.einsum("b i, b j -> b i j", a1_mask, a2_mask)
                    a1_a2_mask = torch.logical_and(a1_a2_mask, pair_mask) if exists(pair_mask) else a1_a2_mask
                    a1_a2_masks.append(a1_a2_mask)

                a1_a2_dist = torch.cdist(a1_coords, a2_coords)
                a1_a2_dists.append(a1_a2_dist)
            full_dists = torch.cat([x.unsqueeze(-1) for x in a1_a2_dists], dim=-1)
            full_mask = None
            if exists(atom_masks):
                full_mask = torch.cat([x.unsqueeze(-1) for x in a1_a2_masks], dim=-1)
        return full_dists.detach(), full_mask

    def forward(
            self,
            pair_output: Tensor,
            atom_ty_map: Dict[str, int],
            atom_coords: Tensor,
            atom_masks: Optional[Tensor] = None,
            pair_mask: Optional[Tensor] = None,
            reduce: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        :param pair_output: Output pair features
        :param atom_ty_map: Dict mapping from atom type to atom position in coordinate tensor
        :param atom_coords: atom coordinates of shape (b,n,a,3) (where atom_ty_map[atom_ty] indexes a dimension)
        :param atom_masks: coordinate mask of shape (b,n,a)
        :param pair_mask: (Optional) mask indicating which pair features to predict distances for (b,n,n)
        :param reduce : whether to take mean of output or return raw scores
        :return: Average cross entropy loss of predictions over all atom types
        """
        assert atom_coords.ndim == 4
        # get predictions
        full_dists, full_mask = self._get_true_dists_n_masks(
            atom_ty_map=atom_ty_map,
            atom_coords=atom_coords,
            atom_masks=atom_masks,
            pair_mask=pair_mask,
        )
        labels = self._to_labels(full_dists)
        if reduce:
            return torch.mean(self.loss_fn(self.net(pair_output)[full_mask], labels[full_mask]))
        else:
            return self.loss_fn(self.net(pair_output), labels), full_mask

    def forward_from_output(self, output: ModelOutput, reduce: bool = True) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run forward from ModelOutput Object"""
        atom_tys = list(self.atom_ty_set)
        atom_ty_map = {a: i for i, a in enumerate(atom_tys)}
        pair_mask = getattr(output, "pair_mask", None)
        native_atom_coords = output.get_atom_coords(native=True, atom_tys=atom_tys).unsqueeze(0)
        native_atom_mask = output.get_atom_mask(native=True, atom_tys=atom_tys).unsqueeze(0)

        return self.forward(
            pair_output=output.pair_output,
            atom_ty_map=atom_ty_map,
            atom_coords=native_atom_coords,
            atom_masks=native_atom_mask,
            pair_mask=pair_mask,
            reduce=reduce
        )
