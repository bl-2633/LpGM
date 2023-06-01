"""Encoder/Decoder Base Classes"""
from torch import nn, Tensor
from abc import abstractmethod
from typing import Tuple, Dict, Any


class EncoderABC(nn.Module):
    """Encoder Base class"""

    def __init__(self):
        super(EncoderABC, self).__init__()

    def foward(
            self,
            scalar_feats: Tensor,
            pair_feats: Tensor,
            coords: Tensor,
            mask: Tensor,
            **kwargs,
    ):
        """Calls encode"""
        return self.encode(
            scalar_feats=scalar_feats,
            pair_feats=pair_feats,
            coords=coords,
            mask=mask,
            **kwargs
        )

    @abstractmethod
    def encode(
            self,
            scalar_feats: Tensor,
            pair_feats: Tensor,
            coords: Tensor,
            mask: Tensor,
            **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Encode a sample
        :param scalar_feats: scalar features sampled from latent distribution
        (b,n,d_latent)
        :param pair_feats: pair features of shape (b,n,n,d_pair)
        :param coords: coordinates of shape (b,n,a,3)
        :param mask: mask of shape (b,n)
        :param kwargs: additional key word arguments
        :return: mean and standard deviation for latent distribution
        """
        pass


class DecoderABC(nn.Module):
    """Decoder base class"""

    def __init__(self):
        super(DecoderABC, self).__init__()

    def foward(
            self,
            scalar_feats: Tensor,
            pair_feats: Tensor,
            coords: Tensor,
            mask: Tensor,
            **kwargs,
    ):
        """Calls decode"""
        return self.decode(
            scalar_feats=scalar_feats,
            pair_feats=pair_feats,
            coords=coords,
            mask=mask,
            **kwargs
        )

    @abstractmethod
    def get_input_from_encoder_input(
            self,
            latent_scalars: Tensor,
            pair_feats: Tensor,
            coords: Tensor,
            mask: Tensor,
            **kwargs
    ) -> Dict[str, Any]:
        """Determine what to provide as input to decode

        should return a dictionary with keys and values for each input parameter
        in encode. the choice of input pair and coordinate features is left to the
        implementing class. latent scalars should probably be passed back unchanged.
        """
        pass

    @abstractmethod
    def decode(
            self,
            scalar_feats: Tensor,
            pair_feats: Tensor,
            coords: Tensor,
            mask: Tensor,
            **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Decode a sample
        :param scalar_feats: scalar features sampled from latent distribution
        (b,n,d_latent)
        :param pair_feats: pair features of shape (b,n,n,d_pair)
        :param coords: coordinates of shape (b,n,a,3)
        :param mask: mask of shape (b,n)
        :param kwargs: additional key word arguments
        :return: decoded sample
        """
        pass
