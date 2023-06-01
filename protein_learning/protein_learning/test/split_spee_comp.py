"""Faster to use k linear projections or one and then split?"""

import torch
from torch import nn, Tensor, tensor, tensor_split # noqa
import time
from torch.optim.adam import Adam
import numpy as np


def exists(x):
    """Returns true iff x is not None.
    """
    return x is not None


def default(x, y):
    """Returns x if x exists, otherwise y.
    """
    return x if x is not None else y


class SplitLinear(nn.Module):
    def __init__(self, dim_in, dim_out, chunks=1, sizes=None):
        super(SplitLinear, self).__init__()
        sizes = tensor(default(sizes, [dim_out // chunks] * chunks))
        assert sum(sizes) == dim_out
        self.sizes = torch.cumsum(sizes, dim=0).long()[:-1]
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        out = self.linear(x)
        return tensor_split(out, self.sizes, dim=-1)


class UnSplitLinear(nn.Module):
    def __init__(self, dim_in, dim_out, chunks=1, sizes=None):
        super(UnSplitLinear, self).__init__()
        sizes = tensor(default(sizes, [dim_out // chunks] * chunks))
        assert sum(sizes) == dim_out
        self.lins = nn.ModuleList(
            [
                nn.Linear(dim_in, s) for s in sizes
            ]
        )

    def forward(self, x):
        return [linr(x) for linr in self.lins]


if __name__ == "__main__":
    N = 1000
    loss = lambda *xs: sum([torch.norm(y) for y in xs])
    device = 'cuda:0'
    dim_in = 350
    dim_out = 2000
    sizes = [200, 400, 600, 200, 400, 200]
    lin_split = SplitLinear(dim_in, dim_out, sizes=sizes).to(device)
    lin_unsplit = UnSplitLinear(dim_in, dim_out, sizes=sizes).to(device)
    optim_split = Adam(lin_split.parameters())
    optim_unsplit = Adam(lin_unsplit.parameters())
    #initialize
    x = torch.randn(dim_in).to(device)
    l_split = loss(*lin_split(x))
    l_unsplit = loss(*lin_unsplit(x))
    l_split.backward()
    l_unsplit.backward()
    optim_split.step()
    optim_unsplit.step()

    start = time.time()
    for _ in range(N):
        x = torch.randn(dim_in).to(device)
        l_split = loss(*lin_split(x))
        l_split.backward()
        optim_split.step()
    print("split : ",np.round(time.time()-start,2))

    start = time.time()
    for _ in range(N):
        x = torch.randn(dim_in).to(device)
        l_unsplit = loss(*lin_unsplit(x))
        l_unsplit.backward()
        optim_unsplit.step()
    print("unsplit : ", np.round(time.time() - start, 2))

