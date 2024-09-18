# Parallel nn.Module, copied from
# https://github.com/pytorch/pytorch/issues/36459

from typing import Callable, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor

class Parallel(nn.ModuleList):
    """Runs modules in parallel on the same input and merges their results."""

    def __init__(self, *modules: nn.Module, merge: Union[str, Callable] = "sum"):
        """Runs modules in parallel on the same input and merges their results.

        Args:
            merge: operation for merging list of results (default: `"sum"`)
        """
        super().__init__(modules)
        self.merge = create_merge(merge)

    def forward(self, x: Tensor) -> Tensor:
        return self.merge([module(x) for module in self])


MERGE_METHODS: Dict[str, Callable] = {
    "cat": lambda xs: torch.cat(xs, dim=1),
    "sum": lambda xs: sum(xs),  # type: ignore
    "prod": lambda xs: reduce(lambda x, y: x * y, xs),  # type: ignore
}


def create_merge(merge: Union[str, Callable]) -> Callable:
    return MERGE_METHODS[merge] if isinstance(merge, str) else merge

'''
def resblock(...):
    return Parallel(
        nn.Identity(),
        nn.Sequential(
            nn.Conv2d(...),
            ...,
        ),
        merge="sum",  # Can also be "prod" or "cat". Default is "sum".
    )
'''
