import traceback
from collections import defaultdict

import numpy as np
import torch

from my_recipe.mydatasets.anno_raw import AnnoRawDataset, MultiRawDataset

__all__ = ["AnnoRawDataset", "MultiRawDataset", "safe_collate_fn"]


def safe_collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)
    try:
        for key, val in tensors.items():
            tensors[key] = torch.stack(val, dim=0)

        # for key, val in non_tensors.items():
        #     non_tensors[key] = np.array(val, dtype=object)
        #
        # @zhonghao: original impl will stack as [[PIL.Image], [PIL.Image]],
        # to avoid this, create empty array and assign values afterwards
        #
        for key, val in non_tensors.items():
            arr = np.empty(len(val), dtype=object)
            arr[:] = val
            non_tensors[key] = arr

    except Exception as e:
        traceback.print_exc()
        raise e

    return {**tensors, **non_tensors}
