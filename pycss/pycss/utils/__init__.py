import numpy as np


def construct_map(vals1D, mask):
    """ Redistribute flattened data to ND array according to mask """
    map_ = np.zeros(mask.shape, dtype=vals1D.dtype).ravel()
    map_[mask.ravel()] = vals1D
    return map_.reshape(mask.shape)
