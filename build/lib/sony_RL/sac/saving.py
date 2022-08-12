import os

import haiku as hk
import numpy as np


def save_params(params, path):
    """
    Save parameters.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    np.savez(path, **params)


def load_params(path):
    """
    Load parameters.
    for some reason, numpy load convert all params to np array format not compatible with JAX
    """

    params = np.load(path, allow_pickle=True)
    new_params = {}
    for files in params.files:
        new_params[files] = params[files][()]
    return new_params
    #return hk.data_structures.to_immutable_dict(np.load(path,allow_pickle=True))