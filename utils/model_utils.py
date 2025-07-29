import numpy as np
import torch
from collections import OrderedDict
from typing import List

def get_parameters(model):
    # if we want to return only the optimized parameters
    # return [val.detach().cpu().numpy() for name, val in model.named_parameters() if 'weight'  in name or 'bias' in name]

    # excluding parameters of BN layers when using FedBN
    return [val.cpu().numpy() for name, val in model.state_dict().items() if 'bn' not in name]

def set_parameters(model, parameters: List[np.ndarray]) -> None:
    # Set model parameters from a list of NumPy ndarrays
    keys = [k for k in model.state_dict().keys() if 'bn' not in k]
    params_dict = zip(keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)