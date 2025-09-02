import numpy as np
import torch
from collections import OrderedDict
from typing import List

def get_parameters(model, include_bn_stats=True):
    """
    Get ALL model parameters for federated aggregation.
    
    Args:
        model: PyTorch model
        include_bn_stats: Always True now - include everything except counters
    """
    # Simple approach: get ALL parameters except num_batches_tracked (it's a counter)
    return [val.cpu().numpy() for name, val in model.state_dict().items() 
            if 'num_batches_tracked' not in name]

def set_parameters(model, parameters: List[np.ndarray], include_bn_stats=True) -> None:
    """
    Set ALL model parameters from federated aggregation.
    
    Args:
        model: PyTorch model  
        parameters: List of parameter arrays including BatchNorm stats
        include_bn_stats: Always True now - expect everything except counters
    """
    # Get parameter names (exclude num_batches_tracked)
    keys = [k for k in model.state_dict().keys() if 'num_batches_tracked' not in k]
    
    if len(keys) != len(parameters):
        raise ValueError(f"Parameter count mismatch: model has {len(keys)} params, "
                        f"received {len(parameters)}")
    
    # Build new state dict
    new_state_dict = OrderedDict()
    for name, param in zip(keys, parameters):
        new_state_dict[name] = torch.tensor(param)
    
    # Keep num_batches_tracked from current model
    for name in model.state_dict().keys():
        if 'num_batches_tracked' in name:
            new_state_dict[name] = model.state_dict()[name]
    
    model.load_state_dict(new_state_dict, strict=True)