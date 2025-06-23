# utils.py

import numpy as np
import torch
import time


def weights_dict_to_list(weights_dict):
    return [weights_dict[k] for k in sorted(weights_dict.keys())]

def weights_list_to_dict(weights_list, model):
    keys = sorted(model.state_dict().keys())
    if len(weights_list) != len(keys):
        raise ValueError(f"Weights list length {len(weights_list)} doesn't match model params count {len(keys)}")
    return {k: w for k, w in zip(keys, weights_list)}

def set_model_weights(model, weights):
    state_dict = model.state_dict()
    new_state_dict = {}

    for name, param in state_dict.items():
        if name not in weights:
            raise KeyError(f"Missing weight for key: {name}")

        weight_tensor = torch.tensor(weights[name])
        if weight_tensor.shape != param.shape:
            raise ValueError(
                f"Shape mismatch for parameter '{name}': expected {param.shape}, got {weight_tensor.shape}"
            )

        new_state_dict[name] = weight_tensor

    model.load_state_dict(new_state_dict)

def parameters_to_ndarrays(parameters):
    ndarrays = []
    for tensor in parameters.tensors:
        if isinstance(tensor, (bytes, bytearray)):
            arr = np.frombuffer(tensor, dtype=np.float32)
        else:
            arr = np.array(tensor)
        ndarrays.append(arr)
    return ndarrays

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = (self.end - self.start) * 1000  # milliseconds
