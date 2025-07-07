# utils.py

import numpy as np
import wandb
import torch
import time
import os
import csv
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset

def weights_dict_to_list(weights_dict):
    return [val.cpu().numpy() if isinstance(val, torch.Tensor) else val for val in weights_dict.values()]

def weights_list_to_dict(weights_list, model):
    keys = list(model.state_dict().keys())
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


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = (self.end - self.start) * 1000  # milliseconds

def append_timing(round_num: int, fwd_time: float, bwd_time: float, path):
    header = ["round", "forward_time_ms", "backward_time_ms"]
    file_exists = os.path.exists(path)

    with open(path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "round": round_num,
            "forward_time_ms": round(fwd_time, 4),
            "backward_time_ms": round(bwd_time, 4),
        })


def init_csv(csv_path):
    if not os.path.isfile(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "accuracy", "f1", "communication_cost_bytes"])

def append_csv(round_num, accuracy, f1_score, communication_cost, csv_path):
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([round_num, accuracy, f1_score, communication_cost])

def log_metrics_wandb(rnd, accuracy, f1, communication_cost):
    wandb.log({
        "round": rnd,
        "accuracy": accuracy,
        "f1": f1,
        "communication_cost_bytes": communication_cost,
    })

def log_timing_wandb(round_num, avg_fwd, avg_bwd):
    wandb.log({
        "round": round_num,
        "client_forward_ms": avg_fwd,
        "server_backward_ms": avg_bwd,
    })
