# fed_ner/metrics.py

import os
import csv
import numpy as np
import wandb
from flwr.common import Parameters

CSV_FILE = "fl_metrics.csv"

def init_csv():
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            csv.writer(f).writerow(["round", "precision", "recall", "f1", "communication_cost_bytes"])

def append_csv(round_num, precision, recall, f1, cost=0):
    with open(CSV_FILE, mode="a", newline="") as f:
        csv.writer(f).writerow([round_num, precision, recall, f1, cost])

def log_to_wandb(round_num, precision, recall, f1, cost=0):
    wandb.log({
        "round": round_num,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "communication_cost_bytes": cost,
    })

def parameters_to_ndarrays(parameters: Parameters):
    return [np.frombuffer(t, dtype=np.float32) if isinstance(t, (bytes, bytearray)) else np.array(t) for t in parameters.tensors]
